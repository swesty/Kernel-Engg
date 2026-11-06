// Kernels 8 and 9: kernel 7 (block tiles, warp tiles, 8x8 thread patches) plus a STAGES-deep cp.async pipeline,
// and optionally register-level fragment prefetch (kernel 9).
// C = A @ B, FP32, all row-major. A is M x K, B is K x N, C is M x N.
// Requires M % BM == 0, N % BN == 0, K % BK == 0, and 16-byte-aligned rows (K % 4 == 0, N % 4 == 0).
//
// Compile-time knobs (prepended as #defines by matmul08.py):
//   BM BN BK WM WN TM TN   tile shapes, as in kernel 7
//   STAGES                 shared-memory ring depth. 1 = no overlap (copy, wait, compute), 3 = cuBLAS's choice
//   ATRANS   (default 0)   1 = store the A tile transposed, [BK][BM+4], by 4-byte cp.async copies, so one k-step's
//                          8 A values per thread are two 128-bit shared loads instead of eight 32-bit ones
//   PREFETCH (default 0)   1 = kernel 9: two register buffers per operand. Step k+1's fragments are loaded from
//                          shared memory while step k's 64 FMAs run (CUTLASS mma_multistage.h, mac_loop_iter).
//                          Needs STAGES >= 2.
//   HOIST    (default 0)   1 = kernel 11 (needs PREFETCH=1): every copy's global source pointer and shared-memory
//                          offset is computed once per thread, then advanced by a constant stride per chunk; ring slots
//                          advance by add-and-wrap instead of t % STAGES; the shared base is converted to a 32-bit
//                          address once. Profiled at n = 4,096, kernel 10's hot loop ran ~200 LEA/IMAD/LDC/IADD3 per
//                          chunk recomputing these (cuBLAS: none of LEA/IMAD/LDC), and they carried most wait stalls.
//   SPREAD   (default 0)   1 = kernel 12 (needs HOIST=1): the next chunk's copies are spread over the BK k-steps of
//                          the current chunk (CUTLASS mma_multistage.h does this) instead of one burst at the chunk
//                          boundary, and each copy address is base + it*stride instead of a running pointer. Profiled
//                          kernel 11: its 16 A copies compiled to a chain of 16 dependent IMAD.WIDE, issued in a burst,
//                          which carried the remaining wait stalls (LDS 41%, LDGSTS 23%, IMAD 16% of them).
//            2 = kernel 13: the same, but branch-free. Every copy is always issued; for a chunk past the end the
//                          copy size is 0 (zero-fill, no bytes read) and the source is clamped to a valid address.
//                          Profiled kernel 12: its per-k-step `if (copy)` made ptxas emit ~24 `@!PT LDS RZ,[RZ]`
//                          filler instructions and ~25 `ULDC`+`MOV` parameter reloads per chunk (+3% instructions).
//                          cuBLAS's SASS uses predicated copies (`LDGSTS ... P0`) instead.
//   FIXED_N  (default 0)   n > 0 = compile for one square size: M = N = K = n become constants, so every address
//                          offset folds into an immediate. cuBLAS cannot do this (one binary, any size): a result with
//                          FIXED_N is a size-specialised kernel and must be reported as such.
//   ORDER    (default 0)   0 = outer product i-outer (row of A x all of B), 1 = j-outer. Changes which FFMA operands
//                          repeat back to back, i.e. register-bank / operand-reuse behaviour (the dispatch stall).
//   BSTRIDED (default 1)   1 = B columns strided by TCOLS*4 (from kernel 10 on); 0 = contiguous h*TN columns, the
//                          layout kernels 8 and 9 were measured with. Lets the older profiles be rebuilt exactly.
//   SWIZZLE  (default 0)   G > 0 = launch blocks in groups of G block-rows (Triton's GROUP_M). Blocks run in roughly
//                          linear-id order; in plain row order, consecutive blocks share one A row-panel but each needs
//                          a different B column-panel, so at n = 4,096 (64 MB per matrix, 50 MB L2) panels are evicted
//                          before reuse. Grouping G rows makes consecutive blocks share both A and B panels.
//
// kernel 7: global -> registers -> shared (LDG then STS), __syncthreads, compute, __syncthreads, repeat.
// kernel 8: cp.async (SASS LDGSTS) copies global -> shared directly and without blocking; STAGES-1 chunks in flight.
//           Profiled at n = 1,024 on the H100, that cut the wait on global memory from 47% to 2.5% of warp cycles, but
//           warps still stalled 0.38 cycles per instruction on shared-memory results (short scoreboard) vs 0.10 for
//           cuBLAS. With one warp per scheduler, nothing else hides those ~30-cycle loads.
// kernel 9: hide them inside the warp: load the next fragment before using the current one.

#ifndef ATRANS
#define ATRANS 0
#endif
#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef SWIZZLE
#define SWIZZLE 0
#endif
#ifndef HOIST
#define HOIST 0
#endif
#ifndef SPREAD
#define SPREAD 0
#endif
#ifndef ORDER
#define ORDER 0
#endif
#ifndef BSTRIDED
#define BSTRIDED 1
#endif
#ifndef FIXED_N
#define FIXED_N 0
#endif

#define NT ((BM / TM) * (BN / TN))      // threads per block
#define TCOLS (WN / TN)                 // thread columns in a warp tile
#define TROWS (WM / TM)                 // thread rows in a warp tile (TROWS * TCOLS == 32)
#if ATRANS
#define ASTRIDE (BM + 4)                // A tile is [BK][BM+4]: +4 keeps rows 16-B aligned and spreads the 4-byte stores
#define A_TILE (BK * ASTRIDE)
#else
#define ASTRIDE (BK + 4)                // A tile is [BM][BK+4], copied as-is in 16-B pieces (cp.async can't transpose)
#define A_TILE (BM * ASTRIDE)
#endif
#define B_TILE (BK * BN)                // B tile is [BK][BN]
#define STAGE_FLOATS (A_TILE + B_TILE)

// ---------------------------------------------------------------------------------------------- cp.async primitives
// cp.async.cg: copy 16 bytes global -> shared, caching in L2 only. The thread does not wait for it.
__device__ __forceinline__ void cp_async16(float* smem, const float* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
// cp.async.ca: copy 4 bytes (.cg only allows 16). Used to scatter A into its transposed layout.
__device__ __forceinline__ void cp_async4(float* smem, const float* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
}
// Close the copies this thread issued so far into one "group".
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// Block this thread until at most N of its groups are still in flight.
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// ---------------------------------------------------------------------------------------- one chunk of K into a stage
// A and B already point at this block's first row / first column. k0 is the chunk's first column of A (row of B).
__device__ __forceinline__ void load_chunk(float* stage, const float* A, const float* B, int K, int N, int k0, int tid) {
    float* As = stage;
    float* Bs = stage + A_TILE;
#if ATRANS
    // BM x BK single floats. Lane l of a warp takes row r = c / BK, column k = c % BK: a warp reads 4 rows x 32 B
    // (whole sectors) and stores to banks (4k + r) mod 32, all different, because the row stride is BM + 4.
    #pragma unroll
    for (int it = 0; it < (BM * BK + NT - 1) / NT; ++it) {
        int c = tid + it * NT;
        if (c < BM * BK) {
            int r = c / BK, k = c % BK;
            cp_async4(As + k * ASTRIDE + r, A + (size_t)r * K + k0 + k);
        }
    }
#else
    // BM x BK floats = BM*BK/4 copies of 16 B, spread over NT threads (a fixed trip count, so it unrolls).
    #pragma unroll
    for (int it = 0; it < (BM * BK / 4 + NT - 1) / NT; ++it) {
        int c = tid + it * NT;
        if (c < BM * BK / 4) {
            int r = c / (BK / 4), q = c % (BK / 4);
            cp_async16(As + r * ASTRIDE + q * 4, A + (size_t)r * K + k0 + q * 4);
        }
    }
#endif
    #pragma unroll
    for (int it = 0; it < (BK * BN / 4 + NT - 1) / NT; ++it) {   // BK x BN floats
        int c = tid + it * NT;
        if (c < BK * BN / 4) {
            int r = c / (BN / 4), q = c % (BN / 4);
            cp_async16(Bs + r * BN + q * 4, B + (size_t)(k0 + r) * N + q * 4);
        }
    }
}

// ------------------------------------------------------------------------------------- which rows a thread owns
// ATRANS = 0: rows strided by TROWS (wr + g, wr + g + TROWS, ...). With A stored [BM][BK+4], contiguous rows would put
//             all 8 row groups of a warp in one bank; strided rows land g * (BK+4) apart, 8 different banks.
// ATRANS = 1: rows contiguous (wr + g*TM ... + TM-1), so a k-step's 8 values are adjacent: two 128-bit loads.
__device__ __forceinline__ int thread_row(int wr, int g, int i) {
#if ATRANS
    return wr + g * TM + i;
#else
    return wr + g + i * TROWS;
#endif
}

// Columns: TN/4 float4 pieces, strided by TCOLS*4 (wc + h*4, wc + h*4 + TCOLS*4, ...). The 8 lanes that share a
// 128-bit load phase then read 8 different 16-B bank groups. With contiguous columns (h*TN) and TCOLS = 8, lanes h and
// h+4 would land TN*4 floats = 32 floats apart, the same bank group: a 2-way conflict.
__device__ __forceinline__ int thread_col(int wc, int h, int j) {      // j = 0, 4, 8, ... < TN
#if BSTRIDED
    return wc + h * 4 + (j / 4) * (TCOLS * 4);
#else
    return wc + h * TN + j;                                             // kernels 8 and 9 as measured
#endif
}

// One k-step's fragments: TM values of A (this thread's rows) and TN values of B (this thread's columns).
__device__ __forceinline__ void load_frag(const float* stage, int kk, float (&a)[TM], float (&b)[TN],
                                          int wr, int wc, int g, int h) {
    const float* As = stage;
    const float* Bs = stage + A_TILE;
#if ATRANS
    #pragma unroll
    for (int i = 0; i < TM; i += 4) {
        float4 v = *reinterpret_cast<const float4*>(As + kk * ASTRIDE + wr + g * TM + i);
        a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
    }
#else
    #pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = As[thread_row(wr, g, i) * ASTRIDE + kk];
#endif
    #pragma unroll
    for (int j = 0; j < TN; j += 4) {
        float4 v = *reinterpret_cast<const float4*>(Bs + kk * BN + thread_col(wc, h, j));
        b[j] = v.x; b[j + 1] = v.y; b[j + 2] = v.z; b[j + 3] = v.w;
    }
}

__device__ __forceinline__ void outer_product(float (&acc)[TM][TN], const float (&a)[TM], const float (&b)[TN]) {
#if ORDER == 0
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
#else
    #pragma unroll
    for (int j = 0; j < TN; ++j)
        #pragma unroll
        for (int i = 0; i < TM; ++i) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
#endif
}

// Kernel 8's inner loop: load a k-step's fragments, then use them (the compiler schedules what it can).
__device__ __forceinline__ void compute_chunk(const float* stage, float (&acc)[TM][TN], int wr, int wc, int g, int h) {
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        float a[TM], b[TN];
        load_frag(stage, kk, a, b, wr, wc, g, h);
        outer_product(acc, a, b);
    }
}

#if HOIST
// ------------------------------------------------------------------------------- kernel 11: the copy plan, hoisted
// The mapping is the same as load_chunk's (copy c = tid + it*NT), rewritten so that everything that does not change
// from chunk to chunk is computed once. Requires NT to divide the copy counts and the copy-row widths (checked on the
// host): then a thread's column inside the tile is fixed and its row advances by a constant per copy.
#if ATRANS
#define A_COPIES (BM * BK / NT)                 // 4-byte copies of A per thread per chunk
#define A_ROWSTEP (NT / BK)                     // tile rows between a thread's consecutive copies
#define A_SSTEP (A_ROWSTEP * 4)                 // bytes between them in shared memory ([BK][BM+4]: rows are adjacent)
#else
#define A_COPIES (BM * BK / 4 / NT)             // 16-byte copies of A per thread per chunk
#define A_ROWSTEP (NT / (BK / 4))
#define A_SSTEP (A_ROWSTEP * ASTRIDE * 4)
#endif
#define B_COPIES (BK * BN / 4 / NT)
#define B_ROWSTEP (NT / (BN / 4))
#define B_SSTEP (B_ROWSTEP * BN * 4)
#define STAGE_BYTES (STAGE_FLOATS * 4)

struct CopyPlan {
    const float* a;        // this thread's first A source for the next chunk to issue
    const float* b;        // same for B
    size_t a_row;          // floats between a thread's consecutive A copies in global memory (A_ROWSTEP rows of K)
    size_t b_row;          // B_ROWSTEP rows of N
    size_t b_chunk;        // BK rows of N: how far B moves per chunk (A moves BK floats)
    unsigned sa, sb;       // byte offsets of this thread's first A and B copy inside a stage
};

__device__ __forceinline__ void cp_async16_s(unsigned s, const float* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async4_s(unsigned s, const float* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(g));
}

__device__ __forceinline__ CopyPlan make_plan(const float* A, const float* B, int K, int N, int tid) {
    CopyPlan P;
#if ATRANS
    const int ar = tid / BK, ak = tid % BK;
    P.a = A + (size_t)ar * K + ak;
    P.sa = (unsigned)(ak * ASTRIDE + ar) * 4u;
#else
    const int ar = tid / (BK / 4), aq = tid % (BK / 4);
    P.a = A + (size_t)ar * K + aq * 4;
    P.sa = (unsigned)(ar * ASTRIDE + aq * 4) * 4u;
#endif
    const int br = tid / (BN / 4), bq = tid % (BN / 4);
    P.b = B + (size_t)br * N + bq * 4;
    P.sb = (unsigned)(A_TILE + br * BN + bq * 4) * 4u;
    P.a_row = (size_t)A_ROWSTEP * K;
    P.b_row = (size_t)B_ROWSTEP * N;
    P.b_chunk = (size_t)BK * N;
    return P;
}

// Issue the next chunk into the stage starting at shared byte address `slot`, then step the plan to the next chunk.
__device__ __forceinline__ void issue_chunk(unsigned slot, CopyPlan& P) {
    const float* pa = P.a;
    #pragma unroll
    for (int it = 0; it < A_COPIES; ++it) {
#if ATRANS
        cp_async4_s(slot + P.sa + it * A_SSTEP, pa);      // it * A_SSTEP is a compile-time immediate
#else
        cp_async16_s(slot + P.sa + it * A_SSTEP, pa);
#endif
        pa += P.a_row;
    }
    const float* pb = P.b;
    #pragma unroll
    for (int it = 0; it < B_COPIES; ++it) {
        cp_async16_s(slot + P.sb + it * B_SSTEP, pb);
        pb += P.b_row;
    }
    P.a += BK;
    P.b += P.b_chunk;
}

#if SPREAD
// Kernel 12: copy number c of a chunk (A copies first, then B), addressed from the chunk's base, no running pointer.
#define TOTAL_COPIES (A_COPIES + B_COPIES)
#define COPIES_PER_STEP ((TOTAL_COPIES + BK - 1) / BK)
__device__ __forceinline__ void issue_copy(unsigned slot, const CopyPlan& P, int c) {   // c is a compile-time constant
    if (c < A_COPIES) {
#if ATRANS
        cp_async4_s(slot + P.sa + c * A_SSTEP, P.a + c * P.a_row);
#else
        cp_async16_s(slot + P.sa + c * A_SSTEP, P.a + c * P.a_row);
#endif
    } else {
        cp_async16_s(slot + P.sb + (c - A_COPIES) * B_SSTEP, P.b + (c - A_COPIES) * P.b_row);
    }
}
#if SPREAD == 2
// cp.async with a source size: bytes past src_size are zero-filled; src_size = 0 reads nothing.
__device__ __forceinline__ void cp_async16_z(unsigned s, const float* g, unsigned src_size) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(g), "r"(src_size));
}
__device__ __forceinline__ void cp_async4_z(unsigned s, const float* g, unsigned src_size) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(g), "r"(src_size));
}
__device__ __forceinline__ void issue_copy_z(unsigned slot, const float* pa, const float* pb, const CopyPlan& P, int c,
                                             unsigned live) {   // live = 1 if the chunk exists, else 0
    if (c < A_COPIES) {
#if ATRANS
        cp_async4_z(slot + P.sa + c * A_SSTEP, pa + c * P.a_row, 4u * live);
#else
        cp_async16_z(slot + P.sa + c * A_SSTEP, pa + c * P.a_row, 16u * live);
#endif
    } else {
        cp_async16_z(slot + P.sb + (c - A_COPIES) * B_SSTEP, pb + (c - A_COPIES) * P.b_row, 16u * live);
    }
}
#endif
#endif
#endif

extern "C" __global__ void __launch_bounds__(NT)
matmul_cp_async(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    extern __shared__ __align__(16) float smem[];          // STAGES * STAGE_FLOATS floats, sized at launch
#if FIXED_N
    M = FIXED_N; N = FIXED_N; K = FIXED_N;                  // size-specialised build: sizes become constants
#endif

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wr = (warp / (BN / WN)) * WM;                 // warp tile's first row inside the block tile
    const int wc = (warp % (BN / WN)) * WN;                 // warp tile's first column
    const int g = lane / TCOLS, h = lane % TCOLS;           // this lane's row group and column group

#if SWIZZLE
    // Linear launch id -> (block row, block col), walking down G block-rows before moving one column right.
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int per_group = SWIZZLE * gridDim.x;
    const int first_row = (bid / per_group) * SWIZZLE;
    const int rows = min((int)gridDim.y - first_row, SWIZZLE);   // the last group may be shorter
    const int brow = first_row + (bid % per_group) % rows;
    const int bcol = (bid % per_group) / rows;
#else
    const int brow = blockIdx.y, bcol = blockIdx.x;
#endif
    A += (size_t)brow * BM * K;
    B += (size_t)bcol * BN;
    C += (size_t)brow * BM * N + (size_t)bcol * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int chunks = K / BK;

#if STAGES == 1
    // Baseline: same instructions, no overlap. Copy, wait for it, compute, repeat (kernel 7's schedule).
    for (int t = 0; t < chunks; ++t) {
        load_chunk(smem, A, B, K, N, t * BK, tid);
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();                                    // everyone's copies have landed
        compute_chunk(smem, acc, wr, wc, g, h);
        __syncthreads();                                    // everyone is done reading before the next copy overwrites
    }
#elif !HOIST
    // Prologue: put chunks 0 .. STAGES-2 in flight. Always commit, even an empty group, so the group count stays exact.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < chunks) load_chunk(smem + s * STAGE_FLOATS, A, B, K, N, s * BK, tid);
        cp_async_commit();
    }
#if !PREFETCH
    for (int t = 0; t < chunks; ++t) {
        // Groups committed so far: STAGES-1+t (chunks 0 .. t+STAGES-2). Allowing STAGES-2 to stay in flight means
        // chunk t has landed, for this thread's copies.
        cp_async_wait<STAGES - 2>();
        // One barrier does two jobs: (1) every thread's copies of chunk t are now visible, and (2) every thread has
        // finished computing on chunk t-1, so its slot ((t-1) % STAGES) is free to refill.
        __syncthreads();
        const int next = t + STAGES - 1;                    // goes into slot next % STAGES == (t-1) % STAGES
        if (next < chunks) load_chunk(smem + (next % STAGES) * STAGE_FLOATS, A, B, K, N, next * BK, tid);
        cp_async_commit();
        compute_chunk(smem + (t % STAGES) * STAGE_FLOATS, acc, wr, wc, g, h);   // overlaps the copies in flight
    }
#else
    // Kernel 9. Registers hold two fragments: buffer kk % 2 is being multiplied while buffer (kk+1) % 2 is loading.
    // The wait + barrier that kernel 8 does at the top of each chunk moves to the LAST k-step of the previous chunk,
    // so the next chunk's first fragment can be loaded before that chunk starts (CUTLASS does the same).
    float a[2][TM], b[2][TN];

    // Chunk 0 must be in shared memory before its first fragment can be read.
    cp_async_wait<STAGES - 2>();                            // committed: chunks 0 .. STAGES-2; chunk 0 has landed
    __syncthreads();
    if (STAGES - 1 < chunks) load_chunk(smem + (STAGES - 1) * STAGE_FLOATS, A, B, K, N, (STAGES - 1) * BK, tid);
    cp_async_commit();                                      // committed: chunks 0 .. STAGES-1
    load_frag(smem, 0, a[0], b[0], wr, wc, g, h);

    for (int t = 0; t < chunks; ++t) {
        const float* stage = smem + (t % STAGES) * STAGE_FLOATS;
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            if (kk < BK - 1) {
                load_frag(stage, kk + 1, a[(kk + 1) % 2], b[(kk + 1) % 2], wr, wc, g, h);
            } else if (t + 1 < chunks) {
                // Last k-step of chunk t. Its fragment is already in registers, so this thread is done reading slot
                // t % STAGES. Committed so far: chunks 0 .. t+STAGES-1; allowing STAGES-2 in flight means chunk t+1
                // has landed. The barrier makes chunk t+1 visible AND proves every thread is done with slot t % STAGES,
                // which chunk t+STAGES refills.
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                const int next = t + STAGES;
                if (next < chunks) load_chunk(smem + (next % STAGES) * STAGE_FLOATS, A, B, K, N, next * BK, tid);
                cp_async_commit();
                load_frag(smem + ((t + 1) % STAGES) * STAGE_FLOATS, 0, a[(kk + 1) % 2], b[(kk + 1) % 2],
                          wr, wc, g, h);
            }
            outer_product(acc, a[kk % 2], b[kk % 2]);       // runs while the loads above are in flight
        }
    }
#endif
#elif !SPREAD
    // Kernel 11: kernel 9's schedule, with the copy plan hoisted and the ring walked by add-and-wrap.
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    CopyPlan P = make_plan(A, B, K, N, tid);
    int issued = 0;                                          // chunks issued so far (they are issued in order)
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (issued < chunks) { issue_chunk(sbase + s * STAGE_BYTES, P); ++issued; }
        cp_async_commit();
    }
    float a[2][TM], b[2][TN];
    cp_async_wait<STAGES - 2>();                            // chunk 0 has landed
    __syncthreads();
    if (issued < chunks) { issue_chunk(sbase + (STAGES - 1) * STAGE_BYTES, P); ++issued; }
    cp_async_commit();
    load_frag(smem, 0, a[0], b[0], wr, wc, g, h);

    unsigned rd = 0, nx = STAGE_BYTES;                      // byte offsets of chunk t's slot and chunk t+1's slot
    for (int t = 0; t < chunks; ++t) {
        const float* stage = smem + rd / 4;
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            if (kk < BK - 1) {
                load_frag(stage, kk + 1, a[(kk + 1) % 2], b[(kk + 1) % 2], wr, wc, g, h);
            } else if (t + 1 < chunks) {
                // Same argument as kernel 9: chunk t+1 has landed, and slot rd (chunk t's) is free for chunk t+STAGES.
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                if (issued < chunks) { issue_chunk(sbase + rd, P); ++issued; }
                cp_async_commit();
                load_frag(smem + nx / 4, 0, a[(kk + 1) % 2], b[(kk + 1) % 2], wr, wc, g, h);
            }
            outer_product(acc, a[kk % 2], b[kk % 2]);
        }
        rd = nx;
        nx = (nx + STAGE_BYTES == STAGES * STAGE_BYTES) ? 0u : nx + STAGE_BYTES;
    }
#else
    // Kernel 12. During chunk t, chunk t+STAGES-1 is copied into the slot chunk t-1 used, a few copies per k-step.
    // That slot is free: the barrier at the end of chunk t-1 came after every thread loaded chunk t-1's last fragment.
    // One commit per chunk, at its last k-step; then committed = chunks 0 .. t+STAGES-1, and wait<STAGES-2> means
    // chunk t+1 has landed. (STAGES = 2: the slot written is chunk t+1's own, which is the chunk being copied.)
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    CopyPlan P = make_plan(A, B, K, N, tid);
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {                  // prologue: chunks 0 .. STAGES-2, issued whole
        if (s < chunks) {
            #pragma unroll
            for (int c = 0; c < TOTAL_COPIES; ++c) issue_copy(sbase + s * STAGE_BYTES, P, c);
            P.a += BK; P.b += P.b_chunk;
        }
        cp_async_commit();
    }
    float a[2][TM], b[2][TN];
    cp_async_wait<STAGES - 2>();                            // chunk 0 has landed
    __syncthreads();
    load_frag(smem, 0, a[0], b[0], wr, wc, g, h);

    unsigned rd = 0;                                        // byte offset of chunk t's slot
    unsigned wr_slot = (STAGES - 1) * STAGE_BYTES;          // chunk t-1's slot (for t = 0: the unused last slot)
    for (int t = 0; t < chunks; ++t) {
        const float* stage = smem + rd / 4;
        const bool copy = t + STAGES - 1 < chunks;          // is there a chunk t+STAGES-1 to copy?
        const unsigned nx = (rd + STAGE_BYTES == STAGES * STAGE_BYTES) ? 0u : rd + STAGE_BYTES;
#if SPREAD == 2
        const unsigned live = copy ? 1u : 0u;               // branch-free: past the end, copy 0 bytes ...
        const float* pa = copy ? P.a : A;                   // ... from a clamped, valid address
        const float* pb = copy ? P.b : B;
#endif
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
#if SPREAD == 2
            #pragma unroll
            for (int c = kk * COPIES_PER_STEP; c < (kk + 1) * COPIES_PER_STEP && c < TOTAL_COPIES; ++c)
                issue_copy_z(sbase + wr_slot, pa, pb, P, c, live);
#else
            if (copy) {
                #pragma unroll
                for (int c = kk * COPIES_PER_STEP; c < (kk + 1) * COPIES_PER_STEP && c < TOTAL_COPIES; ++c)
                    issue_copy(sbase + wr_slot, P, c);
            }
#endif
            if (kk < BK - 1) {
                load_frag(stage, kk + 1, a[(kk + 1) % 2], b[(kk + 1) % 2], wr, wc, g, h);
            } else {
                cp_async_commit();
                if (t + 1 < chunks) {
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    load_frag(smem + nx / 4, 0, a[(kk + 1) % 2], b[(kk + 1) % 2], wr, wc, g, h);
                }
            }
            outer_product(acc, a[kk % 2], b[kk % 2]);
        }
        if (copy) { P.a += BK; P.b += P.b_chunk; }
        wr_slot = rd;                                       // chunk t's slot is chunk (t+1)-1's
        rd = nx;
    }
#endif

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        float* row = C + (size_t)thread_row(wr, g, i) * N;
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
            *reinterpret_cast<float4*>(row + thread_col(wc, h, j)) =
                make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
    }
}
