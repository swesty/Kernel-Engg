# Lab notebook: closing the gap between a hand-written FP32 GEMM and cuBLAS on the H100, and beating it at n = 1,024

Everything done in this folder, in order. Each experiment records:
- the question, the hypothesis, and the prediction made **before** the run;
- the change, the Modal run IDs, and the result files;
- the verdict.

Measured facts and inferences are labelled.

**Setup for every run:**
- **Hardware and software:** NVIDIA H100 80GB HBM3 (sm_90, 132 SMs) on Modal, with `GPU = "H100!"`. Assignment 1's image: CUDA 13.0, torch 2.14.1+cu130, CuPy 14.2 (NVRTC).
- **The problem:** C = A·B, FP32, row-major, square n ∈ {1,024, 2,048, 4,096}. **n is the side of every matrix:** n = 1,024 means a 1,024 × 1,024 A times a 1,024 × 1,024 B, giving a 1,024 × 1,024 C (M = N = K = n, `operands(n)` in `matmul08.py`).
  - The work is 2n³ FLOPs: 2.15 G at n = 1,024, 17.2 G at 2,048 and 137 G at 4,096.
  - Each matrix is 4n² bytes: 4 MB, 16 MB and 64 MB.
  - Only square sizes that are multiples of the tile were tested; there's no edge handling.
- **The baseline:** cuBLAS is `torch.matmul` with TF32 **off**, so both sides use FFMA on the CUDA cores.
- **Timing:** CUDA events, median of 30 launches, with the L2 flushed (256 MB of zeros) before each launch.
- **Correctness:** every kernel is checked against cuBLAS at n = 1,024 and 2,048 (relative error < 1e-4, output pre-filled with NaN).
- **Profiles:** Nsight Compute 2025 (`ncu --set full --import-source yes --clock-control none`), one launch per report.

**Where things are:**
- `results/runs/<tag>.{json,txt}` hold the timing sweeps.
- `results/runs/<tag>.code/` holds the exact code of a run, from k12 on.
- `results/ncu/<set>/` holds the Nsight Compute reports, each set with `summary.txt` and (from 03-rebuild on) `code/`.
- `results/analysis/` holds the scripts and per-SASS-instruction exports used for the source-level diagnoses.
- `results/MANIFEST.md` lists every file. `SCREENSHOTS.md` says which report and Nsight page goes with which figure.

**One code file, many kernels.** Every kernel below is a set of compile-time switches on `matmul_cp_async.cu`, so any earlier kernel can be rebuilt from the current file:
- `STAGES`, `ATRANS`, `PREFETCH`, `SWIZZLE`, `HOIST`, `SPREAD`, `ORDER`, `BSTRIDED`;
- the tile shapes `BM BN BK WM WN TM TN`.

Kernels 8 and 9 were measured with contiguous B columns, so the `/old` variants in `matmul08.py` (`BSTRIDED=0`) rebuild them exactly.

**Results at a glance** (best kernel at each step, % of cuBLAS speed; the cuBLAS time moves 1–2% between runs):

| Step | Kernel | n = 1,024 | n = 2,048 | n = 4,096 | Run |
|---|---|---|---|---|---|
| Book reference | kernel 7 (warp tiles) | 86.6 µs vs 56.1 (65%) | | | book Table 5.4 |
| E1 | k8: 3-stage `cp.async` pipeline | 89% | 81% | 81% | `fc-01M6M5YHA3EST0D4WSF73G1NDA` |
| E2 | k9: register fragment prefetch (+ transposed A) | 97% | 83% | 82% | `fc-01M6M6R5YXZ0T80VJMTWDKEFFV` |
| E3 | block swizzle (L2 reuse) | — | — | no gain (falsified) | profile only |
| E4 | k10: 16×8 / 8×16 thread patches | 92%* | 88% | 90% | `fc-01M6M76Q44PQJT6D8VHK8DSTH1` |
| E5 | k11: hoisted copy-address math | 98% | 94% | 95% | `fc-01M6M7PVKSCQWDCVQDE1CGS4E8` |
| E6 | k12: copies spread over k-steps; FFMA order | 100% | 96% | 95% | `fc-01M6M841CP8WZ8AHJBH6S65J8Z` |
| E7 | k13: branch-free spread; contiguous-B test; size-specialised | 98% | 96% (fixed n) | 96% (fixed n) | `fc-01M6M8PWZYX9QC5S6Z98WFH19B` |
| E8 | **k14: Stream-K (persistent, split leftover K-work)** | **107%** | 92% | 94% | `fc-01M6M8ZEJ5X1FG2STSC9Y57J28` |
| E9 | **confirmation, 15 alternating rounds** | **105.3%** (15/15 rounds faster) | 95.4% (k13f) | 96.4% (k13f) | `fc-01M6M946AGWF2FNH34AY7NK35X` |

**Headline (E9, the number to quote):** at n = 1,024, k14 (128×128 BK16, STAGES=2; generic, not size-specialised) runs in **61.3 µs against cuBLAS's 64.6 µs, 5.3% faster**. The per-round speed ranged from 105.0% to 105.8%, and it was faster in 15 of 15 rounds. At n = 2,048 and 4,096 the best kernels reach 95.4% and 96.4%. Those are size-specialised (`FIXED_N`); the generic ones are about 1 point lower.

\*The n = 1,024 dip at E4 coincides with the B-column remap (see E4).

---

## E0. The question

**Observed (book, saved in the Week 3-1 note):**
- Kernel 7 took 86.6 µs at n = 1,024 on the H100, against 56.1 µs for cuBLAS.
- Nsight showed equal shared-memory efficiency (0.19 passes per FMA), but very different time waiting on global memory: 43% for kernel 7, 7.5% for cuBLAS.
- The cuBLAS kernel is `sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x64x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4`.

**Hypothesis:** there's no secret sauce. `stage3` means three K-chunks in flight with asynchronous global→shared copies, which hide the latency that kernel 7 is stuck waiting on.

**Ruled out before building:**
- **CUDA streams:** they overlap separate kernels; a GEMM is one kernel.
- **Warp shuffles:** they're for reductions, not the GEMM main loop.
- **Bank-conflict swizzles:** passes per FMA already match cuBLAS.

## E1. Kernel 8: multi-stage `cp.async` pipeline

**Change:**
- `cp.async.cg.shared.global` 16-byte copies (SASS `LDGSTS`) into a ring of `STAGES` shared-memory slots.
- Each iteration does `wait_group(STAGES-2)`, then `__syncthreads()`, then issues chunk t+S−1 into chunk t−1's slot, then computes chunk t.
- `STAGES=1` gives the same instructions with no overlap, as the baseline.
- The A tile is stored un-transposed, padded to BK+4, with each thread's rows strided by 8 to avoid bank conflicts, because `cp.async` can't transpose on the way in.

**Prediction:** the global-memory wait drops sharply from S=1 to S=2, and S=3 lands within roughly 10–20% of cuBLAS.

**Runs:**
- `fc-01M6M5XC5PSTCPCBPMBKTCWS03` was accidentally placed on an **H200** by Modal (see "Gotchas"). Its results are `runs/k8-h200.json`, kept but not used.
- `fc-01M6M5YHA3EST0D4WSF73G1NDA` (H100): `runs/k8.json`.
- Profiles: `ncu/01-k8-pipeline-n1024/k8-s1` and `ncu/02-…/k8-s3`.

**Result (measured).** Profile at 128×64 BK8, n = 1,024:

| | S=1 | S=3 | cuBLAS |
|---|---|---|---|
| time (ncu) | 132.4 µs | 79.1 µs | 61.3 µs |
| waiting on global (long-scoreboard share of warp cycles) | 46.8% | **2.5%** | 6.6% |
| issue active | 28.8% | 52.0% | 69.8% |

- **Best timing:** 72.8 µs vs 64.9 µs (89%), at 128×64 BK8 S=4.
- **SASS (S=3):** 9 `LDGSTS`, 0 plain `LDG`.

**Verdict:** confirmed. Pipelining removes the global-memory wait entirely, to below cuBLAS's level. The remaining gap is issue rate, not memory.

**Next lead (measured, CPU-only read of the reports).** In cycles stalled per issued instruction, S=3 vs cuBLAS:
- short scoreboard: **0.38 vs 0.10**
- dispatch: 0.22 vs 0.03
- wait: 0.14 vs 0.04
- total: 1.92 vs 1.43

Instruction counts are equal (40.1 M vs 41.5 M), and so are shared loads (2.10 M vs 2.13 M). There's one warp per scheduler (128 blocks on 132 SMs), so nothing hides a warp's own shared-load latency.

## E2. Kernel 9: register-level fragment prefetch (CUTLASS `mma_multistage.h`, `mac_loop_iter`)

**Reference checked first.** I read Hugging Face's `kernel-builder/skills/cuda-kernels` and found it's about packaging and integration, not GEMM latency hiding. I confirmed in CUTLASS source that `mac_loop_iter` double-buffers warp fragments (`warp_loaded_frag_A_[(warp_mma_k + 1) % 2]`).

**Change:**
- **`PREFETCH=1`:** two register buffers per operand. Step k+1's fragments load from shared memory while step k's 64 FMAs run. The chunk's wait and barrier move to the last k-step of the previous chunk, so the next chunk's first fragment can load early.
- **`ATRANS=1`:** A is stored transposed, using 4-byte `cp.async.ca` copies with a padded stride so the stores are conflict-free. A thread's 8 A values for a k-step are then two 128-bit loads.
- Each switch was measured on its own and together.

**Prediction (stated before the run):** short scoreboard 0.38 → ~0.15, issue active above 60%, ncu time 68–70 µs.

**Run:** `fc-01M6M6R5YXZ0T80VJMTWDKEFFV`, giving `runs/k9.json`. Profiles: `ncu/02-k9-prefetch-n1024/`.

**Result (measured):**

| 128×64 BK8 S=3, n = 1,024 | time | issue | short_sb | dispatch | wait |
|---|---|---|---|---|---|
| k8 | 78.4 µs | 52.0% | 0.38 | 0.22 | 0.14 |
| k8+T (transpose only) | 89.2 µs | 51.0% | 0.39 | 0.13 | 0.25 |
| k9 (prefetch only) | 77.4 µs | 57.7% | 0.26 | 0.12 | 0.18 |
| **k9+T (both)** | **71.5 µs** | **62.4%** | **0.13** | 0.13 | 0.16 |
| cuBLAS | 61.3 µs | 68.7% | 0.10 | 0.03 | 0.04 |

- **Best timing:** 67.6 µs vs 65.3 µs (**97%**), k9+T at 128×64 BK16 S=3.
- At n = 2,048 and 4,096 there was no gain (79–83%).

**Verdict:** the prediction held (0.13 / 62.4% / 71.5 µs). The two changes only work **together**:
- A transposed A without prefetch still stalls on the loads.
- Prefetch without the transpose needs 8 scalar loads per fragment.

**Unexplained:** the `+T` variants are much slower at S=2 and S=4 than at S=3 with the same tile, for example 119 / 92 / 116 µs for k8+T 128×64 BK8.

## E3. Block swizzle: is the large-n gap L2 reuse?

**Hypothesis:** at n = 4,096 each matrix is 64 MB, larger than the 50 MB L2, and blocks launch in plain row order. So B panels are evicted before they're reused.

**Change:** `SWIZZLE=8` launches blocks in groups of 8 block-rows (like Triton's `GROUP_M`). I checked locally that the mapping covers every block exactly once.

**Prediction:** less DRAM traffic and higher speed.

**Profiles:**
- `ncu/03-swizzle-n4096/` is partial: its `n4k-k9` and cuBLAS reports were overwritten by later runs.
- `ncu/03-swizzle-n4096-rebuild/` is the complete set, rebuilt with the original code path.

**Result (measured, rebuild):**

| n = 4,096 | DRAM read | L2 hit | ncu time | waiting on global |
|---|---|---|---|---|
| k9 128×128 BK16 S=2 | 596 MB | 81.3% | 3.1 ms | 0.9% |
| k9+S (swizzled) | 463 MB | 85.6% | 3.5 ms | 1.8% |
| k9+T 128×64 BK16 S=3 | 963 MB | 76.9% | 3.5 ms | 2.7% |
| k9+T+S | 455 MB | 88.3% | 3.3 ms | 5.7% |
| cuBLAS | 342 MB | 89.9% | 2.6 ms | 0.9% |

**Verdict: falsified.** Swizzling cuts DRAM reads by 22–53% and raises the L2 hit rate, but the time doesn't improve, and the wait on global memory is ≤ 6% in every case. Inferred: at roughly 0.3 TB/s effective, against 3.35 TB/s of HBM bandwidth, memory isn't the limit at n = 4,096. The switch stays on (it doesn't hurt).

**Next lead (measured).** cuBLAS changes kernels at n = 4,096: 256 threads, grid 16×32 → a **128×256 tile**, 254 registers, so 128 outputs per thread to our 64. Its overhead instructions (anything that isn't FFMA) are 0.28 G against our 0.52 G.

## E4. Kernel 10: bigger thread patches

**Change:**
- New configs with 16×8 and 8×16 thread patches: 256×128 and 128×256 tiles, 256 threads, about 255 registers, no spills.
- **B-column remap (`BSTRIDED=1`).** With 8 thread columns, contiguous columns would put lanes h and h+4 in the same bank group, a 2-way conflict. Strided columns (`wc + h*4 + j*TCOLS*4`) are conflict-free.

**Run:** `fc-01M6M76Q44PQJT6D8VHK8DSTH1`, giving `runs/k10.json` and `.txt`. Profiles: `ncu/04-bigpatch-n4096/`, plus the k9TS 16×8 report in `ncu/05-…/` (same config and same code for that path).

**Result (measured):**
- n = 4,096: 2,962 µs vs 2,668 µs, **90%**. n = 2,048: 88%.
- At n = 4,096 the big patches match cuBLAS on shared loads (101 M), tile shape and occupancy (2 warps per scheduler).
- Remaining gap: instructions +3.5% (2,513 M vs 2,428 M) and issue rate −8% (83.4% vs 90.1%). 1.035 × 1.08 ≈ 1.12, which matches the time gap.

**Caveat:** k9+T 128×64 BK16 S=3 at n = 1,024 went from 67.6 to 70.7 µs between E2 and E4. That coincides with the remap. It could be the remap or noise; it hasn't been isolated. `k9+T/old` vs `k9+T` would isolate it.

**Source-level diagnosis (measured, `analysis/source-pages/05-n4k-big16x8.sass.csv`):**
- **Integer overhead.** We execute 82 M `LEA`, 56 M `IMAD`, 37 M `LDC` and 27 M `IADD3`, about 200 per 16-k-step chunk. cuBLAS executes 64 M `IADD3` plus about 40 M uniform-datapath ops (`UIADD3`, `UISETP`, `UMOV`, `UIMAD`), and no `LEA`, `IMAD` or `LDC` in its hot code.
- **Wait stalls:** 4.0% of samples vs 0.4%. They land on `LEA` 31%, `LDS` 23%, `LDC` 11%, `LDGSTS` 8%, `IMAD` 7%.
- **Inferred:** at 255 registers, the compiler re-reads `K`/`N` (`LDC`, constant memory) and recomputes the `cp.async` source addresses every chunk.

## E5. Kernel 11: hoist the copy-address math

**Change (`HOIST=1`):**
- `make_plan` computes each thread's copy source pointers and shared offsets once.
- `issue_chunk` advances them by constant strides; each shared offset is a compile-time immediate.
- The ring slots step by add-and-wrap instead of `t % STAGES`.
- The shared base is converted to a 32-bit address once.

**Prediction:**
- address math leaves the hot loop;
- overhead per FFMA 0.170 → ~0.14;
- wait stall → ~0.02;
- n = 4,096 at 94–95%.

**Run:** `fc-01M6M7PVKSCQWDCVQDE1CGS4E8`, giving `runs/k11.json` and `.txt`. Profiles: `ncu/05-k11-hoist-n4096/`.

**Result (measured):**
- **98% / 94% / 95%** at n = 1,024 / 2,048 / 4,096.
- Registers fell from 255 to 231 (16×8) and from 209 to 128 (128×64 BK16).
- At n = 4,096 (128×256 8×16): overhead per FFMA **0.075** (cuBLAS 0.131), ALU pipe 9.3% → 0.9%, instructions 2,308 M, which is **5% fewer than cuBLAS**.
- Issue active is 82.2% vs 90.1%, with dispatch 0.21 vs 0.10 and wait 0.08 vs 0.01.
- k11n, the same change without the transposed A, gains much less (88–94%). The 4-byte transposed copies carried most of the address math.

**Prediction scorecard:**

| Predicted | Result |
|---|---|
| Address math out of the hot loop | ✓ |
| Overhead per FFMA ~0.14 | ✓ beat it (0.075) |
| Wait stall ~0.02 | ✗ (0.08) |
| n = 4,096 at 94–95% | ✓ (95%) |

**Source-level diagnosis of the miss (measured, `analysis/source-pages/05-n4k-k11best.sass.csv`):**
- Wait samples land on `LDS` 41%, `LDGSTS` 23%, `IMAD` 16%.
- The top entries are `IMAD.WIDE R6, R29, 0x40, R20` feeding `LDGSTS`. `0x40` = 16 rows × 4 B, so the running pointer `pa += a_row` compiled into a **serial chain of dependent 64-bit multiply-adds**, issued in one burst at the chunk boundary.
- About 78% of dispatch samples land on `FFMA`.

## E6. Kernel 12: spread the copies; try the other FFMA order

**Change:**
- **`SPREAD=1`:** during chunk t, chunk t+S−1's copies are issued into chunk t−1's slot, a few per k-step, as CUTLASS spreads `AsyncCopyIterationsPerStage` over `kWarpGemmIterations`. Each address is base + c·stride, so there's no chain. There's one commit per chunk.
- **`ORDER=1`:** the outer product loops j-outer instead of i-outer, which changes which FFMA operands repeat back to back. Aimed at the dispatch stall.
- Both were swept independently and together.

**Run:** `fc-01M6M841CP8WZ8AHJBH6S65J8Z`, giving `runs/k12.{json,txt}` and `runs/k12.code/`. Profiles: `ncu/06-k12-n4096/`.

**Result (measured):**

| n | cuBLAS | k11 | k11o | k12 | k12o |
|---|---|---|---|---|---|
| 1,024 | 64.8 | 65.8 (99%) | 68.3 (95%) | 66.4 (98%) | **64.9 (100%)**, 128×64 BK16 S=3 |
| 2,048 | 340.9 | 365.3 (93%) | 369.2 (92%) | **355.6 (96%)**, 256×128 16×8 S=4 | 368.4 (93%) |
| 4,096 | 2,643 | **2,773 (95%)** | 2,808 (94%) | 2,779 (95%) | 2,795 (95%) |

Profile at n = 4,096 (128×256 8×16):

| | instructions | overhead per FFMA | issue | dispatch | wait | short_sb |
|---|---|---|---|---|---|---|
| k11 | 2,308 M | 0.075 | 82.2% | 0.21 | 0.08 | 0.06 |
| k12o | 2,378 M | 0.107 | **84.6%** | **0.15** | **0.05** | 0.13 |
| cuBLAS | 2,428 M | 0.131 | 90.2% | 0.10 | 0.01 | 0.20 |

**Verdict:**
- Spreading **did** cut the stalls it targeted: dispatch 0.21 → 0.15, wait 0.08 → 0.05, issue 82 → 85%.
- But it added 3% instructions: per-copy address math, ALU pipe 0.9% → 2.4%. At n = 4,096 the two effects cancel.
- n = 1,024 reaches **parity** with cuBLAS. The 0.1 µs difference is far inside run-to-run noise, so call it a tie, not a win.
- **ORDER on its own doesn't help.** k11o is slower or equal everywhere; it only helps when combined with SPREAD on the small tile.

## E7. Kernel 13: why spreading cost instructions; two more hypotheses tested

**Diagnosis first (measured).** I diffed the opcode histograms of kernel 11 and kernel 12 on the same tile (`analysis/source-pages/06-*.sass.csv`). SPREAD's +57 M instructions are **not** address math. They are:
- **+25 M `@!PT LDS RZ, [RZ]`:** a shared load with a never-true predicate. It's filler that ptxas emits, and it takes an issue slot but does nothing.
- **+16 M `MOV R, URx` and +11 M `@!UP0 ULDC URx, c[0x0][0x230]`:** a kernel parameter reloaded at each copy site.

Inferred: both come from my per-k-step `if (copy)` branch around the spread copies. ORDER=1 makes it slightly worse (+35 M `LDS` filler).

**Changes:**
- **`SPREAD=2` (k13):** branch-free. Every copy is always issued; past the end, the copy size is 0 (zero-fill, `cp.async … , src_size`) and the source address is clamped. This is the predicated-copy form cuBLAS's SASS uses (`LDGSTS … P0`).
- **`k13c`:** k13 with `BSTRIDED=0`, testing whether the B-column remap from E4 cost speed at n = 1,024.
- **`k13f`:** k13 with `FIXED_N=n`, compiled per size so M = N = K are constants. **Size-specialised:** cuBLAS can't do this, and the report has to say so.

**Run:** `fc-01M6M8PWZYX9QC5S6Z98WFH19B`, giving `runs/k13.{json,txt}` and `runs/k13.code/`. Size-specialised kernels are checked at their own n, 4,096 included.

**Result (measured):**
- **k13 ≈ k11.** Removing the branch didn't buy time.
- **k13c is worse everywhere,** by up to 10% at n = 4,096 (89% vs 95%). The column-remap hypothesis is **falsified**: strided B columns are the better layout, and the E4 dip at n = 1,024 was something else, probably noise.
- **k13f is the best at large n:** 96% at n = 2,048 and 4,096, about +1 point over generic. Confirmed in E9: 95.4% and 96.4%.

**Verdict:** the remaining issue-rate gap doesn't respond to instruction-stream changes in CUDA C anymore. A bigger lever was needed.

## E8. Kernel 14: Stream-K, and the first win over cuBLAS

**Observation (measured, from the Launch Statistics of earlier reports):**
- At n = 4,096, cuBLAS and our best kernels launch 512 tiles onto 132 SMs at one CTA per SM. That's 3 full waves plus 116 tiles, so 16 SMs sit idle for the last quarter. The ideal is 3.88 tile-times instead of 4, a 3% bound.
- At n = 1,024, every kernel so far, cuBLAS included, runs **one warp per scheduler** (128 blocks of 4 warps). That's why short-scoreboard and dispatch stalls show up directly as lost issue slots.

**Change (`STREAMK=1`):**
- **Refactor:** the kernel body became `tile_body(…, brow, bcol, kc0, chunks, atomic)`. `matmul_cp_async` is a thin wrapper, so every earlier variant still builds.
- **`matmul_streamk`:** a persistent kernel. Its grid is the number of CTAs that fit at once, computed from registers, shared memory and threads: 132 × 2 = 264 for the 128×64 and 128×128 tiles.
  - Full waves of tiles are done whole.
  - The leftover tiles' (tile, K-chunk) steps are split evenly over all CTAs (Osama et al., "Stream-K", 2023; used in CUTLASS).
  - Partial tiles are summed with `atomicAdd(float4*)` (sm_90 vector reductions) into a C region that `zero_tiles` clears first. **Both kernels are inside the timed launch.**
- A host-side model checked that every (tile, chunk) pair is covered exactly once, for every shape run.

**Prediction:** about 3% at n = 4,096 from wave quantization. At n = 1,024, 2 CTAs per SM means 2 warps per scheduler, so possibly more.

**Run:** `fc-01M6M8ZEJ5X1FG2STSC9Y57J28`, giving `runs/k14.{json,txt}` and `runs/k14.code/`.

**Result (measured):**
- **n = 1,024:** k14 128×128 BK16 S=2 takes **60.5 µs vs 64.5 µs, 107%**. Many k14 configs are at 100–107%.
- **n = 2,048 and 4,096:** k14 is **worse**, 89–94%. Registers rose (for example, 255 vs 223 at 256×128). Inferred: `tile_body` inlined twice (the full-wave loop and the Stream-K loop) costs register allocation and scheduling quality, which outweighs the 3% quantization bound.

**Profile** (`ncu/07-k14-streamk-n1024/`, n = 1,024):

| | warps per scheduler | issue active | total stall per issue | ncu time |
|---|---|---|---|---|
| k11 128×64 (best non-Stream-K) | 1.00 | 62.6% | 1.60 | 63.8 µs |
| k14 128×64 | 2.71 | 71.2% | 3.81 | 56.7 µs |
| **k14 128×128** | **1.99** | **72.0%** | 2.77 | **54.4 µs** |
| cuBLAS | 1.00 | 69.9% | 1.43 | 61.4 µs |

**Mechanism (measured, then inferred).** Per-warp stalls go *up* (2.77 vs 1.60 cycles per issued instruction), but with two warps per scheduler, one warp issues while the other stalls. So the SM issues on 72% of cycles, more than cuBLAS's 70%. Instruction counts are about equal (38 M vs 41 M), so the time follows the issue rate.

In plain terms: cuBLAS's heuristic picked a one-wave, one-warp-per-scheduler kernel at n = 1,024, and Stream-K's split of the K dimension creates the parallelism that configuration lacks.

**Caveats:**
- The ncu time of k14 excludes `zero_tiles`, which runs as a separate kernel. `::main` and `::confirm` times include it.
- The "tile" column in the profile summary for Stream-K reports isn't a tile shape, because the grid is the persistent CTA count.
- Floating-point summation order differs with atomics: results match cuBLAS to < 1e-4 relative, but runs aren't bit-reproducible.

## E9. Confirmation (`::confirm`)

**Method:** cuBLAS and the candidate kernels are timed in **alternating rounds**: cuBLAS, k1, k2, …, then cuBLAS again, for 15 rounds. Each round is the median of 40 L2-flushed launches. Comparing each kernel with cuBLAS *in the same round* cancels slow drift (clocks, temperature). Each kernel is checked for correctness first.

**Run:** `fc-01M6M946AGWF2FNH34AY7NK35X`, giving `runs/confirm.{json,txt}` and `runs/confirm.code/`.

| n | kernel | median | vs cuBLAS (median; min–max over rounds) | rounds faster |
|---|---|---|---|---|
| 1,024 | cuBLAS | 64.6 µs | — | — |
| 1,024 | **k14 128×128 BK16 S=2** | **61.3 µs** | **105.3%** (105.0–105.8%) | **15/15** |
| 1,024 | k14 128×64 BK16 S=2 | 63.6 µs | 101.7% (101.2–102.2%) | 15/15 |
| 1,024 | k11 128×64 BK16 S=2 | 66.0 µs | 97.9% | 0/15 |
| 2,048 | k13f 256×128 16×8 S=3 (size-specialised) | 357.9 µs | 95.4% | 0/15 |
| 2,048 | k11 256×128 16×8 S=3 | 363.5 µs | 93.9% | 0/15 |
| 4,096 | k13f 256×128 16×8 S=3 (size-specialised) | 2,761 µs | 96.4% | 0/15 |
| 4,096 | k11 128×256 8×16 S=4 | 2,795 µs | 95.2% | 0/15 |

**Verdict:** the n = 1,024 win is real. The spread across rounds is ±0.4%, and the margin is 5.3%.

## Cross-cutting technique: loop unrolling (and what Numba did with the course's loops)

**In our code (source fact).** `matmul_cp_async.cu` has 26 `#pragma unroll`s, and every loop with a compile-time trip count is unrolled:
- the k-step loop over BK (`compute_chunk` and the kernel 9/11/12 main loops);
- the TM × TN outer product;
- fragment loads, copy loops (`A_COPIES`, `B_COPIES`, the spread copies), the prologue, the accumulator zeroing, the epilogue.

Only the loop over K-chunks (`t < chunks`, a runtime count) and the persistent Stream-K loops stay as loops.

Unrolling is load-bearing for three of the techniques:
- the accumulators `acc[TM][TN]` and fragment buffers `a[2][TM]` / `b[2][TN]` can only live in registers if every index is a constant;
- the register double buffer of kernel 9 indexes `a[(kk+1) % 2]`, which is a constant only once kk is;
- kernel 11's copy offsets `it * A_SSTEP` become instruction immediates only after unrolling.

**In the course's Numba kernels (measured).** These come from Shubham's profiles in `lecture10/` (Tesla T4, n = 1,024), read with `ncu --import` and exported to `results/analysis/source-pages/course-lecture10-numba-*.sass.csv`:

| | Registers | Local-memory loads/stores | Static FFMAs in the kernel | Longest FFMA run | Executed instructions | Overhead (non-FFMA) per FFMA |
|---|---|---|---|---|---|---|
| course k5 (Numba) | 122 | 0 | 16 | 16 | 145.1 M | 3.32 |
| course k7 (Numba, 8×8 patch) | 198 | 0 | **64** = one 8×8 outer product | 44 | 69.8 M | **1.08** |
| ours k11 128×256 BK16 8×16 (n = 4,096) | 231 | 0 | **2,048** = 128 × BK 16 | 111 | — | **0.075** |
| ours k14 / k11 at n = 1,024 | 175 / 128 | 0 | — | — | 38 M | **0.127 / 0.136** |

**What that shows:**
- **Numba did unroll the small inner loops** (`for i in range(TM)`, `for j in range(TN)`) on its own, through LLVM/NVVM at its default optimisation level.
  - Measured: zero local-memory traffic, so the `cuda.local.array` accumulators are in registers.
  - Measured: the static FFMA count equals TM·TN.
  - So "we weren't using unrolling" is only partly true for the Numba lab code.
- **Numba did not unroll the k-step loop** (`for dot_k in range(BK)`, with BK = 32 there). Only one outer product's worth of FFMAs exists in the binary, so every k-step pays a loop branch, an index update and address recomputation. Its top non-FFMA instructions are `LDS` 8.4 M, `IMAD` 5.1 M, `IADD3` 3.7 M and `LEA` 3.5 M.
- **The Numba shared-memory reads are scalar.** 16 static `LDS` instructions = TM + TN = 16 single loads per k-step. Inferred: the 4-wide Python loops (`for i in range(0, TM, 4)`) didn't become 128-bit `LDS.128` loads the way our `float4` casts do.
- The book's CUDA C kernels (Week 3-1 note, kernel 5 code) do use `#pragma unroll`. The course's book kernels and its Numba lab code differ here.

**Caveats:**
- **Different hardware and code.** The Numba profiles are from a T4 (sm_75) at n = 1,024, a different GPU and generation from our H100. Instruction counts depend on code structure more than on the GPU, but not entirely.
- **The overhead figure isn't all unrolling.** Overhead per FFMA (1.08 vs 0.13) also includes Numba's 64-bit index arithmetic, scalar shared loads and the lack of hoisting.
- **Not isolated.** We haven't measured unrolling on its own. An `UNROLL=0/1` switch that only drops `#pragma unroll` from the k-step loop would separate it; that's cheap to add if the chapter wants the number.

## Roofline position: arithmetic intensity and the ridge

The data here comes from runs already listed above: `assignment1/results/ceilings.json` (2026-10-30), the DRAM counters in `ncu/06…` and `ncu/07…`, and the E9 medians.

**The ridge on this H100 (measured in Assignment 1):**
- FP32 FMA peak: 61.3 TFLOP/s; HBM copy bandwidth: 3.03 TB/s.
- **Ridge = 20.2 FLOP/byte.** The spec-sheet values (67 TFLOP/s, 3.35 TB/s) give 20.

**Ideal arithmetic intensity** (A and B read once, C written once):
- 2n³ FLOPs ÷ 12n² bytes = **n/6**: that's **171**, **341** and **683** FLOP/byte at n = 1,024, 2,048 and 4,096.

**Measured arithmetic intensity** (FLOPs ÷ DRAM bytes, reads plus writes, including C's write-back):

| n | kernel | DRAM bytes | AI (FLOP/byte) |
|---|---|---|---|
| 1,024 | every kernel (k11, k14, cuBLAS) | ≈ 8.4 MB (A + B) + 4.2 MB (C) | **≈ 170**, the ideal |
| 4,096 | k11 256×128 | 306 + 64 MB | ≈ 371 |
| 4,096 | k11 128×256 | 374 + 64 MB | ≈ 314 |
| 4,096 | cuBLAS | 342 + 64 MB | ≈ 338 |

**Correction, recorded because it was caught in review.** A first pass reported 255 FLOP/byte at n = 1,024, which is above the ideal of 171 and so impossible. At n = 1,024 the 4.2 MB C stays in the 50 MB L2 cache and is written to DRAM only after the kernel ends, so ncu's write counter read about 0 bytes. Counting C's write-back gives about 170.
- The 4.2 MB extra that k14 reads under ncu is probably C itself: ncu flushes caches between kernels, so the atomics re-read the C that `zero_tiles` cleared. That's inferred and unverified.
- At n = 4,096 the ncu write counter saw about 55 MB of C's 64 MB, so the table uses 64 MB.

**Throughput against the flat roof** (E9 medians, as a share of 61.3 TFLOP/s):

| n | ours | cuBLAS |
|---|---|---|
| 1,024 | 35.0 TFLOP/s (57%), k14 | 33.2 TFLOP/s (54%) |
| 2,048 | 48.0 (78%), k13f | 50.3 (82%) |
| 4,096 | 49.8 (81%), k13f | 51.6 (84%) |

**Reading it.** Every kernel sits 8–34× right of the ridge, so the ceiling is the flat compute roof. Past the ridge, more AI buys nothing, and none of these kernels was limited by AI or by DRAM bandwidth. At n = 4,096, k13f needs about 0.13 TB/s of HBM, about 4% of what's available, which explains why E3's 22–53% DRAM cut changed nothing.

Every gain from kernel 8 to kernel 14 moved a point **straight up toward the roof**, by raising issue efficiency:
- latency hiding (k8, k9);
- fewer overhead instructions (k10, k11);
- more warps per scheduler (k14).

None moved a point sideways. Beating cuBLAS means being closer to the same roof than cuBLAS is (57% vs 54%), not exceeding it; no measurement is above 61.3 TFLOP/s. The early kernels were limited by on-chip ceilings, which this HBM roofline doesn't show: shared-memory load latency, and the L2 → SM tile AI of BM·BN / (2(BM+BN)) = 21 / 32 / 43 FLOP/byte for 128×64 / 128×128 / 256×128 tiles.

## Where it stands, and what's left (inferred unless marked)

**Measured (E9):**
- **n = 1,024:** **5.3% faster than cuBLAS**, with a generic kernel (k14).
- **n = 2,048:** 95.4% of cuBLAS (k13f, size-specialised); the generic k11 reaches 93.9%.
- **n = 4,096:** 96.4% of cuBLAS (k13f, size-specialised); the generic k11 reaches 95.2%.

**At large n we run fewer instructions than cuBLAS, with no memory wait.** The gap is issue rate: dispatch stalls (about 75% of them on FFMA) and a small wait stall. CUDA C has no direct control over either; they depend on ptxas's register-bank assignment and scheduling.

**Next options, by expected payoff:**
1. **Stream-K at large n without the register cost.** Use a single tile loop, so `tile_body` is inlined once (the full-wave tiles become one-piece Stream-K items), or `__noinline__` it. That would bring kernel 11's code quality into the persistent kernel. Bound: about 3% at n = 4,096 (3.88 vs 4 waves), which is roughly the gap.
2. **Stream-K plus 2 CTAs per SM at n = 2,048,** where the tiles (128 of 256×128) are fewer than the SMs: the same mechanism that won at n = 1,024.
3. **TMA for tile loading (untried; added 2026-11-08).** Test whether Hopper's Tensor Memory Accelerator can reduce copy instructions and address-register pressure enough to improve issue rate. Start with **B only**, retaining A's existing 4-byte `cp.async` copies and transposed shared-memory layout. This is a hypothesis of incremental benefit, not a measured win or a predicted speedup. See the experiment outline below.
4. **SASS-level register allocation** (CuAssembler, maxas-style). That's outside CUDA C.
5. **Not in scope:** TF32 or tensor cores are a different problem, with different numerics.

### Proposed experiment: TMA for B tile loads

**Status:** proposed, not implemented or benchmarked. The current kernels use ordinary per-thread `cp.async`, not TMA. TMA changes data movement; this experiment keeps FP32 FFMA arithmetic and TF32 off.

**Why it might help:** TMA can move tensor tiles with fewer issuing threads and less per-thread address bookkeeping. Our remaining issue-rate gap makes reduced instruction and register overhead worth testing. However, E1 already hid global-memory latency, and E5 substantially reduced address overhead. Expect limited headroom unless TMA also improves scheduling or occupancy; no numerical gain is established. Hardware background: [NVIDIA Hopper Tuning Guide — Tensor Memory Accelerator](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator).

**First variant:** replace only B's global-to-shared copies in a generic k11 configuration with TMA. Keep tile dimensions, stages, FFMA order, and A's load path fixed where possible. Validate B's shared-memory strides, alignment, and tensor-map layout before implementing the transfer; account explicitly for any padding/layout changes required. Use TMA completion barriers and preserve the separate completion requirements of A's ordinary `cp.async` copies before consuming a stage or reusing it.

**Main risks:** descriptor setup and synchronization can outweigh copy savings, especially for small tiles. A's transposed, padded shared-memory layout currently enables efficient vectorized shared loads; migrating A as well would require a separate layout experiment, not a mechanical replacement of its scatter copies. Lower register use helps occupancy only if it crosses a residency threshold. B-only TMA may leave most copy overhead in A.

**Validation and decision:**
- Compare the TMA variant against the same configuration with ordinary copies at n = 1,024, 2,048, and 4,096 on the pinned H100. Also report against the existing best generic kernel at each size and cuBLAS with TF32 off.
- Check correctness with NaN-prefilled output and the existing relative-error criterion (< 1e-4), including stage reuse and tail handling for the supported shapes.
- Use E9's 15 alternating rounds with medians of 40 L2-flushed launches. Report tensor-map setup separately and state whether it is amortized; include all per-invocation GPU work in timings.
- Profile registers, spills, achieved occupancy, instruction counts, issue-active rate, dispatch/wait stalls, and shared-memory bank conflicts. Save a distinct run tag and source snapshot.
- Keep the variant only if timing gains exceed run-to-run noise without correctness regressions. If it helps, test integration with k14/Stream-K separately; investigate A's layout only as a subsequent experiment. Record a neutral or negative result too.

## Gotchas found along the way (each cost a run)

- **Modal GPU upgrade:** `GPU = "H100"` may be upgraded to an H200, and the first kernel 8 run landed on one. `"H100!"` pins a real H100.
- **ncu clock lock:** `ncu` on Modal fails with "Failed to lock GPU clock frequencies" unless you pass `--clock-control none`. Performance counters do work.
- **CPU-only report reading:** `ncu -i report --page raw --csv` and `--page source --csv --print-source sass` run without a GPU, in a CPU-only Modal container (`analysis/scripts/`).
- **Profile overwrites:** early profile runs reused filenames and overwrote each other. Since E6, every profile set and sweep writes its own folder or tag and a code snapshot, and refuses to overwrite.
- **My own errors:**
  - A `sed` edit once put a comment inside the ncu argument list, which dropped the kernel filter, so that run profiled the random-number kernel. Caught from the kernel names and re-run.
  - Swizzling alone, without the bigger patches, was briefly expected to help. It didn't, and E3 records that.
