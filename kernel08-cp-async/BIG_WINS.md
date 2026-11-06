# Big wins beyond the course kernels

The course (and Simon Boehm's post, which it follows up to warp tiling) stops at **kernel 7**. On the H100 at n = 1,024, kernel 7 takes 86.6 µs against cuBLAS's 56.1 µs, about **65%** of cuBLAS (book Table 5.4). Everything below is new relative to the course kernels, ordered by size of effect. All of it was measured on an H100 80GB HBM3, FP32, with TF32 off, on square n × n matrices. Experiment numbers (E1–E9) point to `LAB_NOTEBOOK.md`.

| # | Win | What it is | Measured effect | Where |
|---|---|---|---|---|
| 1 | **Multi-stage `cp.async` pipeline** (k8) | Global → shared copies that skip registers and don't block (`LDGSTS`), into a ring of 2–4 shared-memory slots, so the next chunks load while the current one computes. Boehm planned it; neither he nor the course built it. | Waiting on global memory **46.8% → 2.5%** of warp cycles. Best time at n = 1,024: 99.8 → 72.8 µs (**1.37×**), to **89%** of cuBLAS. | E1 |
| 2 | **Register fragment prefetch + transposed A** (k9+T) | Two register buffers per operand: step k+1's values load from shared memory while step k's 64 FMAs run (CUTLASS `mac_loop_iter`). A is stored transposed with 4-byte copies, so a fragment is two 128-bit loads. Neither change helps alone. | Short-scoreboard stall **0.38 → 0.13** cycles per instruction; issue active 52% → 62%. n = 1,024: 89% → **97%**. | E2 |
| 3 | **Hoisted copy-address math** (k11) | Each thread's copy pointers and shared offsets computed once and advanced by constant strides; ring slots step by add-and-wrap. Found by reading per-instruction stall samples on the SASS Source page (`LEA`, `IMAD`, `LDC` in the hot loop). | Overhead per FFMA **0.170 → 0.075** (cuBLAS 0.131); ALU pipe 9.3% → 0.9%; registers 255 → 231. n = 4,096: 90% → **95%**; n = 2,048: 88% → 94%; n = 1,024: 98%. | E4–E5 |
| 4 | **Stream-K** (k14) | A persistent kernel: full waves of tiles done whole, then the leftover tiles' K-work split evenly across all resident blocks, with float4 atomics into pre-zeroed tiles. At n = 1,024 it fits 2 blocks per SM, so each scheduler has 2 warps where cuBLAS has 1. | **n = 1,024: 105.3% of cuBLAS** (61.3 vs 64.6 µs), faster in **15/15** alternating rounds. Warps per scheduler 1.0 → 2.0; issue active 63% → 72% (cuBLAS 70%). | E8–E9 |
| 5 | **Bigger thread patches** (k10: 16×8 and 8×16) | 128 outputs per thread instead of 64 (cuBLAS's own shape at n = 4,096), with strided B columns so the 8 thread columns don't cause bank conflicts. | Shared loads **336 M → 101 M** (matches cuBLAS). n = 4,096: 82% → **90%**; n = 2,048: 83% → 88%. | E4 |
| 6 | **Full loop unrolling, including the k-step loop** (cross-cutting) | `#pragma unroll` on every constant-trip loop (26 of them). This keeps accumulators and double buffers in registers and turns copy offsets into immediates. The course's Numba lab code unrolled only the inner TM×TN loops by itself, not the k-step loop. | Static FFMAs in the kernel: **2,048 vs 64** (Numba k7). Overhead per FFMA **0.13 vs 1.08**. That gap mixes unrolling with other differences; unrolling on its own wasn't isolated. | notebook, "loop unrolling" |
| 7 | Size specialisation (k13f) | M = N = K compiled in as constants, so address math folds into immediates. **Not like-for-like:** cuBLAS is one binary for any size. | About **+1 point** at large n: **95.4%** (n = 2,048), **96.4%** (n = 4,096) in E9. | E7, E9 |

**Final standings** (E9, `results/runs/confirm.txt`):

| n | Result |
|---|---|
| 1,024 | **105.3%** of cuBLAS (k14, generic) |
| 2,048 | 95.4% (k13f, size-specialised); 93.9% generic |
| 4,096 | 96.4% (k13f, size-specialised); 95.2% generic |

**Tried, and not wins.** Keep these in the write-up; they're part of the method.
- **Block swizzling for L2 reuse (E3).** DRAM reads fell 22–53%, but the time didn't change, because the kernels are compute-bound. That matches Boehm's experience.
- **Copies spread across the k-steps (E6/E7).** Stalls fell but instructions rose, so the net effect was zero. The diagnosis surfaced compiler filler (`@!PT LDS RZ,[RZ]`) caused by a branch.
- **j-outer FFMA order (E6):** equal or worse.
- **Contiguous B columns (E7):** up to 10% worse. The strided layout from win 5 is right.
- **Stream-K at n ≥ 2,048 (E8):** 89–94%, worse than without it. Inferred cause: register pressure from inlining the tile body twice. This is the top open lead.
