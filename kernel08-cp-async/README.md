# Kernels 8–14: from a `cp.async` pipeline to beating cuBLAS at n = 1,024 (FP32, H100)

> **For the write-up:**
> - [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md): every experiment, with hypothesis, prediction, run ID, result and verdict.
> - [`SCREENSHOTS.md`](SCREENSHOTS.md): which `.ncu-rep` file and which Nsight Compute page goes with each figure.
> - [`results/MANIFEST.md`](results/MANIFEST.md): every result file.
> - [`BIG_WINS.md`](BIG_WINS.md): one-page summary of the wins beyond the course kernels, with measured effects.
> - [`CHAPTER_HANDOFF.md`](CHAPTER_HANDOFF.md): the outline, figure list and sources for writing the chapter in a new session.
>
> The sections below were written as the work happened, so the later sections supersede the earlier ones.
>
> **Current best** (`::confirm`, run `fc-01M6M946AGWF2FNH34AY7NK35X`):
> - **n = 1,024:** k14 (Stream-K) at 61.3 µs vs cuBLAS 64.6 µs, **5.3% faster**, in 15 of 15 rounds.
> - **n = 2,048:** 95.4% of cuBLAS (size-specialised k13f).
> - **n = 4,096:** 96.4% of cuBLAS (size-specialised k13f).
>
> Details are in LAB_NOTEBOOK.md, E7–E9.

Kernel 7 (warp tiles, 16-byte loads) reached 86.6 µs at n = 1,024 on the H100, against 56.1 µs for cuBLAS. Both kernels used shared memory equally well (0.19 passes per FMA). The difference was time spent waiting on global memory: 43% of warp time for kernel 7, 7.5% for cuBLAS. The cuBLAS kernel's name, `sm80_xmma_gemm_…_tilesize128x64x8_stage3_…_ffma`, says how it avoids that wait: it keeps three K-chunks in flight while it computes. This folder builds the same pipeline (kernel 8), then the register-level prefetch from CUTLASS's mainloop (kernel 9), and measures both.

| File | What it does |
|---|---|
| `matmul_cp_async.cu` | The kernel. Tiles are loaded with `cp.async.cg` (SASS `LDGSTS`), which copies global memory straight into shared memory without registers and without blocking. The shared memory is a ring of `STAGES` slots. `STAGES = 1` compiles the same code with no overlap, as the baseline. Two more switches: `PREFETCH=1` (kernel 9: two register buffers per operand, step k+1 loaded while step k computes) and `ATRANS=1` (A stored transposed via 4-byte `cp.async`, so a fragment is two 128-bit loads). |
| `matmul08.py` | Modal app. `::main` checks correctness, times every variant × config × STAGES against cuBLAS and counts SASS instructions. `::profile` saves Nsight Compute reports to your laptop. |
| `results/` | `runs/` (timing sweeps, `.json` + `.txt` + `.code/`), `ncu/<set>/` (Nsight Compute reports + `summary.txt` + `code/`), `analysis/` (SASS-level exports and scripts). See `results/MANIFEST.md`. |

## Run it

The script uses Assignment 1's Modal image (same digest, same pinned packages) and its Modal client:

```bash
cd kernel08-cp-async
../assignment1/.venv/bin/modal run matmul08.py::main                         # ~5 min (SWEEP x CONFIGS x STAGES), ~30 cents
../assignment1/.venv/bin/modal run matmul08.py::profile --set 07-k14-streamk-n1024   # ~2 min; results/ncu/<set>/
../assignment1/.venv/bin/modal run matmul08.py::confirm   # winners vs cuBLAS, 15 alternating rounds; results/runs/confirm.*
```

Open the `.ncu-rep` files in the Nsight Compute GUI on your laptop (a free download that doesn't need a GPU). `SCREENSHOTS.md` lists which pair to open for each figure. Neither command overwrites: each run writes its own tag or folder and a snapshot of the code it used.

Settings marked `TUNE` in `matmul08.py`:
- `CONFIGS`: tile shapes
- `VARIANTS` and `SWEEP`: which switch combinations exist, and which `::main` times
- `STAGES_LIST`
- `SIZES`
- `PROFILE_SETS`: named profile sets (01–05 rebuild the earlier experiments)
- `GPU`

**Two Modal details that cost a run each to find:**
- **`GPU = "H100!"`.** The `!` pins a real H100. A plain `"H100"` request may be upgraded to an H200, and that happened on the first run here.
- **`ncu --clock-control none`.** Modal's GPUs refuse ncu's clock lock (`Failed to lock GPU clock frequencies`), but performance counters do work.

## Kernel 8 results (H100 80GB HBM3, 2026-11-04, torch 2.14.1+cu130)

`::main`, best of each kind (time in µs, % = share of cuBLAS speed):

| n | cuBLAS | best S = 1 (no overlap) | best pipelined | speedup from pipelining | pipelined vs cuBLAS |
|---|---|---|---|---|---|
| 1,024 | 64.9 | 99.8 (128×64 BK16) | 72.8 (128×64 BK8, S=4) | 1.37× | 89% |
| 2,048 | 343.3 | 568.6 (128×128 BK16) | 422.0 (128×128 BK16, S=4) | 1.35× | 81% |
| 4,096 | 2,661 | 3,780 (128×64 BK16) | 3,266 (128×128 BK16, S=4) | 1.16× | 81% |

`::profile`, cuBLAS's own shape (128×64, BK 8), n = 1,024:

| Report | Time | Waiting on global (long-scoreboard share of warp cycles) | Issue active |
|---|---|---|---|
| S = 1 | 132.4 µs | 46.8% | 28.8% |
| S = 3 | 79.1 µs | **2.5%** | 52.0% |
| cuBLAS | 61.3 µs | 6.6% | 69.8% |

**What this shows:**
- Pipelining removes the wait on global memory. It drops from 47% to 2.5%, below cuBLAS's 6.6%.
- What separates kernel 8 from cuBLAS now is **how often warps issue instructions** (52% vs 70% of cycles), not memory. That points at instruction scheduling and shared-memory read overhead, the next things to work on.
- The S = 1 baseline isn't the book's kernel 7 exactly. It loads A un-transposed (rows padded to BK+4, each thread's rows strided by 8), because `cp.async` can't transpose on the way in.
- Run-to-run noise is a few percent. At n = 2,048, the 128×64 BK8 S=3 result moved from 483 to 584 µs between two runs.

SASS for 128×64 BK8: S = 3 has 9 `LDGSTS`, 3 `LDGDEPBAR`, 1 `DEPBAR`, and **0 plain `LDG`**. No global data passes through registers.

## Kernel 9: register fragment prefetch (same H100 setup, run `fc-01M6M6R5YXZ0T80VJMTWDKEFFV`)

Kernel 8's profile left one big stall: **short scoreboard**, warps waiting on their own shared-memory loads, at 0.38 cycles per issued instruction against cuBLAS's 0.10. With one warp per scheduler (128 blocks on 132 SMs at n = 1,024), no other warp hides those loads, so the hiding has to happen inside the warp. Kernel 9 follows CUTLASS `mma_multistage.h` (`mac_loop_iter`):
- it keeps **two register buffers** per operand, and loads step k+1's fragment while step k's 64 FMAs run;
- it moves the chunk's wait and barrier to the **last k-step of the previous chunk**, so the next chunk's first fragment is already loading when that chunk starts.

`::main`, best of each variant (µs, % of cuBLAS speed). All 56 kernels match cuBLAS:

| n | cuBLAS | k8 | k8+T (transpose only) | k9 (prefetch only) | **k9+T (both)** |
|---|---|---|---|---|---|
| 1,024 | 65.3 | 73.2 (89%) | 71.1 (92%) | 69.7 (94%) | **67.6 (97%)**, 128×64 BK16 S=3 |
| 2,048 | 341.2 | 418.7 (81%) | 426.0 (80%) | 412.9 (83%) | 429.9 (79%) |
| 4,096 | 2,662 | 3,271 (81%) | 3,339 (80%) | 3,261 (82%) | 3,390 (79%) |

`::profile`, 128×64 BK8 S=3, n = 1,024. Stalls are cycles per issued instruction, and "total" includes the issue cycle itself:

| Report | Regs | Time | Issue active | Total | short_sb | dispatch | wait | Waiting on global |
|---|---|---|---|---|---|---|---|---|
| k8 | 164 | 78.4 µs | 52.0% | 1.93 | 0.38 | 0.22 | 0.14 | 2.5% |
| k8+T | 165 | 89.2 µs | 51.0% | 1.96 | 0.39 | 0.13 | 0.25 | 0.8% |
| k9 | 125 | 77.4 µs | 57.7% | 1.73 | 0.26 | 0.12 | 0.18 | 2.8% |
| **k9+T** | 167 | **71.5 µs** | **62.4%** | 1.60 | **0.13** | 0.13 | 0.16 | 3.2% |
| cuBLAS | 224 | 61.3 µs | 68.7% | 1.43 | 0.10 | 0.03 | 0.04 | 6.7% |

**What this shows:**
- **The prediction held.** Before the run, I predicted short scoreboard 0.38 → ~0.15, issue active above 60%, and 68–70 µs under ncu. Measured for k9+T: 0.13, 62.4% and 71.5 µs.
- **The two changes only pay off together.**
  - Transposing alone (k8+T) is *slower*: the same short-scoreboard stall, plus a larger wait stall.
  - Prefetch alone (k9) helps a little. Without the transpose, prefetching an A fragment takes 8 separate 32-bit loads.
  - With both, a whole k-step is four 128-bit loads (two for A, two for B), issued a full step ahead.
- **The gap left at n = 1,024 is about 3%.** In the profile it shows as dispatch stall (0.13 vs 0.03) and wait stall (0.16 vs 0.04). That's instruction-level scheduling inside ptxas, where cuBLAS's hand tuning lives. The 224 vs 167 registers suggest cuBLAS spends registers on even deeper prefetch.
- **At n = 2,048 and 4,096, kernel 9 doesn't help** (79–83%). The limit there is something else. My guess is L2 reuse: 64 MB matrices against a 50 MB L2, with 2,048 blocks launched in plain row order. The next experiment is block swizzling, plus a profile at n = 4,096 comparing `dram__bytes_read` against cuBLAS.
- **Unexplained:** the `+T` variants are much slower at S = 2 and S = 4 than at S = 3 with the same tile (for example, k8+T 128×64 BK8 takes 119 / 92 / 116 µs at S = 2 / 3 / 4). I haven't looked into it.

## Bigger thread patches, and what the n = 4,096 profile says (run `fc-01M6M76Q44PQJT6D8VHK8DSTH1`)

Two changes in this run:
- **Bigger thread patches.** A 16×8 or 8×16 patch gives each thread 128 outputs instead of 64. cuBLAS's n = 4,096 kernel uses 256 threads, a 128×256 tile and 254 registers.
- **Strided B columns** (`thread_col`). Needed because with 8 thread columns, contiguous columns would cause 2-way bank conflicts.

All 104 kernels match cuBLAS, and none spill to local memory.

| n | cuBLAS | best before (8×8 patches) | best now | config |
|---|---|---|---|---|
| 1,024 | 64.7 | 67.6 (97%) | 70.5 (92%) | k9TS 128×64 BK16 S=3 |
| 2,048 | 341.4 | 412.9 (83%) | 387.7 (88%) | k9TS 256×128 BK16 16×8 S=4 |
| 4,096 | 2,668 | 3,261 (82%) | **2,962 (90%)** | k9TS 256×128 BK16 16×8 S=4 |

**Caveat at n = 1,024:** the same k9+T config measured 67.6 µs before the column change and 70.7 µs after. The new column layout may cost something at the 8×8 shape, or it may be noise; that's untested.

`::profile --n 4096`, best kernel vs cuBLAS:

| | Instructions | Overhead per FFMA | LDS | Issue active | Warps per scheduler | Dispatch stall | Wait stall |
|---|---|---|---|---|---|---|---|
| 8×8 k9 | 2,669 M | 0.243 | 336 M | 87.4% | 3.9 | 0.23 | 0.15 |
| **16×8 k9TS** | 2,513 M | 0.170 | 101 M | 83.4% | 2.0 | 0.23 | 0.09 |
| cuBLAS | 2,428 M | 0.131 | 101 M | 90.1% | 2.0 | 0.10 | 0.01 |

**What changed:** the big patches match cuBLAS on shared loads (101 M), tile shape and occupancy.

**What's left (11%):** about 3.5% more instructions and about 8% lower issue rate. Per-SASS-instruction stall sampling from the source page of the same reports:
- **Extra integer instructions.** Ours executes 82 M `LEA`, 56 M `IMAD`, 37 M `LDC` and 27 M `IADD3`, about 200 per 16-k-step chunk. cuBLAS runs 64 M `IADD3` plus about 40 M uniform-datapath ops (`UIADD3`, `UISETP`, `UMOV`, `UIMAD`), and **no `LEA`, `IMAD` or `LDC`** in its hot code.
  - `LDC` is a load from constant memory, the kernel parameters such as `K` and `N`. Inferred: at 255 registers, the compiler re-reads `K`/`N` and recomputes the `cp.async` source addresses every chunk instead of keeping them in registers.
- **Wait stalls:** 4.0% of samples for ours vs 0.4% for cuBLAS. They fall on `LEA` (31%), `LDS` (23%), `LDC` (11%), `LDGSTS` (8%) and `IMAD` (7%), which is the same address math as above.
- **Dispatch stalls:** 9.5% of samples vs 4.5%, and about 73% of them land on `FFMA` in both kernels. My inference is register-bank conflicts between FFMA operands. ncu's SASS view doesn't show `.reuse` flags for either kernel, so that can't be checked from these reports.

**Next experiment:** compute each thread's `cp.async` source pointers once, before the main loop, and advance them by a constant stride per chunk (one 64-bit add each). Prediction: `LEA`, `IMAD` and `LDC` largely disappear from the hot loop, the wait stall drops toward 0.02, and overhead per FFMA drops toward 0.14.

## Kernel 11: hoisted copy plan (runs `fc-01M6M7PVKSCQWDCVQDE1CGS4E8` and `::profile --n 4096`)

`HOIST=1`, built on the PREFETCH path:
- Each thread computes its `cp.async` source pointers and shared-memory offsets **once** (`make_plan`), then advances them by constant strides per chunk (`issue_chunk`). Per copy that's one 64-bit pointer add, and the shared offset is a compile-time immediate.
- The ring slots advance by add-and-wrap instead of `t % STAGES`.
- The shared base is converted to a 32-bit address once.

All 72 kernels match cuBLAS, and none spill.

| n | cuBLAS | kernel 10 best (k9TS) | **kernel 11 best** | config |
|---|---|---|---|---|
| 1,024 | 65.1 | 71.5 (91%) | **66.7 (98%)** | k11 128×64 BK16 S=2 |
| 2,048 | 345.1 | 388.6 (89%) | **366.5 (94%)** | k11 256×128 BK16 16×8 S=3 |
| 4,096 | 2,663 | 2,998 (89%) | **2,799 (95%)** | k11 128×256 BK16 8×16 S=3 |

**Registers fell** from 255 to 231 (256×128 16×8) and from 209 to 128 (128×64 BK16), because the compiler no longer keeps recomputing addresses.

**k11n**, the same change without the transposed A, gains much less (88–94%). The 4-byte transposed copies are what had the expensive per-copy address math, so they gain most from hoisting.

`::profile --n 4096`:

| | Instructions | Overhead per FFMA | ALU pipe | FMA pipe | Issue active | Dispatch | Wait |
|---|---|---|---|---|---|---|---|
| kernel 10 (256×128 16×8) | 2,513 M | 0.170 | 9.3% | 73.1% | 83.4% | 0.23 | 0.09 |
| kernel 11 (same tile) | 2,322 M | 0.081 | 1.0% | 75.4% | 80.9% | 0.24 | 0.07 |
| kernel 11 best (128×256 8×16) | 2,308 M | **0.075** | 0.9% | 76.9% | 82.2% | 0.21 | 0.08 |
| cuBLAS | 2,428 M | 0.131 | 7.5% | 79.7% | 90.1% | 0.10 | 0.01 |

**Prediction scorecard:**

| Predicted | Measured | Result |
|---|---|---|
| Address math leaves the hot loop | ALU pipe 9.3% → 0.9% | ✓ |
| Overhead per FFMA drops to ~0.14 | 0.075, below cuBLAS's 0.131 | ✓ (beat it) |
| Wait stall drops to ~0.02 | 0.08 | ✗ (barely moved) |
| n = 4,096 reaches 94–95% | 95% | ✓ |

**What's left (5% at n = 4,096, 2% at n = 1,024):**
- Kernel 11 now runs **5% fewer instructions than cuBLAS**. The whole gap is issue rate: 82% vs 90%.
- That comes from **dispatch stalls** (0.21 vs 0.10, about 73% of them on FFMA in the last source-level read) and the **wait stall** that didn't move (0.08 vs 0.01).
- Both sit below CUDA C: register-bank assignment and instruction scheduling inside ptxas. Things to try next:
  - Reorder the outer product (j-outer, or a zig-zag) and watch the dispatch stall.
  - Read where the wait samples land now in the kernel 11 source page.

## Kernel 12: copies spread over the k-steps, and FFMA order (run `fc-01M6M841CP8WZ8AHJBH6S65J8Z`)

The full write-up is E6 in `LAB_NOTEBOOK.md`. The source page of kernel 11 showed its 16 A-copy addresses compiled into a chain of dependent `IMAD.WIDE`, issued in one burst at the chunk boundary. Two switches came out of that:
- **`SPREAD=1`** spreads the next chunk's copies over the k-steps, as CUTLASS does, with each address computed from the base.
- **`ORDER=1`** runs the outer product j-outer, aimed at the FFMA dispatch stall.

| n | cuBLAS | best k11 | best k12 / k12o |
|---|---|---|---|
| 1,024 | 64.8 µs | 65.8 (99%) | **64.9 (100%)**, k12o 128×64 BK16 S=3 |
| 2,048 | 340.9 µs | 365.3 (93%) | **355.6 (96%)**, k12 256×128 16×8 S=4 |
| 4,096 | 2,643 µs | **2,773 (95%)** | 2,779 (95%) |

**Profile at n = 4,096** (`results/ncu/06-k12-n4096/`):
- Spreading cut the stalls it targeted: dispatch 0.21 → 0.15, wait 0.08 → 0.05, issue 82% → 85% (cuBLAS 90%).
- It also added 3% instructions, so the net effect is zero at n = 4,096.
- n = 1,024 is at **parity**, within noise. ORDER on its own doesn't help.
- Where the gap stands, and the options not yet tried, are at the end of `LAB_NOTEBOOK.md`.
