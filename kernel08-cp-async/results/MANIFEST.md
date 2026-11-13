# results/ manifest

Experiment IDs (E1–E6) refer to `../LAB_NOTEBOOK.md`. All runs are on an H100 80GB HBM3 unless marked.

## runs/: timing sweeps (`modal run matmul08.py::main`)

| File | Experiment | Modal run | Contents |
|---|---|---|---|
| `k8-h200.json` | E1 | `fc-01M6M5XC5PSTCPCBPMBKTCWS03` | **H200** (Modal upgraded a plain "H100" request). Kept, but not used. |
| `k8.json` | E1 | `fc-01M6M5YHA3EST0D4WSF73G1NDA` | kernel 8, 4 configs × S = 1–4 |
| `k9.json` | E2 | `fc-01M6M6R5YXZ0T80VJMTWDKEFFV` | k8, k8+T, k9, k9+T × 4 configs × S |
| `k10.json`, `k10.txt` | E4 | `fc-01M6M76Q44PQJT6D8VHK8DSTH1` | k8, k9, k9+T, k9TS × 8 configs (adds 16×8 / 8×16) |
| `k11.json`, `k11.txt` | E5 | `fc-01M6M7PVKSCQWDCVQDE1CGS4E8` | k9TS, k11, k11n × 8 configs |
| `k12.json`, `k12.txt`, `k12.code/` | E6 | `fc-01M6M841CP8WZ8AHJBH6S65J8Z` | k11, k11o, k12, k12o × 8 configs; `k12.code/` holds the exact `.cu`/`.py` plus SHA256SUMS |
| `k13.json`, `k13.txt`, `k13.code/` | E7 | `fc-01M6M8PWZYX9QC5S6Z98WFH19B` | k11, k12, k13, k13c, k13f (size-specialised, one build per n) × 5 configs |
| `k14.json`, `k14.txt`, `k14.code/` | E8 | `fc-01M6M8ZEJ5X1FG2STSC9Y57J28` | k11, k13f, k14 (Stream-K), k14f × 5 configs |
| `confirm.json`, `confirm.txt`, `confirm.code/` | E9 | `fc-01M6M946AGWF2FNH34AY7NK35X` | the beat claim: 15 alternating rounds against cuBLAS |

**Code for the earlier runs.** Runs before k12 didn't save code snapshots. Their kernels can be rebuilt from today's `matmul_cp_async.cu` through switches:
- the variants in `matmul08.py → VARIANTS`;
- the `/old` variants (`BSTRIDED=0`), which reproduce the kernels 8 and 9 B-column layout.

Every run from k12 on saves its code next to its results.

## ncu/: Nsight Compute reports (`modal run matmul08.py::profile --set <name>`)

| Folder | Experiment | Reports | Notes |
|---|---|---|---|
| `01-k8-pipeline-n1024/` | E1 | `k8-s1` | Original kernel 8 code. The matching original `k8-s3`/cuBLAS were overwritten; equivalents are in 02. |
| `02-k9-prefetch-n1024/` | E1–E2 | `k8-s3`, `k8t-s3`, `k9-s3`, `k9t-s3`, `k8-cublas` | 128×64 BK8 S=3, n = 1,024 |
| `03-swizzle-n4096/` | E3 | `n4k-k9s`, `n4k-k9t`, `n4k-k9ts` | **Partial.** The set's `n4k-k9` and cuBLAS were overwritten later. |
| `03-swizzle-n4096-rebuild/` | E3 | `n4k-k9`, `n4k-k9s`, `n4k-k9t`, `n4k-k9ts`, `n4k-cublas` | **Complete rebuild** using the `/old` code path, with `code/`, `summary.txt` and `raw-metrics.csv.txt`. Use this one for figures. |
| `04-bigpatch-n4096/` | E4 | `n4k-k9` (128×128 BK8 S=4), `n4k-big8x16` | The set's `n4k-big16x8`/cuBLAS were overwritten by 05, which has the same config and the same code for that path. |
| `05-k11-hoist-n4096/` | E4–E5 | `n4k-big16x8` (k9TS), `n4k-k11`, `n4k-k11best`, `n4k-cublas` | |
| `06-k12-n4096/` | E6 | `n4k-k11-256x128`, `n4k-k12-256x128`, `n4k-k11-128x256`, `n4k-k12o-128x256`, `n4k-cublas` | With `code/`, `summary.txt` and `raw-metrics.csv.txt` |
| `07-k14-streamk-n1024/` | E8 | `n1k-k11-128x64`, `n1k-k14-128x64`, `n1k-k14-128x128`, `k8-cublas` | With `code/`. The Stream-K reports cover `matmul_streamk` only (`zero_tiles` isn't profiled). |

To rebuild any set completely: `modal run matmul08.py::profile --set <name> --tag <name>-rebuild` (sets are defined in `matmul08.py → PROFILE_SETS`).

## analysis/: source-level diagnosis

- `source-pages/*.sass.csv`: per-SASS-instruction exports (`ncu -i … --page source --csv --print-source sass`). They hold instructions executed and warp-stall samples per reason.
  - `05-n4k-big16x8` and `05-n4k-cublas`: the E4 diagnosis.
  - `05-n4k-k11best` and `05-n4k-cublas-b`: the E5 diagnosis. Same cuBLAS report, exported a second time.
  - `06-n4k-k11-*` vs `06-n4k-k12-256x128` and `06-n4k-k12o-128x256`: the E7 diagnosis (filler instructions).
  - `course-lecture10-numba-k5-T4` and `-k7-T4`: the course's Numba kernels (Shubham's `lecture10/*.ncu-rep`, Tesla T4), for the loop-unrolling comparison. Each file holds several kernels; the matmul is the `__main__::matmul…` section.
- `scripts/`:
  - `ncu_read*.py` and `ncu_src.py`: the CPU-only Modal jobs that ran `ncu --import` on saved reports. They point at the report paths as they were at the time, before the reorganisation into `ncu/`.
  - `ncu_numba*.py`: reads the lecture10 Numba reports (registers, local memory, static and executed SASS).
  - `sass_hotspots.py`: summarizes a `.sass.csv` (stall shares, an opcode table, and the hottest wait-stalled instructions).

## Complete numbered-kernel coverage — 2026-11-13

[Collection index](ncu/coverage-k8-k14-20261113-v2/README.md): 24 full Nsight reports covering k8–k14 plus cuBLAS at n=1024, 2048, 4096; 21 passing correctness checks; separate unprofiled timing CSV; source snapshots and report hashes. Each k14 report includes initialization. This is representative generic coverage, not an exhaustive configuration sweep. The earlier non-v2 folder is a preserved incomplete attempt.
