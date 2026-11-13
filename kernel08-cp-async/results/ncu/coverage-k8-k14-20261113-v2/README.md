# Complete H100 collection: kernels 8–14

Completed 2026-11-13. Modal app: https://modal.com/apps/swesty/main/ap-4hwRLA2xK8xClpxJ0yXAMa

- **24 Nsight Compute reports**: seven numbered kernels plus cuBLAS at each of n=1024, 2048, 4096.
- **21 correctness checks passed**, finite output after NaN initialization, relative error < 1e-4.
- **24 unprofiled timing rows**, 15 alternating rounds × 40 L2-flushed launches per round.
- H100 80GB HBM3 confirmed for each size; each size's timing and profiling used one GPU allocation.
- Each `k14.ncu-rep` includes `zero_tiles` and `matmul_streamk`. Timings include both launches.
- Manifest SHA256 checksums and source snapshot checksums verified locally.

## Open the data

- `n1024/`, `n2048/`, `n4096/`: open `.ncu-rep` in Nsight Compute. Each folder also contains complete raw CSV exports and detailed timing/correctness JSON.
- [timings.csv](timings.csv): combined unprofiled timings, including explicit numbered-kernel labels.
- [metrics-summary.csv](metrics-summary.csv): selected Nsight counters in long format with original units. Full counters remain in each raw export and report.
- [manifest.json](manifest.json): report hashes, captured kernel names and GPU/run identities.
- `code/`: exact harness and CUDA source used by this run.

## Scope and interpretation

This covers one representative generic configuration per numbered kernel at three sizes, **not every variant or tile/stage combination**. k10 is the larger-thread-patch `k9TS` path; k13 is generic, not size-specialized `k13f`. See the per-size `timings.json` `runs` mapping for exact parameters. These are not necessarily the best configurations at each size.

Use unprofiled `timings.csv` for performance and the Nsight reports for mechanisms. Nsight replay timings are separate from application timing. GPU clocks were not locked. cuBLAS has TF32 disabled and uses profiler markers around one warmed-up matmul, avoiding assumptions about kernel names.

The earlier `coverage-k8-k14-20261113` folder is an incomplete first attempt, superseded by this collection. Its cuBLAS n=2048 report was missed by the old name filter; this run fixes that with profiler markers.
