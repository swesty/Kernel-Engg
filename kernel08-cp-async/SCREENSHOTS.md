# Screenshot guide for the report (Nsight Compute GUI)

Every figure below names the report file(s), the Nsight Compute page and section to capture, the baseline to set (if any), and what the screenshot should show. All paths are under `results/ncu/`. The numbers to look for come from `LAB_NOTEBOOK.md`, and each set's `summary.txt` has them in text form for the captions.

**How to compare two reports:**
1. Open both `.ncu-rep` files.
2. In the report you want as the reference, click **Add Baseline** in the toolbar, or use Baseline → Add Baseline.
3. Switch to the other report. Every metric on its **Details** page now shows the difference against the baseline, with bars.

You can stack several baselines; each gets a colour.

**Pages used:**
- **Details:** the summary sections.
- **Source:** set **View: SASS**. Then add or sort by the columns "Warp Stall Sampling (All Samples)", "Instructions Executed" and the per-reason `stall_*` columns.
- **Raw:** every metric. Use it for the exact names cited in the text.

---

### Fig 1. Pipelining removes the wait on global memory (E1)
- **Reports:** `01-k8-pipeline-n1024/k8-s1.ncu-rep` vs `02-k9-prefetch-n1024/k8-s3.ncu-rep`, with `k8-s1` as the baseline.
- **Details → Warp State Statistics:** the stall chart. Long Scoreboard dominates in S=1 and nearly disappears in S=3.
- **Details → GPU Speed Of Light Throughput:** duration 132 → 79 µs.
- **Caption:** long scoreboard 46.8% → 2.5% of warp cycles; issue active 28.8% → 52.0%.
- **Optional third panel:** add `02-…/k8-cublas.ncu-rep` (6.6% long scoreboard) to show S=3 waits *less* than cuBLAS.

### Fig 2. What's left after pipelining: short-scoreboard stalls (E1 → E2)
- **Reports:** `02-k9-prefetch-n1024/k8-s3`, with `02-…/k8-cublas` as the baseline.
- **Details → Warp State Statistics:** Stall Short Scoreboard is far larger than cuBLAS's (0.38 vs 0.10 cycles per issued instruction).
- **Details → Instruction Statistics:** instruction counts are nearly equal (40.1 M vs 41.5 M), so the problem is stalls, not work.
- **Details → Launch Statistics / Occupancy:** 128 blocks of 128 threads gives one warp per scheduler, which is why nothing hides the stall.

### Fig 3. Register fragment prefetch (E2)
- **Reports:** `02-k9-prefetch-n1024/k8-s3`, `k8t-s3`, `k9-s3` and `k9t-s3`, with `k8-s3` as the baseline.
- **Details → Warp State Statistics** for each:
  - short scoreboard 0.38 → 0.39 (transpose only) / 0.26 (prefetch only) / **0.13** (both);
  - issue active 52% → 62%.
- **Source (SASS), `k9t-s3`:** the `LDS.128` for step k+1 issued before step k's FFMA run, which is the prefetch visible in the instruction order.

### Fig 4. The L2 hypothesis, falsified (E3)
- **Reports:** use the complete rebuild, `03-swizzle-n4096-rebuild/`: `n4k-k9t` vs `n4k-k9ts`, and `n4k-k9` vs `n4k-k9s`.
- **Details → Memory Workload Analysis → Memory Chart:** DRAM read bytes drop (963 → 455 MB; 596 → 463 MB) and the L2 hit rate rises.
- **Details → GPU Speed Of Light Throughput:** duration doesn't improve (3.5 → 3.3 ms and 3.1 → 3.5 ms).
- **Caption:** fewer bytes from DRAM, same time, so memory wasn't the limit.

### Fig 5. cuBLAS changes shape at n = 4,096; bigger thread patches (E3 → E4)
- **Reports:** `04-bigpatch-n4096/n4k-k9` (8×8 patches) vs `05-k11-hoist-n4096/n4k-big16x8` (16×8), with `05-…/n4k-cublas` as the baseline.
- **Details → Launch Statistics:** cuBLAS runs 256 threads, grid 16 × 32 (a 128×256 tile) and 254 registers.
- **Details → Instruction Statistics:** shared loads 336 M → 101 M, matching cuBLAS.
- **Caption:** n = 4,096 rises from 82% to 90% of cuBLAS.

### Fig 6. The address-math overhead, found on the Source page (E4 → E5)
- **Report:** `05-k11-hoist-n4096/n4k-big16x8`, **Source page, View: SASS**, sorted by Instructions Executed, then by `stall_wait`.
  - Show the integer ops in the main loop: `LEA`, `IMAD`, `LDC` (constant loads of `K`/`N`) and `IADD3` around the `LDGSTS`.
- **Same view of `05-…/n4k-cublas`**, for contrast: only `IADD3` and the uniform-datapath ops (`UIADD3`, `UISETP`, `UMOV`), with no `LEA`/`IMAD`/`LDC`.
- **Text version:** `results/analysis/source-pages/05-*.sass.csv`, summarized by `analysis/scripts/sass_hotspots.py`.

### Fig 7. Hoisting the copy plan (E5)
- **Reports:** `05-k11-hoist-n4096/n4k-k11` (same tile) and `n4k-k11best`, with `n4k-big16x8` as the baseline.
- **Details → Compute Workload Analysis:** the ALU pipe drops from 9.3% to about 1%.
- **Details → Instruction Statistics:** instructions 2,513 M → 2,308 M, below cuBLAS's 2,428 M.
- **Details → Launch Statistics:** registers 255 → 231.
- **Caption:** n = 4,096 at 95%, n = 1,024 at 98%.

### Fig 8. The serial pointer chain kernel 11 still had (E5 → E6)
- **Report:** `05-k11-hoist-n4096/n4k-k11best`, **Source page, SASS**, sorted by `stall_wait`.
- **What to capture:** the run of `IMAD.WIDE Rx, R29, 0x40, Ry`, each feeding the next and then an `LDGSTS`. That's the dependent 64-bit pointer chain, issued in one burst.

### Fig 9. Spreading the copies; where the gap is now (E6)
- **Reports:** `06-k12-n4096/n4k-k11-128x256` vs `n4k-k12o-128x256`, with `06-…/n4k-cublas` as the baseline.
- **Details → Warp State Statistics:**
  - dispatch stall 0.21 → 0.15 (cuBLAS 0.10);
  - wait 0.08 → 0.05 (cuBLAS 0.01);
  - issue active 82% → 85% (cuBLAS 90%).
- **Details → Instruction Statistics:** instructions 2,308 M → 2,378 M (cuBLAS 2,428 M).
- **Caption:** fewer stalls, more instructions. Net zero at n = 4,096; parity with cuBLAS at n = 1,024.
- **Source page of `n4k-k12o-128x256`, sorted by `stall_dispatch`:** about three quarters of the dispatch samples sit on `FFMA`, the last 5%.

### Fig 10. Why the leftover gap didn't move: compiler filler (E7)
- **Report:** `06-k12-n4096/n4k-k12-256x128`, **Source page, SASS**.
- **What to capture:** the `@!PT LDS RZ, [RZ]` lines (never-true predicate, issue slot spent) and the `@!UP0 ULDC` + `MOV` parameter reloads around the spread `LDGSTS` copies.
- **Contrast:** the same region in `06-…/n4k-k11-256x128`.
- **Text version:** `analysis/source-pages/06-*.sass.csv`.

### Fig 11. Beating cuBLAS at n = 1,024: Stream-K doubles the warps per scheduler (E8)
- **Reports:** `07-k14-streamk-n1024/n1k-k11-128x64` vs `n1k-k14-128x128`, with `07-…/k8-cublas` as the baseline.
- **Details → Scheduler Statistics:** active warps per scheduler 1.0 → 2.0; issue slots busy 63% → 72% (cuBLAS 70%).
- **Details → Warp State Statistics:** per-warp stalls go *up* (2.77 vs 1.60 cycles per instruction), but the second warp covers them. This is the counter-intuitive panel worth explaining.
- **Details → Launch Statistics:** grid 264 persistent CTAs (2 per SM) for k14, against cuBLAS's 128 blocks.
- **Details → GPU Speed Of Light:** duration 63.8 µs (k11) and 54.4 µs (k14) vs 61.4 µs (cuBLAS). Under ncu, k14's `zero_tiles` pre-pass is a separate kernel and isn't included; the `::confirm` times are what to quote.

### Fig 12 (not a screenshot): the confirmation table
- `results/runs/confirm.txt`: 15 alternating rounds, with k14 faster than cuBLAS in 15 of 15 (105.0–105.8%). Use it as a table or a strip plot.

---

**Tips for the captures:**
- Nsight Compute's Details sections can be expanded. Use **Copy as image** on a section's context menu, if your version has it; otherwise take a window screenshot.
- Keep the report name visible in the tab, so each figure can be traced to its file.
- The times shown by ncu are under replay and differ slightly from the `::main` timings. Quote `::main` times (from `results/runs/*.txt`) for performance and ncu numbers for mechanisms. The notebook does the same.
