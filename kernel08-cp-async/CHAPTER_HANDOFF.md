# Handoff: writing the "beyond kernel 7" chapter (for a fresh session)

**Goal.** Write a chapter in the style of the course book's GEMM chapter (book ch. 5, as captured in the vault note `3-Resources/courses/Kernel Engineering/Week 3-1 - Kernel Engineering - GEMM worklog II.md`). It should explain every optimisation from kernel 8 to kernel 14, with figures, and pick up where the course stopped: kernel 7 at 86.6 µs vs cuBLAS 56.1 µs. Shawn will also pull Nsight Compute screenshots for it. Nothing in this file has been written as chapter prose yet.

## Read these first (sources of truth)

| File | What it gives the chapter |
|---|---|
| `LAB_NOTEBOOK.md` | Every experiment E0–E9: question, hypothesis, the prediction made before the run, change, run ID, result tables, verdict, plus the roofline section and the open options. **The chapter's facts come from here.** |
| `SCREENSHOTS.md` | Figs 1–12: which `.ncu-rep`, which Nsight page and section, which baseline, and the numbers each caption needs. |
| `results/MANIFEST.md` | Every result file mapped to its experiment and run; notes which early reports were overwritten and rebuilt. |
| `matmul_cp_async.cu` | The one kernel file. Each kernel is a set of switches; the comment block at the top documents every switch and why it exists. Code excerpts for the chapter come from here: `load_chunk`, `load_frag`, `make_plan` and `issue_chunk`, `issue_copy_z`, the kernel 9/11/12 main loops in `tile_body`, `matmul_streamk`. |
| `matmul08.py` | `VARIANTS` maps every kernel name to its switches; `PROFILE_SETS`; `::main`, `::profile`, `::confirm`. |
| `BIG_WINS.md` | One-page ranked summary of the wins beyond the course kernels: the chapter's "key takeaways" and the blog's spine. |
| `results/runs/confirm.txt` | **The numbers to quote** for the final standings. |

**Style to copy, from the vault note:**
- numbered sections, each going idea → mechanism → code → measured;
- short paragraphs;
- figures with dense captions that carry the numbers ("*Book Fig 5.31 — … 43% vs 7.5% …*");
- a "Key takeaways" list at the top and a "verification flags" section.

**Vault rules apply if the chapter goes in the vault:**
- filenames of 75 characters or fewer;
- **never hard-wrap prose**: one paragraph per line;
- images in `ZETA/Attachments`. A plugin converts new PNGs to `<hash>_MD5.jpeg` and can break links, so check every `![[…]]` afterwards (memory: `reference_local_images_plus_converts_attachments`).

## Proposed outline (one section per rung of the ladder)

Figure types: **[D]** data chart generated from `results/` · **[C]** concept diagram, drawn programmatically · **[S]** Nsight screenshot (the number is the Fig in `SCREENSHOTS.md`).

0. **Where the course left off.** Kernel 7 vs cuBLAS, and the cuBLAS kernel name decoded (`tilesize128x64x8_stage3_…_ffma`). Defines n (square, M = N = K) and the method: timing, correctness and profiling rules, predictions written before each run.
   - [D] the whole ladder at n = 1,024 / 2,048 / 4,096 as % of cuBLAS, from kernel 7 to kernel 14 (data: `results/runs/k8…k14.json`, `confirm.json`; book K7 from Week 3-1).
1. **Kernel 8: the `cp.async` pipeline (E1).** `LDGSTS`, the ring of STAGES slots, why one barrier does two jobs.
   - [C] a timeline of copy and compute for S=1 vs S=3.
   - [C] the ring buffer: which slot is filled while which is read.
   - [S] Fig 1, long scoreboard 46.8% → 2.5%.
2. **What pipelining exposed (E1 → E2).** Short scoreboard, one warp per scheduler.
   - [S] Fig 2.
   - [D] stall breakdown, k8 vs cuBLAS, stacked bars (data: `ncu/02…/` raw metrics; the stall table in E1).
3. **Kernel 9: register fragment prefetch, plus transposed A (E2).** CUTLASS `mac_loop_iter`; the wait and barrier moved to the last k-step; why the transpose only pays together with prefetch; bank math for the padded, strided layouts.
   - [C] a two-register-buffer timeline.
   - [C] the A-tile layouts: un-transposed with padding and strided rows, vs transposed with 4-byte copies.
   - [S] Fig 3.
   - [D] the 2×2 factorial (k8, k8+T, k9, k9+T): time and short scoreboard.
4. **A falsified hypothesis: L2 reuse and block swizzling (E3).** Keep it in; it shows the method. Fewer DRAM bytes, same time.
   - [C] launch order, row-major vs grouped.
   - [S] Fig 4.
   - [D] DRAM MB vs time for the four variants (data: `ncu/03-swizzle-n4096-rebuild/summary.txt`).
5. **Kernel 10: bigger thread patches (E4).** cuBLAS changes shape at n = 4,096; register AI 2.0 → 2.67; strided B columns for bank conflicts.
   - [S] Fig 5.
   - [D] shared loads per kernel (336 M → 101 M).
6. **Kernel 11: hoisting the address math (E4 → E5).** Finding `LEA`, `IMAD` and `LDC` on the Source page; registers 255 → 231.
   - [S] Figs 6 and 7.
   - [D] overhead instructions per FFMA (0.243 → 0.170 → 0.075, cuBLAS 0.131) and the ALU pipe.
   - Include the prediction scorecard, with the wait-stall miss.
6b. **Sidebar: loop unrolling** (notebook section). Why every constant-trip loop is unrolled, what it enables (registers, the double-buffer index, immediates), and what Numba did on its own, measured from the course's lecture10 profiles. [D] static FFMAs, 64 vs 2,048, and overhead per FFMA, 1.08 vs 0.13.
7. **Kernel 12 and 13: spreading copies, and compiler filler (E6–E7).** The serial `IMAD.WIDE` chain; spreading cuts stalls but adds instructions; `@!PT LDS RZ,[RZ]` and `ULDC` filler from a branch; branch-free zero-fill copies; contiguous-B and size-specialisation tests. Shows where CUDA C stops giving.
   - [S] Figs 8 and 10.
   - [D] the opcode delta, k11 → k12 (data: `analysis/source-pages/06-*.sass.csv` via `analysis/scripts/sass_hotspots.py`).
8. **Kernel 14: Stream-K, and beating cuBLAS at n = 1,024 (E8–E9).** Wave quantisation (3.88 vs 4 waves); the persistent kernel; the split of leftover K-work; atomics and `zero_tiles`; why it wins at n = 1,024 (two warps per scheduler; per-warp stalls go *up*, SM issue goes up); why it loses at n ≥ 2,048 for now.
   - [C] the Stream-K work split: tiles × K-chunks coloured by CTA (generate it from the host-side model in `LAB_NOTEBOOK.md` E8).
   - [C] the SM / scheduler occupancy picture, 1 vs 2 warps.
   - [S] Fig 11.
   - [D] the confirmation strip plot: 15 rounds, k14 vs cuBLAS (data: `results/runs/confirm.json`).
9. **The roofline view (notebook section "Roofline position").** n/6 ideal AI, the ridge at 20.2, all points far right; gains moved points up, not right. Include the 255 → 170 correction as a verification flag.
   - [D] the roofline: measured ridge, k-ladder points, cuBLAS points.
10. **What's left.** Issue rate, register-bank assignment, SASS-level tools; open options 1–2 from the notebook.

**Key numbers to quote** (from `confirm.txt`):
- n = 1,024: k14 61.3 µs vs cuBLAS 64.6 µs, 105.3% (105.0–105.8%), faster in 15/15 rounds.
- n = 2,048: 95.4%, size-specialised k13f; the generic k11 reaches 93.9%.
- n = 4,096: 96.4%, size-specialised k13f; the generic k11 reaches 95.2%.

Quote `::main` and `::confirm` times for performance and ncu numbers for mechanisms; ncu replays each kernel, so its times differ.

## Making the figures

- **[D] and [C] charts:** load the `dataviz` skill first, as the session rules require. matplotlib 3.10 is in `../assignment1/.venv` (`../assignment1/.venv/bin/python`). Put generated figures and their script in `figures/`, so they regenerate from `results/`.
- **[S] screenshots:** Shawn takes these in the Nsight Compute GUI, following `SCREENSHOTS.md`.
- **Book figures referenced:** the course's ch. 5 figures (`KE-ch5-fig5.*.png`) are already in the vault, if the chapter wants to cite the kernel 7 state.

## Open items the chapter must state honestly

- Generic kernels at n ≥ 2,048 are 94–95% of cuBLAS, and size-specialised ones 95–96%. The win is only at n = 1,024.
- Some reports in `ncu/01`, `03` and `04` are partial because of early overwrites (see MANIFEST). `03-…-rebuild` is complete.
- Unexplained: the `+T` variants are slow at S = 2 and 4 in E2.
- Stream-K's atomics make results non-bit-reproducible (they match to < 1e-4).
- Untested: rectangular shapes, sizes that aren't tile multiples, and any GPU other than the H100.
- Nothing is committed to git yet. Ask Shawn before committing.

## Also planned: a blog post built from the same material

Shawn wants a blog post too, framed as the sequel to Simon Boehm's [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) (Dec 2022). The seed, with title options and the story arc, is in the vault: `2-Areas/Writing/Newsletter Seeds.md`, "Ready to Write" #3.

**Facts checked from his post (2026-11-04):**
- Best kernel: 10 (warptiling), **93.7%** of cuBLAS (21,779 vs 23,250 GFLOP/s), FP32, **RTX A6000, 4092²**.
- He skipped double buffering: "putting off hunting the last 6%".
- Thread swizzling gave no improvement for him.
- cuBLAS uses split-K at small sizes.

**Fairness rule for the post:** his 93.7% (A6000, n = 4092) and our 105.3% (H100, n = 1,024) aren't the same measurement. Say so in the opening. The like-for-like baseline is the course's kernel 7 on the H100, at 65%.

The chapter's figures should be made so the post can reuse them. The blog repo is gitea `swesty/blogs` (blog.swesty.net).

### Lineage: Boehm's kernels → the course's kernels → ours (checked against his post, 2026-11-04)

| Technique | Boehm (A6000) | Course (H100) | Ours (k8–k14) |
|---|---|---|---|
| Coalesced global loads | k2 | k2 | inherited; now `cp.async` 16-byte copies (A: 4-byte copies when transposed) |
| Shared-memory tiling | k3 | k3 | inherited; became a STAGES-deep ring |
| 2D thread patch, register-cached outer product (`regM`/`regN`) | k5 (8×8) | k5 | inherited (`load_frag` + `outer_product`); bigger 16×8 / 8×16 patches in k10 |
| `float4` / `LDS.128` vectorisation | k6 | k6 | inherited |
| A stored transposed in shared memory | k6 (scalar strided stores) | k6 | same idea, done with 4-byte `cp.async` (`ATRANS`); measured: it pays only together with prefetch (E2) |
| Autotuned BM/BN/BK/TM/TN | k9 (~400 combinations) | k7 | config × STAGES sweeps every run |
| Warp tiling | k10 (93.7%) | k7 (65% on H100) | inherited |
| Shared-memory bank conflicts | k7/k8: removed conflicts but slower, so dropped | — | free versions: padding (BK+4 / BM+4) plus strided rows and columns; contiguous B was up to 10% slower (E7) |
| Thread/block swizzle for L2 | tried, no gain (L2 hit ≈ 80%), removed | — | tried, no gain (E3: DRAM −22–53%, same time; compute-bound) |
| GMEM→SMEM double buffering | **planned** ("kernel 11, WIP") | — | **k8**: multi-stage `cp.async` ring (S = 2–4) |
| SMEM→register double buffering | **planned** | — | **k9**: two-buffer fragment prefetch (CUTLASS `mac_loop_iter`) |
| Address-math hoisting, found on the SASS Source page | — | — | k11 |
| Copies spread over k-steps; branch-free zero-fill | — | — | k12 / k13 |
| Size specialisation (`FIXED_N`) | — | — | k13f (flagged as not like-for-like) |
| Split-K / Stream-K | observed cuBLAS using split-K at n = 256 | — | **k14**: Stream-K, the win at n = 1,024 |
| Loop unrolling | not checked in his post | book CUDA C: `#pragma unroll`; Numba lab: inner TM×TN loops auto-unrolled, **k-step loop not unrolled**, scalar `LDS` | everything with a constant trip count (26 pragmas): k-step loop, outer product, copies, epilogue. Static FFMAs 2,048 vs Numba's 64; overhead per FFMA 0.13 vs 1.08 (notebook, "Cross-cutting technique: loop unrolling") |
| Inspection method | SASS/PTX on Godbolt | Nsight Compute | Nsight Compute, plus per-SASS stall sampling (`analysis/`) |
