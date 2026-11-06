"""Kernels 8 and 9 (kernel 7 + a cp.async pipeline, + register fragment prefetch), measured against cuBLAS on one H100.

    modal run matmul08.py::main        # correctness, timing sweep and SASS check in one GPU run (~3 min)
    modal run matmul08.py::profile --set 06-k12-n4096   # .ncu-rep + summary.txt + code/ to results/ncu/<set>/

Kernel 7 reached 86.6 us at n = 1,024 vs cuBLAS 56.1 us; its warps spent 43% of their time waiting on global memory,
cuBLAS's 7.5%. The cuBLAS kernel is "..._tilesize128x64x8_stage3_..._ffma": three K-chunks in flight.
  kernel 8  cp.async (SASS LDGSTS) copies global -> shared without registers and without blocking, into a ring of
            STAGES slots. Measured: wait on global 47% (STAGES=1) -> 2.5% (STAGES=3); 89% of cuBLAS at n = 1,024.
            Left over: 0.38 cycles per instruction stalled on shared-memory results (short scoreboard) vs 0.10.
  kernel 9  PREFETCH=1: two register buffers per operand; step k+1's fragments load while step k's FMAs run
            (CUTLASS mma_multistage.h, mac_loop_iter). ATRANS=1 stores A transposed so a fragment is two 128-bit loads.

cuBLAS here is torch.matmul in FP32 with TF32 off, so both sides use FFMA on the CUDA cores. Times are the median of
30 launches by CUDA events, L2 flushed before each (256 MB of zeros), as in Assignment 1. The image is Assignment 1's
(same digest, same pinned packages), so it is already built if you ran a1.
"""
import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GPU = "H100!"   # "!" pins a real H100: plain "H100" may be upgraded to an H200. TUNE: any sm_80+ GPU; a T4 has no cp.async
RESULTS = "k13"  # default tag: results/runs/<tag>.json + .txt

app = modal.App("k8-cp-async")
image = (modal.Image.from_registry(
             "nvidia/cuda:13.0.0-devel-ubuntu24.04"
             "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([])
         .pip_install_from_requirements(str(HERE.parent / "assignment1" / "image-requirements.txt"))
         .add_local_file(HERE / "matmul_cp_async.cu", "/root/matmul_cp_async.cu"))

with image.imports():
    import cupy as cp
    import numpy as np
    import torch

KERNEL_FILE = "/root/matmul_cp_async.cu"

# TUNE: block tile BMxBN, chunk depth BK, warp tile WMxWN, thread patch TMxTN. The first is cuBLAS's own shape.
CONFIGS = {
    "128x64 BK8":   dict(BM=128, BN=64,  BK=8,  WM=64, WN=32, TM=8, TN=8),   # 128 threads, 2x2 warps
    "128x128 BK8":  dict(BM=128, BN=128, BK=8,  WM=64, WN=32, TM=8, TN=8),   # 256 threads, 2x4 warps
    "128x64 BK16":  dict(BM=128, BN=64,  BK=16, WM=64, WN=32, TM=8, TN=8),
    "128x128 BK16": dict(BM=128, BN=128, BK=16, WM=64, WN=32, TM=8, TN=8),
    # Bigger thread patches (128 outputs per thread instead of 64): more FMAs per shared load and per address calc.
    # cuBLAS's n = 4,096 kernel uses 256 threads, a 128 x 256 tile and 254 registers.
    "256x128 BK8 16x8":  dict(BM=256, BN=128, BK=8,  WM=64, WN=64, TM=16, TN=8),    # 256 threads
    "256x128 BK16 16x8": dict(BM=256, BN=128, BK=16, WM=64, WN=64, TM=16, TN=8),
    "128x256 BK8 8x16":  dict(BM=128, BN=256, BK=8,  WM=64, WN=64, TM=8,  TN=16),   # cuBLAS's tile shape
    "128x256 BK16 8x16": dict(BM=128, BN=256, BK=16, WM=64, WN=64, TM=8,  TN=16),
}
# The two switches, separately and together, so each one's effect can be read off the table.
VARIANTS = {
    "k8":   dict(ATRANS=0, PREFETCH=0),    # kernel 8 as profiled
    "k8+T": dict(ATRANS=1, PREFETCH=0),    # + A stored transposed (4-byte cp.async)
    "k9":   dict(ATRANS=0, PREFETCH=1),    # + register fragment prefetch
    "k9+T": dict(ATRANS=1, PREFETCH=1),    # both
    "k9+S": dict(ATRANS=0, PREFETCH=1, SWIZZLE=8),   # + grouped launch order for L2 reuse
    "k9TS": dict(ATRANS=1, PREFETCH=1, SWIZZLE=8),   # all three
    "k11":  dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1),   # + hoisted copy plan, add-and-wrap ring
    "k11n": dict(ATRANS=0, PREFETCH=1, SWIZZLE=8, HOIST=1),   # same, A not transposed
    "k11o": dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1, ORDER=1),             # kernel 11, j-outer FFMA order
    "k12":  dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1, SPREAD=1),            # + copies spread over k-steps
    "k12o": dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1, SPREAD=1, ORDER=1),   # both
    "k13":  dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1, SPREAD=2),            # spread, branch-free (zero-fill)
    "k13c": dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1, SPREAD=2, BSTRIDED=0),  # + contiguous B columns
    "k13f": dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, HOIST=1, SPREAD=2, FIXED_N="per_n"),  # + SIZE-SPECIALISED (see .cu)
    # Kernels 8 and 9 exactly as first measured (contiguous B columns), for rebuilding the early profiles:
    "k8/old":   dict(ATRANS=0, PREFETCH=0, BSTRIDED=0),
    "k8+T/old": dict(ATRANS=1, PREFETCH=0, BSTRIDED=0),
    "k9/old":   dict(ATRANS=0, PREFETCH=1, BSTRIDED=0),
    "k9+T/old": dict(ATRANS=1, PREFETCH=1, BSTRIDED=0),
    "k9+S/old": dict(ATRANS=0, PREFETCH=1, SWIZZLE=8, BSTRIDED=0),
    "k9TS/old": dict(ATRANS=1, PREFETCH=1, SWIZZLE=8, BSTRIDED=0),
}
SWEEP = ["k11", "k12", "k13", "k13c", "k13f"]   # TUNE: which VARIANTS ::main times (earlier sweeps: results/runs/)
SWEEP_CONFIGS = ["128x64 BK16", "128x128 BK16", "256x128 BK16 16x8", "128x256 BK8 8x16", "128x256 BK16 8x16"]  # TUNE
STAGES_LIST = [1, 2, 3, 4]      # TUNE: 1 = no overlap (kernel 7's schedule), 3 = cuBLAS's choice; S=1 only for k8
SIZES = [1024, 2048, 4096]      # TUNE: square n x n x n


# ============================================================================================== helpers (on the GPU)
def smem_bytes(d):
    """Dynamic shared memory for one set of defines (must match A_TILE + B_TILE in the .cu file)."""
    a_tile = d["BK"] * (d["BM"] + 4) if d.get("ATRANS") else d["BM"] * (d["BK"] + 4)
    return d["STAGES"] * (a_tile + d["BK"] * d["BN"]) * 4


def why_invalid(d, optin):
    """Why a set of defines can't run, or None if it can."""
    BM, BN, BK, WM, WN, TM, TN = (d[k] for k in ("BM", "BN", "BK", "WM", "WN", "TM", "TN"))
    nt = (BM // TM) * (BN // TN)
    if any(x % 4 for x in (BK, BN, WN, TN, TM)):
        return "BK, BN, WN, TM, TN must be multiples of 4 (16-byte copies and loads)"
    if BM % WM or BN % WN or WM % TM or WN % TN:
        return "tiles must divide evenly"
    if (WM // TM) * (WN // TN) != 32:
        return "a warp tile must hold exactly 32 thread patches"
    if (BM // WM) * (BN // WN) * 32 != nt or nt > 1024:
        return "warp tiles must cover the block tile with <= 1,024 threads"
    if d.get("PREFETCH") and d["STAGES"] < 2:
        return "PREFETCH refills a slot while the next is read: needs STAGES >= 2"
    if d.get("SPREAD") and not d.get("HOIST"):
        return "SPREAD is built on the HOIST path"
    if d.get("HOIST"):
        if not d.get("PREFETCH"):
            return "HOIST is built on the PREFETCH path"
        a_copies, a_width = (BM * BK, BK) if d.get("ATRANS") else (BM * BK // 4, BK // 4)
        if a_copies % nt or nt % a_width or (BK * BN // 4) % nt or nt % (BN // 4):
            return "HOIST needs NT to divide the copy counts and the copy-row widths"
    if smem_bytes(d) > optin:
        return f"{smem_bytes(d)} B of shared memory > {optin} B limit"
    return None


def build(src, d):
    """Compile one set of defines with NVRTC; return (bind, registers). bind(A, B, C) -> zero-argument launch."""
    head = "".join(f"#define {k} {v}\n" for k, v in d.items())
    kern = cp.RawKernel(head + src, "matmul_cp_async", options=("-std=c++17",))
    smem = smem_bytes(d)
    if smem > 48 * 1024:
        kern.max_dynamic_shared_size_bytes = smem          # opt in above the default 48 KB (this also compiles)
    else:
        kern.compile()
    nt = (d["BM"] // d["TM"]) * (d["BN"] // d["TN"])

    def bind(A, B, C):
        (M, K), N = A.shape, B.shape[1]
        assert M % d["BM"] == 0 and N % d["BN"] == 0 and K % d["BK"] == 0, "sizes must be multiples of the tile"
        args = (cp.from_dlpack(A), cp.from_dlpack(B), cp.from_dlpack(C), np.int32(M), np.int32(N), np.int32(K))
        grid, block = (N // d["BN"], M // d["BM"]), (nt,)
        return lambda: kern(grid, block, args, shared_mem=smem)
    return bind, (kern.num_regs, kern.local_size_bytes)


_SCRUB = []


def time_us(fn, reps=30, warmup=3):
    """Median GPU time of fn in microseconds; the L2 is flushed before each launch."""
    import statistics

    if not _SCRUB:
        _SCRUB.append(torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda"))
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        _SCRUB[0].zero_()                     # flush; also keeps the GPU busy while Python issues the launch
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def operands(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(n, n, device="cuda", generator=g)
    B = torch.randn(n, n, device="cuda", generator=g)
    return A, B, torch.empty(n, n, device="cuda")


def sass_counts(d, cc):
    """Compile the same file with nvcc and count the instructions that tell the story."""
    import subprocess

    defs = [f"-D{k}={v}" for k, v in d.items()]
    out = "/tmp/k.cubin"
    subprocess.run(["nvcc", f"-arch=sm_{cc}", "-cubin", "-o", out, *defs, KERNEL_FILE], check=True)
    sass = subprocess.run(["cuobjdump", "-sass", out], capture_output=True, text=True, check=True).stdout
    pats = {"LDGSTS": r"\bLDGSTS\b", "LDG": r"\bLDG\.", "LDS.32": r"\bLDS\s", "LDS.128": r"\bLDS\.128\b",
            "FFMA": r"\bFFMA\b", "BAR": r"\bBAR\.SYNC"}
    return {k: len(re.findall(v, sass)) for k, v in pats.items()}


# ========================================================================================================= the run
@app.function(gpu=GPU, image=image, timeout=2400)
def run_gpu():
    torch.backends.cuda.matmul.allow_tf32 = False     # cuBLAS must do true FP32 (FFMA), like our kernel
    torch.backends.cudnn.allow_tf32 = False
    p = torch.cuda.get_device_properties(0)
    cc = p.major * 10 + p.minor
    facts = {"gpu": p.name, "sm": cc, "sms": p.multi_processor_count, "torch": str(torch.__version__),
             "run_id": modal.current_function_call_id(), "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    print(f"{p.name} sm_{cc} ({p.multi_processor_count} SMs), run {facts['run_id']}")
    assert cc >= 80, f"sm_{cc} has no cp.async (needs sm_80+): set GPU to H100, A100 or L4"

    src = open(KERNEL_FILE).read()
    optin = cp.cuda.runtime.deviceGetAttribute(97, 0)  # cudaDevAttrMaxSharedMemoryPerBlockOptin
    kernels, regs, skipped = {}, {}, []
    for v in SWEEP:
        vd = VARIANTS[v]
        per_n = vd.get("FIXED_N") == "per_n"                 # size-specialised: one build per problem size
        for name in SWEEP_CONFIGS:
            c = CONFIGS[name]
            for s in STAGES_LIST:
                if s == 1 and not v.startswith("k8"):
                    continue                    # S=1 is only kernel 8's no-overlap baseline
                for n0 in (SIZES if per_n else [0]):
                    d = dict(c, STAGES=s, **dict(vd, FIXED_N=n0) if per_n else vd)
                    if not per_n:
                        d = dict(c, STAGES=s, **vd)
                    why = why_invalid(d, optin)
                    if why:
                        skipped.append(f"{v} {name} S={s}: {why}")
                        continue
                    kernels[(v, name, s, n0)], regs[(v, name, s)] = build(src, d)
    print(f"compiled {len(kernels)} kernels")

    # Correctness: C starts as NaN, so an unwritten tile fails too.
    failures = []
    for n in (1024, 2048, 4096):
        A, B, C = operands(n, seed=n)
        ref = A @ B
        for key, bind in kernels.items():
            if key[3] != n and not (key[3] == 0 and n < 4096):   # generic kernels: 1,024 and 2,048; fixed: own n
                continue
            C.fill_(float("nan"))
            bind(A, B, C)()
            torch.cuda.synchronize()
            err = ((C - ref).abs().max() / ref.abs().max()).item()
            if not (bool(torch.isfinite(C).all()) and err < 1e-4):
                failures.append(f"n={n} {key[0]} {key[1]} S={key[2]}: relative error {err:.2e}")
        del A, B, C, ref
    print("all kernels match cuBLAS" if not failures else f"{len(failures)} FAILURES")
    # (generic kernels are checked at n = 1,024 and 2,048; size-specialised ones at their own n, 4,096 included)

    rows = []
    for n in SIZES:
        A, B, C = operands(n)
        cublas = time_us(lambda: torch.matmul(A, B, out=C))
        rows.append({"variant": "cuBLAS", "config": "", "stages": 0, "n": n, "us": cublas})
        for (v, name, s, n0), bind in kernels.items():
            if n0 in (0, n):
                rows.append({"variant": v, "config": name, "stages": s, "n": n, "us": time_us(bind(A, B, C))})
        print(f"  n={n} timed")

    first = next(iter(CONFIGS.values()))
    sass = {v: sass_counts(dict(first, STAGES=3, **{k: (1024 if x == "per_n" else x) for k, x in VARIANTS[v].items()}), cc)
            for v in SWEEP}
    return {"facts": facts, "failures": failures, "skipped": skipped, "rows": rows, "sass": sass,
            "regs": {f"{k[0]}|{k[1]}|{k[2]}": r[0] for k, r in regs.items()},
            "spill": {f"{k[0]}|{k[1]}|{k[2]}": r[1] for k, r in regs.items()}}


@app.local_entrypoint()
def main(tag: str = RESULTS):
    """Runs the sweep; writes results/runs/<tag>.json (raw) and <tag>.txt (the tables printed here)."""
    runs = HERE / "results" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if (runs / f"{tag}.json").exists():
        tag = f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}"     # never overwrite an earlier run
    lines = []

    def say(x=""):
        print(x)
        lines.append(str(x))

    snapshot(runs / f"{tag}.code")
    r = run_gpu.remote()
    (runs / f"{tag}.json").write_text(json.dumps(r, indent=1))
    f = r["facts"]
    say(f"\n{f['gpu']} (sm_{f['sm']}, {f['sms']} SMs), torch {f['torch']}, {f['utc']}, run {f['run_id']}")
    say(f"skipped {len(r['skipped'])} invalid combinations (PREFETCH needs STAGES >= 2, shared-memory limits)")
    if r["failures"]:
        say("CORRECTNESS FAILURES (do not trust the timings below):")
        for x in r["failures"]:
            say(f"  {x}")
    else:
        say("correctness: every kernel matches cuBLAS (relative error < 1e-4): generic kernels at n = 1,024 and 2,048, size-specialised ones at their own n")

    rows = r["rows"]
    cub = {row["n"]: row["us"] for row in rows if row["variant"] == "cuBLAS"}
    t = {(row["variant"], row["config"], row["stages"], row["n"]): row["us"] for row in rows}
    keys = list(dict.fromkeys((row["variant"], row["config"], row["stages"]) for row in rows
                              if row["variant"] != "cuBLAS"))
    say(f"\n{'variant':<7}{'config':<19}{'S':>3}{'regs':>5}{'spill':>6}"
          + "".join(f"{f'n={n} us':>13}{'%cuB':>6}" for n in SIZES))
    say(f"{'cuBLAS':<7}{'':<19}{'':>3}{'':>5}{'':>6}" + "".join(f"{cub[n]:13.1f}{100.0:6.0f}" for n in SIZES))
    for k in keys:
        kid = f"{k[0]}|{k[1]}|{k[2]}"
        line = f"{k[0]:<7}{k[1]:<19}{k[2]:>3}{r['regs'][kid]:>5}{r['spill'][kid]:>6}"
        line += "".join(f"{t[(*k, n)]:13.1f}{100 * cub[n] / t[(*k, n)]:6.0f}" for n in SIZES)
        say(line)

    say("\nheadline: best of each variant (any config, any STAGES)")
    for n in SIZES:
        parts = []
        for v in SWEEP:
            best = min(((t[(*k, n)], k) for k in keys if k[0] == v), default=None)
            if best:
                parts.append(f"{v} {best[0]:7.1f} ({100 * cub[n] / best[0]:.0f}%, {best[1][1]} S={best[1][2]})")
        say(f"  n={n:5d}  cuBLAS {cub[n]:7.1f} | " + " | ".join(parts))

    say(f"\nSASS of '{next(iter(CONFIGS))}' S=3 per variant (LDS.128 = 128-bit shared loads):")
    for v, c in r["sass"].items():
        say(f"  {v:<5} " + "  ".join(f"{k} {n}" for k, n in c.items()))
    (runs / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    say(f"\nsaved results/runs/{tag}.json and .txt")


# ============================================================== Nsight Compute reports you open on your laptop
# ncu runs on the GPU machine and writes .ncu-rep files; their bytes come back to your laptop as the function's return
# value and land in results/. Open them in the Nsight Compute GUI (File -> Open, or drag in; it is a free download
# and needs no GPU). Open two reports and use "Add Baseline" to get the side-by-side of the book's Fig 5.31.
# TUNE: named profile sets: set name -> (n, {report name: (variant, config, STAGES)}). cuBLAS is added to every set.
# Sets 01-05 rebuild the earlier experiments with their original code paths (the "/old" variants); 06 is new.
#     modal run matmul08.py::profile --set 06-k12-n4096
PROFILE_SETS = {
    "01-k8-pipeline-n1024": (1024, {"k8-s1": ("k8/old", "128x64 BK8", 1), "k8-s3": ("k8/old", "128x64 BK8", 3)}),
    "02-k9-prefetch-n1024": (1024, {"k8-s3": ("k8/old", "128x64 BK8", 3), "k8t-s3": ("k8+T/old", "128x64 BK8", 3),
                                    "k9-s3": ("k9/old", "128x64 BK8", 3), "k9t-s3": ("k9+T/old", "128x64 BK8", 3)}),
    "03-swizzle-n4096": (4096, {"n4k-k9": ("k9/old", "128x128 BK16", 2), "n4k-k9s": ("k9+S/old", "128x128 BK16", 2),
                                "n4k-k9t": ("k9+T/old", "128x64 BK16", 3), "n4k-k9ts": ("k9TS/old", "128x64 BK16", 3)}),
    "04-bigpatch-n4096": (4096, {"n4k-k9": ("k9", "128x128 BK8", 4), "n4k-big16x8": ("k9TS", "256x128 BK16 16x8", 4),
                                 "n4k-big8x16": ("k9TS", "128x256 BK8 8x16", 4)}),
    "05-k11-hoist-n4096": (4096, {"n4k-big16x8": ("k9TS", "256x128 BK16 16x8", 4),
                                  "n4k-k11": ("k11", "256x128 BK16 16x8", 4),
                                  "n4k-k11best": ("k11", "128x256 BK16 8x16", 3)}),
    "06-k12-n4096": (4096, {"n4k-k11-256x128": ("k11", "256x128 BK16 16x8", 4),     # control, same tile as k12
                            "n4k-k12-256x128": ("k12", "256x128 BK16 16x8", 4),     # k12's best at n = 2,048 / 4,096
                            "n4k-k11-128x256": ("k11", "128x256 BK16 8x16", 4),     # k11's best at n = 4,096
                            "n4k-k12o-128x256": ("k12o", "128x256 BK16 8x16", 3)}),
}


def snapshot(dst):
    """Copy the exact kernel and harness this run used next to its results, with SHA-256s."""
    import hashlib
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    sums = []
    for f in (HERE / "matmul_cp_async.cu", HERE / "matmul08.py"):
        shutil.copy2(f, dst / f.name)
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    (dst / "SHA256SUMS").write_text("\n".join(sums) + "\n")


PROFILE_SCRIPT = r"""
import sys, json, numpy as np
which, n, d = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
if which == "cublas":
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    A = torch.randn(n, n, device="cuda"); B = torch.randn(n, n, device="cuda"); C = torch.empty(n, n, device="cuda")
    torch.matmul(A, B, out=C); torch.cuda.synchronize()
else:
    import cupy as cp
    src = "".join(f"#define {k} {v}\n" for k, v in d.items()) + open("/root/matmul_cp_async.cu").read()
    k = cp.RawKernel(src, "matmul_cp_async", options=("-std=c++17",))
    a_tile = d["BK"] * (d["BM"] + 4) if d["ATRANS"] else d["BM"] * (d["BK"] + 4)
    smem = d["STAGES"] * (a_tile + d["BK"] * d["BN"]) * 4
    if smem > 48 * 1024:
        k.max_dynamic_shared_size_bytes = smem
    A = cp.random.randn(n, n, dtype=cp.float32); B = cp.random.randn(n, n, dtype=cp.float32); C = cp.empty_like(A)
    nt = (d["BM"] // d["TM"]) * (d["BN"] // d["TN"])
    k((n // d["BN"], n // d["BM"]), (nt,), (A, B, C, np.int32(n), np.int32(n), np.int32(n)), shared_mem=smem)
    cp.cuda.runtime.deviceSynchronize()
"""
TIME, ISSUE = "gpu__time_duration.sum", "smsp__issue_active.avg.pct_of_peak_sustained_active"
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")  # one per stall reason


@app.function(gpu=GPU, image=image, timeout=900)
def profile_gpu(n, runs):
    import shutil
    import subprocess

    ncu = shutil.which("ncu") or next(iter(sorted(Path("/usr/local/cuda").glob("**/ncu"))), None) \
        or next(iter(sorted(Path("/opt/nvidia").glob("**/ncu"))), None)
    if not ncu:
        return {"error": "ncu is not in this image", "reports": {}}
    Path("/tmp/p.py").write_text(PROFILE_SCRIPT)
    targets = {name: dict(CONFIGS[c], STAGES=s, **VARIANTS[v]) for name, (v, c, s) in runs.items()}
    cublas_name = "k8-cublas" if n == 1024 else f"n{n // 1024}k-cublas"
    targets[cublas_name] = {}
    reports, log = {}, []
    for name, d in targets.items():
        which = "cublas" if name == cublas_name else "ours"
        kernel_filter = "regex:gemm" if which == "cublas" else "matmul_cp_async"   # cuBLAS names contain "gemm"
        # --clock-control none: Modal's GPUs refuse ncu's clock lock ("Failed to lock GPU clock frequencies").
        res = subprocess.run([str(ncu), "--set", "full", "--import-source", "yes", "--clock-control", "none",
                              "--kernel-name", kernel_filter, "--launch-count", "1", "-f", "-o", f"/tmp/{name}",
                              "python", "/tmp/p.py", which, str(n), json.dumps(d)],
                             capture_output=True, text=True)
        rep = Path(f"/tmp/{name}.ncu-rep")
        if not rep.exists():
            log.append(f"--- {name}: no report\n" + "\n".join((res.stdout + res.stderr).splitlines()[-15:]))
            continue
        reports[f"{name}.ncu-rep"] = rep.read_bytes()
        raw = subprocess.run([str(ncu), "-i", str(rep), "--page", "raw", "--csv"], capture_output=True, text=True).stdout
        log.append(f"--- {name}\n{raw.strip()}")
    return {"error": None, "reports": reports, "log": "\n".join(log)}


def dram_mb(num, units, head):
    """DRAM bytes read, in MB, whatever unit ncu chose for the column."""
    col = "dram__bytes_read.sum"
    if col not in head:
        return float("nan")
    scale = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}.get(units[head.index(col)], float("nan"))
    return num(col) * scale


@app.local_entrypoint()
def profile(set: str = "06-k12-n4096", tag: str = ""):
    """Profiles PROFILE_SETS[set] + cuBLAS; writes results/ncu/<tag>/ (.ncu-rep, summary.txt, code/), never overwrites."""
    import csv
    import io

    n, runs = PROFILE_SETS[set]
    tag = tag or set
    out = HERE / "results" / "ncu" / tag
    if out.exists():
        out = out.with_name(f"{tag}-{time.strftime('%H%M%S')}")
    out.mkdir(parents=True)
    lines = []

    def say(x=""):
        print(x)
        lines.append(str(x))

    snapshot(out / "code")
    r = profile_gpu.remote(n, runs)
    if r["error"]:
        say(r["error"])
        return
    say(f"profile set {set}, n = {n}; runs: " + ", ".join(f"{k} = {v}" for k, v in runs.items()))
    for name, data in r["reports"].items():
        (out / name).write_bytes(data)
        say(f"saved {out.relative_to(HERE)}/{name} ({len(data) / 1e6:.1f} MB)")
    # A short table from each report's raw page, so you see the headline without opening the GUI.
    # Stall columns are cycles a warp spent stalled per instruction it issued, by reason; "total" includes the issue
    # cycle itself, so issue rate ~ 1 / total with one warp per scheduler. "global %" = long scoreboard / total, the
    # measure behind the book's 43% (kernel 7) vs 7.5% (cuBLAS).
    say(f"\nn = {n} (ncu replays each kernel, so its time differs a little from ::main)")
    say(f"{'report':<11}{'regs':>5}{'time us':>9}{'issue %':>9}{'total':>7}{'short_sb':>9}{'dispatch':>9}"
          f"{'wait':>6}{'barrier':>8}{'global %':>9}{'DRAM MB':>9}{'L2 hit':>8}")
    for block in r["log"].split("--- ")[1:]:
        name, _, body = block.partition("\n")
        if name.endswith("no report"):
            say(f"{name}; ncu said:\n{body}")
            continue
        rows = list(csv.reader(io.StringIO(body)))
        head, units, vals = rows[0], rows[1], rows[-1]          # row 1 holds units; the last row is the launch
        num = lambda col: float(vals[head.index(col)].replace(",", "")) if col in head else float("nan")
        st = {STALL.match(h).group(1): num(h) for h in head if STALL.match(h)}
        total = sum(st.values())
        t_us = num(TIME) / (1000.0 if units[head.index(TIME)] == "ns" else 1.0)
        say(f"{name:<11}{num('launch__registers_per_thread'):5.0f}{t_us:9.1f}{num(ISSUE):8.1f}%{total:7.2f}"
              f"{st.get('short_scoreboard', 0):9.2f}{st.get('dispatch_stall', 0):9.2f}{st.get('wait', 0):6.2f}"
              f"{st.get('barrier', 0):8.2f}{100 * st.get('long_scoreboard', 0) / total:8.1f}%"
              f"{dram_mb(num, units, head):9.1f}{num('lts__t_sector_hit_rate.pct'):7.1f}%")
    # Second table: where the issue slots go. Every kernel does the same n^3 / 32 warp-level FFMAs, so "other per FFMA"
    # = (all instructions - FFMAs) / FFMAs is pure overhead: shared loads, address math, copies, loop control.
    say(f"\n{'report':<13}{'inst M':>8}{'other/FFMA':>11}{'LDS M':>7}{'LDS bank cf M':>14}{'FMA pipe':>9}"
          f"{'ALU pipe':>9}{'warps/sched':>12}{'tile':>12}")
    ffma = n**3 / 32 / 1e6
    for block in r["log"].split("--- ")[1:]:
        name, _, body = block.partition("\n")
        if name.endswith("no report"):
            continue
        rows = list(csv.reader(io.StringIO(body)))
        head, vals = rows[0], rows[-1]
        num = lambda col: float(vals[head.index(col)].replace(",", "")) if col in head else float("nan")
        inst = num("smsp__inst_executed.sum") / 1e6
        tile = f"{int(n / num('launch__grid_dim_y'))}x{int(n / num('launch__grid_dim_x'))}"
        say(f"{name:<13}{inst:8.0f}{(inst - ffma) / ffma:11.3f}{num('smsp__sass_inst_executed_op_shared_ld.sum') / 1e6:7.0f}"
              f"{num('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum') / 1e6:14.2f}"
              f"{num('sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active'):8.1f}%"
              f"{num('sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active'):8.1f}%"
              f"{num('sm__warps_active.avg.pct_of_peak_sustained_active') / 100 * 16:12.2f}{tile:>12}")
    say("\nopen the .ncu-rep files in the Nsight Compute GUI; pick one as baseline to compare the others against it")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    (out / "raw-metrics.csv.txt").write_text(r["log"])        # every raw metric of every report, for the write-up
