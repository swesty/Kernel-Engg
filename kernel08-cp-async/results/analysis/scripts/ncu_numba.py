"""CPU-only: inspect the course's Numba kernel-7 profile for loop unrolling and local-memory use."""
from pathlib import Path
import modal
R = Path.home() / "Developer/kernel-engineering/Kernel-Engg/lecture10"
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04"
         "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([]).add_local_file(str(R / "matmul-07-optimal.ncu-rep"), "/reps/k7.ncu-rep")
         .add_local_file(str(R / "matmul-05.ncu-rep"), "/reps/k5.ncu-rep"))
app = modal.App("numba-unroll-check")
@app.function(image=image, timeout=600)
def read():
    import subprocess, shutil
    ncu = shutil.which("ncu") or str(next(Path("/usr/local/cuda").glob("**/ncu")))
    out = {}
    for n in ("k7", "k5"):
        raw = subprocess.run([ncu, "-i", f"/reps/{n}.ncu-rep", "--page", "raw", "--csv"], capture_output=True, text=True).stdout
        src = subprocess.run([ncu, "-i", f"/reps/{n}.ncu-rep", "--page", "source", "--csv", "--print-source", "sass"], capture_output=True, text=True)
        out[n] = (raw, src.stdout or src.stderr[-1500:])
    return out
@app.local_entrypoint()
def main():
    import csv, io, re, collections
    for n, (raw, src) in read.remote().items():
        rows = list(csv.reader(io.StringIO(raw))); head, vals = rows[0], rows[-1]
        g = lambda c: vals[head.index(c)] if c in head else "n/a"
        print(f"=== course {n}: {g('Kernel Name')[:60]} on {g('device__attribute_display_name')}")
        for c in ("launch__registers_per_thread", "smsp__sass_inst_executed_op_local_ld.sum", "smsp__sass_inst_executed_op_local_st.sum",
                  "smsp__inst_executed.sum", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "gpu__time_duration.sum"):
            print(f"   {c}: {g(c)}")
        srows = list(csv.reader(io.StringIO(src)))
        hi = next((i for i, r in enumerate(srows) if "Source" in r), None)
        if hi is None:
            print("   no source page:", src[:300]); continue
        sh = srows[hi]; lines = [dict(zip(sh, r)) for r in srows[hi + 1:] if len(r) == len(sh)]
        ops = collections.Counter(re.sub(r"^@!?U?P\w+\s+", "", l["Source"].strip()).split()[0].split(".")[0] for l in lines if l["Source"].strip())
        # longest run of consecutive FFMA in the static SASS = evidence of an unrolled TMxTN outer product
        run = best = 0
        for l in lines:
            op = re.sub(r"^@!?U?P\w+\s+", "", l["Source"].strip()).split()[0] if l["Source"].strip() else ""
            run = run + 1 if op.startswith("FFMA") else 0; best = max(best, run)
        print(f"   static SASS: {len(lines)} lines; FFMA {ops['FFMA']}, LDS {ops['LDS']}, LDL {ops['LDL']}, STL {ops['STL']}, BRA {ops['BRA']}; longest consecutive FFMA run {best}")
