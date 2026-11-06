"""CPU-only Modal job: read the local .ncu-rep files with ncu --import and dump the metrics that explain issue rate."""
from pathlib import Path
import modal
R = Path.home() / "Developer/kernel-engineering/Kernel-Engg/kernel08-cp-async/results"
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04"
         "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([]).add_local_dir(str(R), "/reps", ignore=["*.json"]))
app = modal.App("k8-ncu-read")
KEEP = ["stalled_", "inst_executed.sum", "inst_executed_op_shared", "inst_executed_op_global", "inst_executed_op_ldgsts",
        "bank_conflicts", "launch__registers", "launch__occupancy_limit", "warps_active.avg.pct", "launch__block_size",
        "launch__grid_size", "launch__shared_mem", "pipe_fma", "inst_executed_pipe_lsu", "sm__throughput", "issue_active",
        "achieved_occupancy", "ipc", "smsp__thread_inst_executed_per_inst", "dram__throughput"]
@app.function(image=image, timeout=300)
def read():
    import csv, io, subprocess, shutil
    ncu = shutil.which("ncu") or str(next(Path("/usr/local/cuda").glob("**/ncu")))
    out = {}
    for name in ("k8-s1", "k8-s3", "k8-cublas"):
        raw = subprocess.run([ncu, "-i", f"/reps/{name}.ncu-rep", "--page", "raw", "--csv"], capture_output=True, text=True).stdout
        rows = list(csv.reader(io.StringIO(raw)))
        out[name] = {h: (u, v) for h, u, v in zip(rows[0], rows[1], rows[-1]) if any(k in h for k in KEEP)}
    return out
@app.local_entrypoint()
def main():
    d = read.remote()
    names = list(d)
    keys = sorted(set().union(*[set(v) for v in d.values()]))
    def f(x):
        try: return float(x.replace(",", ""))
        except: return None
    for k in keys:
        vals = [d[n].get(k, ("", ""))[1] for n in names]
        if "stalled_" in k and not k.endswith("_per_issue_active.ratio"): continue
        if all((f(v) or 0) == 0 for v in vals): continue
        print(f"{k[:78]:<78} " + " ".join(f"{v:>14}" for v in vals))
