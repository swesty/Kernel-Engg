"""CPU-only Modal job: per-SASS-instruction source page of two .ncu-rep files -> opcode histogram + stall reasons."""
from pathlib import Path
import modal
R = Path.home() / "Developer/kernel-engineering/Kernel-Engg/kernel08-cp-async/results"
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04"
         "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([]).add_local_file(str(R / "ncu/06-k12-n4096/n4k-k12o-128x256.ncu-rep"), "/reps/ours.ncu-rep")
         .add_local_file(str(R / "ncu/06-k12-n4096/n4k-k11-128x256.ncu-rep"), "/reps/cublas.ncu-rep"))
app = modal.App("k12-ncu-src")
@app.function(image=image, timeout=600)
def read():
    import csv, io, subprocess, shutil
    ncu = shutil.which("ncu") or str(next(Path("/usr/local/cuda").glob("**/ncu")))
    out = {}
    for name in ("ours", "cublas"):
        raw = subprocess.run([ncu, "-i", f"/reps/{name}.ncu-rep", "--page", "source", "--csv", "--print-source", "sass"],
                             capture_output=True, text=True)
        out[name] = raw.stdout if raw.stdout.strip() else raw.stderr[-2000:]
    return out
@app.local_entrypoint()
def main():
    import csv, io, collections, json
    d = read.remote()
    for name, txt in d.items():
        Path(f"/private/tmp/claude-501/-Users-swesty-obsidian-New-Start/f78b04a5-e834-4aff-a658-146ec0387dfb/scratchpad/k8/src12_{name}.csv").write_text(txt)
        rows = list(csv.reader(io.StringIO(txt)))
        hi = next(i for i, r in enumerate(rows) if "Source" in r)
        head = rows[hi]
        print(f"=== {name}: {len(rows) - hi - 1} SASS lines; columns:")
        print([h for h in head][:60])
