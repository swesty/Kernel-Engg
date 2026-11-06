"""CPU-only: inspect the course's Numba kernel-7 profile for loop unrolling and local-memory use."""
from pathlib import Path
import modal
R = Path.home() / "Developer/kernel-engineering/Kernel-Engg/lecture10"
image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04"
         "@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d", add_python="3.12")
         .entrypoint([]).add_local_file(str(R / "matmul-07-optimal.ncu-rep"), "/reps/k7.ncu-rep")
         .add_local_file(str(R / "matmul-05.ncu-rep"), "/reps/k5.ncu-rep"))
app = modal.App("numba-unroll-check2")
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
    for n, (raw, src) in read.remote().items():
        Path(f"/private/tmp/claude-501/-Users-swesty-obsidian-New-Start/f78b04a5-e834-4aff-a658-146ec0387dfb/scratchpad/k8/numba_{n}_src.csv").write_text(src)
        Path(f"/private/tmp/claude-501/-Users-swesty-obsidian-New-Start/f78b04a5-e834-4aff-a658-146ec0387dfb/scratchpad/k8/numba_{n}_raw.csv").write_text(raw)
        print(n, len(src))
