"""Summarize an ncu SASS source page (ncu -i X.ncu-rep --page source --csv --print-source sass > X.sass.csv).

    python sass_hotspots.py results/analysis/source-pages/05-n4k-k11best.sass.csv

Prints: stall-reason shares, an opcode table (executed instructions, share of wait and dispatch samples), and the
hottest wait-stalled instructions with the instruction before each. This is the analysis behind the kernel 10-12
diagnoses in LAB_NOTEBOOK.md.
"""
import collections
import csv
import re
import sys


def load(path):
    rows = list(csv.reader(open(path)))
    hi = next(i for i, r in enumerate(rows) if "Source" in r)
    head, data = rows[hi], []
    for r in rows[hi + 1:]:
        if len(r) != len(head):
            continue
        d = dict(zip(head, r))
        f = lambda k: float(d.get(k, "0").replace(",", "") or 0)
        src = d["Source"].strip()
        op = re.sub(r"^@!?U?P\w+\s+", "", src).split()[0] if src else "?"
        data.append(dict(src=src, op=op, ex=f("Instructions Executed"), samp=f("Warp Stall Sampling (All Samples)"),
                         **{k: f(f"stall_{k}") for k in ("selected", "not_selected", "dispatch", "wait", "short_sb",
                                                          "long_sb", "barrier", "math")}))
    return data


def main(path):
    D = load(path)
    T = sum(x["samp"] for x in D)
    reasons = ["selected", "not_selected", "dispatch", "wait", "short_sb", "long_sb", "barrier", "math"]
    print(f"{path}: {len(D)} SASS lines, {T:.0f} samples")
    print("  " + "  ".join(f"{r} {100 * sum(x[r] for x in D) / T:.1f}%" for r in reasons))
    by = collections.defaultdict(collections.Counter)
    for x in D:
        b = by[x["op"].split(".")[0]]
        b["ex"] += x["ex"]
        for r in reasons:
            b[r] += x[r]
    W, Dp = sum(x["wait"] for x in D), sum(x["dispatch"] for x in D)
    print(f"{'opcode':<9}{'exec M':>9}{'wait share':>12}{'dispatch share':>16}")
    for op, c in sorted(by.items(), key=lambda kv: -kv[1]["ex"])[:12]:
        print(f"{op:<9}{c['ex'] / 1e6:9.1f}{100 * c['wait'] / max(W, 1):11.1f}%{100 * c['dispatch'] / max(Dp, 1):15.1f}%")
    print("hottest wait-stalled instructions (wait = fixed-latency dependency):")
    for i in sorted(range(len(D)), key=lambda i: -D[i]["wait"])[:6]:
        print(f"  {D[i]['wait']:6.0f}  {D[i]['src'][:58]:<58}  after: {D[i - 1]['src'][:50] if i else ''}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
