#!/usr/bin/env python3
"""Shared-memory bank-conflict calculator for tiled stencil kernels.

Model: a block of BX x BY threads (warp lanes run along x first) reads a
shared-memory map whose rows are S elements long. Each pixel occupies C
consecutive elements (e.g. C=3 for interleaved R,G,B). Each thread reads
one component of its pixel at offset (dy, dx) inside a patch that has a
halo of H pixels on every side.

  element index  = (ty + dy) * S + C * (tx + dx) + comp
  bank           = (element index * ELEM_BYTES // WORD_BYTES) mod BANKS
  ticks per read = max over banks of (distinct words requested in that bank)

Two lanes reading the same word are a broadcast, not a conflict.

Closed-form rule (C odd, 4-byte elements, BX a power of two <= 32):
  conflict-free  <=>  the largest power of two dividing S is exactly BX
  (for BX = 32: any S works, because a warp is one row)

Examples:
  bank_layout.py --bx 16 --by 16 --c 3 --halo 2 --stride 60   # lecture, 2 ticks
  bank_layout.py --bx 16 --by 16 --c 3 --halo 2 --search-pad  # finds 80
  bank_layout.py --c 3 --halo 2 --threads 256 --search-width  # finds 32x8
"""
import argparse
from collections import defaultdict


def warp_ticks(bx, by, stride, c, halo, comp=0, elem_bytes=4,
               word_bytes=4, banks=32, warp=32):
    """Worst ticks per read over every warp and every stencil offset."""
    n = bx * by
    worst = 1
    span = 2 * halo + 1
    for dy in range(span):
        for dx in range(span):
            for w0 in range(0, n, warp):
                words_in_bank = defaultdict(set)
                for t in range(w0, min(w0 + warp, n)):
                    tx, ty = t % bx, t // bx
                    elem = (ty + dy) * stride + c * (tx + dx) + comp
                    word = elem * elem_bytes // word_bytes
                    words_in_bank[word % banks].add(word)
                worst = max(worst, max(len(s) for s in words_in_bank.values()))
    return worst


def min_stride(bx, c, halo):
    return c * (bx + 2 * halo)


def closed_form(bx, c, stride, elem_bytes=4, banks=32):
    """Return True/False from the 2-adic rule, or None when it doesn't apply."""
    if elem_bytes != 4 or c % 2 == 0 or bx > banks or banks % bx:
        return None
    if bx == banks:
        return True
    return (stride & -stride) == bx  # lowest set bit of S equals BX


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bx", type=int, default=16, help="block width (threads)")
    p.add_argument("--by", type=int, default=16, help="block height (threads)")
    p.add_argument("--c", type=int, default=3, help="elements per pixel")
    p.add_argument("--halo", type=int, default=2, help="halo radius in pixels")
    p.add_argument("--comp", type=int, default=0, help="component read (0=R)")
    p.add_argument("--elem-bytes", type=int, default=4, help="bytes per element")
    p.add_argument("--stride", type=int, help="row stride in elements (default: unpadded)")
    p.add_argument("--search-pad", action="store_true", help="find smallest conflict-free padding")
    p.add_argument("--max-pad", type=int, default=64)
    p.add_argument("--search-width", action="store_true", help="try every block width, no padding")
    p.add_argument("--threads", type=int, default=256, help="threads per block for --search-width")
    a = p.parse_args()

    kw = dict(c=a.c, halo=a.halo, comp=a.comp, elem_bytes=a.elem_bytes)

    if a.search_width:
        print(f"{'block':>7} {'stride':>6} {'ticks':>5} {'loads/out':>9} {'smem B':>7}")
        for bx in range(1, 33):
            by = max(1, a.threads // bx)
            s = min_stride(bx, a.c, a.halo)
            t = warp_ticks(bx, by, s, **kw)
            loads = (bx + 2 * a.halo) * (by + 2 * a.halo) / (bx * by)
            smem = s * (by + 2 * a.halo) * a.elem_bytes
            mark = "  <- conflict-free" if t == 1 else ""
            print(f"{bx:>3}x{by:<3} {s:>6} {t:>5} {loads:>9.3f} {smem:>7}{mark}")
        return

    base = min_stride(a.bx, a.c, a.halo)
    if a.search_pad:
        for pad in range(a.max_pad + 1):
            s = base + pad
            if warp_ticks(a.bx, a.by, s, **kw) == 1:
                print(f"smallest conflict-free stride: {s} (unpadded {base} + pad {pad})")
                return
        print(f"no conflict-free stride within +{a.max_pad}")
        return

    s = a.stride or base
    t = warp_ticks(a.bx, a.by, s, **kw)
    cf = closed_form(a.bx, a.c, s, a.elem_bytes)
    print(f"block {a.bx}x{a.by}, stride {s}: worst {t} tick(s) per read")
    if cf is not None:
        print(f"closed-form rule predicts: {'conflict-free' if cf else 'conflicts'}")


if __name__ == "__main__":
    main()
