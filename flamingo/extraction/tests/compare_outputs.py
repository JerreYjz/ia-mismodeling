#!/usr/bin/env python3
"""Numerically compare two directories of Pds-style text outputs.

usage: compare_outputs.py <dirA> <dirB> [rtol]
Prints, per file, the max relative difference per column (scaled by the
column's max |value|); files above rtol are flagged.
"""
import os
import sys

import numpy as np


def load(path):
    rows = [ln.split() for ln in open(path) if ln.strip()]
    return np.array(rows, dtype=float)


def main(a, b, rtol=1e-4):
    bad = 0
    names = sorted(os.path.relpath(os.path.join(r, f), a)
                   for r, _, fs in os.walk(a) for f in fs if f.endswith(".dat"))
    for name in names:
        x, y = load(os.path.join(a, name)), load(os.path.join(b, name))
        if x.shape != y.shape:
            print(f"SHAPE {name}: {x.shape} vs {y.shape}")
            bad += 1
            continue
        scale = np.maximum(np.abs(x).max(axis=0), 1e-300)
        rel = (np.abs(x - y) / scale).max(axis=0)
        flag = "  <-- differs" if rel.max() > rtol else ""
        bad += bool(flag)
        print(f"{name:32s} max rel diff per column: " + " ".join(f"{r:.1e}" for r in rel) + flag)
    print(f"{len(names)} files compared, {bad} above rtol={rtol}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(float(t) for t in sys.argv[3:]))
