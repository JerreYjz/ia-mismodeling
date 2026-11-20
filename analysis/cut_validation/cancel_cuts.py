#!/usr/bin/env python3
"""The reliable cuts (probes cs, 2x2 and 3x2; the results of earlier runs
are kept in cancel_cuts.json, those of the truths given are replaced) of the bias-cancellation study, from bias_pieces.json and
fisher_minimize.json: for each probe, model and truth, the largest K0 at
which, at K0 and every smaller K0 of the scan,
    total   the MAP figure of bias is below TOL;
    pieces  also every piece of the zshape split is below TOL;
    shape   also every piece but the amplitude one is below TOL (the cut the
            scale dependence alone allows, when the redshift-amplitude error
            fails at every cut).
"-" when it fails already at the largest scales of the scan.
    python3 cancel_cuts.py [truth ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
K0 = ["0.2", "0.25", "0.3", "0.35", "0.4", "0.5", "0.6"]
MODELS = ["NLA", "TATT", "NLAa", "TATTa", "NLAf", "TATTf"]
TOL = 0.3


def cut(ok):
    best = "-"
    for k, o in zip(K0, ok):
        if o is None or not o:
            break
        best = k
    return best


def main():
    R = json.load(open(os.path.join(HERE, "bias_pieces.json")))["runs"]
    M = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
    path = os.path.join(HERE, "cancel_cuts.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for tr in sys.argv[1:] or ["S1", "lin"]:
        for p in ("cs", "2x2", "3x2"):
            for m in MODELS:
                keys = [f"{p}+cmb|{m}|{k}|{tr}" for k in K0]
                if not all(k in R and k in M for k in keys):
                    continue
                tot = [M[k]["flatprior"]["fob_map"] for k in keys]
                big = [R[k]["largest_zshape_fob"] for k in keys]
                shp = [max(v["fob"] for q, v in R[k]["zshape"].items() if q != "z|amplitude") for k in keys]
                amp = [R[k]["zshape"]["z|amplitude"]["fob"] for k in keys]
                c = dict(total=cut([t < TOL for t in tot]),
                         pieces=cut([t < TOL and b < TOL for t, b in zip(tot, big)]),
                         shape=cut([s < TOL for s in shp]))
                out[f"{p}|{m}|{tr}"] = c
                print(f"{tr:4s}{p:4s}{m:6s} cut total {c['total']:5s} pieces {c['pieces']:5s} shape {c['shape']:5s}| "
                      + " ".join(f"{t:.2f}/{b:.2f}/{a:.2f}" for t, b, a in zip(tot, big, amp)))
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
