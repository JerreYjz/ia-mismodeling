#!/usr/bin/env python3
"""data.json of plots/cut_combo: the 3x2pt test of the separate cuts.

Each probe's cut is first set on that probe alone (cosmic shear: the xi_+-
cut; 2x2pt: the gamma_t cut, 0.5 on the stitched truth). The pair is then
tested in 3x2pt, and when it fails one cut is lowered at a time:
    xi    the cosmic-shear cut lowered, gamma_t at 0.5;
    gt    the gamma_t cut lowered, cosmic shear at its own cut (0.35 on the
          stitched truth, 0.6 on the E minus B truth).
For NLA and TATT on both truths: the MAP figure of bias in (Omega_m, S8)
(fisher_minimize.json, every parameter free, CMB prior), the figure of merit
1/sqrt(det C) of (Omega_m, S8) in thousands (pieces_maps/), the parameters at
a prior bound (within 2% of the prior range of a bound: fisher_minimize.py's
own flag, 0.1%, missed fits pressed against eta2 = 5 at 4.93 to 4.98), and
the probe split of the linear shift (probe_split.json:
the cosmic-shear rows and the gamma_t rows).
    ~/cocoa/Cocoa/.local/bin/python make_combo_figure.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OWN = {"S1": "0.35", "EB": "0.6"}
XI = ["0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.5", "0.6"]
GT = ["0.2", "0.25", "0.3", "0.4", "0.5"]


# prior ranges of the free nuisance parameters (fisher_minimize.py)
BOUNDS = {"roman_A1_1": (-5, 5), "roman_A1_2": (-5, 5), "roman_A2_1": (-5, 5), "roman_A2_2": (-5, 5),
          "roman_BTA_1": (0, 2), "hm_alpha": (-1, 2), **{f"roman_B1_{i}": (0.4, 4.0) for i in range(1, 9)}}


def at_bound(best, tol=0.02):
    return [p for p, (lo, hi) in BOUNDS.items()
            if p in best and min(best[p] - lo, hi - best[p]) < tol*(hi - lo)]


def key(m, xi, tr, gt):
    return f"3x2+cmb|{m}|{xi}|{tr}" + ("" if gt == "0.3" else f"|gt{gt}")


def main():
    M = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
    P = json.load(open(os.path.join(HERE, "probe_split.json")))["runs"]
    lines = {}
    for m in ("NLA", "TATT"):
        for tr in ("S1", "EB"):
            for scan, grid in (("xi", XI), ("gt", GT)):
                pts = []
                for v in grid:
                    k = key(m, v, tr, "0.5") if scan == "xi" else key(m, OWN[tr], tr, v)
                    if scan == "xi" and float(v) > float(OWN[tr]):
                        continue
                    if k not in M:
                        continue
                    f = os.path.join(HERE, "pieces_maps", k.replace("|", "_").replace("+", "p") + ".npz")
                    fom = float(1/np.sqrt(np.linalg.det(np.load(f)["C2"]))/1e3) if os.path.exists(f) else None
                    pts.append(dict(cut=float(v), key=k, fob=M[k]["flatprior"]["fob_map"],
                                    mu=M[k]["flatprior"]["mu_map"], fom=fom, at_bound=at_bound(M[k]["best"]),
                                    xi_rows=P[k]["xi"]["fob"] if k in P else None,
                                    gt_rows=P[k]["gt"]["fob"] if k in P else None))
                lines[f"{m}|{tr}|{scan}"] = pts
    d = dict(about=f"Producer: make_combo_figure.py.\n{__doc__}", threshold=0.3, own=OWN, lines=lines)
    os.makedirs(os.path.join(HERE, "plots", "cut_combo"), exist_ok=True)
    json.dump(d, open(os.path.join(HERE, "plots", "cut_combo", "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
