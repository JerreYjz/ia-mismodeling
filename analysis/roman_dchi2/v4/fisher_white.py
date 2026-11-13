#!/usr/bin/env python3
"""Cost in precision of the white II nuisance: Fisher sigma(S8), FoM with and
without a free white-template amplitude a_w (flat prior width 10/sqrt 12, as
mask_scan.py), for every mask of the three families, cases A and B.
Needs jacobians.npz (saved by roman_strategies.py) and the S8 gradient.
    python fisher_white.py <cocoa data dir>"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
import bias_analysis as B   # noqa: E402
import bias_v2 as V         # noqa: E402
sys.path.insert(0, HERE)
from roman_strategies import load_masks  # noqa: E402

os.chdir(COCOA)
cov = np.load("analysis/cov_gauss_d3a.npy")
TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]
Jz = np.load(os.path.join(HERE, "jacobians.npz"))
cl = B.CL(os.path.join(COCOA, "lib"), sys.argv[1])
cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
g = []
for n in B.CP:
    pp, pm = dict(B.FID), dict(B.FID)
    pp[n] += B.STEP[n]
    pm[n] -= B.STEP[n]
    g.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
g = np.array(g)
masks = load_masks()
out = {}
for key, m in masks.items():
    if not key.startswith("scan:") and not key.startswith("n1:"):
        continue
    row = {}
    C = cov[np.ix_(m, m)]
    Ci = np.linalg.inv(C)
    for mdl in ("NLA", "TATT"):
        for case in ("A", "B"):
            nc = 2 if case == "A" else 5
            for white in (False, True):
                cols = [Jz[mdl][:, :nc], Jz[mdl][:, 5:]]
                pr = [0.0]*2 + ([] if case == "A" else [1/B.PRIOR_B[n]**2 for n in ("ns", "H0", "omegab")])
                pr += [1/V.IA_SIG[n]**2 for n in B.IAP[mdl]]
                if white:
                    cols.append(TPL[:, None])
                    pr.append(1/(10/np.sqrt(12))**2)
                Jm = np.hstack(cols)[m]
                Cv = np.linalg.inv(Jm.T @ Ci @ Jm + np.diag(pr))
                G = np.zeros((2, Jm.shape[1]))
                G[0, 0] = 1.0
                G[1, :nc] = g[:nc]
                St = G @ Cv @ G.T
                row[f"{mdl}_{case}_{'w' if white else 'nw'}"] = dict(sOm=float(np.sqrt(St[0, 0])), sS8=float(np.sqrt(St[1, 1])),
                                                                    FoM=float(1/np.sqrt(np.linalg.det(St))))
    out[key] = dict(npts=int(m.sum()), **row)
json.dump(out, open(os.path.join(HERE, "fisher_white.json"), "w"), indent=1)
for key in ("n1:0.25", "n1:0.3", "n1:0.35", "n1:0.4", "n1:0.5", "scan:flat_0.3", "scan:flat_0.4", "scan:flat_0.5"):
    r = out[key]
    print(key, r["npts"], " ".join(f"{mdl}{c}: {r[f'{mdl}_{c}_nw']['sS8']:.4f}->{r[f'{mdl}_{c}_w']['sS8']:.4f}"
                                  for mdl in ("NLA", "TATT") for c in ("A", "B")))
