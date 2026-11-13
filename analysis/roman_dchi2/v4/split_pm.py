#!/usr/bin/env python3
"""Separate cuts for xi_+ and xi_- (the white II term lives in xi_- only).

Linear-response figure of bias for masks that take the xi_+ rows from the
1/D-family mask K0+ and the xi_- rows from the mask K0- (K0- = 0: no xi_-):
    delta theta = G F^-1 J^T C^-1 r,   r = d_FLAMINGO - model(fiducial, IA_ref),
IA refitted (the Jacobian includes the IA parameters), case A. Validated
against the full non-linear fits on the diagonal K0+ = K0-.
Also the white term alone (template) for the same masks.
    python split_pm.py <cocoa data dir>"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
V3 = os.path.join(HERE, "..", "v3")
sys.path.insert(0, COCOA)
import bias_analysis as B   # noqa: E402
import bias_v2 as V         # noqa: E402

os.chdir(COCOA)
cov = np.load("analysis/cov_gauss_d3a.npy")
R = np.load("run/cosmolike_xi_ia.npz")
TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]
Jz = np.load(os.path.join(HERE, "jacobians.npz"))
rs = json.load(open(os.path.join(HERE, "roman_strategies.json")))
IA_REF = rs["IA_REF"]
S = np.load(os.path.join(V3, "masks_scan.npz"))
cl = B.CL(os.path.join(COCOA, "lib"), sys.argv[1])
cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
g = []
for n in ("omegam", "As_1e9"):
    pp, pm = dict(B.FID), dict(B.FID)
    pp[n] += B.STEP[n]
    pm[n] -= B.STEP[n]
    g.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
d_fl = cl.model("NLA", B.FID, [0.0, 0.0]) + R["xi_flam"]
resid = {m: d_fl - cl.model(m, B.FID, IA_REF[m]) for m in ("NLA", "TATT")}
K0S = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
KM = (0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
ms = json.load(open(os.path.join(V3, "mask_scan.json")))
out = []
for mdl in ("NLA", "TATT"):
    for kp in K0S:
        for km in KM:
            m = np.zeros(1080, bool)
            m[:540] = S[f"D1_{kp:g}"][:540]
            if km > 0:
                m[540:] = S[f"D1_{km:g}"][540:]
            Jm = np.hstack([Jz[mdl][:, :2], Jz[mdl][:, 5:]])[m]
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            pr = [0.0, 0.0] + [1/V.IA_SIG[n]**2 for n in B.IAP[mdl]]
            Cv = np.linalg.inv(Jm.T @ Ci @ Jm + np.diag(pr))
            G = np.zeros((2, Jm.shape[1]))
            G[0, 0] = 1.0
            G[1, :2] = g
            St = G @ Cv @ G.T
            Bl = G @ Cv @ Jm.T @ Ci
            dt = Bl @ resid[mdl][m]
            dw = Bl @ TPL[m]
            fob = float(np.sqrt(dt @ np.linalg.solve(St, dt)))
            row = dict(model=mdl, K0p=kp, K0m=km, npts=int(m.sum()), n_xim=int(m[540:].sum()), FoB=fob,
                       dS8=float(dt[1]), sS8=float(np.sqrt(St[1, 1])), FoM=float(1/np.sqrt(np.linalg.det(St))),
                       white_FoB=float(np.sqrt(dw @ np.linalg.solve(St, dw))))
            if kp == km:
                full = [x for x in ms if x["family"] == "D1" and abs(x["K0"] - kp) < 1e-9][0]
                row["FoB_full_fit"] = full[f"{mdl}_A"]["FoB"]
            out.append(row)
json.dump(out, open(os.path.join(HERE, "split_pm.json"), "w"), indent=1)
for mdl in ("NLA", "TATT"):
    print(mdl, "validation (diagonal): linear vs full fit:",
          [(x["K0p"], round(x["FoB"], 2), round(x["FoB_full_fit"], 2)) for x in out if x["model"] == mdl and "FoB_full_fit" in x])
    print(mdl, "FoB [rows K0+, cols K0-]", KM)
    for kp in K0S:
        print(f"  {kp:4g}", " ".join(f"{x['FoB']:.2f}/{x['sS8']*1e3:.1f}" for x in out if x["model"] == mdl and x["K0p"] == kp))
