#!/usr/bin/env python3
"""Linear-response figure of bias and Fisher precision for every mask, with a
choice of covariance (Gaussian of these notes, or the production example1_cov).

delta theta = G F^-1 J^T C^-1 r, r = d_FLAMINGO - model(fiducial, IA_ref);
J = [cosmology, IA (, white template)] from v4/jacobians.npz; IA priors the
flat-prior width / sqrt 12; case A (Omega_m, A_s) and B (+ n_s, H0, Omega_b
with Gaussian priors). Validated against the full fits of the Gaussian runs.
Masks: the three K0 families (v3/masks_scan.npz) and separate xi_+/xi_- masks.
    python linear_cov.py <cocoa data dir> <cov npy (xi block)> <tag>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
V3 = os.path.join(HERE, "..", "v3")
V4 = os.path.join(HERE, "..", "v4")
sys.path.insert(0, COCOA)
import bias_analysis as B   # noqa: E402
import bias_v2 as V         # noqa: E402

data, covf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(COCOA)
cov = np.load(covf)
R = np.load("run/cosmolike_xi_ia.npz")
TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]
Jz = np.load(os.path.join(V4, "jacobians.npz"))
IA_REF = json.load(open(os.path.join(V4, "roman_strategies.json")))["IA_REF"]
S = np.load(os.path.join(V3, "masks_scan.npz"))
cl = B.CL(os.path.join(COCOA, "lib"), data)
cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
g = []
for n in B.CP:
    pp, pm = dict(B.FID), dict(B.FID)
    pp[n] += B.STEP[n]
    pm[n] -= B.STEP[n]
    g.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
g = np.array(g)
xi_ia = R["xi_flam"]
if os.environ.get("TRUTH"):                       # file:key, e.g. run/xi_flam_white.npz:GI_II1
    f_, k_ = os.environ["TRUTH"].split(":")
    xi_ia = np.load(f_)[k_]
d_fl = cl.model("NLA", B.FID, [0.0, 0.0]) + xi_ia
resid = {m: d_fl - cl.model(m, B.FID, IA_REF[m]) for m in ("NLA", "TATT")}


def linfit(mdl, case, m, white=False):
    nc = 2 if case == "A" else 5
    cols = [Jz[mdl][:, :nc], Jz[mdl][:, 5:]]
    pr = [0.0, 0.0] + ([] if case == "A" else [1/B.PRIOR_B[n]**2 for n in ("ns", "H0", "omegab")])
    pr += [1/V.IA_SIG[n]**2 for n in B.IAP[mdl]]
    if white:
        cols.append(TPL[:, None])
        pr.append(1/(10/np.sqrt(12))**2)
    Jm = np.hstack(cols)[m]
    C = cov[np.ix_(m, m)]
    Ci = np.linalg.inv(C)
    Cv = np.linalg.inv(Jm.T @ Ci @ Jm + np.diag(pr))
    G = np.zeros((2, Jm.shape[1]))
    G[0, 0] = 1.0
    G[1, :nc] = g[:nc]
    St = G @ Cv @ G.T
    Bl = G @ Cv @ Jm.T @ Ci
    dt = Bl @ resid[mdl][m]
    r = resid[mdl][m]
    rp = r - Jm @ (Cv @ (Jm.T @ Ci @ r))
    return dict(FoB=float(np.sqrt(dt @ np.linalg.solve(St, dt))), dOm=float(dt[0]/np.sqrt(St[0, 0])), dS8=float(dt[1]/np.sqrt(St[1, 1])),
                dS8abs=float(dt[1]), sS8=float(np.sqrt(St[1, 1])), sOm=float(np.sqrt(St[0, 0])),
                FoM=float(1/np.sqrt(np.linalg.det(St))), chi2=float(rp @ Ci @ rp))


out = []
for key in S.files:
    if key in ("z2", "D"):
        continue
    fam, K0 = key.split("_")
    m = S[key].astype(bool)
    row = dict(kind="family", family=fam, K0=float(K0), npts=int(m.sum()))
    for mdl in ("NLA", "TATT"):
        for case in ("A", "B"):
            row[f"{mdl}_{case}"] = linfit(mdl, case, m)
        row[f"{mdl}w_A"] = linfit(mdl, "A", m, white=True)
        row[f"{mdl}w_B"] = linfit(mdl, "B", m, white=True)
    out.append(row)
for kp in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    for km in (0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        m = np.zeros(1080, bool)
        m[:540] = S[f"D1_{kp:g}"][:540]
        if km > 0:
            m[540:] = S[f"D1_{km:g}"][540:]
        row = dict(kind="split", K0p=kp, K0m=km, npts=int(m.sum()))
        for mdl in ("NLA", "TATT"):
            for case in ("A", "B"):
                row[f"{mdl}_{case}"] = linfit(mdl, case, m)
        out.append(row)
json.dump(out, open(os.path.join(HERE, f"linear_{tag}.json"), "w"), indent=1)
for fam in ("D1", "D23", "flat"):
    print(f"--- {fam} ({tag}): K0 npts | FoB A NLA/TATT | FoB B | +white A | sS8 A | chi2 NLA/TATT")
    for x in sorted([x for x in out if x["kind"] == "family" and x["family"] == fam], key=lambda x: x["K0"]):
        print(f"  {x['K0']:4g} {x['npts']:4d} | {x['NLA_A']['FoB']:.2f}/{x['TATT_A']['FoB']:.2f} | {x['NLA_B']['FoB']:.2f}/{x['TATT_B']['FoB']:.2f} | "
              f"{x['NLAw_A']['FoB']:.2f}/{x['TATTw_A']['FoB']:.2f} | {x['NLA_A']['sS8']:.4f} | {x['NLA_A']['chi2']:.2f}/{x['TATT_A']['chi2']:.2f}")
