#!/usr/bin/env python3
"""Decompose python-vs-cosmolike differences for model - FLAMINGO xi_IA:
same masks (cosmolike IR/UV, n0 at k_UV = k_max), with and without IA B modes
in the python calculation, and python IR/UV masks applied to cosmolike xi."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import roman_limber as RL
R = np.load("run/cosmolike_xi_ia.npz"); d = np.load("../v2/fits.npz")
cov = np.load("analysis/cov_gauss_d3a.npy")
XI2 = np.load("../v2/xi/xi_ia.npz")
Mpy = np.load("../v2/roman/masks_real.npy")       # python IR/UV masks (FLAMINGO total), n0
M0 = R["masks_n0"]
kg = R["kgrid"]; zs = R["zs"]; kd = d["k"][0]
th = np.tile(np.arange(15), 72)
def q(dv, m=None):
    m = np.ones(dv.size, bool) if m is None else m
    return float(dv[m] @ np.linalg.solve(cov[np.ix_(m, m)], dv[m]))
cosmo = RL.Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
S = RL.RomanShear(cosmo, sys.argv[1])
lo = np.geomspace(1e-4, 0.05, 30, endpoint=False); hi = np.geomspace(kd[-1]*1.02, 2e3, 20)
m = kd >= 0.05; kk = np.concatenate([lo, kd[m], hi])
def tab(Rtab, bb, hi_mode):
    out = np.zeros((len(zs), 3, len(kk)))
    for s in range(len(zs)):
        for x in range(3):
            if x == 2 and not bb: continue
            r = Rtab[s, x, m]
            top = np.zeros(len(hi)) if hi_mode == "zero" else np.full(len(hi), r[-1])
            out[s, x] = np.concatenate([np.full(len(lo), r[0]), r, top])
    return {X: RL.ratio_table(zs, kk, out[:, i]) for i, X in enumerate(("dE", "EE", "BB"))}
def pyxi(Rtab, bb, hi_mode):
    CE, CB = S.cl(tab(Rtab, bb, hi_mode), gg=False); return S.xi(CE, CB)
xt_nb = pyxi(d["Rtrue"], False, "ratio"); xt_bb = pyxi(d["Rtrue"], True, "ratio")
print("model kmax | mask | cosmolike(noBB) | python(noBB) | python(+BB) ")
for mdl, j in (("NLA", 3), ("NLA", 4), ("TATT", 4), ("TATT", 6)):
    Rm = d[f"{mdl}_R"][:, j]
    xm_nb = pyxi(Rm, False, "ratio"); xm_bb = pyxi(Rm, True, "ratio")
    dc = R[f"xi_{mdl}_{j}"] - R["xi_flam"]
    for lab, mm in (("cosmolike n0 mask", M0[j]), ("python n0 mask", Mpy[j]), ("theta>=15.8'", th >= 6), ("all", None)):
        print(f"{mdl} {kg[j]:.2f} | {lab:18s} | {q(dc, mm):8.2f} | {q(xm_nb - xt_nb, mm):8.2f} | {q(xm_bb - xt_bb, mm):8.2f}")
print("kept points cosmolike n0 vs python n0:", [int(x.sum()) for x in M0[:7]], [int(x.sum()) for x in Mpy[:7]])
