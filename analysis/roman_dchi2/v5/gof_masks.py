#!/usr/bin/env python3
"""Roman fits (all parameters free, cases A and B, FLAMINGO-noise FoB error) on
the masks built from the simulation goodness-of-fit cuts (v3/masks_v3.npz:
NLA_GoF_x1, TATT_GoF_x1), with a chosen covariance.
    python gof_masks.py <cocoa data dir> <cov npy> <tag>"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
V3 = os.path.join(HERE, "..", "v3")
sys.path.insert(0, COCOA)
sys.path.insert(0, V3)
import bias_analysis as B                              # noqa: E402
from cosmolike_xi_ia import KMIN                       # noqa: E402
import roman_test as RT                                # noqa: E402

data, covf, tag = sys.argv[1:4]
os.chdir(COCOA)
cov = np.load(covf)
R = np.load("run/cosmolike_xi_ia.npz")
Mz = np.load(os.path.join(V3, "masks_v3.npz"))
fits = np.load("../v2/fits.npz")
zs, kd = fits["zs"], fits["k"][0]
cl = B.CL(os.path.join(COCOA, "lib"), data)
cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
h = cl.h
d_fl = cl.model("NLA", B.FID, [0.0, 0.0]) + R["xi_flam"]
J_cells = np.load("analysis/J_cells.npy")
bk, Pf, err = fits["b"], fits["PNL"], fits["err"]
nodes = np.where(kd >= KMIN)[0]
PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
sig = []
for s in range(len(zs)):
    for x in range(2):
        conv = PNLd[s]/(bk[s]*Pf[s]) if x == 0 else PNLd[s]/Pf[s]
        sig.extend((err[s, x]*conv)[nodes])
Jcells_S = J_cells*np.array(sig)[None, :]
out = []
for shape in ("NLA_GoF", "TATT_GoF"):
    m = Mz[f"{shape}_x1"].astype(bool)
    kz = {z: float(np.interp(z, Mz["z2"], Mz[f"kuv_{shape}"])) for z in (0.5, 1.0, 2.0)}
    C = cov[np.ix_(m, m)]
    Ci = np.linalg.inv(C)
    Lw = np.linalg.inv(np.linalg.cholesky(C))
    print(f"[{shape}: {int(m.sum())} pts; k_UV(0.5,1,2) = {kz[0.5]:.3f} {kz[1.0]:.3f} {kz[2.0]:.3f}]", flush=True)
    for case in ("A", "B"):
        free, prior = RT.CASES[case]
        rn = RT.fit(cl, "NLA", d_fl, Ci, Lw, m, free, prior, print, Jcells_S=Jcells_S)
        rt = RT.fit(cl, "TATT", d_fl, Ci, Lw, m, free, prior, print, nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=Jcells_S)
        out.append(dict(shape=shape, case=case, npts=int(m.sum()), kuv=kz,
                        **{mdl: dict(FoB=r["FoB"], FoB_sig=r.get("FoB_sig"), dOm=r["dOm"]/r["sOm"], dS8=r["dS8"]/r["sS8"],
                                     sS8=r["sS8"], chi2=r["chi2_bestfit"], at_bound=r["at_bound"]) for mdl, r in (("NLA", rn), ("TATT", rt))}))
        json.dump(out, open(os.path.join(HERE, f"gof_masks_{tag}.json"), "w"), indent=1)
print("done", flush=True)
