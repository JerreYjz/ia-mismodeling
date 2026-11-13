#!/usr/bin/env python3
"""Full non-linear Roman fits (roman_test.fit, all parameters free, case A and
B, FLAMINGO-noise FoB uncertainty) at the separate xi_+ / xi_- masks singled
out by the linear scan of split_pm.py.
    python verify_split.py <cocoa data dir>"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
V3 = os.path.join(HERE, "..", "v3")
sys.path.insert(0, COCOA)
sys.path.insert(0, V3)
import bias_analysis as B                              # noqa: E402
from cosmolike_xi_ia import KMIN                       # noqa: E402
import roman_test as RT                                # noqa: E402

CONF = [("NLA", 0.3, 0.25), ("NLA", 0.35, 0.25), ("NLA", 0.5, 0.0), ("TATT", 0.4, 0.25), ("TATT", 0.4, 0.0), ("TATT", 0.5, 0.0)]
if os.environ.get("CONF"):
    CONF = [(c.split(":")[0], float(c.split(":")[1]), float(c.split(":")[2])) for c in os.environ["CONF"].split(",")]
TAG = os.environ.get("OUTTAG", "")


def main():
    t0 = time.time()
    os.chdir(COCOA)
    cov = np.load(os.environ.get("COV", "analysis/cov_gauss_d3a.npy"))
    R = np.load("run/cosmolike_xi_ia.npz")
    S = np.load(os.path.join(V3, "masks_scan.npz"))
    fits = np.load("../v2/fits.npz")
    zs, kd = fits["zs"], fits["k"][0]
    cl = B.CL(os.path.join(COCOA, "lib"), sys.argv[1])
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
    logf = open(os.path.join(HERE, f"verify_split{TAG}_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    out = []
    for mdl, kp, km in CONF:
        m = np.zeros(1080, bool)
        m[:540] = S[f"D1_{kp:g}"][:540]
        if km > 0:
            m[540:] = S[f"D1_{km:g}"][540:]
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        log(f"[{mdl} K0+ {kp} K0- {km}: {int(m.sum())} pts ({time.time() - t0:.0f}s)]")
        for case in ("A", "B"):
            free, prior = RT.CASES[case]
            nla = RT.fit(cl, "NLA", d_fl, Ci, Lw, m, free, prior, log)
            r = nla if mdl == "NLA" else RT.fit(cl, "TATT", d_fl, Ci, Lw, m, free, prior, log,
                                                nla_sol=(nla["best"]["A1"], nla["best"]["eta1"]), Jcells_S=Jcells_S)
            if mdl == "NLA":
                r = RT.fit(cl, "NLA", d_fl, Ci, Lw, m, free, prior, log, Jcells_S=Jcells_S)
            out.append(dict(model=mdl, K0p=kp, K0m=km, case=case, npts=int(m.sum()), FoB=r["FoB"], FoB_sig=r.get("FoB_sig"),
                            chi2=r["chi2_bestfit"], dOm=r["dOm"]/r["sOm"], dS8=r["dS8"]/r["sS8"], sS8=r["sS8"], sOm=r["sOm"],
                            at_bound=r["at_bound"], ia=[r["best"][n] for n in B.IAP[mdl]]))
            json.dump(out, open(os.path.join(HERE, f"verify_split{TAG}.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
