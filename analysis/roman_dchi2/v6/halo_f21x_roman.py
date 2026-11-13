#!/usr/bin/env python3
"""Roman figure of bias for Fortuna+21 variants on the v6 template path:
  halo_rb    F21 with red/blue centrals as in F21 Eqs. 3-4 (II 2-halo without
             the red-blue cross term): II 2-halo amplitude r A_s^2, r >= 1/2
  halo_ext   F21 with a chosen satellite grid point / windows (templates2_<tag>)
Same pipeline as halo_roman.py (masks, cases, priors, FoB noise).
    python halo_f21x_roman.py <cocoa data dir> <cov> <outtag> <model> [template tag]
model: halo_rb | halo_x (plain F21 structure on another template set)
For a new template set, build it with build_templates() from a table npz
(env TPL_TABLE, TPL_K1H, TPL_K2H).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_roman as HR                                   # noqa: E402
import halo_ia as H                                       # noqa: E402
B = HR.B
from cosmolike_xi_ia import KMIN                          # noqa: E402

HR.BOUNDS["r"] = (0.5, 5.0)
HR.IAP.update({"halo_rb": ["A1", "eta1", "r", "a1h", "eta1h"], "halo_x": ["A1", "eta1", "a1h", "eta1h"]})


class F21X(HR.IAModel):
    def __init__(self, T, tab, mdl):
        super().__init__(T, tab, "halo")
        self.mdl = mdl
        if mdl == "halo_rb":
            self.starts = [[5.3, 0.5, r, a, e] for r in (1.0, 1.6) for a, e in ((1e-3, 0.0), (0.05, 0.0), (0.3, -2.0))]

    def xi(self, ia):
        if self.mdl != "halo_rb":
            return super().xi(ia)
        A1, e1, r, a1h, e1h = ia
        A = A1*self.x**e1
        v = A @ self.T["GI2h"] + r*(A**2) @ self.T["II2h"]
        return v + super().xi([0.0, 0.0, a1h, e1h])


def main():
    data, covf, tag, mdl = sys.argv[1:5]
    ttag = sys.argv[5] if len(sys.argv) > 5 else "halo"
    t0 = time.time()
    os.chdir(HR.COCOA)
    cov = np.load(covf)
    tabfile = os.environ.get("TPL_TABLE", "halo_tables.npz")
    tab = dict(np.load(os.path.join(HERE, tabfile)))
    R = np.load("run/cosmolike_xi_ia.npz")
    cl = B.CL(os.path.join(HR.COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    logf = open(os.path.join(HERE, f"halo_f21x_roman_{tag}_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    if "TPL_K1H" in os.environ:
        H.K1H = float(os.environ["TPL_K1H"])
    if "TPL_K2H" in os.environ:
        H.K2H = float(os.environ["TPL_K2H"])
    T = HR.templates(cl, tab, os.environ.get("TPL_NOWIN", "0") == "1", ttag)
    M = F21X(T, tab, mdl)
    log(f"[{tag}] model {mdl}, templates {ttag} (table {tabfile}, k1h {H.K1H}, k2h {H.K2H}) ({time.time() - t0:.0f}s)")
    Jc, gS8 = [], []
    for n in B.CP:
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        Jc.append((cl.model("NLA", pp, [0.0, 0.0]) - cl.model("NLA", pm, [0.0, 0.0]))/(2*B.STEP[n]))
        gS8.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
    Jc, gS8 = np.array(Jc).T, np.array(gS8)
    fits = np.load("../v2/fits.npz")
    kd = fits["k"][0]
    J_cells = np.load("analysis/J_cells.npy")
    bk, Pf, err = fits["b"], fits["PNL"], fits["err"]
    zs = tab["zs"]
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    sig = []
    for s in range(len(zs)):
        for xx in range(2):
            conv = PNLd[s]/(bk[s]*Pf[s]) if xx == 0 else PNLd[s]/Pf[s]
            sig.extend((err[s, xx]*conv)[nodes])
    Jcells_S = J_cells*np.array(sig)[None, :]
    S = np.load(os.path.join(HR.V3, "masks_scan.npz"))
    Mz = np.load(os.path.join(HR.V3, "masks_v3.npz"))
    masks = [(f"K0/D {K0:g}", S[f"D1_{K0:g}"].astype(bool), dict(family="D1", K0=K0)) for K0 in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)]
    masks += [(f"const {K0:g}", S[f"flat_{K0:g}"].astype(bool), dict(family="flat", K0=K0)) for K0 in (0.2, 0.3, 0.4, 0.5)]
    masks += [(f"GoF {sh}", Mz[f"{sh}_GoF_x1"].astype(bool), dict(family="GoF", shape=sh)) for sh in ("NLA", "TATT")]
    out = []
    for lab, m, meta in masks:
        log(f"[{lab}: {int(m.sum())} pts ({time.time() - t0:.0f}s)]")
        for case in ("A", "B"):
            r = HR.fit(cl, M, d_fl, xi_gg, Jc, gS8, cov, m, case, Jcells_S, log)
            r.update(mask=lab, **meta)
            out.append(r)
        json.dump(out, open(os.path.join(HERE, f"halo_f21x_roman_{tag}.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
