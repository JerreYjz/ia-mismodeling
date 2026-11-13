#!/usr/bin/env python3
"""Roman figure of bias for the red/blue halo-model hybrid: red centrals NLA +
blue centrals tidal torquing (cosmolike's native TATT, A1, eta1, A2, eta2,
b_TA) + red satellites (F21 1-halo templates, a1h, eta1h), and the native
NLA / TATT run through the identical pipeline (halo_roman.fit) as references.

  xi_IA = [xi_TATT(A1, eta1, A2, eta2, bTA) - xi_gg] + sum_s [a_s T_GI1h(a_s) + a_s^2 T_II1h(a_s)]
Satellite templates: templates2_<SATTAG>.npz (default "halo": F21, b = -2,
with the F21 f_1h window). IA held at the fiducial cosmology (as all v6 runs);
the cosmology enters through the lensing-only response.
    python halo_hyb_roman.py <cocoa data dir> [cov] [tag] [models, comma list]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_roman as HR                                   # noqa: E402
B = HR.B
from cosmolike_xi_ia import KMIN                          # noqa: E402

HR.IAP.update({"NLA_nat": ["A1", "eta1"], "TATT_nat": ["A1", "eta1", "A2", "eta2", "bTA"],
               "hyb": ["A1", "eta1", "A2", "eta2", "bTA", "a1h", "eta1h"],
               "hyb0": ["A1", "eta1", "A2", "eta2", "a1h", "eta1h"]})


class Native:
    def __init__(self, cl, xi_gg, mdl):
        self.cl, self.xi_gg, self.mdl = cl, xi_gg, mdl
        self.starts = [[3.5, 0.7]] if mdl == "NLA_nat" else [[3.5, 0.7, 0.0, 0.0, 0.2], [3.5, 0.7, -1.0, 0.0, 0.5], [3.5, 0.7, 1.0, 0.0, 0.5]]

    def xi(self, ia):
        if self.mdl == "NLA_nat":
            return self.cl.model("NLA", B.FID, [ia[0], ia[1]]) - self.xi_gg
        return self.cl.model("TATT", B.FID, list(ia[:5])) - self.xi_gg


class Hybrid:
    def __init__(self, cl, xi_gg, T, tab, bta=True):
        self.cl, self.xi_gg, self.T = cl, xi_gg, T
        self.mdl = "hyb" if bta else "hyb0"
        self.bta = bta
        self.x = (1 + tab["zs"])/1.62
        self.la = np.log(np.maximum(tab["a_grid"], 1e-6))
        base = [[3.5, 0.7, -1.0, 0.0] + ([0.5] if bta else []), [3.5, 0.7, 0.0, 0.0] + ([0.2] if bta else []),
                [3.5, 0.7, 1.0, 0.0] + ([0.5] if bta else [])]
        self.starts = [base[0] + [1e-3, 0.0], base[1] + [0.05, 0.0], base[0] + [0.3, -2.0]]

    def xi(self, ia):
        if self.bta:
            tatt = list(ia[:5])
            a1h, e1h = ia[5], ia[6]
        else:
            tatt = list(ia[:4]) + [0.0]
            a1h, e1h = ia[4], ia[5]
        v = self.cl.model("TATT", B.FID, tatt) - self.xi_gg
        a = np.clip(a1h*self.x**e1h, 0.0, np.exp(self.la[-1]))
        for s in range(len(self.x)):
            if a[s] <= 0:
                continue
            xx = np.log(max(a[s], 1e-6))
            j = int(np.clip(np.searchsorted(self.la, xx), 1, self.la.size - 1))
            w = np.clip((xx - self.la[j - 1])/(self.la[j] - self.la[j - 1]), 0, 1)
            g = self.T["GI1h"][s, j - 1]*(1 - w) + self.T["GI1h"][s, j]*w
            q = self.T["II1h"][s, j - 1]*(1 - w) + self.T["II1h"][s, j]*w
            v = v + a[s]*g + a[s]**2*q
        return v


def main():
    data = sys.argv[1]
    covf = sys.argv[2] if len(sys.argv) > 2 else "analysis/cov_example1_xi.npy"
    tag = sys.argv[3] if len(sys.argv) > 3 else "example1"
    which = sys.argv[4].split(",") if len(sys.argv) > 4 else ["NLA_nat", "TATT_nat", "hyb0", "hyb"]
    sattag = os.environ.get("SATTAG", "halo")
    t0 = time.time()
    os.chdir(HR.COCOA)
    cov = np.load(covf)
    tab = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
    R = np.load("run/cosmolike_xi_ia.npz")
    cl = B.CL(os.path.join(HR.COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    logf = open(os.path.join(HERE, f"halo_hyb_roman_{tag}_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    Jc, gS8 = [], []
    for n in B.CP:
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        Jc.append((cl.model("NLA", pp, [0.0, 0.0]) - cl.model("NLA", pm, [0.0, 0.0]))/(2*B.STEP[n]))
        gS8.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
    Jc, gS8 = np.array(Jc).T, np.array(gS8)
    T = dict(np.load(os.path.join(HERE, f"templates2_{sattag}.npz")))
    t1 = time.time()
    cl.model("TATT", B.FID, [3.5, 0.7, -1.0, 0.0, 0.5])
    log(f"[{tag}] one TATT evaluation: {time.time() - t1:.2f}s; satellites from templates2_{sattag}")
    models = {"NLA_nat": Native(cl, xi_gg, "NLA_nat"), "TATT_nat": Native(cl, xi_gg, "TATT_nat"),
              "hyb0": Hybrid(cl, xi_gg, T, tab, bta=False), "hyb": Hybrid(cl, xi_gg, T, tab, bta=True)}
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
    masks = [(f"K0/D {K0:g}", S[f"D1_{K0:g}"].astype(bool), dict(family="D1", K0=K0)) for K0 in (0.3, 0.35, 0.4, 0.5, 0.25)]
    masks += [(f"const {K0:g}", S[f"flat_{K0:g}"].astype(bool), dict(family="flat", K0=K0)) for K0 in (0.4, 0.3)]
    path = os.path.join(HERE, f"halo_hyb_roman_{tag}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["model"], r["mask"], r["case"]) for r in out}
    for lab, m, meta in masks:
        log(f"[{lab}: {int(m.sum())} pts ({time.time() - t0:.0f}s)]")
        for case in ("B", "A"):
            for k in which:
                if (k, lab, case) in done:
                    continue
                r = HR.fit(cl, models[k], d_fl, xi_gg, Jc, gS8, cov, m, case, Jcells_S, log)
                r.update(mask=lab, **meta)
                out.append(r)
                json.dump(out, open(path, "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
