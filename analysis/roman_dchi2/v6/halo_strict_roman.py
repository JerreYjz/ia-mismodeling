#!/usr/bin/env python3
"""Roman figure of bias for the strict halo-model IA (halo_strict.py), same
pipeline as halo_roman.py (sim_IA templates at the fiducial cosmology,
lensing-only cosmology response, same masks, cases, priors, FoB noise).
Templates per snapshot s (amplitudes A_s = A1 x^eta1, a_s = a1h x^eta1h):
  GIc  : -C1 f_rc I1m P_lin            (A_s)
  IIc  :  C1^2 f_rc^2 P_lin            (A_s^2)
  GIs  : -(I1m P_lin Q(a) + S_dI(a))   (a_s)
  IIs  :  P_lin Q(a)^2 + S_II(a)       (a_s^2)
  IIx  :  2 C1 f_rc P_lin Q(a)         (A_s a_s)
P_lin: cosmolike's linear total-matter spectrum at the fiducial cosmology.
    python halo_strict_roman.py <cocoa data dir> [cov npy] [tag]
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
from cosmolike_xi_ia import KW, KMIN, write_set           # noqa: E402

HR.IAP["halo_strict"] = ["A1", "eta1", "a1h", "eta1h"]


def templates(cl, tab, tag="strict"):
    cache = os.path.join(HERE, f"templates2_{tag}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    zs = tab["zs"]
    sns = np.load("../v2/fits.npz")["sns"]
    h = cl.h
    idx = np.searchsorted(tab["k_all"], KW)
    assert np.allclose(tab["k_all"][idx], KW)
    nsn, na = len(zs), len(tab["a_grid"])
    out = dict(GIc=np.zeros((nsn, 1080)), IIc=np.zeros((nsn, 1080)),
               GIs=np.zeros((nsn, na, 1080)), IIs=np.zeros((nsn, na, 1080)), IIx=np.zeros((nsn, na, 1080)))
    zero = np.zeros((nsn, len(KW)))

    def run(PdE, PEE):
        p = write_set("run/tables_halo", f"tpl_{tag}", sns, PdE, PEE)
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
        v = cl.xi_sim()
        p0 = write_set("run/tables_halo", f"tpl_{tag}_zero", sns, zero, zero)
        cl.use_sim_tables(p0, sns, [float(z) for z in zs], 1)
        return v - cl.xi_sim()
    cl.at(B.FID)
    for s, z in enumerate(zs):
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))
        C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
        PL = cl.PKL.P(z, KW*h)*h**3
        frc = tab["hod"][s, 5]
        I1 = tab["I1m"][s][idx]
        a = zero.copy(); a[s] = -C1*frc*I1*PL
        out["GIc"][s] = run(a, zero)
        b = zero.copy(); b[s] = (C1*frc)**2*PL
        out["IIc"][s] = run(zero, b)
        for ia in range(na):
            Q = tab["Q"][s, ia][idx]
            a = zero.copy(); a[s] = -(I1*PL*Q + tab["SdI"][s, ia][idx])
            out["GIs"][s, ia] = run(a, zero)
            b = zero.copy(); b[s] = PL*Q**2 + tab["SII"][s, ia][idx]
            out["IIs"][s, ia] = run(zero, b)
            b = zero.copy(); b[s] = 2*C1*frc*PL*Q
            out["IIx"][s, ia] = run(zero, b)
        print(f"  templates {tag}: snapshot {s} done", flush=True)
    np.savez(cache, **out)
    return out


class StrictModel:
    mdl = "halo_strict"

    def __init__(self, T, tab):
        self.T = T
        self.x = (1 + tab["zs"])/1.62
        self.la = np.log(np.maximum(tab["a_grid"], 1e-6))

    def xi(self, ia):
        A = ia[0]*self.x**ia[1]
        v = A @ self.T["GIc"] + (A**2) @ self.T["IIc"]
        a = np.clip(ia[2]*self.x**ia[3], 0.0, np.exp(self.la[-1]))
        for s in range(len(self.x)):
            if a[s] <= 0:
                continue
            xx = np.log(max(a[s], 1e-6))
            j = int(np.clip(np.searchsorted(self.la, xx), 1, self.la.size - 1))
            w = np.clip((xx - self.la[j - 1])/(self.la[j] - self.la[j - 1]), 0, 1)
            it = lambda K: self.T[K][s, j - 1]*(1 - w) + self.T[K][s, j]*w
            v = v + a[s]*it("GIs") + a[s]**2*it("IIs") + A[s]*a[s]*it("IIx")
        return v


def main():
    data = sys.argv[1]
    covf = sys.argv[2] if len(sys.argv) > 2 else "analysis/cov_example1_xi.npy"
    tag = sys.argv[3] if len(sys.argv) > 3 else "example1"
    t0 = time.time()
    os.chdir(HR.COCOA)
    cov = np.load(covf)
    tab = dict(np.load(os.path.join(HERE, os.environ.get("HS_TABLES", "halo_strict_tables.npz"))))
    R = np.load("run/cosmolike_xi_ia.npz")
    cl = B.CL(os.path.join(HR.COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    logf = open(os.path.join(HERE, f"halo_strict_roman_{tag}_log.txt"), "w")

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
    T = templates(cl, tab, os.environ.get("HS_TPL", "strict"))
    M = StrictModel(T, tab)
    log(f"templates done ({time.time() - t0:.0f}s)")
    # validation: full table set vs template sum
    sns = np.load("../v2/fits.npz")["sns"]
    zs = tab["zs"]
    ia = [5.0, 0.5, 0.05, 0.0]
    x = (1 + zs)/1.62
    idx = np.searchsorted(tab["k_all"], KW)
    la = np.log(np.maximum(tab["a_grid"], 1e-6))
    PdE = np.zeros((len(zs), len(KW)))
    PEE = np.zeros_like(PdE)
    cl.at(B.FID)
    for s, z in enumerate(zs):
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))
        C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
        PL = cl.PKL.P(z, KW*h)*h**3
        A, a = ia[0]*x[s]**ia[1], ia[2]*x[s]**ia[3]
        j = int(np.clip(np.searchsorted(la, np.log(a)), 1, la.size - 1))
        w = (np.log(a) - la[j - 1])/(la[j] - la[j - 1])
        it = lambda K: tab[K][s, j - 1][idx]*(1 - w) + tab[K][s, j][idx]*w
        g = A*C1*tab["hod"][s, 5] + a*it("Q")
        PdE[s] = -(tab["I1m"][s][idx]*PL*g + a*it("SdI"))
        PEE[s] = PL*g**2 + a**2*it("SII")
    p = write_set("run/tables_halo", "check_strict", sns, PdE, PEE)
    cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
    full = cl.xi_sim()
    p0 = write_set("run/tables_halo", "check_strict_zero", sns, 0*PdE, 0*PEE)
    cl.use_sim_tables(p0, sns, [float(z) for z in zs], 1)
    full = full - cl.xi_sim()
    tsum = M.xi(ia)
    log(f"validation: strict full tables vs template sum, chi2 = {float((full - tsum) @ np.linalg.solve(cov, full - tsum)):.3e} "
        "(nonzero part from interpolating Q^2 linearly in a)")
    fits = np.load("../v2/fits.npz")
    kd = fits["k"][0]
    J_cells = np.load("analysis/J_cells.npy")
    bk, Pf, err = fits["b"], fits["PNL"], fits["err"]
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
        json.dump(out, open(os.path.join(HERE, f"halo_strict_roman_{tag}.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
