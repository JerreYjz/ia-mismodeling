#!/usr/bin/env python3
"""Why is the FoB the same for NLA and TATT?  Suspect: the injection path.

sim_IA interpolates the injected P(k, z) linearly in z between the 13 FLAMINGO
snapshots (z = 0 ... 4, steps up to 1).  The model side (native cosmolike NLA /
TATT) is continuous in z.  A z-interpolation error in the data vector is a
residual that neither model can absorb, so it would bias both the same way.

Tests (all cosmolike, Gaussian covariance, production masks):
  A. NLA power law (A1 3.5, eta 0.68) injected as tables on the 13 snapshot
     redshifts vs on a fine z grid (dz 0.025), both against native NLA:
     Delta chi2 and the per-pair structure of the difference.
  B. Null fits: data = GG + NLA tables (coarse / fine), model = native NLA.
     A non-zero FoB on the coarse tables is the injection-path bias.
  C. FLAMINGO on the fine grid: ratios P_IA/P_NL at the 13 snapshots, cubic
     spline (and, as a variant, linear) in z per k, times cosmolike's P_NL.
  D. Bias fits (case A) on the fine-grid FLAMINGO, NLA and TATT.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B                     # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set, extend_ratio   # noqa: E402

C1RHOCRIT = 0.01389
ZFINE = np.round(np.arange(0.0, 4.0001, 0.025), 4)
PAIRS = B.Cosmolike.PAIRS


def main():
    data = sys.argv[1]
    t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    Ci_all = np.linalg.inv(cov)
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    fits = np.load("../v2/fits.npz")
    zs, sns, kd = fits["zs"], fits["sns"], fits["k"][0]
    cl = B.CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (
        cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h, Om = cl.h, B.FID["omegam"]
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    logf = open("analysis/bias_diag_log.txt", "w")
    out = {}

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    def D(z):
        return np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))

    def PNLw(z):
        return cl.PKNL.P(z, KW*h)*h**3

    def dchi2(v, m=None):
        if m is None:
            return float(v @ Ci_all @ v)
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    def run_tables(name, zlist, PdE, PEE):
        sn = [1000 + i for i in range(len(zlist))]
        p = write_set("run/tables_diag", name, sn, np.array(PdE), np.array(PEE))
        cl.at(B.FID)
        cl.use_sim_tables(p, sn, [float(z) for z in zlist], 1)
        return cl.xi_sim() - xi_gg

    def report(lab, v):
        s = f"{lab}: Delta chi2 all {dchi2(v):.3f}; masks " + ", ".join(
            f"{K0}/D {dchi2(v, M1[K1.index(K0)]):.3f}" for K0 in (0.25, 0.35, 0.5))
        log(s)

    # ---------------- A. NLA tables, coarse vs fine z ----------------
    A1, ETA = 3.5, 0.68
    nat = cl.model("NLA", B.FID, [A1, ETA]) - xi_gg

    def nla_tables(zlist):
        PdE, PEE = [], []
        for z in zlist:
            C1 = A1*((1 + z)/1.62)**ETA*Om*C1RHOCRIT/D(z)
            P = PNLw(z)
            PdE.append(-C1*P)
            PEE.append(C1**2*P)
        return PdE, PEE

    log("A. NLA power law as sim tables vs native NLA")
    tab_c = run_tables("nla_coarse", zs, *nla_tables(zs))
    tab_f = run_tables("nla_fine", ZFINE, *nla_tables(ZFINE))
    report("   13 snapshot z (as the FLAMINGO injection)", tab_c - nat)
    report("   fine z grid, dz 0.025                     ", tab_f - nat)
    # per-pair structure: ratio tables/native for xi_+ at theta >= 54'
    th = np.tile(np.arange(15), 72)
    log("   xi_+ (theta >= 54'), median (tables_coarse / native - 1) per bin pair [%]:")
    rows = []
    for ip, (i, j) in enumerate(PAIRS):
        sel = np.zeros(1080, bool)
        sel[ip*15:(ip + 1)*15] = th[ip*15:(ip + 1)*15] >= 10
        rows.append((i, j, 100*np.median(tab_c[sel]/nat[sel] - 1), 100*np.median(tab_f[sel]/nat[sel] - 1)))
    for i in range(8):
        log("     " + " ".join(f"({a},{b}) {c:+5.2f}/{d:+5.2f}" for a, b, c, d in rows if a == i))
    out["A"] = dict(coarse=dchi2(tab_c - nat), fine=dchi2(tab_f - nat), pairs=rows)

    # ---------------- B. null fits on the injection path ----------------
    log("B. null fits: data = GG + NLA tables, model = native NLA (case A)")
    out["B"] = []
    for lab, tab in (("coarse", tab_c), ("fine", tab_f)):
        for K0 in (0.25, 0.35, 0.5):
            m = M1[K1.index(K0)]
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            Lw = np.linalg.inv(np.linalg.cholesky(C))
            log(f"   [{lab} tables, mask {K0}/D]")
            r = B.fit(cl, "NLA", xi_gg + tab, Ci, Lw, m, ["omegam", "As_1e9"], {}, log)
            r.update(tables=lab, mask=K0)
            out["B"].append(r)
    json.dump(out, open("analysis/bias_diag.json", "w"), indent=1)

    # ---------------- C. FLAMINGO on the fine z grid ----------------
    log("C. FLAMINGO ratios interpolated in z onto the fine grid")
    T, b, Pfit, Ns = fits["T"], fits["b"], fits["PNL"], fits["Ns_true"]
    rdE = np.array([extend_ratio(kd, T[s, 0]/(b[s]*Pfit[s])) for s in range(len(zs))])   # (13, nKW)
    rEE = np.array([extend_ratio(kd, (T[s, 1] - Ns[s])/Pfit[s]) for s in range(len(zs))])
    xi_fine = {}
    for how in ("spline", "linear"):
        if how == "spline":
            fdE, fEE = CubicSpline(zs, rdE, axis=0)(ZFINE), CubicSpline(zs, rEE, axis=0)(ZFINE)
        else:
            fdE = np.array([np.interp(ZFINE, zs, rdE[:, i]) for i in range(KW.size)]).T
            fEE = np.array([np.interp(ZFINE, zs, rEE[:, i]) for i in range(KW.size)]).T
        PdE = [fdE[i]*PNLw(z) for i, z in enumerate(ZFINE)]
        PEE = [fEE[i]*PNLw(z) for i, z in enumerate(ZFINE)]
        xi_fine[how] = run_tables(f"flam_fine_{how}", ZFINE, PdE, PEE)
        report(f"   FLAMINGO fine ({how:6s}) - FLAMINGO coarse (13 snapshots)", xi_fine[how] - R["xi_flam"])
    report("   FLAMINGO fine spline - fine linear", xi_fine["spline"] - xi_fine["linear"])
    np.savez("run/xi_flam_fine.npz", spline=xi_fine["spline"], linear=xi_fine["linear"], zfine=ZFINE)

    # ---------------- D. bias on the fine-grid FLAMINGO ----------------
    log("D. bias fits, data = GG + FLAMINGO (fine z grid), case A")
    out["D"] = []
    for how in ("spline", "linear"):
        for K0 in (0.25, 0.35, 0.5):
            m = M1[K1.index(K0)]
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            Lw = np.linalg.inv(np.linalg.cholesky(C))
            for mdl in ("NLA", "TATT"):
                log(f"   [FLAMINGO fine {how}, mask {K0}/D]")
                r = B.fit(cl, mdl, xi_gg + xi_fine[how], Ci, Lw, m, ["omegam", "As_1e9"], {}, log)
                r.update(interp=how, mask=K0)
                out["D"].append(r)
            json.dump(out, open("analysis/bias_diag.json", "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")
    logf.close()


if __name__ == "__main__":
    main()
