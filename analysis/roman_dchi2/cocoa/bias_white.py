#!/usr/bin/env python3
"""Decompose the Roman shift: is it the k-structure of FLAMINGO's II excess?

Variants of the injected truth (13 snapshots, as production), case A fits
with the bounded priors of bias_v2:
  prod        FLAMINGO P_dE and P_EE (smoothed)                      [bias_v2 "E"]
  GI_II1      FLAMINGO P_dE, II = P_dE^2 / P_NL (single tidal field, q = 1)
  GI_II1_W    as GI_II1 plus a white II excess X_s per snapshot (the
              error-weighted mean of X(k) over 0.05 <= k <= 0.5): FLAMINGO
              with its II excess made exactly white
  GI_II1_Wk   as GI_II1 plus the measured, k-dependent X(k) (smoothed-truth
              version): this is "prod" again, as a check of the decomposition
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B                                   # noqa: E402
import bias_v2 as V                                         # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set, extend_ratio   # noqa: E402

MASKS = (0.25, 0.35, 0.5)


def main():
    data = sys.argv[1]
    t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    f = np.load("../v2/fits.npz")
    zs, sns, kd = f["zs"], f["sns"], f["k"][0]
    y, err, b, Pf, Ns, T = f["y"], f["err"], f["b"], f["PNL"], f["Ns_true"], f["T"]
    cl = B.CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (
        cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    PNLw = {float(z): cl.PKNL.P(z, KW*h)*h**3 for z in zs}
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    logf = open("analysis/bias_white_log.txt", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    def q(v, m):
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    def run_tables(name, rdE, rEE):
        PdE = [rdE[s]*PNLw[float(z)] for s, z in enumerate(zs)]
        PEE = [rEE[s]*PNLw[float(z)] for s, z in enumerate(zs)]
        p = write_set("run/tables_diag", name, sns, np.array(PdE), np.array(PEE))
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
        return cl.xi_sim() - xi_gg

    # ratios to FLAMINGO's P_NL (fit convention), then extended to KW; the
    # table convention is ratio x cosmolike P_NL, as in cosmolike_xi_ia.py
    sel = (kd >= KMIN) & (kd <= 0.5)
    rdE_s = T[:, 0]/(b*Pf)                       # smoothed P_dE / P_NL
    rEE_s = (T[:, 1] - Ns[:, None])/Pf           # smoothed (P_EE - Ns) / P_NL
    X_meas = (y[:, 1] - Ns[:, None]) - (y[:, 0]/b)**2/Pf
    dX = np.sqrt(err[:, 1]**2 + (2*(y[:, 0]/b)*err[:, 0]/b/Pf)**2)
    Xw = np.array([np.sum(X_meas[s, sel]/dX[s, sel]**2)/np.sum(1/dX[s, sel]**2) for s in range(13)])
    chi_flat = np.array([np.sum(((X_meas[s, sel] - Xw[s])/dX[s, sel])**2) for s in range(13)])
    log("white II excess per snapshot X_s [(Mpc/h)^3], chi2 of X(k) = const over " + f"{int(sel.sum())} nodes:")
    for s, z in enumerate(zs):
        log(f"   z {z:4.2f}: X {Xw[s]:6.3f} (Ns {Ns[s]:6.3f}), chi2 {chi_flat[s]:5.1f}")
    variants = {}
    variants["GI_II1"] = (rdE_s, rdE_s**2)
    variants["GI_II1_W"] = (rdE_s, rdE_s**2 + Xw[:, None]/Pf)
    variants["GI_II1_Wk"] = (rdE_s, rdE_s**2 + (T[:, 1] - Ns[:, None] - T[:, 0]**2/b**2/Pf)/Pf)
    xi = {}
    for name, (rd, re) in variants.items():
        rd_w = np.array([extend_ratio(kd, rd[s]) for s in range(13)])
        re_w = np.array([extend_ratio(kd, re[s]) for s in range(13)])
        xi[name] = run_tables(name, rd_w, re_w)
        log(f"{name:10s}: Delta chi2 vs production FLAMINGO on masks " + ", ".join(
            f"{K0}/D {q(xi[name] - R['xi_flam'], M1[K1.index(K0)]):.3f}" for K0 in MASKS))
    np.savez("run/xi_flam_white.npz", **xi)
    out = []
    for K0 in MASKS:
        m = M1[K1.index(K0)]
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        for name in ("GI_II1", "GI_II1_W", "GI_II1_Wk"):
            log(f"[{name}, mask {K0}/D]")
            rn = V.fit(cl, "NLA", xi_gg + xi[name], Ci, Lw, m, ["omegam", "As_1e9"], log)
            rt = V.fit(cl, "TATT", xi_gg + xi[name], Ci, Lw, m, ["omegam", "As_1e9"], log,
                       nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]))
            for r in (rn, rt):
                r.update(variant=name, mask=K0)
                out.append(r)
        json.dump(out, open("analysis/bias_white.json", "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
