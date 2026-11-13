#!/usr/bin/env python3
"""Where does the common NLA/TATT S8 shift come from?  (injection path is
clean: bias_diag.py, null FoB 0.01)

  E. NLA with a free amplitude per source bin (IA_REDSHIFT_BINNING, 8 A_i):
     if the FoB drops, the bias is the power-law z evolution, not the k shape.
  F. GI / II split of the FLAMINGO truth: (a) FLAMINGO P_dE with an
     NLA-consistent P_EE = P_dE^2 / P_NL; (b) FLAMINGO P_EE with
     P_dE = -sqrt(P_EE P_NL).  Which half carries the shift?
  G. Fit on large angles only (theta >= 54', within the mask): a shift that
     survives is not a small-scale effect.
  H. FLAMINGO with NLA stitched below k_t (0.1, 0.2, 0.3 h/Mpc; per-snapshot
     NLA amplitude of the k_max 0.29 fit): the low-k FLAMINGO modes are the
     noisiest and Roman weighs them most.
All on the production 13-snapshot injection, case A (Omega_m, A_s free).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B                     # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set, extend_ratio   # noqa: E402

C1RHOCRIT = 0.01389
B.IAP["NLA8"] = [f"A1_{i}" for i in range(8)]
B.IA_START["NLA8"] = [2.5, 3.0, 3.3, 3.5, 3.7, 3.9, 4.2, 4.6]


def model8(self, mdl, p, ia):
    self.at(p)
    if mdl == "NLA8":
        return self.xi_native(0, A1=list(ia), A2=[0.0]*8, BTA=[0.0]*8, evolution=2)
    return _model_orig(self, mdl, p, ia)


_model_orig = B.CL.model
B.CL.model = model8


def main():
    data = sys.argv[1]
    t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    fits = np.load("../v2/fits.npz")
    zs, sns, kd = fits["zs"], fits["sns"], fits["k"][0]
    cl = B.CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (
        cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h, Om = cl.h, B.FID["omegam"]
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    logf = open("analysis/bias_diag2_log.txt", "w")
    out = {}

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    def D(z):
        return np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))

    PNLw = {float(z): cl.PKNL.P(z, KW*h)*h**3 for z in zs}

    def dchi2(v, m):
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    def run_tables(name, PdE, PEE):
        p = write_set("run/tables_diag", name, sns, np.array(PdE), np.array(PEE))
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
        return cl.xi_sim() - xi_gg

    def fitmask(mdl, d, m, lab):
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        log(f"   [{lab}]")
        return B.fit(cl, mdl, d, Ci, Lw, m, ["omegam", "As_1e9"], {}, log)

    # ---------------- E. per-bin NLA ----------------
    log("E. NLA with a free amplitude per source bin, data = GG + FLAMINGO")
    out["E"] = []
    for K0 in (0.25, 0.35, 0.5):
        r = fitmask("NLA8", d_fl, M1[K1.index(K0)], f"mask {K0}/D")
        r.update(mask=K0)
        out["E"].append(r)
    json.dump(out, open("analysis/bias_diag2.json", "w"), indent=1)

    # ---------------- F. GI / II split ----------------
    log("F. GI / II split of the FLAMINGO truth (13 snapshots, as production)")
    T, b, Pfit, Ns = fits["T"], fits["b"], fits["PNL"], fits["Ns_true"]
    rdE = [extend_ratio(kd, T[s, 0]/(b[s]*Pfit[s])) for s in range(len(zs))]
    rEE = [extend_ratio(kd, (T[s, 1] - Ns[s])/Pfit[s]) for s in range(len(zs))]
    P = [PNLw[float(z)] for z in zs]
    variants = {
        "GI_flam_II_nla": ([rdE[s]*P[s] for s in range(13)], [rdE[s]**2*P[s] for s in range(13)]),
        "II_flam_GI_nla": ([-np.sqrt(np.maximum(rEE[s], 0))*P[s] for s in range(13)],
                           [rEE[s]*P[s] for s in range(13)]),
    }
    # H. NLA stitched below k_t
    j29 = int(np.argmin(np.abs(fits["kgrid"] - 0.29)))
    a_nla = fits["NLA_x_true"][:, j29, 0]
    for kt in (0.1, 0.2, 0.3):
        PdE, PEE = [], []
        for s, z in enumerate(zs):
            c1 = a_nla[s]*Om*C1RHOCRIT/D(z)
            lo = KW < kt
            rd, re = rdE[s].copy(), rEE[s].copy()
            rd[lo], re[lo] = -c1, c1**2
            PdE.append(rd*P[s])
            PEE.append(re*P[s])
        variants[f"stitch_nla_below_{kt}"] = (PdE, PEE)
    xi_var = {}
    for name, (PdE, PEE) in variants.items():
        xi_var[name] = run_tables(name, PdE, PEE)
        log(f"   {name:24s}: Delta chi2 vs production FLAMINGO on masks " + ", ".join(
            f"{K0}/D {dchi2(xi_var[name] - R['xi_flam'], M1[K1.index(K0)]):.3f}" for K0 in (0.25, 0.35, 0.5)))
    np.savez("run/xi_flam_variants.npz", **xi_var)
    out["FH"] = []
    for name in xi_var:
        for K0 in (0.35,):
            for mdl in ("NLA", "TATT"):
                r = fitmask(mdl, xi_gg + xi_var[name], M1[K1.index(K0)], f"{name}, mask {K0}/D")
                r.update(variant=name, mask=K0)
                out["FH"].append(r)
        json.dump(out, open("analysis/bias_diag2.json", "w"), indent=1)

    # ---------------- G. large angles only ----------------
    log("G. fit on theta >= 54' only (within the mask), data = GG + FLAMINGO")
    th = np.tile(np.arange(15), 72)
    out["G"] = []
    for K0 in (0.35, 0.5):
        m = M1[K1.index(K0)] & (th >= 10)
        for mdl in ("NLA", "TATT"):
            r = fitmask(mdl, d_fl, m, f"mask {K0}/D and theta >= 54', {int(m.sum())} points")
            r.update(mask=K0)
            out["G"].append(r)
    json.dump(out, open("analysis/bias_diag2.json", "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")
    logf.close()


if __name__ == "__main__":
    main()
