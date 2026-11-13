#!/usr/bin/env python3
"""The Roman test on stitched FLAMINGO truths (control variants).

Truth variants, per snapshot, as ratios R_X = P_X / P_NL on the data k grid
(26 points, 0.009 - 17.7 h/Mpc), then sim_IA tables on the wide grid:
  prod : production truth (cosmolike_xi_ia 'flam'): smoothing spline of the
         measured FLAMINGO ratios on k >= 0.05, edge value held below 0.05,
         white shape noise N_s = B-mode plateau subtracted from P_EE.
  S1   : notes' stitch with the spline on the simulation side:
         R = (1 - w) R_NLA + w R_spline, w = logistic in ln k at k_t(z)
         (width 0.30), R_NLA = (-C1, C1^2) from the notes' NLA fit on
         0.05 <= k <= 0.3; N_s = B-mode plateau.
  S2   : as S1 with the notes' N_s (from the NLA + N_s fit on 0.05 - 0.3).
  S3   : the notes' recipe literally (build_spectra 'truth_nla'): measured
         simulation side, notes' N_s.
  S4   : production truth above k = 0.05, NLA (-C1, C1^2) below.
For each variant, mask and model: bias_v2/roman_test fit with every
parameter free (case A: Omega_m, A_s; case B for S1 on the K0/D masks),
FoB in (Omega_m, S8), its FLAMINGO-noise uncertainty, chi2 at the best fit;
and Delta chi2_Roman between the variant and the production truth.
    python stitched_test.py <cocoa data dir>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B                              # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set        # noqa: E402
from fit_stitched_kmax import KT, SIGMA_W, weight      # noqa: E402
import roman_test as RT                                # noqa: E402

VARIANTS = ["S1", "S2", "S3", "S4"]
LABEL = {"prod": "production (spline, $N_s$ = B plateau)",
         "S1": "stitch: NLA below $k_t$, spline above, $N_s$ = B plateau",
         "S2": "stitch: NLA below $k_t$, spline above, notes' $N_s$",
         "S3": "notes' recipe: NLA below $k_t$, measured above, notes' $N_s$",
         "S4": "production above 0.05, NLA below"}


def extend_full(kd, R, kw=KW):
    """Ratio on the full data grid -> wide grid (linear in ln k, edges held)."""
    return np.interp(np.log(kw), np.log(kd), R)


def build_truths(fits, spec):
    """Per-variant ratios (nsn, 2, nk): [R_dE, R_EE] on the data grid."""
    kd = fits["k"][0]
    T, b, Pf, Ns_bb = fits["T"], fits["b"], fits["PNL"], fits["Ns_true"]
    sns = fits["sns"]
    Ns_nla = spec["Ns_nla"]
    th = spec["truth_highk"]
    nsn = len(sns)
    out = {v: np.zeros((nsn, 2, kd.size)) for v in ["prod"] + VARIANTS}
    meta = dict(C1=[], w=[])
    for s in range(nsn):
        m = kd >= KMIN
        rdE = T[s, 0]/(b[s]*Pf[s])
        rEE_bb = (T[s, 1] - Ns_bb[s])/Pf[s]
        rEE_nla = (T[s, 1] - Ns_nla[s])/Pf[s]
        # production: edge value held below KMIN
        prod = np.array([np.where(m, rdE, rdE[m][0]), np.where(m, rEE_bb, rEE_bb[m][0])])
        out["prod"][s] = prod
        mC1 = th[s, 0, 0]                 # -C1 (w ~ 1e-4 at the first grid point)
        C1sq = th[s, 1, 0]
        w = weight(kd, KT[int(sns[s])], SIGMA_W)
        meta["C1"].append(-mC1)
        meta["w"].append(w)
        spl_dE = np.where(m, rdE, mC1)    # spline where measured, NLA below (w -> 0 there anyway)
        out["S1"][s] = np.array([(1 - w)*mC1 + w*spl_dE, (1 - w)*C1sq + w*np.where(m, rEE_bb, C1sq)])
        out["S2"][s] = np.array([(1 - w)*mC1 + w*spl_dE, (1 - w)*C1sq + w*np.where(m, rEE_nla, C1sq)])
        out["S3"][s] = spec["truth_nla"][s, :2]
        out["S4"][s] = np.array([np.where(m, rdE, mC1), np.where(m, rEE_bb, C1sq)])
    return kd, out, meta


def main():
    data = sys.argv[1]
    t0 = time.time()
    os.chdir(COCOA)
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    Mz = np.load(os.path.join(HERE, "masks_v3.npz"))
    fits = np.load("../v2/fits.npz")
    spec = np.load("../spectra.npz")
    zs, sns = fits["zs"], fits["sns"]
    kd, truths, meta = build_truths(fits, spec)
    np.savez(os.path.join(HERE, "stitched_truths.npz"), k=kd, zs=zs, sns=sns, C1=np.array(meta["C1"]),
             w=np.array(meta["w"]), **{f"R_{v}": truths[v] for v in truths})
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])

    def xi_of(v):
        PdE = np.array([extend_full(kd, truths[v][s, 0])*PNLw[s] for s in range(len(zs))])
        PEE = np.array([extend_full(kd, truths[v][s, 1])*PNLw[s] for s in range(len(zs))])
        p = write_set("run/tables_stitch", v, sns, PdE, PEE)
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
        return cl.xi_sim() - xi_gg
    xi = {v: xi_of(v) for v in ["prod"] + VARIANTS}
    np.savez(os.path.join(HERE, "stitched_xi.npz"), **xi, xi_flam=R["xi_flam"])
    print(f"rebuilt production truth vs stored xi_flam: max |diff|/max|xi| = "
          f"{np.max(np.abs(xi['prod'] - R['xi_flam']))/np.max(np.abs(R['xi_flam'])):.2e}", flush=True)

    # noise propagation (as roman_test), sim side weighted by w for the stitches
    J_cells = np.load("analysis/J_cells.npy")
    bk, Pf, err = fits["b"], fits["PNL"], fits["err"]
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    sig, wsig = [], []
    for s in range(len(zs)):
        for x in range(2):
            conv = PNLd[s]/(bk[s]*Pf[s]) if x == 0 else PNLd[s]/Pf[s]
            sig.extend((err[s, x]*conv)[nodes])
            wsig.extend(meta["w"][s][nodes])
    sig, wsig = np.array(sig), np.array(wsig)
    JS = {"S4": J_cells*sig[None, :]}
    for v in ("S1", "S2", "S3"):
        JS[v] = J_cells*(sig*wsig)[None, :]
    logf = open(os.path.join(HERE, "stitched_test_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    part = sys.argv[2] if len(sys.argv) > 2 else "all"          # all | k0 | flat (parallel runs)
    masks = [(f"K0/D {K0}", M1[K1.index(K0)], dict(family="K0/D", K0=K0)) for K0 in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)]
    masks += [(f"flat x{f:g}", Mz[f"TATT_GoF_x{f:g}"], dict(family="flat", factor=f)) for f in (1.0, 1.4, 2.0)]
    if part != "all":
        masks = [mm for mm in masks if (mm[2]["family"] == "K0/D") == (part == "k0")]
    suffix = "" if part == "all" else f"_{part}"
    out = []
    for lab, m, mt in masks:
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        log(f"[{lab}: {int(m.sum())} points  ({time.time() - t0:.0f}s)]")
        for v in VARIANTS:
            dv = (xi[v] - xi["prod"])[m]
            dchi2 = float(dv @ Ci @ dv)
            d = xi_gg + xi[v]
            cases = ["A", "B"] if (v == "S1" and mt["family"] == "K0/D") else ["A"]
            for case in cases:
                free, prior = RT.CASES[case]
                log(f"  {v}: Delta chi2_Roman(variant - production) {dchi2:.2f}; case {case}")
                rn = RT.fit(cl, "NLA", d, Ci, Lw, m, free, prior, log, Jcells_S=JS[v])
                rt = RT.fit(cl, "TATT", d, Ci, Lw, m, free, prior, log,
                            nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=JS[v])
                for r in (rn, rt):
                    r.update(mask=lab, case=case, variant=v, dchi2_vs_prod=dchi2, **mt)
                    out.append(r)
                json.dump(out, open(os.path.join(HERE, f"stitched_test{suffix}.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
