#!/usr/bin/env python3
"""The Roman test, as the chain would see it.

For each mask (production K0/D(z) family and the z-dependent TATT-shaped
cuts x f), each model (NLA, TATT) and two cosmology settings
  A: Omega_m, A_s free
  B: Omega_m, A_s, n_s, H0, Omega_b free, the last three with Gaussian priors
     of the width of the chain's flat priors (range / sqrt 12)
fit every parameter (IA bounded to the chain priors, several starts) to the
Roman xi_+- data vector with the FLAMINGO IA injected; report the (Omega_m,
S8) shift in sigma (FoB, IA marginalised with the flat-prior width), its
uncertainty from the simulation's error bars, and chi2 at the best fit.
Controls at every mask and case:
  TATT on a TATT truth (Part 1a fiducial)     -> must be ~0 (matched model)
  NLA  on a TATT truth                        -> the Part 1 mismodelling bias
  NLA  on an NLA truth through the injection path (sim tables) -> must be ~0
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
import bias_analysis as B                              # noqa: E402
import bias_v2 as V                                    # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set        # noqa: E402

C1RHOCRIT = 0.01389
CASES = {"A": (["omegam", "As_1e9"], {}), "B": (B.CP, B.PRIOR_B)}


def fit(cl, mdl, d, Ci, Lw, m, free, prior, log, nla_sol=(3.5, 0.68), Jcells_S=None):
    """bias_v2.fit with Gaussian cosmology priors (case B)."""
    nia = len(B.IAP[mdl])
    names = free + B.IAP[mdl]
    nf = len(free)
    lo = np.array([V.BOUNDS[n][0] for n in names])
    hi = np.array([V.BOUNDS[n][1] for n in names])
    x_fid = np.array([B.FID[n] for n in free])
    psig = np.array([prior.get(n, np.inf) for n in free])
    has_p = np.isfinite(psig)

    def ia_model(ia):
        return cl.model(mdl, B.FID, ia)[m]

    def Jc_at(ia):
        J = np.zeros((m.sum(), nf))
        for i, n in enumerate(free):
            pp, pm = dict(B.FID), dict(B.FID)
            pp[n] += B.STEP[n]
            pm[n] -= B.STEP[n]
            J[:, i] = (cl.model(mdl, pp, ia) - cl.model(mdl, pm, ia))[m]/(2*B.STEP[n])
        return J
    best = None
    for s0 in V.starts(mdl, nla_sol):
        r0 = least_squares(lambda ia: Lw @ (d[m] - ia_model(ia)), s0, bounds=(lo[nf:], hi[nf:]), x_scale="jac", max_nfev=600)
        c = float(r0.fun @ r0.fun)
        if best is None or c < best[0]:
            best = (c, r0.x)
    chi2_ia, ia0 = best
    x = np.concatenate([x_fid, ia0])
    for rep in range(2):
        Jc = Jc_at(x[nf:])

        def resid(x):
            dc = x[:nf] - x_fid
            rp = np.where(has_p, dc/np.where(has_p, psig, 1.0), 0.0)[has_p]
            return np.concatenate([Lw @ (d[m] - ia_model(x[nf:]) - Jc @ dc), rp])
        r = least_squares(resid, x, bounds=(lo, hi), x_scale="jac", max_nfev=2000)
        x = r.x
    pc = dict(B.FID)
    pc.update(zip(free, x[:nf]))
    ia = x[nf:]
    res_exact = (d - cl.model(mdl, pc, ia))[m]
    chi2 = float(res_exact @ Ci @ res_exact)
    at_bound = [n for n, v, l, h in zip(names, x, lo, hi) if (v - l) < 1e-3*(h - l) or (h - v) < 1e-3*(h - l)]
    Jia = np.zeros((m.sum(), nia))
    for i in range(nia):
        e = np.zeros(nia)
        e[i] = 0.05
        Jia[:, i] = (ia_model(ia + e) - ia_model(ia - e))/0.1
    J = np.hstack([Jc, Jia])
    Pinv = np.diag([0.0 if not np.isfinite(v) else 1/v**2 for v in psig] + [1/V.IA_SIG[n]**2 for n in B.IAP[mdl]])
    Cov = np.linalg.inv(J.T @ Ci @ J + Pinv)
    s8_fid, s8_fit = cl.s8(B.FID), cl.s8(pc)
    G = np.zeros((2, len(names)))
    G[0, names.index("omegam")] = 1.0
    for i, n in enumerate(free):
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        G[1, i] = (cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n])
    St = G @ Cov @ G.T
    dth = np.array([pc["omegam"] - B.FID["omegam"], s8_fit - s8_fid])
    fob = float(np.sqrt(dth @ np.linalg.solve(St, dth)))
    out = dict(model=mdl, free=free, chi2_ia_only=chi2_ia, chi2_bestfit=chi2, best=dict(zip(names, x.tolist())),
               at_bound=at_bound, npts=int(m.sum()), dOm=float(dth[0]), dS8=float(dth[1]),
               sOm=float(np.sqrt(St[0, 0])), sS8=float(np.sqrt(St[1, 1])), FoB=fob)
    if Jcells_S is not None:
        Best = Cov @ J.T @ Ci
        Bn = Best @ Jcells_S[m]
        Vn = G @ Bn @ Bn.T @ G.T
        g = np.linalg.solve(St, dth)/max(fob, 1e-12)
        out["FoB_sig"] = float(np.sqrt(g @ Vn @ g))
    log(f"    {mdl:4s} case {'A' if nf == 2 else 'B'}: chi2 {chi2:6.2f}; dOm {dth[0]/out['sOm']:+.2f}s dS8 {dth[1]/out['sS8']:+.2f}s; "
        f"FoB {fob:.2f}" + (f" +- {out['FoB_sig']:.2f}" if "FoB_sig" in out else "") + f"; IA {np.round(ia, 3)}"
        + (f"  AT BOUND {at_bound}" if at_bound else ""))
    return out


def main():
    data = sys.argv[1]
    t0 = time.time()
    os.chdir(COCOA)
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    Mz = np.load(os.path.join(HERE, "masks_v3.npz"))
    fits = np.load("../v2/fits.npz")
    zs, sns, kd = fits["zs"], fits["sns"], fits["k"][0]
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    d_tatt = cl.model("TATT", B.FID, B.TATT_FID)
    # NLA truth through the injection path (13 snapshot tables)
    A1, ETA = 3.5, 0.68
    PdE, PEE = [], []
    for z in zs:
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))
        C1 = A1*((1 + z)/1.62)**ETA*B.FID["omegam"]*C1RHOCRIT/Dz
        P = cl.PKNL.P(z, KW*h)*h**3
        PdE.append(-C1*P)
        PEE.append(C1**2*P)
    p = write_set("run/tables_diag", "nla_truth", sns, np.array(PdE), np.array(PEE))
    cl.at(B.FID)
    cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
    d_nla = xi_gg + (cl.xi_sim() - xi_gg)
    # noise propagation matrix
    J_cells = np.load("analysis/J_cells.npy")
    T, bk, Pf, err = fits["T"], fits["b"], fits["PNL"], fits["err"]
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    sig = []
    for s in range(len(zs)):
        for x in range(2):
            conv = PNLd[s]/(bk[s]*Pf[s]) if x == 0 else PNLd[s]/Pf[s]
            sig.extend((err[s, x]*conv)[nodes])
    Jcells_S = J_cells*np.array(sig)[None, :]
    logf = open(os.path.join(HERE, "roman_test_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    masks = [(f"K0/D {K0}", M1[K1.index(K0)], dict(family="K0/D", K0=K0)) for K0 in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)]
    masks += [(f"flat x{f:g}", Mz[f"TATT_GoF_x{f:g}"], dict(family="flat", factor=f)) for f in (0.7, 1.0, 1.4, 2.0)]
    out = []
    for lab, m, meta in masks:
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        log(f"[{lab}: {int(m.sum())} points  ({time.time() - t0:.0f}s)]")
        for case, (free, prior) in CASES.items():
            log(f"  data = GG + FLAMINGO")
            rn = fit(cl, "NLA", d_fl, Ci, Lw, m, free, prior, log, Jcells_S=Jcells_S)
            rt = fit(cl, "TATT", d_fl, Ci, Lw, m, free, prior, log, nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=Jcells_S)
            log(f"  controls")
            c1 = fit(cl, "TATT", d_tatt, Ci, Lw, m, free, prior, log, nla_sol=(3.8, 0.57))
            c2 = fit(cl, "NLA", d_tatt, Ci, Lw, m, free, prior, log, nla_sol=(3.8, 0.57))
            c3 = fit(cl, "NLA", d_nla, Ci, Lw, m, free, prior, log, nla_sol=(3.5, 0.68))
            for r, tag in ((rn, "NLA_on_FLAMINGO"), (rt, "TATT_on_FLAMINGO"), (c1, "TATT_on_TATT"), (c2, "NLA_on_TATT"), (c3, "NLA_on_NLA_injected")):
                r.update(mask=lab, case=case, test=tag, **meta)
                out.append(r)
            json.dump(out, open(os.path.join(HERE, "roman_test.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
