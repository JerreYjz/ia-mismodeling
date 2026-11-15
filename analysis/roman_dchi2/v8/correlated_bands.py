#!/usr/bin/env python3
"""The IA error budget as a band in k, for random errors correlated in k.

correlated_budget.py lets the random error run over every k with the same
rms, so its budget is one number per correlation length. Here the error is
confined to one decade of k centred on k_c, and the budget eps_95(k_c) is a
band in k, like the bands of clean_budget.py.

Perturbation. At the 31 NODES of correlated_budget, the fractional error of
P_X (X = GI or II, the same at every z) is
    delta_i = w(k_i; k_c) g_i,   <g_i g_j> = eps^2 exp(-(ln k_i - ln k_j)^2 / ell^2),
with the window w = 1 within a decade around k_c and smooth edges,
    w = [1 + exp(-(ln(k/k_c) + ln 10/2)/0.25)]^-1 [1 + exp((ln(k/k_c) - ln 10/2)/0.25)]^-1,
splined in ln k as before. eps is the rms of the error at k_c.
Budget. eps_95(k_c): the rms at which 95% of such errors keep |dOmega_m| and
|dS8| below 0.3 sigma, by linear response (quantile of max|z|, 4e5 draws).
Where it exceeds LIN_MAX, it is found again by refitting 200 errors (common
draws) with the IA amplitude and evolution exact and the cosmology linear
(the fast refit of correlated_budget), by bisection in ln eps on the
fraction inside, between eps_lin/3 and min(3 eps_lin, 1).
Capped at 1 (100%): no limit.
Output: correlated_bands.json, correlated_bands_log.txt.

    python correlated_bands.py <cocoa so dir> <cocoa_roman_real/data>
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_budget as CB                                     # noqa: E402
from clean_budget import T, B, KW                             # noqa: E402
from correlated_budget import NODES, spline_basis, corr, chol  # noqa: E402

TOL = 0.3
ELLS = (0.3, 1.0, 3.0)
HALF = 0.5*np.log(10.0)
EDGE = 0.25
LIN_MAX = 0.05
NDRAW = 400_000
NREFIT = 200
NBISECT = 7
RNG_SEED = 20261115


def window(kc):
    x = np.log(NODES/kc)
    return 1/(1 + np.exp(np.clip(-(x + HALF)/EDGE, -60, 60)))/(1 + np.exp(np.clip((x - HALF)/EDGE, -60, 60)))


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    logf = open(os.path.join(HERE, "correlated_bands_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])                   # before any sim table
    T.GG["fid"] = gg
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    cov = np.load("analysis/cov_example1_xi.npy")
    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {"none": np.ones(1080, bool), "K0/D 0.3": M1[K1.index(0.3)]}
    Hf = CB.tents_on(KW, CB.FINE)
    S_fine = spline_basis(CB.FINE)
    rng = np.random.default_rng(RNG_SEED)
    Z = rng.standard_normal((NDRAW, 2))
    G0 = rng.standard_normal((NREFIT, len(NODES)))           # common draws of the refit search
    free_t, iap = ["omegam", "As_1e9"], ["A", "eta"]
    x_fid = {**{n: B.FID[n] for n in free_t}, "A": 1.0, "eta": 0.0}
    sigA, sigE = T.SIG_IA["A"], T.SIG_IA["eta"]
    out = dict(about=__doc__, nodes_h_per_Mpc=NODES.tolist(), k_c=NODES.tolist(), ells=list(ELLS),
               tolerance=TOL, truths={})

    for tr in CB.TRUTHS:
        model = CB.Truth(cl, st, tr)
        P = model.pnl(B.FID)
        PdE, PEE = model.RdE*P, model.REE*P
        zero = np.zeros_like(PdE)
        fine = {"GI": np.array([T.TB_xi(cl, model, PdE*Hf[j], zero, f"cw_{tr}_GI_{j}")
                                for j in range(len(CB.FINE))]),
                "II": np.array([T.TB_xi(cl, model, zero, PEE*Hf[j], f"cw_{tr}_II_{j}")
                                for j in range(len(CB.FINE))])}
        node_resp = {X: S_fine @ fine[X] for X in ("GI", "II")}
        nz = len(model.zs)
        rsnap = {"GI": [], "II": []}
        for s_ in range(nz):
            only = np.zeros(nz)
            only[s_] = 1.0
            rsnap["GI"].append(T.TB_xi(cl, model, only[:, None]*PdE, zero, f"cw_{tr}_sGI_{s_}"))
            rsnap["II"].append(T.TB_xi(cl, model, zero, only[:, None]*PEE, f"cw_{tr}_sII_{s_}"))
        rsnap = {X: np.array(v) for X, v in rsnap.items()}
        x0 = (1 + model.zs)/T.X0
        xi_truth = rsnap["GI"].sum(0) + rsnap["II"].sum(0)
        J = T.jacobian(model, free_t, iap, x_fid)
        Jc = J[:, :2]
        G = T.s8_row(cl, free_t, iap, x_fid)
        Gc = T.s8_row(cl, free_t, [], x_fid)
        Pi = T.prior_inv(free_t, iap, {})
        log(f"[{time.time() - t0:.0f}s] truth {tr}: responses and Jacobian done")
        tout = {}
        for cut, m in masks.items():
            Ci = np.linalg.inv(cov[np.ix_(m, m)])
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            Jm = J[m]
            Finv = np.linalg.inv(Jm.T @ Ci @ Jm + Pi)
            sig = np.sqrt(np.diag(G @ Finv @ G.T))
            Proj = (G @ Finv @ Jm.T @ Ci)/sig[:, None]

            def fast_shift(dd):
                def resid(p):
                    a = p[2]*x0**p[3]
                    mv = gg[m] + (a @ rsnap["GI"] + a**2 @ rsnap["II"])[m] + Jc[m] @ p[:2]
                    return np.concatenate([Lw @ (dd[m] - mv), [(p[2] - 1)/sigA, p[3]/sigE]])
                r = least_squares(resid, np.array([0.0, 0.0, 1.0, 0.0]), x_scale=[0.01, 0.05, 0.05, 0.1],
                                  max_nfev=400, xtol=1e-12, ftol=1e-12)
                return (Gc @ r.x[:2])/sig

            cres = {"sigma": sig.tolist()}
            for X in ("GI", "II"):
                A = Proj @ node_resp[X][:, m].T
                for ell in ELLS:
                    L = chol(corr(ell))
                    lin, eps95, note = [], [], []
                    for kc in NODES:
                        W = np.diag(window(kc))
                        Sd = A @ W @ corr(ell) @ W @ A.T
                        zmax = np.max(np.abs(Z @ chol(Sd).T), axis=1)
                        e_lin = float(min(TOL/max(np.quantile(zmax, 0.95), 1e-300), 1.0))
                        lin.append(e_lin)
                        if e_lin <= LIN_MAX:
                            eps95.append(e_lin)
                            note.append("linear")
                            continue
                        shapes = (G0 @ L.T)*window(kc)                  # (NREFIT, 31), unit rms

                        def frac(e):
                            ins = 0
                            for dv in shapes:
                                dd = gg + xi_truth + e*(dv @ node_resp[X])
                                ins += np.max(np.abs(fast_shift(dd))) < TOL
                            return ins/NREFIT

                        if frac(1.0) >= 0.95:
                            eps95.append(1.0)
                            note.append("refit, no limit")
                            continue
                        lo, hi = np.log(e_lin/3), np.log(min(3*e_lin, 1.0))
                        if frac(np.exp(hi)) >= 0.95:
                            lo = hi
                        for _ in range(NBISECT):
                            if lo == hi:
                                break
                            mid = 0.5*(lo + hi)
                            lo, hi = (mid, hi) if frac(np.exp(mid)) >= 0.95 else (lo, mid)
                        eps95.append(float(np.exp(0.5*(lo + hi))))
                        note.append("refit")
                    cres[f"{X}|{ell:g}"] = dict(linear=lin, eps95=eps95, method=note)
                    log(f"  {tr} {cut:9s} {X} ell {ell:g}: " + " ".join(f"{v:.3f}" for v in eps95) +
                        f"  ({time.time() - t0:.0f}s)")
            tout[cut] = cres
        out["truths"][tr] = tout
        json.dump(out, open(os.path.join(HERE, "correlated_bands.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
