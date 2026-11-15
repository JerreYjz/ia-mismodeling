#!/usr/bin/env python3
"""The IA error budget for random errors correlated in k.

Question. The error of the FLAMINGO IA spectrum is not one coherent shape
but a random function of k whose values at nearby k are correlated. How
large may it be before Roman xi_+- moves Omega_m or S8 by 0.3 sigma?

Perturbation. At the NODES (31 points, 10 per decade, 0.01 to 10 h/Mpc) the
fractional errors delta_i of P_X are a Gaussian random vector,
    <delta_i delta_j> = eps^2 C_ij,   C_ij = exp(-(ln k_i - ln k_j)^2 / ell^2),
eps the rms error at each node and ell the correlation length in ln k. The
perturbed spectrum is P_X(k, z) [1 + f(k)], the same at every z, with f the
cubic spline in ln k through (k_i, delta_i), held flat outside the nodes.
X = GI or II.
Model and criterion as clean_budget.py: the truth-as-model (IA amplitude A,
evolution eta, Omega_m, A_s free), example1_cov, |dOmega_m| < 0.3 sigma and
|dS8| < 0.3 sigma.
Budget. The spline is linear in delta and xi is linear in the tables, so the
shifts in units of sigma are d = A_X delta with A_X (2 x 31) exact (spline
basis on the 61 tents of clean_budget, one cosmolike run per tent). d is
Gaussian with covariance eps^2 A_X C A_X^T; the criterion holds for a
fraction q of the errors when eps < eps_q = 0.3 / Q_q(max_i |z_i|),
z ~ N(0, A_X C A_X^T), Q_q the q quantile (by 4e5 draws).
Checks. (1) Direct injection: realizations of f evaluated on the table grid
(no tents), injected through cosmolike and refitted by Gauss-Newton, against
the linear shifts. (2) Beyond linear response: at eps_95 the fraction of 300
realizations inside the criterion, each refitted with the IA amplitude and
evolution exact (the analytic IA model of nonlinear_tol.py).
Output: correlated_budget.json, correlated_budget_log.txt.

    python correlated_budget.py <cocoa so dir> <cocoa_roman_real/data>
"""
import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_budget as CB                                     # noqa: E402
from clean_budget import T, B, KW                             # noqa: E402

TOL = 0.3
NODES = np.logspace(-2, 1, 31)
ELLS = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
QS = (0.68, 0.95)
NDRAW = 400_000
NMC = 300
N_DIRECT = 4
RNG_SEED = 20261114


def spline_basis(k):
    """S (len(NODES), len(k)): f(k) = delta @ S, the cubic spline in ln k, flat outside."""
    x = np.clip(np.log(k), np.log(NODES[0]), np.log(NODES[-1]))
    return np.array([CubicSpline(np.log(NODES), e)(x) for e in np.eye(len(NODES))])


def corr(ell):
    d = np.log(NODES)[:, None] - np.log(NODES)[None, :]
    return np.exp(-d**2/ell**2)


def chol(C):
    w, V = np.linalg.eigh(C)
    return V*np.sqrt(np.clip(w, 0, None))                     # L L^T = C, safe when C is near singular


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    logf = open(os.path.join(HERE, "correlated_budget_log.txt"), "w")

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
    S_fine = spline_basis(CB.FINE)                            # (31, 61)
    S_kw = spline_basis(KW)
    rng = np.random.default_rng(RNG_SEED)
    Z = rng.standard_normal((NDRAW, 2))
    free_t, iap = ["omegam", "As_1e9"], ["A", "eta"]
    x_fid = {**{n: B.FID[n] for n in free_t}, "A": 1.0, "eta": 0.0}
    s8f = cl.s8(dict(B.FID))
    sigA, sigE = T.SIG_IA["A"], T.SIG_IA["eta"]
    out = dict(about=__doc__, nodes_h_per_Mpc=NODES.tolist(), ells=list(ELLS), quantiles=list(QS),
               tolerance=TOL, truths={})

    for tr in CB.TRUTHS:
        model = CB.Truth(cl, st, tr)
        P = model.pnl(B.FID)
        PdE, PEE = model.RdE*P, model.REE*P
        zero = np.zeros_like(PdE)
        fine = {"GI": np.array([T.TB_xi(cl, model, PdE*Hf[j], zero, f"cb_{tr}_GI_{j}")
                                for j in range(len(CB.FINE))]),
                "II": np.array([T.TB_xi(cl, model, zero, PEE*Hf[j], f"cb_{tr}_II_{j}")
                                for j in range(len(CB.FINE))])}
        node_resp = {X: S_fine @ fine[X] for X in ("GI", "II")}  # (31, 1080) per unit delta_i
        nz = len(model.zs)
        rsnap = {"GI": [], "II": []}
        for s_ in range(nz):
            only = np.zeros(nz)
            only[s_] = 1.0
            rsnap["GI"].append(T.TB_xi(cl, model, only[:, None]*PdE, zero, f"cb_{tr}_sGI_{s_}"))
            rsnap["II"].append(T.TB_xi(cl, model, zero, only[:, None]*PEE, f"cb_{tr}_sII_{s_}"))
        rsnap = {X: np.array(v) for X, v in rsnap.items()}
        x0 = (1 + model.zs)/T.X0
        xi_truth = rsnap["GI"].sum(0) + rsnap["II"].sum(0)
        log(f"[{time.time() - t0:.0f}s] truth {tr}: tent and snapshot responses done")
        J = T.jacobian(model, free_t, iap, x_fid)
        Jc = J[:, :2]
        G = T.s8_row(cl, free_t, iap, x_fid)
        Gc = T.s8_row(cl, free_t, [], x_fid)
        Pi = T.prior_inv(free_t, iap, {})
        tout = {}
        for cut, m in masks.items():
            Ci = np.linalg.inv(cov[np.ix_(m, m)])
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            Jm = J[m]
            Finv = np.linalg.inv(Jm.T @ Ci @ Jm + Pi)
            sig = np.sqrt(np.diag(G @ Finv @ G.T))
            Proj = (G @ Finv @ Jm.T @ Ci)/sig[:, None]
            cres = dict(sigma=sig.tolist(), budget={}, verify_direct=[], verify_nonlinear={})

            def fast_shift(dd):
                def resid(p):
                    a = p[2]*x0**p[3]
                    mv = gg[m] + (a @ rsnap["GI"] + a**2 @ rsnap["II"])[m] + Jc[m] @ p[:2]
                    return np.concatenate([Lw @ (dd[m] - mv), [(p[2] - 1)/sigA, p[3]/sigE]])
                r = least_squares(resid, np.array([0.0, 0.0, 1.0, 0.0]), x_scale=[0.01, 0.05, 0.05, 0.1],
                                  max_nfev=400, xtol=1e-12, ftol=1e-12)
                return (Gc @ r.x[:2])/sig

            for X in ("GI", "II"):
                A = Proj @ node_resp[X][:, m].T                        # (2, 31)
                rows = []
                for ell in ELLS:
                    C = corr(ell)
                    Sd = A @ C @ A.T
                    zmax = np.max(np.abs(Z @ chol(Sd).T), axis=1)
                    epsq = {f"{q:.2f}": float(TOL/np.quantile(zmax, q)) for q in QS}
                    rms = np.sqrt(np.diag(Sd))
                    rows.append(dict(ell=ell, eps=epsq, rms_shift_per_unit_eps=rms.tolist(),
                                     shift_corr=float(Sd[0, 1]/np.prod(rms)),
                                     limiting=["Omega_m", "S8"][int(np.argmax(rms))]))
                    log(f"  {tr} {cut:9s} {X} ell {ell:4.2f}: eps_68 {epsq['0.68']:.4f}  eps_95 "
                        f"{epsq['0.95']:.4f}  (rms shift per unit eps: Om {rms[0]:.2f}, S8 {rms[1]:.2f})")
                cres["budget"][X] = rows
                cres[f"A_{X}"] = A.tolist()
                # check 2: beyond linear response, at eps_95
                nl = []
                for row in rows:
                    if row["ell"] not in (0.3, 1.0, 3.0):
                        continue
                    eps = row["eps"]["0.95"]
                    L = chol(corr(row["ell"]))
                    dl = rng.standard_normal((NMC, len(NODES))) @ L.T
                    inside_lin, inside_nl, worst = 0, 0, []
                    for dv in dl:
                        dd = gg + xi_truth + eps*(dv @ node_resp[X])
                        sh = fast_shift(dd)
                        lin = eps*(A @ dv)
                        inside_lin += np.max(np.abs(lin)) < TOL
                        inside_nl += np.max(np.abs(sh)) < TOL
                        worst.append(float(np.max(np.abs(sh - lin))))
                    nl.append(dict(ell=row["ell"], eps=eps, frac_linear=inside_lin/NMC,
                                   frac_refit=inside_nl/NMC, max_abs_diff=max(worst)))
                    log(f"  {tr} {cut:9s} {X} ell {row['ell']:.1f} at eps_95 {eps:.4f}: inside "
                        f"{inside_lin/NMC:.3f} (linear), {inside_nl/NMC:.3f} (refit); max |refit - "
                        f"linear| {max(worst):.3f} sigma  ({time.time() - t0:.0f}s)")
                cres["verify_nonlinear"][X] = nl
            # check 1: direct injection through cosmolike, Gauss-Newton refit
            for X in ("GI", "II"):
                A = np.array(cres[f"A_{X}"])
                row = [r for r in cres["budget"][X] if r["ell"] == 1.0][0]
                eps = row["eps"]["0.95"]
                L = chol(corr(1.0))
                for i in range(N_DIRECT):
                    dv = rng.standard_normal(len(NODES)) @ L.T
                    fk = 1 + eps*(dv @ S_kw)
                    dd = gg + T.TB_xi(cl, model, PdE*(fk if X == "GI" else 1.0),
                                      PEE*(fk if X == "II" else 1.0), f"cb_{tr}_ver")
                    x = CB.gn_refit(model, cl, dd, Ci, m, free_t, iap, x_fid)
                    full = [(x["omegam"] - B.FID["omegam"])/sig[0],
                            (cl.s8(T.full_p({k: x[k] for k in free_t})) - s8f)/sig[1]]
                    lin = (eps*(A @ dv)).tolist()
                    cres["verify_direct"].append(dict(X=X, ell=1.0, eps=eps, full=full, linear=lin))
                    log(f"  {tr} {cut:9s} direct {X} ell 1.0 eps {eps:.4f} #{i}: full ({full[0]:+.3f}, "
                        f"{full[1]:+.3f}) linear ({lin[0]:+.3f}, {lin[1]:+.3f})")
            tout[cut] = cres
        out["truths"][tr] = tout
        json.dump(out, open(os.path.join(HERE, "correlated_budget.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
