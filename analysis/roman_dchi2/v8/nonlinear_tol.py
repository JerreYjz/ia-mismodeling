#!/usr/bin/env python3
"""Non-linear tolerances for the large errors of clean_budget.py.

The linear response of clean_budget.py is exact for small errors and checked
by full refits there, but where the linear tolerance is large (II at large
scales with a cut reaches 90%) the refit moves far and linear response
fails: II goes as the square of the IA amplitude. Every tolerance whose
linear value exceeds LIN_MAX is therefore found again by refitting.

Refit. The truth-as-model with the IA amplitude A and evolution eta exact
and the cosmology (Omega_m, A_s) through its response at the fiducial, J_c;
Gaussian priors on A and eta as in clean_budget; scipy least_squares. The IA
part is evaluated exactly without cosmolike runs: the sim tables interpolate
linearly in z between snapshots, so
    xi_IA(A, eta) = sum_s a_s r_GI,s + a_s^2 r_II,s,   a_s = A ((1+z_s)/1.62)^eta,
with r_X,s the response to the truth's spectrum X at snapshot s alone (26
cosmolike runs), and the perturbed data are xi_IA + eps sum_m f_m r_X,m with
the 61 tent responses of clean_budget. Checked against model() and against
the full Gauss-Newton refit of clean_budget at one large error.
Search. For each sign s = +1 (spectrum too high) and -1 (too low) separately,
the eps at which max(|dOmega_m|/sigma, |dS8|/sigma) = 0.3, by bracketing and
bisection on (0, 1]; if the shift stays below 0.3 at eps = 1 the tolerance is
1 (no limit within 100%).
Output: nonlinear_tol.json (tol_up, tol_down per truth, cut, shape|X, node).

    python nonlinear_tol.py <cocoa so dir> <cocoa_roman_real/data>
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
from clean_budget import T, B, KW, NODES                      # noqa: E402

TOL = 0.3
LIN_MAX = 0.10
NBISECT = 12


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    src = json.load(open(os.path.join(HERE, "clean_budget.json")))
    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])
    T.GG["fid"] = gg
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    cov = np.load("analysis/cov_example1_xi.npy")
    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {"none": np.ones(1080, bool), "K0/D 0.3": M1[K1.index(0.3)]}
    Hf = CB.tents_on(KW, CB.FINE)
    shp = CB.shapes()
    free_t = ["omegam", "As_1e9"]
    x_fid = {**{n: B.FID[n] for n in free_t}, "A": 1.0, "eta": 0.0}
    sigA, sigE = T.SIG_IA["A"], T.SIG_IA["eta"]
    out = dict(about=__doc__, lin_max=LIN_MAX, truths={})
    log = open(os.path.join(HERE, "nonlinear_tol_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    for tr in CB.TRUTHS:
        model = CB.Truth(cl, st, tr)
        P = model.pnl(B.FID)
        PdE, PEE = model.RdE*P, model.REE*P
        Jc = T.jacobian(model, free_t, [], x_fid)                     # (1080, 2) at A = 1, eta = 0
        G = T.s8_row(cl, free_t, [], x_fid)
        nz = len(model.zs)
        zero = np.zeros_like(PdE)
        rsnap = {"GI": [], "II": []}
        for s_ in range(nz):
            only = np.zeros(nz)
            only[s_] = 1.0
            rsnap["GI"].append(T.TB_xi(cl, model, only[:, None]*PdE, zero, f"nl_{tr}_sGI_{s_}"))
            rsnap["II"].append(T.TB_xi(cl, model, zero, only[:, None]*PEE, f"nl_{tr}_sII_{s_}"))
        rsnap = {X: np.array(v) for X, v in rsnap.items()}
        fine = {"GI": np.array([T.TB_xi(cl, model, PdE*Hf[mm], zero, f"nl_{tr}_fGI_{mm}")
                                for mm in range(len(CB.FINE))]),
                "II": np.array([T.TB_xi(cl, model, zero, PEE*Hf[mm], f"nl_{tr}_fII_{mm}")
                                for mm in range(len(CB.FINE))])}
        x0 = (1 + model.zs)/CB.T.X0

        def m_ia(A, eta):
            a = A*x0**eta
            return gg + a @ rsnap["GI"] + (a**2) @ rsnap["II"]

        xi_truth = rsnap["GI"].sum(0) + rsnap["II"].sum(0)
        chk = np.max(np.abs(m_ia(1.1, 0.3) - model(dict(B.FID), {"A": 1.1, "eta": 0.3})))/np.max(np.abs(xi_truth))
        say(f"[{time.time() - t0:.0f}s] {tr}: analytic IA model vs cosmolike at A = 1.1, eta = 0.3: {chk:.1e}")
        tout = {}
        for cut, m in masks.items():
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            sig = np.array(src["truths"][tr]["cuts"][cut]["sigma"])

            def refit(dd):
                def resid(p):
                    dc, A, eta = p[:2], p[2], p[3]
                    mv = m_ia(A, eta)[m] + Jc[m] @ dc
                    return np.concatenate([Lw @ (dd[m] - mv), [(A - 1)/sigA, eta/sigE]])
                r = least_squares(resid, np.array([0.0, 0.0, 1.0, 0.0]), x_scale=[0.01, 0.05, 0.05, 0.1],
                                  max_nfev=400, xtol=1e-12, ftol=1e-12)
                dth = G @ r.x[:2]
                return dth/sig

            def shift(shape, X, j, eps):
                f = shp[(shape, j)]
                dd = gg + xi_truth + eps*((f @ fine["GI"] if X in ("GI", "both") else 0)
                                          + (f @ fine["II"] if X in ("II", "both") else 0))
                return refit(dd)

            if tr == "S1" and cut == "K0/D 0.3":
                s = shift("below", "II", 4, 0.9189)
                say(f"[check] S1 K0/D 0.3 below II k 0.1 eps 0.919: fast refit ({s[0]:+.3f}, {s[1]:+.3f}); "
                    f"full Gauss-Newton refit gave (-0.278, -0.715)")
            cres = {}
            lin = src["truths"][tr]["cuts"][cut]["tol"]
            for key, tl in lin.items():
                shape, X = key.split("|")
                if X == "both":
                    continue
                up, down = list(tl), list(tl)
                for j, t in enumerate(tl):
                    if t <= LIN_MAX:
                        continue
                    for sgn in (+1, -1):
                        g = lambda e: np.max(np.abs(shift(shape, X, j, sgn*e))) - TOL
                        if g(1.0) < 0:
                            res = 1.0
                        else:
                            lo, hi = 0.0, 1.0
                            e = min(t, 0.99)
                            ge = g(e)
                            if ge < 0:
                                lo = e
                            else:
                                hi = e
                            for _ in range(NBISECT):
                                mid = 0.5*(lo + hi)
                                lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
                            res = 0.5*(lo + hi)
                        (up if sgn > 0 else down)[j] = float(res)
                    say(f"  {tr} {cut:9s} {key:9s} k {NODES[j]:.3f}: linear {t:.3f} -> up {up[j]:.3f}, "
                        f"down {down[j]:.3f}  ({time.time() - t0:.0f}s)")
                cres[key] = dict(linear=tl, up=up, down=down)
            tout[cut] = cres
            out["truths"][tr] = tout
            json.dump(out, open(os.path.join(HERE, "nonlinear_tol.json"), "w"), indent=1)
    say(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
