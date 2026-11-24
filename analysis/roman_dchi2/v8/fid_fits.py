#!/usr/bin/env python3
"""NLA and TATT fitted to the FLAMINGO data vector at the fiducial cosmology.

For each truth (S1, prod) and cut (none, k_UV = 0.3/D(z)), the IA parameters
of cosmolike's native NLA (A1, eta1) or TATT (A1, eta1, A2, eta2, b_TA) are
fitted to Roman xi_+- with the cosmology held at the FLAMINGO D3A fiducial
(least squares with the example1 covariance, several starts, chain-prior
bounds as roman_test). The error of the fitted model against the truth,
    Delta_X(k, z) = P_X^model(k, z) / P_X^truth(k, z) - 1,   X = GI, II,
is evaluated at the core snapshots (0.2 <= z <= 2) on k_dense of
clean_budget, in the convention of the truth tables
(tatt_fit/fit_tatt_flamingo.TATTModel, model_errors.model_ratio).
A truth with a B-mode II table (CP, v3/cp_truth.py) has its B part,
(xi_+, -xi_-), in the data vector (v9/bmode_tables.py); native TATT carries
its own B modes, NLA none. The error is that of the E tables.
Output: fid_fits.json (--truths adds or replaces only those truths).

    python fid_fits.py <cocoa so dir> <cocoa_roman_real/data> [--truths CP]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_budget as CB                                         # noqa: E402
from clean_budget import T, B, ME, KW                             # noqa: E402
import bias_v2 as V                                               # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "v9"))
import bmode_tables as BT                                         # noqa: E402


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    truths = CB.TRUTHS
    if "--truths" in sys.argv:
        truths = tuple(sys.argv[sys.argv.index("--truths") + 1].split(","))
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    cl = B.CL(so_dir, data, threads=8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])
    T.GG["fid"] = gg
    st = BT.load_truths(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    zs, kd = st["zs"], st["k"]
    core = [s for s, z in enumerate(zs) if CB.ZCORE[0] - 1e-9 <= z <= CB.ZCORE[1] + 1e-9]
    cov = np.load("analysis/cov_example1_xi.npy")
    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {"none": np.ones(1080, bool), "K0/D 0.3": M1[K1.index(0.3)]}
    tm = ME.F.TATTModel(halofit="takahashi")
    path = os.path.join(HERE, "fid_fits.json")
    out = json.load(open(path)) if os.path.exists(path) else dict(fits={})
    out.update(about=__doc__, k_dense=CB.KDENSE.tolist(), z_core=[float(zs[s]) for s in core])
    for tr in truths:
        model = CB.Truth(cl, st, tr)
        full = model
        if BT.has_b(st, tr):
            full = BT.EBModel(model, CB.Truth(cl, st, tr + "_B"), CB.Truth(cl, st, "ZERO"))
        d = full(dict(B.FID), {"A": 1.0, "eta": 0.0})
        for cut, m in masks.items():
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            nla_sol = (3.5, 0.68)
            for mdl in ("NLA", "TATT"):
                names = B.IAP[mdl]
                lo = np.array([V.BOUNDS[n][0] for n in names])
                hi = np.array([V.BOUNDS[n][1] for n in names])
                best = None
                for s0 in V.starts(mdl, nla_sol):
                    r = least_squares(lambda ia: Lw @ (d[m] - cl.model(mdl, B.FID, ia)[m]), s0,
                                      bounds=(lo, hi), x_scale="jac", max_nfev=600)
                    c = float(r.fun @ r.fun)
                    if best is None or c < best[0]:
                        best = (c, r.x)
                chi2, ia = best
                if mdl == "NLA":
                    nla_sol = (ia[0], ia[1])
                bp = dict(zip(names, ia.tolist()))
                err = {"GI": [], "II": []}
                for s in core:
                    Rt = np.array([np.interp(np.log(CB.KDENSE), np.log(kd), model.R[s, i]) for i in (0, 1)])
                    Rm = ME.model_ratio(tm, float(zs[s]), CB.KDENSE, bp, mdl)
                    err["GI"].append((Rm[0]/Rt[0] - 1).tolist())
                    err["II"].append((Rm[1]/Rt[1] - 1).tolist())
                at_bound = [n for n, v, l, h in zip(names, ia, lo, hi)
                            if (v - l) < 1e-3*(h - l) or (h - v) < 1e-3*(h - l)]
                out["fits"][f"{tr}|{cut}|{mdl}"] = dict(best=bp, chi2=chi2, npts=int(m.sum()),
                                                        at_bound=at_bound, error=err)
                print(f"[{time.time() - t0:.0f}s] {tr} {cut:9s} {mdl:4s}: chi2 {chi2:.1f} ({int(m.sum())} pts); "
                      f"IA {np.round(ia, 3)}" + (f"  AT BOUND {at_bound}" if at_bound else ""), flush=True)
                json.dump(out, open(path, "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
