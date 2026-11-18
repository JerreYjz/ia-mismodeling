#!/usr/bin/env python3
"""NLA and TATT fitted to the 3x2pt FLAMINGO data vector, for the v9 bands.

For each truth and cut of ia_base_3x2_common.npz, NLA (A1, eta1) and TATT
(A1, eta1, A2, eta2, b_TA) are fitted to the data vector of the base (GG +
FLAMINGO IA in xi_+- and gamma_t) at the fiducial cosmology and biases, IA
parameters only, with the Gaussian 3x2pt covariance on the cut (as
v8/fid_fits.py for xi_+-; least_squares from the starts of bias_v2, inside
its bounds). Stored: the best fit, chi2, the parameters at a bound, and the
error of each fit, err[X] = P_X^model/P_X^truth - 1 on FINE at every
snapshot (X = GI, II), the known drift of ia_budget_3x2.py.
Output: fits_3x2.json.

    python fits_3x2.py <cosmolike so dir> <cocoa_roman_real/data> [--truths prod] [--cuts a,b]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v8"))
sys.path.insert(0, HERE)
import clean_budget as CB                                          # noqa: E402
from clean_budget import T, B, ME                                  # noqa: E402
import bias_v2 as V                                                # noqa: E402
import ia_fisher as IF                                             # noqa: E402
from ia_base_3x2 import CL3, BN                                    # noqa: E402

MODELS = ("NLA", "TATT")
CUTS = ("cs|none", "cs|0.3", "3x2|none", "3x2|0.3", "3x2b|none", "3x2b|0.3",
        "cs|0.2", "3x2|0.2", "3x2b|0.2")


def ia_vector(cl, mdl, ia):
    if mdl == "NLA":
        return cl.dv_native(0, A1=[ia[0], ia[1]], A2=[0.0, 0.0], BTA=[0.0])
    return cl.dv_native(1, A1=[ia[0], ia[1]], A2=[ia[2], ia[3]], BTA=[ia[4]])


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    arg = lambda f, d: tuple(sys.argv[sys.argv.index(f) + 1].split(",")) if f in sys.argv else d
    truths, cuts = arg("--truths", CB.TRUTHS), arg("--cuts", CUTS)
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    cl = CL3(so_dir, data, threads=8)
    cl.at(B.FID)
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    tm = ME.F.TATTModel(halofit="takahashi")
    common = np.load(os.path.join(HERE, "ia_base_3x2_common.npz"))
    cov = common["cov"]
    path = os.path.join(HERE, "fits_3x2.json")
    out = json.load(open(path)) if os.path.exists(path) else dict(about=__doc__, truths={})
    out["about"] = __doc__
    for tr in truths:
        base = np.load(os.path.join(HERE, f"ia_base_3x2_{tr}.npz"))
        d, zs = base["xi0"], base["zs"]
        Rt = np.array([[np.interp(np.log(IF.FINE), np.log(st["k"]), st[f"R_{tr}"][s, i]) for i in (0, 1)]
                       for s in range(len(zs))])
        tout = out["truths"].setdefault(tr, {})
        for cut in cuts:
            m = common[f"mask|{cut}"]
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            nla_sol = (3.5, 0.68)
            row = {}
            for mdl in MODELS:
                tf = time.time()
                names = B.IAP[mdl]
                lo = np.array([V.BOUNDS[n][0] for n in names])
                hi = np.array([V.BOUNDS[n][1] for n in names])
                best = None
                for s0 in V.starts(mdl, nla_sol):
                    r = least_squares(lambda ia: Lw @ (d[m] - ia_vector(cl, mdl, ia)[m]), s0,
                                      bounds=(lo, hi), x_scale="jac", max_nfev=600)
                    c = float(r.fun @ r.fun)
                    if best is None or c < best[0]:
                        best = (c, r.x)
                chi2, ia = best
                if mdl == "NLA":
                    nla_sol = (ia[0], ia[1])
                bp = dict(zip(names, ia.tolist()))
                at_bound = [n for n, v, l, h in zip(names, ia, lo, hi)
                            if (v - l) < 1e-3*(h - l) or (h - v) < 1e-3*(h - l)]
                err = {X: np.zeros((len(zs), len(IF.FINE))) for X in IF.XS}
                for s in range(len(zs)):
                    Rm = ME.model_ratio(tm, float(zs[s]), IF.FINE, bp, mdl)
                    for i, X in enumerate(IF.XS):
                        err[X][s] = Rm[i]/Rt[s, i] - 1
                row[mdl] = dict(best=bp, chi2=chi2, npts=int(m.sum()), at_bound=at_bound,
                                err={X: e.tolist() for X, e in err.items()})
                print(f"[{time.time() - t0:.0f}s] {tr} {cut:9s} {mdl:4s} ({time.time() - tf:.0f}s): "
                      f"chi2 {chi2:.2f}/{int(m.sum())}, " +
                      ", ".join(f"{n} {v:+.3f}" for n, v in bp.items()) +
                      (f"  AT BOUND {at_bound}" if at_bound else ""), flush=True)
            tout[cut] = row
            json.dump(out, open(path, "w"))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
