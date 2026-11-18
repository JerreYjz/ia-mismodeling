#!/usr/bin/env python3
"""The scale cut from the v9 base: NLA and TATT shifts against K_0.

For each truth (S1, prod) and each cut of the production family
k_UV = K_0/D(z) (masks_n1 of run/cosmolike_xi_ia.npz, K_0 = 0.15 to 0.8, and
no cut), NLA (A1, eta1) and TATT (A1, eta1, A2, eta2, b_TA) are refitted to
the FLAMINGO data vector at the fiducial cosmology with that mask, as
v8/fid_fits.py. The error of each fit, Delta_X = P^model/P^truth - 1 on the
tents at every snapshot (as ia_base.py), is the known drift; the rigid fit of
the base with the same mask gives
    mu         the linear shift of (Omega_m, S8) in units of their errors;
    fob        the 2D figure of bias sqrt(dp^T C^-1 dp) in (Omega_m, S8);
    room       sigma_max of a random error on top (GI and II, exp kernel,
               ell = 1, L_z = 0.3 and inf), zero once |mu| >= 0.3;
    sigma, fom the marginalised errors and 1/sqrt(det C) of (Omega_m, S8);
plus the model chi2, the fitted IA parameters and those at a prior bound.
Output: scale_cut_scan.json.

    python scale_cut_scan.py <cocoa so dir> <cocoa_roman_real/data> [--truths prod]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(HERE, "..", "v8")
sys.path.insert(0, V8)
sys.path.insert(0, HERE)
import clean_budget as CB                                          # noqa: E402
from clean_budget import T, B, ME                                  # noqa: E402
import bias_v2 as V                                                # noqa: E402
import ia_fisher as IF                                             # noqa: E402

MODELS = ("NLA", "TATT")
ROOM = dict(ell=1.0, kind="exp", lzs=(0.3, np.inf))


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    truths = CB.TRUTHS
    if "--truths" in sys.argv:
        truths = tuple(sys.argv[sys.argv.index("--truths") + 1].split(","))
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    cl = B.CL(so_dir, data, threads=8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])                       # before any sim table
    T.GG["fid"] = gg
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    cov = np.load("analysis/cov_example1_xi.npy")
    R0 = np.load("run/cosmolike_xi_ia.npz")
    cuts = [("none", np.ones(1080, bool))] + [(f"{float(k):g}", m) for k, m in
                                              zip(R0["K_n1"], R0["masks_n1"])]
    tm = ME.F.TATTModel(halofit="takahashi")
    out = dict(about=__doc__, K0=[c for c, _ in cuts], room=dict(ell=ROOM["ell"], kind=ROOM["kind"],
               lzs=["0.3", "inf"]), truths={})
    nfit, ntot = 0, len(truths)*len(cuts)*len(MODELS)
    for tr in truths:
        base = IF.Base(os.path.join(HERE, f"ia_base_{tr}.npz"), os.path.join(HERE, "ia_base_common.npz"))
        zs = base.zs
        Rmap = base.knob_map(IF.XS)
        Ks = {lz: IF.residual_cov(IF.NODES, zs, ROOM["ell"], lz, ROOM["kind"], IF.XS, 0.0)
              for lz in ROOM["lzs"]}
        model = CB.Truth(cl, st, tr)
        d = model(dict(B.FID), {"A": 1.0, "eta": 0.0})
        assert np.allclose(d, base.xi0, rtol=1e-6, atol=1e-12*np.max(np.abs(d)))
        Rt = np.array([[np.interp(np.log(IF.FINE), np.log(model.kd), model.R[s, i]) for i in (0, 1)]
                       for s in range(len(zs))])
        tout = {}
        for cut, m in cuts:
            rigid, _ = IF.rigid_fit(base, m)
            A = rigid.shift(Rmap)
            row = dict(npts=int(m.sum()), sigma=rigid.sig.tolist(), fom=IF.fom_pair(rigid),
                       dropped=rigid.dropped, models={})
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            nla_sol = (3.5, 0.68)
            for mdl in MODELS:
                tf = time.time()
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
                at_bound = [n for n, v, l, h in zip(names, ia, lo, hi)
                            if (v - l) < 1e-3*(h - l) or (h - v) < 1e-3*(h - l)]
                c_knobs = np.zeros(base.nb)
                for s in range(len(zs)):
                    Rm = ME.model_ratio(tm, float(zs[s]), IF.FINE, bp, mdl)
                    for i, X in enumerate(IF.XS):
                        # an analysis model wrong by Delta moves the fit as the data moving by -Delta
                        c_knobs[base.iknob[X][s]] = -(Rm[i]/Rt[s, i] - 1)
                mu = rigid.shift(c_knobs)
                dp = mu*rigid.sig
                fob = float(np.sqrt(dp @ np.linalg.solve(rigid.C2, dp)))
                rooms = {f"{lz:g}": IF.rms_budget(A, Ks[lz], mu)[0] for lz in ROOM["lzs"]}
                row["models"][mdl] = dict(best=bp, chi2=chi2, at_bound=at_bound, mu=mu.tolist(),
                                          fob=fob, room=rooms)
                nfit += 1
                print(f"[{time.time() - t0:.0f}s] {tr} K0 {cut:>4s} {mdl:4s} ({nfit}/{ntot}, "
                      f"{time.time() - tf:.0f}s): chi2 {chi2:.1f}/{int(m.sum())}, mu ({mu[0]:+.3f}, "
                      f"{mu[1]:+.3f}), FoB {fob:.3f}, room {rooms['0.3']:.4f}/{rooms['inf']:.4f}, "
                      f"FoM {row['fom']:.0f}" + (f"  AT BOUND {at_bound}" if at_bound else ""), flush=True)
            tout[cut] = row
            out["truths"][tr] = tout
            json.dump(out, open(os.path.join(HERE, "scale_cut_scan.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
