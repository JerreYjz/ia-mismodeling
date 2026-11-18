#!/usr/bin/env python3
"""The GGL scale cut from the v9 3x2pt base: NLA and TATT shifts against K0_ggl.

Cosmic shear is held at its cut, K0_cs = 0.2 (masks_n1, the robust cut of
scale_cut_scan.py), w at K0 = 0.2, and gamma_t is cut at k_UV = K0_ggl/D(z)
for K0_ggl = 0.1 to 0.5, in two configurations: every lens-source pair
("all") and sources behind lenses only ("behind"). The reference is xi_+-
alone at K0_cs = 0.2 ("cs"). At each mask NLA and TATT are refitted to the
3x2pt FLAMINGO data vector at the fiducial cosmology and biases (IA
parameters only, as fits_3x2.py); the error of each fit on the tents at
every snapshot is the known drift, and the rigid fit of the base (cosmology,
A, eta, the 8 biases) with the same mask gives
    mu      the shift of (Omega_m, S8) in units of their errors, and its
            parts from the GI and the II errors (mu_GI + mu_II = mu);
    fob     the 2D figure of bias in (Omega_m, S8);
    room    sigma_max of a random GI and II error on top (exp, ell = 1,
            L_z = 0.3 and inf), zero once |mu| >= 0.3;
    sigma, fom, chi2, the fitted IA parameters and those at a bound.
Gaussian 3x2pt covariance throughout. Output: scale_cut_scan_3x2.json.

    python scale_cut_scan_3x2.py <cosmolike so dir> <cocoa_roman_real/data> [--truths prod]
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
from ia_base_3x2 import CL3, NXI, NGT                              # noqa: E402
from fits_3x2 import ia_vector, MODELS                             # noqa: E402

K0_CS, K0_GC = "0.2", "0.2"
K0_GGL = ("0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.5")
ROOM = dict(ell=1.0, kind="exp", lzs=(0.3, np.inf))


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    truths = CB.TRUTHS
    if "--truths" in sys.argv:
        truths = tuple(sys.argv[sys.argv.index("--truths") + 1].split(","))
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    cl = CL3(so_dir, data, threads=8)
    cl.at(B.FID)
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    tm = ME.F.TATTModel(halofit="takahashi")
    common = os.path.join(HERE, "ia_base_3x2_common.npz")
    c = np.load(common)
    cov = c["cov"]

    def mask(config, kg):
        m = np.zeros(cov.shape[0], bool)
        m[:NXI] = c[f"xi|{K0_CS}"]
        if config != "cs":
            m[NXI:NXI + NGT] = c[f"gt|{kg}"] & (c["behind"] if config == "behind" else True)
            m[NXI + NGT:] = c[f"w|{K0_GC}"]
        return m

    runs = [("cs", None)] + [(cf, kg) for cf in ("all", "behind") for kg in K0_GGL]
    out = dict(about=__doc__, K0_cs=K0_CS, K0_gc=K0_GC, K0_ggl=list(K0_GGL),
               room=dict(ell=ROOM["ell"], kind=ROOM["kind"], lzs=["0.3", "inf"]), truths={})
    for tr in truths:
        base = IF.Base(os.path.join(HERE, f"ia_base_3x2_{tr}.npz"), common)
        zs, d = base.zs, base.xi0
        Rmap = base.knob_map(IF.XS)
        Ks = {lz: IF.residual_cov(IF.NODES, zs, ROOM["ell"], lz, ROOM["kind"], IF.XS, 0.0)
              for lz in ROOM["lzs"]}
        Rt = np.array([[np.interp(np.log(IF.FINE), np.log(st["k"]), st[f"R_{tr}"][s, i]) for i in (0, 1)]
                       for s in range(len(zs))])
        tout = out["truths"].setdefault(tr, {})
        for config, kg in runs:
            key = config if kg is None else f"{config}|{kg}"
            m = mask(config, kg)
            rigid, _ = IF.rigid_fit(base, m)
            A = rigid.shift(Rmap)
            row = dict(npts=dict(xi=int(m[:NXI].sum()), gt=int(m[NXI:NXI + NGT].sum()),
                                 w=int(m[NXI + NGT:].sum())),
                       sigma=rigid.sig.tolist(), fom=IF.fom_pair(rigid), models={})
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            nla_sol = (3.5, 0.68)
            for mdl in MODELS:
                tf = time.time()
                names = B.IAP[mdl]
                lo = np.array([V.BOUNDS[n][0] for n in names])
                hi = np.array([V.BOUNDS[n][1] for n in names])
                best = None
                for s0 in V.starts(mdl, nla_sol):
                    r = least_squares(lambda ia: Lw @ (d[m] - ia_vector(cl, mdl, ia)[m]), s0,
                                      bounds=(lo, hi), x_scale="jac", max_nfev=600)
                    cc = float(r.fun @ r.fun)
                    if best is None or cc < best[0]:
                        best = (cc, r.x)
                chi2, ia = best
                if mdl == "NLA":
                    nla_sol = (ia[0], ia[1])
                bp = dict(zip(names, ia.tolist()))
                at_bound = [n for n, v, l, h in zip(names, ia, lo, hi)
                            if (v - l) < 1e-3*(h - l) or (h - v) < 1e-3*(h - l)]
                ck = {X: np.zeros(base.nb) for X in IF.XS}
                for s in range(len(zs)):
                    Rm = ME.model_ratio(tm, float(zs[s]), IF.FINE, bp, mdl)
                    for i, X in enumerate(IF.XS):
                        # an analysis model wrong by Delta moves the fit as the data moving by -Delta
                        ck[X][base.iknob[X][s]] = -(Rm[i]/Rt[s, i] - 1)
                parts = {X: rigid.shift(v) for X, v in ck.items()}
                mu = parts["GI"] + parts["II"]
                dp = mu*rigid.sig
                fob = float(np.sqrt(dp @ np.linalg.solve(rigid.C2, dp)))
                rooms = {f"{lz:g}": IF.rms_budget(A, Ks[lz], mu)[0] for lz in ROOM["lzs"]}
                row["models"][mdl] = dict(best=bp, chi2=chi2, at_bound=at_bound, mu=mu.tolist(),
                                          mu_GI=parts["GI"].tolist(), mu_II=parts["II"].tolist(),
                                          fob=fob, room=rooms)
                print(f"[{time.time() - t0:.0f}s] {tr} {key:12s} {mdl:4s} ({time.time() - tf:.0f}s): "
                      f"chi2 {chi2:.2f}/{int(m.sum())}, mu ({mu[0]:+.3f}, {mu[1]:+.3f}) "
                      f"[GI {parts['GI'][1]:+.3f}, II {parts['II'][1]:+.3f} in S8], FoB {fob:.3f}, "
                      f"room {rooms['inf']:.4f}, FoM {row['fom']:.0f}; " +
                      ", ".join(f"{n} {v:+.2f}" for n, v in bp.items()) +
                      (f"  AT BOUND {at_bound}" if at_bound else ""), flush=True)
            tout[key] = row
            json.dump(out, open(os.path.join(HERE, "scale_cut_scan_3x2.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
