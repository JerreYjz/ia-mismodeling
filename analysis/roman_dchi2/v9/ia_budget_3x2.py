#!/usr/bin/env python3
"""Stage 2 of the v9 IA error budget for 3x2pt: global budget and bands.

Inputs: ia_base_3x2_<truth>.npz and ia_base_3x2_common.npz (ia_base_3x2.py),
fits_3x2.json (fits_3x2.py, optional: the NLA and TATT drifts), and
ia_budget.json (the cosmic-shear v9 budget, for the regression check).
Output: ia_budget_3x2.json, ia_budget_3x2_log.txt.

The rigid fit is that of ia_budget.py with the 8 linear biases free (flat):
cosmology (Omega_m, A_s), template (A, eta, Gaussian SIG_IA), b_1..b_8. For
each truth and cut (cs|K: xi_+- alone; 3x2|K: xi_+-, gamma_t at K, w at 0.2;
3x2b|K: gamma_t of sources behind lenses only; K = none, 0.2, 0.3), all
with the Gaussian 3x2pt covariance:
 1. global  sigma_max and sigma_95 of a random error over every k and z
            (exp kernel, ell = 0.3, 1, 3, L_z = 0.3, inf), X = GI, II, both;
 2. band    the error confined around k_c (Gaussian window, width 0.3 in
            ln k): sigma_max(k_c), sigma_95(k_c), for GI, II, both;
 3. drift   (with fits_3x2.json) the NLA and TATT errors fitted with that cut
            as known drifts: shift mu, FoB, the room left on top.
Regression: cs|none and cs|0.3 with the example1 xi covariance in place of
the Gaussian one must reproduce ia_budget.json (none, K03) exactly.

    python ia_budget_3x2.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ia_fisher as IF                                             # noqa: E402

TRUTHS = ("S1", "prod")
CUTS = ("cs|none", "cs|0.2", "cs|0.3", "3x2|none", "3x2|0.2", "3x2|0.3",
        "3x2b|none", "3x2b|0.2", "3x2b|0.3")
SETS = {"GI": ("GI",), "II": ("II",), "both": ("GI", "II")}
GLOBAL = dict(kind="exp", ells=(0.3, 1.0, 3.0), lzs=(0.3, np.inf))
BAND = dict(kind="exp", ells=(0.3, 1.0, 3.0), lzs=(0.3, np.inf), wk=0.3)
ROOM = dict(ell=1.0, kind="exp", lzs=(0.3, np.inf))
NXI = 1080


def lz_key(lz):
    return "inf" if not np.isfinite(lz) else f"{lz:g}"


def budgets(base, mask, rmaps, zs, log, tag):
    rigid, _ = IF.rigid_fit(base, mask)
    A = {k: rigid.shift(r) for k, r in rmaps.items()}
    res = dict(npts=int(mask.sum()), sigma=rigid.sig.tolist(), fom=IF.fom_pair(rigid),
               dropped=rigid.dropped)
    glob = []
    for sk, Xs in SETS.items():
        for ell in GLOBAL["ells"]:
            for lz in GLOBAL["lzs"]:
                K = IF.residual_cov(IF.NODES, zs, ell, lz, GLOBAL["kind"], Xs, 0.0)
                smax, lim, rms = IF.rms_budget(A[sk], K)
                glob.append(dict(set=sk, ell=ell, lz=lz_key(lz), sigma_max=smax, limit=lim,
                                 sigma_95=IF.box_budget(A[sk], K)))
    res["global"] = glob
    band = {}
    for sk, Xs in SETS.items():
        for ell in BAND["ells"]:
            for lz in BAND["lzs"]:
                K = IF.residual_cov(IF.NODES, zs, ell, lz, BAND["kind"], Xs, 0.0)
                smax, s95 = [], []
                for kc in IF.NODES:
                    Aw = A[sk]*IF.window(IF.NODES, zs, Xs, kc=kc, wk=BAND["wk"])[None, :]
                    smax.append(IF.rms_budget(Aw, K)[0])
                    s95.append(IF.box_budget(Aw, K))
                band[f"{sk}|{ell:g}|{lz_key(lz)}"] = dict(sigma_max=smax, sigma_95=s95)
    res["band"] = band
    g1 = {(r["set"], r["ell"], r["lz"]): r for r in glob}
    log(f"  {tag:10s} n {res['npts']:4d}  sigma(Om, S8) {rigid.sig[0]:.4f} {rigid.sig[1]:.4f}  "
        f"FoM {res['fom']:7.0f}  global both ell 1: sigma_max {g1[('both', 1.0, '0.3')]['sigma_max']:.4f} "
        f"(Lz 0.3) {g1[('both', 1.0, 'inf')]['sigma_max']:.4f} (inf)  band GI min "
        f"{min(band['GI|1|inf']['sigma_max']):.3f}  II min {min(band['II|1|inf']['sigma_max']):.3f}")
    return rigid, A, res


def main():
    t0 = time.time()
    logf = open(os.path.join(HERE, "ia_budget_3x2_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    common = os.path.join(HERE, "ia_base_3x2_common.npz")
    fits_path = os.path.join(HERE, "fits_3x2.json")
    fits = json.load(open(fits_path)) if os.path.exists(fits_path) else None
    cs = json.load(open(os.path.join(HERE, "ia_budget.json")))
    out = dict(about=__doc__, tol=IF.TOL, nodes=IF.NODES.tolist(), cuts=list(CUTS), sets=SETS,
               global_=dict(GLOBAL, lzs=[lz_key(z) for z in GLOBAL["lzs"]]),
               band=dict(BAND, lzs=[lz_key(z) for z in BAND["lzs"]]),
               room=dict(ROOM, lzs=[lz_key(z) for z in ROOM["lzs"]]), truths={}, regression={})
    for tr in TRUTHS:
        base = IF.Base(os.path.join(HERE, f"ia_base_3x2_{tr}.npz"), common)
        zs = base.zs
        out["zs"] = zs.tolist()
        log(f"[{time.time() - t0:.0f}s] {tr}: base {base.B.shape}, npar {base.npar}")
        rmaps = {k: base.knob_map(Xs) for k, Xs in SETS.items()}

        # regression on the cosmic-shear v9 budget: same cut, example1 xi covariance
        cov_g = base.cov
        cov_x = cov_g.copy()
        cov_x[:NXI, :NXI] = np.load(os.path.join(HERE, "ia_base_common.npz"))["cov"]
        base.cov = cov_x
        reg = {}
        for cut, old in (("cs|none", "none"), ("cs|0.3", "K03")):
            _, _, r = budgets(base, base.masks[cut], rmaps, zs, log, f"{cut} ex1")
            ref = cs["truths"][tr][old]
            d_sig = float(np.max(np.abs(np.array(r["sigma"])/np.array(ref["sigma"]) - 1)))
            b_new = np.array(r["band"]["GI|1|inf"]["sigma_max"] + r["band"]["II|1|inf"]["sigma_max"])
            b_old = np.array(ref["band"]["GI|1|inf"]["sigma_max"] + ref["band"]["II|1|inf"]["sigma_max"])
            reg[cut] = dict(sigma=d_sig, band=float(np.max(np.abs(b_new/b_old - 1))))
        out["regression"][tr] = reg
        log(f"  regression vs ia_budget.json: " + ", ".join(f"{c} sigma {v['sigma']:.1e} band {v['band']:.1e}"
                                                         for c, v in reg.items()))
        base.cov = cov_g

        tout = {}
        for cut in CUTS:
            rigid, A, res = budgets(base, base.masks[cut], rmaps, zs, log, cut)
            if fits is not None and cut in fits["truths"].get(tr, {}):
                Kr = {lz: IF.residual_cov(IF.NODES, zs, ROOM["ell"], lz, ROOM["kind"], IF.XS, 0.0)
                      for lz in ROOM["lzs"]}
                Rmap = base.knob_map(IF.XS)
                Ab = rigid.shift(Rmap)
                res["models"] = {}
                for mdl, v in fits["truths"][tr][cut].items():
                    c = np.zeros(base.nb)
                    for X in IF.XS:
                        c[base.iknob[X].ravel()] = -np.array(v["err"][X]).ravel()
                    mu = rigid.shift(c)
                    dp = mu*rigid.sig
                    fob = float(np.sqrt(dp @ np.linalg.solve(rigid.C2, dp)))
                    room = {lz_key(lz): IF.rms_budget(Ab, Kr[lz], mu)[0] for lz in ROOM["lzs"]}
                    res["models"][mdl] = dict(mu=mu.tolist(), fob=fob, room=room)
                    log(f"    {mdl:4s}: mu ({mu[0]:+.3f}, {mu[1]:+.3f}), FoB {fob:.3f}, room "
                        f"{room['0.3']:.4f} / {room['inf']:.4f}")
            tout[cut] = res
        out["truths"][tr] = tout
        json.dump(out, open(os.path.join(HERE, "ia_budget_3x2.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
