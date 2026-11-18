#!/usr/bin/env python3
"""Stage 2 of the v9 IA error budget: every budget from the base (numpy only).

Inputs: ia_base_<truth>.npz and ia_base_common.npz (ia_base.py), and, for
the comparison with v8, ../v8/correlated_budget.json.
Output: ia_budget.json and ia_budget_log.txt.

For each truth (S1, prod) and cut (none, k_UV = 0.3/D(z)) the rigid fit
(cosmology + template A, eta) is built once, F = M^T G_K M + priors, and its
response A to a residual delta(X, s, i) is one matrix product. Then:

 1. global     sigma_max (eta criterion) and sigma_95 (box criterion of v8)
               for a random error over every k and z, for X = GI, II, both
               (rho = 0: independent, 1: the same fractional error), kernels
               exp (the bias-prior notes) and sqexp (v8), ell_k in ln k and
               L_z in z (inf: the same error at every z).
 2. band       the error confined around k_c (Gaussian window of width 0.3
               in ln k, the one-range bump): sigma_max(k_c), sigma_95(k_c).
 3. map        confined around (k_c, z_c) (z window: Gaussian of width 0.25):
               sigma_max(k_c, z_c).
 4. drift      the NLA and TATT errors as known drifts (each fitted with the
               cut of the panel): their shifts, eta and coverage of the rigid
               fit, alone and with a random error on top, and the room left,
               sigma_max = sqrt(0.3^2 - mu^2)/rms.
 5. marginal   the fit that carries the residual with prior sigma_fit: the
               price (sigma(Omega_m), sigma(S8), FoM against the rigid fit)
               and eta(sigma_true, sigma_fit), without and with the NLA drift;
               the smallest sigma_fit that passes each sigma_true.
 6. v8         sqexp, L_z = inf, no window, X alone: sigma_95 against v8's
               eps_95 (correlated_budget.json), which must agree to its
               Monte Carlo noise (4e5 draws).

    python ia_budget.py [--quick]      (--quick: fewer kernels, for a test)
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
CUTS = ("none", "K03")
CUT_V8 = {"none": "none", "K03": "K0/D 0.3"}
SETS = {"GI": (("GI",), 0.0), "II": (("II",), 0.0), "both": (("GI", "II"), 0.0),
        "both_rho1": (("GI", "II"), 1.0)}
KINDS = ("exp", "sqexp")
ELLS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)
LZS = (0.1, 0.3, 1.0, np.inf)
BAND = dict(ells=(0.3, 1.0, 3.0), lzs=(0.3, np.inf), kind="exp", wk=0.3)
MAP = dict(ell=1.0, lz=0.3, kind="exp", wk=0.3, wz=0.25, sets=("GI", "II"))
DRIFT = dict(models=("NLA", "TATT"), ell=1.0, lz=0.3, kind="exp", sigmas=(0.0, 0.01, 0.02, 0.05))
MARG = dict(ell=1.0, lz=0.3, kind="exp", set="both",
            sigma_fit=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
            sigma_true=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2))


def lz_key(lz):
    return "inf" if not np.isfinite(lz) else f"{lz:g}"


def main():
    quick = "--quick" in sys.argv
    kinds, ells, lzs = (("exp",), (0.3, 1.0), (0.3, np.inf)) if quick else (KINDS, ELLS, LZS)
    t0 = time.time()
    logf = open(os.path.join(HERE, "ia_budget_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    common = os.path.join(HERE, "ia_base_common.npz")
    v8 = json.load(open(os.path.join(HERE, "..", "v8", "correlated_budget.json")))
    out = dict(about=__doc__, tol=IF.TOL, eta_max=IF.ETA_MAX, nodes=IF.NODES.tolist(),
               sets={k: [list(v[0]), v[1]] for k, v in SETS.items()}, kinds=list(kinds),
               ells=list(ells), lzs=[lz_key(z) for z in lzs], band=dict(BAND, lzs=[lz_key(z) for z in BAND["lzs"]]),
               map=MAP, drift=DRIFT, marginal=MARG, truths={})
    for tr in TRUTHS:
        path = os.path.join(HERE, f"ia_base_{tr}.npz")
        if not os.path.exists(path):
            log(f"no {path}: run ia_base.py first")
            continue
        base = IF.Base(path, common)
        zs = base.zs
        out["zs"] = zs.tolist()
        log(f"[{time.time() - t0:.0f}s] {tr}: base {base.B.shape}; checks {base_checks(path)}")
        rmaps = {k: base.knob_map(v[0]) for k, v in SETS.items() if k != "both_rho1"}
        rmaps["both_rho1"] = rmaps["both"]
        tout = {}
        for cut in CUTS:
            rigid, G = IF.rigid_fit(base, base.masks[cut])
            A = {k: rigid.shift(r) for k, r in rmaps.items()}         # (2, n_res) in sigma
            cres = dict(sigma=rigid.sig.tolist(), fom=IF.fom_pair(rigid), dropped=rigid.dropped)
            log(f"[{time.time() - t0:.0f}s] {tr} {cut}: sigma(Om, S8) = {rigid.sig[0]:.4f}, "
                f"{rigid.sig[1]:.4f}")

            # 1. global
            glob = []
            for sk, (Xs, rho) in SETS.items():
                for kind in kinds:
                    for ell in ells:
                        for lz in lzs:
                            K = IF.residual_cov(IF.NODES, zs, ell, lz, kind, Xs, rho)
                            smax, lim, rms = IF.rms_budget(A[sk], K)
                            s95 = IF.box_budget(A[sk], K)
                            glob.append(dict(set=sk, kind=kind, ell=ell, lz=lz_key(lz), sigma_max=smax,
                                             limit=lim, rms_per_unit=rms, sigma_95=s95))
            cres["global"] = glob
            for row in glob:
                if row["kind"] == "exp" and row["ell"] in (0.3, 1.0, 3.0):
                    log(f"  global {row['set']:9s} exp ell {row['ell']:g} Lz {row['lz']:>3s}: "
                        f"sigma_max {row['sigma_max']:.4f} ({row['limit']}), sigma_95 {row['sigma_95']:.4f}")

            # 2. band in k_c
            band = {}
            for sk in ("GI", "II", "both"):
                Xs, rho = SETS[sk]
                for ell in BAND["ells"]:
                    for lz in BAND["lzs"]:
                        K = IF.residual_cov(IF.NODES, zs, ell, lz, BAND["kind"], Xs, rho)
                        smax, s95 = [], []
                        for kc in IF.NODES:
                            w = IF.window(IF.NODES, zs, Xs, kc=kc, wk=BAND["wk"])
                            Aw = A[sk]*w[None, :]
                            smax.append(IF.rms_budget(Aw, K)[0])
                            s95.append(IF.box_budget(Aw, K))
                        band[f"{sk}|{ell:g}|{lz_key(lz)}"] = dict(sigma_max=smax, sigma_95=s95)
                        log(f"  band {sk:4s} ell {ell:g} Lz {lz_key(lz):>3s}: sigma_max " +
                            " ".join(f"{v:.3f}" for v in smax[::3]))
            cres["band"] = band

            # 3. map in (k_c, z_c)
            mp = {}
            for sk in MAP["sets"]:
                Xs, rho = SETS[sk]
                K = IF.residual_cov(IF.NODES, zs, MAP["ell"], MAP["lz"], MAP["kind"], Xs, rho)
                grid = np.zeros((len(zs), len(IF.NODES)))
                for a, zc in enumerate(zs):
                    for b, kc in enumerate(IF.NODES):
                        w = IF.window(IF.NODES, zs, Xs, kc=kc, zc=zc, wk=MAP["wk"], wz=MAP["wz"])
                        grid[a, b] = IF.rms_budget(A[sk]*w[None, :], K)[0]
                mp[sk] = grid.tolist()
                log(f"  map {sk}: sigma_max min {grid.min():.4f}, median {np.median(grid):.4f}")
            cres["map"] = mp

            # 4. known drifts (NLA, TATT)
            Xs, rho = SETS["both"]
            K = IF.residual_cov(IF.NODES, zs, DRIFT["ell"], DRIFT["lz"], DRIFT["kind"], Xs, rho)
            dr = {}
            for mdl in DRIFT["models"]:
                mu = rigid.shift(base.drift_knobs(f"{cut}|{mdl}"))
                room, lim, _ = IF.rms_budget(A["both"], K, mu)
                rows = []
                for s in DRIFT["sigmas"]:
                    sc = IF.scores(rigid, A["both"], s**2*K, mu)
                    rows.append(dict(sigma_true=s, **sc))
                dr[mdl] = dict(shift_sigma=mu.tolist(), room_sigma_max=room, room_limit=lim, scores=rows)
                log(f"  drift {mdl}: shift (Om, S8) = ({mu[0]:+.3f}, {mu[1]:+.3f}) sigma; eta "
                    f"{rows[0]['omegam']['eta']:.3f}, {rows[0]['S8']['eta']:.3f}; room {room:.4f}")
            cres["drift"] = dr

            # 5. the marginalised fit
            Xs, rho = SETS[MARG["set"]]
            Rm = rmaps[MARG["set"]]
            K = IF.residual_cov(IF.NODES, zs, MARG["ell"], MARG["lz"], MARG["kind"], Xs, rho)
            mu_nla = None
            marg = []
            for sf in MARG["sigma_fit"]:
                fit = rigid if sf == 0 else IF.marginalised_fit(base, G, Rm, sf**2*K)
                Af = fit.shift(Rm)
                mu_n = fit.shift(base.drift_knobs(f"{cut}|NLA"))
                row = dict(sigma_fit=sf, sigma=fit.sig.tolist(), sigma_ratio=(fit.sig/rigid.sig).tolist(),
                           fom_ratio=IF.fom_pair(fit)/IF.fom_pair(rigid), dropped=fit.dropped,
                           nla_shift=mu_n.tolist(), eta={}, eta_nla={})
                for st in MARG["sigma_true"]:
                    row["eta"][f"{st:g}"] = IF.scores(fit, Af, st**2*K)
                    row["eta_nla"][f"{st:g}"] = IF.scores(fit, Af, st**2*K, mu_n)
                marg.append(row)
                log(f"  marginal sigma_fit {sf:g}: sigma ratio (Om, S8) = ({row['sigma_ratio'][0]:.3f}, "
                    f"{row['sigma_ratio'][1]:.3f}), FoM ratio {row['fom_ratio']:.3f}, NLA shift "
                    f"({mu_n[0]:+.3f}, {mu_n[1]:+.3f})")
            need = {}
            for st in MARG["sigma_true"]:
                ok = [r["sigma_fit"] for r in marg
                      if max(r["eta"][f"{st:g}"][p]["eta"] for p in IF.SCORED) <= IF.ETA_MAX]
                need[f"{st:g}"] = min(ok) if ok else None
            cres["marginal"] = dict(rows=marg, smallest_passing_sigma_fit=need)
            log(f"  marginal: smallest passing sigma_fit per sigma_true {need}")

            # 6. against v8
            comp = []
            if cut in CUT_V8 and tr in v8["truths"]:
                bud = v8["truths"][tr][CUT_V8[cut]]["budget"]
                for X in ("GI", "II"):
                    for row in bud[X]:
                        K = IF.residual_cov(IF.NODES, zs, row["ell"], np.inf, "sqexp", (X,), 0.0)
                        s95 = IF.box_budget(A[X], K)
                        comp.append(dict(X=X, ell=row["ell"], v8_eps95=row["eps"]["0.95"], v9_sigma95=s95,
                                         ratio=s95/row["eps"]["0.95"]))
                r = np.array([c["ratio"] for c in comp])
                log(f"  v8 check: sigma_95 / v8 eps_95 = {r.min():.4f} to {r.max():.4f} "
                    f"({len(r)} cases)")
            cres["v8_check"] = comp
            tout[cut] = cres
        out["truths"][tr] = tout
        json.dump(out, open(os.path.join(HERE, "ia_budget.json"), "w"), indent=1, default=float)
    log(f"done ({time.time() - t0:.0f}s)")


def base_checks(path):
    d = np.load(path, allow_pickle=False)
    return json.loads(str(d["checks"]))


if __name__ == "__main__":
    main()
