#!/usr/bin/env python3
"""Step 1: NLA and TATT fits to FLAMINGO vs k_max, per snapshot.

Spectra: P_hE, P_EE, P_BB with FLAMINGO's error bars (diagonal covariance
C_FL), KMIN = 0.05 <= k <= k_max. Free parameters per snapshot:
    NLA : a1, Ns            TATT : a1, a2, b_TA, Ns
(Ns: white shape noise shared by P_EE and P_BB; cosmolike conventions,
P_hE = b(k) P_dE, fit_tatt_flamingo.TATTModel). Each fit is done twice:
  raw   : to the measured spectra   -> chi2_raw, dof, p  (goodness of fit)
  truth : to the smoothed spectra   -> noise-free misfit
For the truth fit, Delta chi2_FL(k_max; K) = sum_k<=K [(model - truth)/sigma]^2
is stored for every evaluation scale K on the same grid (K = k_max is the
in-fit misfit; K > k_max measures extrapolation). The truth-fit spectra are
what step 2 propagates to Roman.

Also stored: A_FL(K) = Delta chi2_FL per eps^2 for a pure amplitude change
of the truth (P_dE -> (1+eps) P_dE, P_EE, P_BB -> (1+eps)^2), the P(k)
side of the amplitude exchange rate.

    python fit_models.py --ps-dir PS --out fits.npz
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flamingo_truth as FT  # noqa: E402
F = FT.F

FREE = {"NLA": ["a1", "Ns"], "TATT": ["a1", "a2", "bTA", "Ns"]}
LO = {"a1": -60, "a2": -100, "bTA": -20, "Ns": -50}
KGRID_LO, KGRID_HI = 0.11, 3.1


def predict(model, s, K, th):
    pr = model.predict(s["z"], s["k"], th["a1"], th["a2"], th["bTA"], 1.0, K=K,
                       PNL=s["PNL"], Ns=th["Ns"], bk=s["b"])
    return {p: pr[p] for p in FT.PROBES}


def fit(model, s, K, mdl, y, kmax, warm):
    k, err = s["k"], s["err"]
    m = (k >= FT.KMIN) & (k <= kmax*(1 + 1e-9))
    names = FREE[mdl]

    def th_of(x):
        th = dict(a1=0.0, a2=0.0, bTA=0.0, Ns=0.0)
        th.update(zip(names, x))
        return th

    def resid(x):
        pr = predict(model, s, K, th_of(x))
        return np.concatenate([((y[p] - pr[p])/err[p])[m] for p in FT.PROBES])

    a1g = F.SNAPSHOTS[s["sn"]][4]*F.C1RHOCRIT_REPORT/F.C1RHOCRIT
    starts = []
    if mdl == "TATT":
        for a2 in (-2.0, 0.0, 2.0):
            for b in (-1.0, 0.0, 1.0):
                starts.append([a1g, a2, b, s["Ns"]])
    else:
        starts.append([a1g, s["Ns"]])
    if warm is not None:
        starts.append(list(warm))
    lo = np.array([LO[n] for n in names], float)
    best = None
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo + 1e-6, -lo - 1e-6), bounds=(lo, -lo),
                          x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    c2 = float(best.fun @ best.fun)
    dof = best.fun.size - len(names)
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    return dict(x=best.x, th=th_of(best.x), chi2=c2, dof=dof,
                p=float(chi2_dist.sf(c2, dof)) if dof > 0 else np.nan,
                err=np.sqrt(np.abs(np.diag(cov))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="fits.npz")
    ap.add_argument("--halofit", default="takahashi")
    a = ap.parse_args()
    print("CAMB + FAST-PT ...", flush=True)
    model = F.TATTModel(halofit=a.halofit)
    sns = sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    out = {"sns": np.array(sns), "zs": np.array([F.SNAPSHOTS[s][0] for s in sns])}
    acc = {}

    def put(key, val):
        acc.setdefault(key, []).append(val)

    for sn in sns:
        s = FT.load(a.ps_dir, sn, model)
        k = s["k"]
        R, chi_s = FT.smooth_truth(s)
        T = FT.truth_observed(s, R)
        K = model.kernels(s["z"], k)
        grid = k[(k >= KGRID_LO) & (k <= KGRID_HI)]
        inK = np.array([(k >= FT.KMIN) & (k <= kk*(1 + 1e-9)) for kk in grid])  # [nK, nk]
        put("k", k)
        put("y", np.array([s["y"][p] for p in FT.PROBES]))
        put("err", np.array([s["err"][p] for p in FT.PROBES]))
        put("T", np.array([T[p] for p in FT.PROBES]))
        put("Rtrue", FT.roman_ratios(s, R))
        put("b", s["b"])
        put("PNL", s["PNL"])
        put("Ns_true", s["Ns"])
        put("chi2_smooth", [chi_s[p] for p in FT.PROBES])
        put("npts_smooth", int((k >= FT.KMIN).sum()))
        # amplitude exchange rate, P(k) side
        dl = {"hE": T["hE"], "EE": 2*(T["EE"] - s["Ns"]), "BB": 2*(T["BB"] - s["Ns"])}
        w2 = sum(((dl[p]/s["err"][p])**2) for p in FT.PROBES)
        put("AFL", inK.astype(float) @ w2)
        # raw-data noise of the truth: chi2 of the data about the smooth truth
        r2 = sum((((s["y"][p] - T[p])/s["err"][p])**2) for p in FT.PROBES)
        put("noiseFL", inK.astype(float) @ r2)
        for mdl in ("NLA", "TATT"):
            wr = wt = None
            rows = {key: [] for key in ("chi2_raw", "dof", "p_raw", "x_raw", "e_raw",
                                        "lam", "x_true", "dFL", "R")}
            for kmax in grid:
                fr = fit(model, s, K, mdl, s["y"], kmax, wr)
                ft = fit(model, s, K, mdl, T, kmax, wt)
                wr, wt = fr["x"], ft["x"]
                pr = predict(model, s, K, ft["th"])
                r2m = sum((((pr[p] - T[p])/s["err"][p])**2) for p in FT.PROBES)
                th0 = dict(ft["th"], Ns=0.0)
                p0 = model.predict(s["z"], k, th0["a1"], th0["a2"], th0["bTA"], 1.0, K=K,
                                   PNL=s["PNL"], Ns=0.0, bk=s["b"])
                rows["chi2_raw"].append(fr["chi2"])
                rows["dof"].append(fr["dof"])
                rows["p_raw"].append(fr["p"])
                rows["x_raw"].append(fr["x"])
                rows["e_raw"].append(fr["err"])
                rows["lam"].append(ft["chi2"])
                rows["x_true"].append(ft["x"])
                rows["dFL"].append(inK.astype(float) @ r2m)
                rows["R"].append(np.array([p0["dE"], p0["EE"], p0["BB"]])/s["PNL"])
            for key, v in rows.items():
                put(f"{mdl}_{key}", np.array(v))
        print(f"sn{sn} z={s['z']:.2f}: smooth chi2/N = "
              + ", ".join(f"{p} {c/max(1, int((k >= FT.KMIN).sum())):.2f}"
                          for p, c in zip(FT.PROBES, [chi_s[p] for p in FT.PROBES]))
              + f";  NLA lam(K=kmax) {np.round(np.array(acc['NLA_lam'][-1][:8]), 1)}", flush=True)
        out["kgrid"] = grid
    for key, v in acc.items():
        out[key] = np.array(v)
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
