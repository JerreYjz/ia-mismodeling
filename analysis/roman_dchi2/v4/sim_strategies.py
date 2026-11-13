#!/usr/bin/env python3
"""Simulation-level scale-cut strategies: definitions, calibration, scatter.

Production truth (measured FLAMINGO spectra, N_s fixed to the B plateau, three
probes, 0.05 <= k <= k_max), fitter of simfit.py (reproduces v3/v3.npz exactly).

1. GoF (p-rule): last k_max before two consecutive p < 0.05.
2. Parameter stability (the A1-stability idea), three versions:
     naive-ref : |a1(k_max) - a1(k_ref)| > sigma_a1(k_ref), k_ref = 0.09
     naive-seq : |a1(k_j) - a1(k_{j-1})| > sigma_a1(k_{j-1})
     nested    : Delta = theta_j - theta_{j-1}, V_Delta = V_{j-1} - V_j (the
                 covariance of the difference of nested efficient fits),
                 Delta^T V_Delta^+ Delta > chi2_0.95(n_par)  [all fitted
                 parameters; for NLA the a1 test at 95 per cent]
   each with the two-consecutive-failure rule.
3. Simulation volume: the noise-free (Asimov) version of the p-rule when the
   simulation error bars shrink by sqrt(f) (f = volume factor):
     chi2_exp(k_max) = nu + f chi2_T(k_max),   p_A = P(chi2_nu > chi2_exp),
   chi2_T = min over parameters of the fit to the smoothed truth T.
4. Simulation-noise scatter of every cut: N_REAL realisations y = T + sigma N(0,1).
5. Calibration (false-cut rate): realisations of a truth that IS the model
   (NLA: NLA fitted to T at k <= 0.16; TATT: TATT fitted to T at k <= 0.29),
   y = M + sigma N; fraction of realisations in which each rule cuts at
   k_max < 3.04, and where.
Outputs: sim_strategies.json, sim_cuts.csv, fig_sim_volume, fig_sim_noise,
fig_sim_stability, fig_sim_calib (220 dpi PNG + PDF).
    python sim_strategies.py [N_REAL]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import chi2 as chi2_dist, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simfit as S  # noqa: E402

N_REAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
RULES = ("GoF", "naive-ref", "naive-seq", "nested")
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
FGRID = np.geomspace(1/64, 64, 49)
CZ = np.where(S.core)[0]
NK = len(S.kg)


def scan(s, data, mdl):
    """Fits at every k_max: chi2, dof, x, cov."""
    out = [S.fit(s, data, mdl, k) for k in S.kg]
    return out


def cuts_from_scan(sc, mdl):
    """All rule cuts from one k_max scan."""
    c2 = np.array([o["chi2"] for o in sc])
    dof = np.array([o["dof"] for o in sc])
    p = chi2_dist.sf(c2, dof)
    a1 = np.array([o["x"][0] for o in sc])
    sa1 = np.array([np.sqrt(o["cov"][0, 0]) for o in sc])
    ok = {"GoF": p >= 0.05}
    ok["naive-ref"] = np.abs(a1 - a1[0]) <= sa1[0]
    seq = np.ones(NK, bool)
    for j in range(1, NK):
        seq[j] = abs(a1[j] - a1[j - 1]) <= sa1[j - 1]
    ok["naive-seq"] = seq
    nest = np.ones(NK, bool)
    npar = len(S.FREE[mdl])
    thr = chi2_dist.isf(0.05, npar)
    for j in range(1, NK):
        xa = sc[j - 1]["x"][:npar] if mdl == "TATT" else sc[j - 1]["x"][:1]
        xb = sc[j]["x"][:npar] if mdl == "TATT" else sc[j]["x"][:1]
        Vd = sc[j - 1]["cov"] - sc[j]["cov"]
        w, U = np.linalg.eigh(0.5*(Vd + Vd.T))
        keep = w > 1e-10*max(w.max(), 1e-30)
        dlt = U.T @ (xb - xa)
        stat = float(np.sum(dlt[keep]**2/w[keep]))
        nest[j] = stat <= chi2_dist.isf(0.05, int(keep.sum())) if keep.any() else True
    ok["nested"] = nest
    return {r: S.cut_rule(ok[r]) for r in RULES}, p


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261109)
    res = dict(zs=S.zs.tolist(), kgrid=S.kg.tolist(), core=S.core.tolist(), N_REAL=N_REAL)

    # ---------- 1, 2: rules on the measured spectra ----------
    meas = {}
    scans = {}
    for mdl in ("NLA", "TATT"):
        meas[mdl] = {r: [] for r in RULES}
        scans[mdl] = []
        for s in range(S.nsn):
            sc = scan(s, S.y[s], mdl)
            scans[mdl].append(sc)
            c, p = cuts_from_scan(sc, mdl)
            for r in RULES:
                meas[mdl][r].append(c[r])
        chk = np.allclose(np.nan_to_num(meas[mdl]["GoF"]), np.nan_to_num(S.v3[f"cut_{mdl}_GoF"]))
        print(f"{mdl}: GoF cuts reproduce v3: {chk}", flush=True)
        assert chk
    res["measured"] = meas
    res["a1_scan"] = {mdl: [[float(o["x"][0]) for o in scans[mdl][s]] for s in range(S.nsn)] for mdl in scans}
    res["a1_err"] = {mdl: [[float(np.sqrt(o["cov"][0, 0])) for o in scans[mdl][s]] for s in range(S.nsn)] for mdl in scans}

    # ---------- 3: simulation volume (Asimov) ----------
    vol = {}
    chi2T = {}
    for mdl in ("NLA", "TATT"):
        cT = np.zeros((S.nsn, NK))
        dof = np.zeros(NK)
        for s in range(S.nsn):
            for j, k in enumerate(S.kg):
                o = S.fit(s, S.T[s], mdl, k)
                cT[s, j], dof[j] = o["chi2"], o["dof"]
        chi2T[mdl] = cT
        cut_f = np.zeros((S.nsn, len(FGRID)))
        comb_f = np.zeros(len(FGRID))
        for i, fv in enumerate(FGRID):
            for s in range(S.nsn):
                pA = chi2_dist.sf(dof + fv*cT[s], dof)
                cut_f[s, i] = S.cut_rule(pA >= 0.05)
            nuc = dof*len(CZ)
            pc = chi2_dist.sf(nuc + fv*cT[CZ].sum(0), nuc)
            comb_f[i] = S.cut_rule(pc >= 0.05)
        vol[mdl] = dict(cut=cut_f.tolist(), comb=comb_f.tolist(), chi2T=cT.tolist(), dof=dof.tolist())
    res["volume"] = dict(f=FGRID.tolist(), **vol)
    print(f"volume done ({time.time() - t0:.0f}s)", flush=True)

    # ---------- 4, 5: realisations ----------
    truths = {}
    for mdl, ka in (("NLA", 0.1615), ("TATT", 0.2908)):
        M = np.zeros_like(S.T)
        for s in range(S.nsn):
            o = S.fit(s, S.T[s], mdl, ka)
            M[s] = S.pred(s, *o["x"])
        truths[mdl] = M
    real = {}
    for tag, truth_of in (("noise", lambda mdl: S.T), ("calib", lambda mdl: truths[mdl])):
        real[tag] = {}
        for mdl in ("NLA", "TATT"):
            Tt = truth_of(mdl)
            arr = np.full((N_REAL, S.nsn, len(RULES)), np.nan)
            comb = np.full(N_REAL, np.nan)
            for r in range(N_REAL):
                pc = np.zeros(NK)
                c2c = np.zeros(NK)
                dofc = np.zeros(NK)
                for s in CZ:
                    yr = Tt[s] + S.err[s]*rng.standard_normal(S.err[s].shape)
                    sc = scan(s, yr, mdl)
                    c, p = cuts_from_scan(sc, mdl)
                    arr[r, s] = [c[q] for q in RULES]
                    c2c += np.array([o["chi2"] for o in sc])
                    dofc += np.array([o["dof"] for o in sc])
                comb[r] = S.cut_rule(chi2_dist.sf(c2c, dofc) >= 0.05)
                if r % 50 == 0:
                    print(f"  {tag} {mdl} realisation {r} ({time.time() - t0:.0f}s)", flush=True)
            real[tag][mdl] = dict(cuts=arr.tolist(), comb=comb.tolist())
    res["realisations"] = real
    json.dump(res, open(os.path.join(HERE, "sim_strategies.json"), "w"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
