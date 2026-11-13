#!/usr/bin/env python3
"""FLAMINGO IA spectra used everywhere in v2: data, errors and a noise-free
smoothed version ("truth"), per snapshot, on the fold-combined k grid.

Per snapshot (fit_tatt_flamingo.load_snapshot conversions):
    P_hE, P_EE, P_BB measured, with FLAMINGO's error bars (mode scatter);
    folds: nfold 1 for k < 1.2, 4 for 1.2 <= k < 9, 16 above (to 17.7 h/Mpc);
    b(k) = sqrt(P_hh / P_NL) (shot-noise subtracted, 3-point smoothed), the
    same deterministic bias the models use, P_hE = b P_dE;
    Ns: white shape noise of P_EE, P_BB = median P_BB of nfold 4 at 5-10 h/Mpc.
Truth: a weighted cubic smoothing spline in ln k of each ratio
    hE: P_hE / (b P_NL),  EE: (P_EE - Ns) / P_NL,  BB: (P_BB - Ns) / P_NL
over k >= KMIN, smoothing chosen so chi2(data - spline) = number of points
(the spline removes the FLAMINGO noise, not the k dependence). Below KMIN the
ratios are held at their KMIN value (linear-scale NLA-like behaviour).

In the P(k) chi2 the truth is compared in observed units:
    hE: b P_NL R_hE,  EE: P_NL R_EE + Ns,  BB: P_NL R_BB + Ns.
"""
import os
import sys

import numpy as np
from scipy.interpolate import UnivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402

FOLDS = [(1, 0.0, 1.2), (4, 1.2, 9.0), (16, 9.0, 1e9)]
KMIN = 0.05
PROBES = ("hE", "EE", "BB")


def load(ps, sn, model):
    ks, P, S = [], {p: [] for p in ("hh",) + PROBES}, {p: [] for p in PROBES}
    for nf, lo, hi in FOLDS:
        d = F.load_snapshot(ps, sn, nf)
        m = (d["k"] >= lo) & (d["k"] < hi)
        ks.append(d["k"][m])
        for p in P:
            P[p].append(d[p][0][m])
        for p in S:
            S[p].append(d[p][1][m])
    k = np.concatenate(ks)
    o = np.argsort(k)
    k = k[o]
    P = {p: np.concatenate(v)[o] for p, v in P.items()}
    S = {p: np.concatenate(v)[o] for p, v in S.items()}
    z = F.SNAPSHOTS[sn][0]
    PNL = model.Pnl(z, k)
    b = F.scale_bias({"hh": (P["hh"], None)}, PNL)
    d4 = F.load_snapshot(ps, sn, 4)
    sel = (d4["k"] > 5) & (d4["k"] < 10)
    Ns = float(np.median(d4["BB"][0][sel]))
    return dict(sn=sn, z=z, k=k, PNL=PNL, b=b, Ns=Ns, y=P, err=S)


def norm(s, p):
    """Observed spectrum = norm * R (+ Ns for EE, BB)."""
    return s["b"]*s["PNL"] if p == "hE" else s["PNL"]


def smooth_truth(s):
    k = s["k"]
    m = k >= KMIN
    lk = np.log(k)
    R, chi = {}, {}
    for p in PROBES:
        n = norm(s, p)
        y = (s["y"][p] - (s["Ns"] if p != "hE" else 0.0))/n
        e = s["err"][p]/n
        sp = UnivariateSpline(lk[m], y[m], w=1.0/e[m], k=3, s=float(m.sum()))
        r = sp(lk)
        r[~m] = sp(lk[m][0])
        R[p] = r
        chi[p] = float(np.sum(((y[m] - r[m])/e[m])**2))
    return R, chi


def truth_observed(s, R):
    return {p: norm(s, p)*R[p] + (s["Ns"] if p != "hE" else 0.0) for p in PROBES}


def roman_ratios(s, R):
    """R_X = P_X / P_NL for the Limber step (X = dE, EE, BB); white noise out."""
    return np.array([R["hE"], R["EE"], R["BB"]])
