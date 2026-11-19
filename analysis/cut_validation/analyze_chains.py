#!/usr/bin/env python3
"""Posterior bias of the cut-validation chains, exported for plotting.

For each chain root it reads the Cobaya samples (first 30% of each MPI chain
dropped), and computes, against the D3A truth (build_datavectors.json):
    sigma    the posterior standard deviations of (Omega_m, S8)
    mu       (posterior mean - truth)/sigma, per parameter
    fob      the 2D figure of bias sqrt(dp^T C^-1 dp) with C the posterior
             covariance of (Omega_m, S8)
    R-1      the last Gelman-Rubin value of the .progress file
    map      the same shift and FoB of the minimizer's maximum a posteriori
             (<root>.minimum.txt, when present), still in posterior sigmas
and, next to it, the Fisher prediction of the same shift from
v9/scale_cut_scan.json (rigid fit, Omega_m and A_s free only). It writes
plots/<name>/data.json: the 2D posterior densities (getdist), the 68/95%
levels, the S8 trace of every MPI chain and the numbers above.

Run from ~/cocoa/Cocoa after start_cocoa.sh:
    python <this> <name> <chain root> [<chain root> ...]
"""
import glob
import json
import os
import sys

import numpy as np
from getdist import loadMCSamples

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.expanduser("~/cocoa/Cocoa/projects/roman_real/chains/cutval")
PARS = ("omegam", "S8")
IAP = ("roman_A1_1", "roman_A1_2", "roman_A2_1", "roman_A2_2", "roman_BTA_1")
FISHER_KEY = {"NLA_on_flam": "NLA", "TATT_on_flam": "TATT"}


def main():
    name, roots = sys.argv[1], sys.argv[2:]
    b = json.load(open(os.path.join(HERE, "build_datavectors.json")))
    truth = b["truth"]
    scan = json.load(open(os.path.join(HERE, "..", "roman_dchi2", "v9", "scale_cut_scan.json")))
    srow = scan["truths"]["prod"]["0.2"]
    out = dict(about=f"{__doc__}\nProducer: analyze_chains.py {' '.join(sys.argv[1:])}",
               truth=[truth[p] for p in PARS], pars=list(PARS), chains={})
    for root in roots:
        path = os.path.join(CHAINS, root)
        s = loadMCSamples(path, settings={"ignore_rows": 0.3})
        mean = s.getMeans([s.index[p] for p in PARS])
        C = s.cov(list(PARS))
        sig = np.sqrt(np.diag(C))
        dp = mean - np.array(out["truth"])
        mu = dp/sig
        fob = float(np.sqrt(dp @ np.linalg.solve(C, dp)))
        dens = s.get2DDensityGridData("omegam", "S8", num_plot_contours=2)
        prog = None
        if os.path.exists(path + ".progress"):
            rows = [l.split() for l in open(path + ".progress") if not l.startswith("#")]
            prog = np.array([[float(r[0]), np.nan, float(r[2]), float(r[3])] for r in rows]) if rows else None
        trace = []
        for f in sorted(glob.glob(path + ".[0-9]*.txt")):
            hdr = open(f).readline().lstrip("#").split()
            c = np.loadtxt(f, ndmin=2)
            trace.append(c[:, hdr.index("S8")].tolist())
        ia = {p: [float(s.mean(p)), float(s.std(p))] for p in IAP if p in s.index and s.std(p) > 0}
        row = dict(n_samples=int(s.numrows), weight_sum=float(s.norm), mean=mean.tolist(),
                   sigma=sig.tolist(), mu=mu.tolist(), fob=fob, ia=ia,
                   rminus1=(float(prog[-1, 3]) if prog is not None else None),
                   acceptance=(float(prog[-1, 2]) if prog is not None else None),
                   density=dict(x=dens.x.tolist(), y=dens.y.tolist(), P=dens.P.tolist(),
                                levels=[float(v) for v in dens.contours]),
                   trace=trace)
        for k, m in FISHER_KEY.items():
            if k in root:
                f = srow["models"][m]
                row["fisher"] = dict(mu=f["mu"], fob=f["fob"], sigma=srow["sigma"],
                                     shift=(np.array(f["mu"])*np.array(srow["sigma"])).tolist())
        mfile = path + ".minimum.txt"
        if os.path.exists(mfile):
            # the MAP of cobaya-run --minimize: its shift has no projection effect;
            # quoted in units of the posterior sigma, as the mean
            hdr = open(mfile).readline().lstrip("#").split()
            v = np.loadtxt(mfile, ndmin=2)[0]
            pm = np.array([v[hdr.index(p)] for p in PARS])
            dm = pm - np.array(out["truth"])
            row["map"] = dict(point=pm.tolist(), mu=(dm/sig).tolist(),
                              fob=float(np.sqrt(dm @ np.linalg.solve(C, dm))),
                              chi2=float(sum(v[i] for i, h in enumerate(hdr) if h.startswith("chi2__"))),
                              ia={p: float(v[hdr.index(p)]) for p in IAP if p in hdr})
        out["chains"][root] = row
        if "map" in row:
            print(f"{root} MAP: mu ({row['map']['mu'][0]:+.3f}, {row['map']['mu'][1]:+.3f}), "
                  f"FoB {row['map']['fob']:.3f}, chi2 {row['map']['chi2']:.3f}")
        print(f"{root}: {row['n_samples']} rows, R-1 {row['rminus1']}, acc {row['acceptance']}; "
              f"sigma ({sig[0]:.4f}, {sig[1]:.4f}), mu ({mu[0]:+.3f}, {mu[1]:+.3f}), FoB {fob:.3f}"
              + (f"; Fisher mu ({row['fisher']['mu'][0]:+.3f}, {row['fisher']['mu'][1]:+.3f}), "
                 f"FoB {row['fisher']['fob']:.3f}" if "fisher" in row else "")
              + "; " + ", ".join(f"{p} {v[0]:+.3f}+-{v[1]:.3f}" for p, v in ia.items()))
    d = os.path.join(HERE, "plots", name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "data.json"), "w"))


if __name__ == "__main__":
    main()
