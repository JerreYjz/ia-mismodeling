#!/usr/bin/env python3
"""data.json of the two minimizer-plus-Fisher figures, from fisher_minimize.json.

plots/cutval_fisher_bias: mu(Omega_m), mu(S8) of every model and cut, on
    both truths, for the MAP and for the linear estimate of the posterior
    mean (flat priors as Gaussians), in units of the flat-prior Fisher sigma.
plots/cutval_setups: the 2D figure of bias of NLA and TATT at both cuts, MAP
    and linear posterior mean, for cosmic shear and 3x2pt, each without and
    with the CMB prior (n_s, Omega_b h^2, H0), on both truths.
plots/cutval_nlaz: the per-bin amplitudes of NLAz at the mean redshift of
    each source bin (example1.nz) against the NLA power law fitted at the
    same cut, and the (n_s, H0) of every MAP.

    python make_fisher_figures.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
MODELS, CUTS, TRUTHS = ("NLA", "NLAz", "TATT"), ("0.2", "0.25"), ("prod", "S1")
ABOUT = f"Producer: make_fisher_figures.py from fisher_minimize.json.\n{__doc__}"


def main():
    bias = dict(about=ABOUT, models=list(MODELS), cuts=list(CUTS), truths=list(TRUTHS), threshold=0.3,
                rows={})
    for tr in TRUTHS:
        for m in MODELS:
            for K in CUTS:
                r = R[f"{m}|{K}|{tr}"]["flatprior"]
                bias["rows"][f"{m}|{K}|{tr}"] = dict(map=r["mu_map"], mean=r["mu_mean_linear"],
                                                    fob_map=r["fob_map"], fob_mean=r["fob_mean_linear"])
    json.dump(bias, open(os.path.join(HERE, "plots", "cutval_fisher_bias", "data.json"), "w"), indent=1)

    nz = np.loadtxt(os.path.expanduser("~/cocoa/Cocoa/projects/roman_real/data/example1.nz"))
    zbar = [float(np.trapz(nz[:, 0]*nz[:, i], nz[:, 0])/np.trapz(nz[:, i], nz[:, 0])) for i in range(1, 9)]
    nlaz = dict(about=ABOUT, zbar=zbar, pivot=1.62, cuts=list(CUTS), truths=list(TRUTHS), amps={}, powerlaw={},
                maps={}, truth=dict(ns=0.967, H0=68.1))
    for tr in TRUTHS:
        for K in CUTS:
            b = R[f"NLAz|{K}|{tr}"]["best"]
            nlaz["amps"][f"{K}|{tr}"] = [b[f"roman_A1_{i}"] for i in range(1, 9)]
            n = R[f"NLA|{K}|{tr}"]["best"]
            nlaz["powerlaw"][f"{K}|{tr}"] = [n["roman_A1_1"], n["roman_A1_2"]]
            for m in MODELS:
                bb = R[f"{m}|{K}|{tr}"]["best"]
                nlaz["maps"][f"{m}|{K}|{tr}"] = [bb["ns"], bb["H0"]]
    json.dump(nlaz, open(os.path.join(HERE, "plots", "cutval_nlaz", "data.json"), "w"), indent=1)
    print("zbar", np.round(zbar, 3))

    setups = [("cs", "cosmic shear"), ("cs+cmb", "cosmic shear + CMB prior"),
              ("3x2", "3x2pt"), ("3x2+cmb", "3x2pt + CMB prior")]
    out = dict(about=ABOUT, setups=[s for s, _ in setups], titles=[t for _, t in setups],
               models=["NLA", "TATT"], cuts=list(CUTS), truths=list(TRUTHS), threshold=0.3, rows={})
    for su, _ in setups:
        for tr in TRUTHS:
            for m in ("NLA", "TATT"):
                for K in CUTS:
                    key = f"{m}|{K}|{tr}" if su == "cs" else f"{su}|{m}|{K}|{tr}"
                    if key not in R:
                        continue
                    f = R[key]["flatprior"]
                    out["rows"][f"{su}|{m}|{K}|{tr}"] = dict(
                        fob_map=f["fob_map"], fob_mean=f["fob_mean_linear"], mu_map=f["mu_map"],
                        mu_mean=f["mu_mean_linear"], sigma=f["sigma"], chi2=R[key]["chi2_min"],
                        npts=R[key]["npts"], at_bound=R[key]["at_bound"],
                        starts=[r["chi2"] for r in R[key]["starts"]])
    json.dump(out, open(os.path.join(HERE, "plots", "cutval_setups", "data.json"), "w"), indent=1)
    print("setups rows", len(out["rows"]))


if __name__ == "__main__":
    main()
