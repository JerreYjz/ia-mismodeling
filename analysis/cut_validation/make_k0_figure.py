#!/usr/bin/env python3
"""data.json of plots/cutval_k0scan from fisher_minimize.json.

3x2pt with the CMB prior, xi_+- cut K0 = 0.2 to 0.4 (gamma_t behind lenses at
0.3, w at 0.2), NLA and TATT on both truths: the 2D figure of bias of the MAP
(equal to the linear posterior-mean estimate within 0.03 with the prior),
sigma(S8), and TATT's A2 and b_TA at the MAP.

    python make_k0_figure.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
K0 = ["0.2", "0.25", "0.3", "0.35", "0.4"]


def main():
    out = dict(about=f"Producer: make_k0_figure.py from fisher_minimize.json.\n{__doc__}",
               K0=[float(k) for k in K0], threshold=0.3, truths=["prod", "S1"], models=["NLA", "TATT"], lines={})
    for tr in out["truths"]:
        for m in out["models"]:
            rows = [R.get(f"3x2+cmb|{m}|{k}|{tr}") for k in K0]
            out["lines"][f"{m}|{tr}"] = dict(
                fob=[r["flatprior"]["fob_map"] if r else None for r in rows],
                fob_mean=[r["flatprior"]["fob_mean_linear"] if r else None for r in rows],
                mu=[r["flatprior"]["mu_map"] if r else None for r in rows],
                sigma_S8=[r["flatprior"]["sigma"][1] if r else None for r in rows],
                chi2=[r["chi2_min"] if r else None for r in rows],
                npts=[r["npts"] if r else None for r in rows],
                A2=[r["best"].get("roman_A2_1") if r else None for r in rows],
                bTA=[r["best"].get("roman_BTA_1") if r else None for r in rows],
                at_bound=[r["at_bound"] if r else None for r in rows])
    json.dump(out, open(os.path.join(HERE, "plots", "cutval_k0scan", "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
