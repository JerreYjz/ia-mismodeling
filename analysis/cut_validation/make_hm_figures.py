#!/usr/bin/env python3
"""data.json of plots/hm_k0scan and plots/hm_alpha from fisher_minimize.json.

The halo-model scan (run_hm_scan.sh, jobs_hm.txt): NLA, TATT and their
halo-model variants NLAa, TATTa (A1 terms on I_1 P_lin + alpha (P_NL -
I_1 P_lin), alpha free) against the cut K0 of xi_+-, cosmic shear and 3x2pt
(gamma_t behind lenses at 0.3, w at 0.2), both with the CMB prior, on the
production and stitched FLAMINGO truths. Every number is the MAP with the
flat priors as Gaussians of the same variance (the "flatprior" block of
fisher_minimize.py), as in plots/cutval_k0scan.

    hm_k0scan  the 2D figure of bias and chi2_min per model, probe, truth, K0
               (NLA, TATT and the alpha-free variants)
    hm_fixed   the same for NLA, TATT and NLAf, TATTf: alpha held at
               FLAMINGO's value per snapshot (ALPHA_FLAMINGO of
               fisher_minimize.py)
    hm_gt      3x2pt: the 2D figure of bias against the gamma_t cut (0.3 to
               0.6) at the xi_+- cuts K0 = 0.25 and 0.3, NLA, TATT, NLAf, TATTf
    hm_alpha   alpha at the MAP with its Fisher error (sigma_all), per model,
               probe, truth, K0, and the chi2 gain over the parent model

    python make_hm_figures.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
K0 = ["0.2", "0.25", "0.3", "0.35", "0.4", "0.5", "0.6"]
MODELS = ["NLA", "TATT", "NLAa", "TATTa"]
FIXED = ["NLA", "TATT", "NLAf", "TATTf"]
PROBES = ["cs", "3x2"]
TRUTHS = ["prod", "S1"]
PARENT = {"NLAa": "NLA", "TATTa": "TATT"}


def row(probe, m, k, tr):
    return R.get(f"{probe}+cmb|{m}|{k}|{tr}")


def main():
    about = f"Producer: make_hm_figures.py from fisher_minimize.json.\n{__doc__}"
    scan = dict(about=about, K0=[float(k) for k in K0], threshold=0.3, models=MODELS, probes=PROBES,
                truths=TRUTHS, lines={})
    fixed = dict(scan, models=FIXED, lines={})
    alpha = dict(about=about, K0=[float(k) for k in K0], models=["NLAa", "TATTa"], probes=PROBES,
                 truths=TRUTHS, physical=[0.0, 1.0], bounds=[-1.0, 2.0], lines={})
    for p in PROBES:
        for tr in TRUTHS:
            for m in MODELS + ["NLAf", "TATTf"]:
                rows = [row(p, m, k, tr) for k in K0]
                (scan if m in MODELS else fixed)["lines"][f"{p}|{m}|{tr}"] = dict(
                    fob=[r["flatprior"]["fob_map"] if r else None for r in rows],
                    fob_mean=[r["flatprior"]["fob_mean_linear"] if r else None for r in rows],
                    mu=[r["flatprior"]["mu_map"] if r else None for r in rows],
                    sigma=[r["flatprior"]["sigma"] if r else None for r in rows],
                    chi2=[r["chi2_min"] if r else None for r in rows],
                    npts=[r["npts"] if r else None for r in rows],
                    at_bound=[r["at_bound"] if r else None for r in rows])
                if m in PARENT:
                    par = [row(p, PARENT[m], k, tr) for k in K0]
                    alpha["lines"][f"{p}|{m}|{tr}"] = dict(
                        alpha=[r["best"]["hm_alpha"] if r else None for r in rows],
                        sigma=[r["sigma_all"]["hm_alpha"] if r else None for r in rows],
                        dchi2=[(q["chi2_min"] - r["chi2_min"]) if (r and q) else None for r, q in zip(rows, par)],
                        A1=[r["best"]["roman_A1_1"] if r else None for r in rows],
                        eta1=[r["best"]["roman_A1_2"] if r else None for r in rows],
                        A2=[r["best"].get("roman_A2_1") if r else None for r in rows],
                        bTA=[r["best"].get("roman_BTA_1") if r else None for r in rows])
    for m in ("NLA", "TATT"):
        for k, v in scan["lines"].items():
            if f"|{m}|" in k:
                fixed["lines"][k] = v
    GT = ["0.3", "0.4", "0.5", "0.6"]
    gtd = dict(about=about, K0=[float(g) for g in GT], threshold=0.3, models=FIXED, probes=["0.25", "0.3"],
               truths=TRUTHS, xlabel="gamma_t", lines={})
    for xi in gtd["probes"]:
        for tr in TRUTHS:
            for m in FIXED:
                rows = [R.get(f"3x2+cmb|{m}|{xi}|{tr}" + ("" if g == "0.3" else f"|gt{g}")) for g in GT]
                gtd["lines"][f"{xi}|{m}|{tr}"] = dict(
                    fob=[r["flatprior"]["fob_map"] if r else None for r in rows],
                    mu=[r["flatprior"]["mu_map"] if r else None for r in rows],
                    sigma=[r["flatprior"]["sigma"] if r else None for r in rows],
                    chi2=[r["chi2_min"] if r else None for r in rows],
                    npts=[r["npts"] if r else None for r in rows],
                    at_bound=[r["at_bound"] if r else None for r in rows])
    for name, d in (("hm_k0scan", scan), ("hm_alpha", alpha), ("hm_fixed", fixed), ("hm_gt", gtd)):
        os.makedirs(os.path.join(HERE, "plots", name), exist_ok=True)
        json.dump(d, open(os.path.join(HERE, "plots", name, "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
