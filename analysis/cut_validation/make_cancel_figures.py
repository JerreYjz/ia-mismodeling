#!/usr/bin/env python3
"""data.json of the bias-cancellation figures from bias_pieces.json.

Every number is a 2D figure of bias in (Omega_m, S8), every parameter free,
the CMB prior, the flat priors as Gaussians (as fisher_minimize.py). For each
probe (cs, 2x2, 3x2), model and truth, against the xi_+- cut K0:
    total     the minimizer's MAP figure of bias (fisher_minimize.json), and
              mu_map its signed shift (Omega_m, S8) in sigma
    linear    the linear figure of bias of the direct model error
    pieces    the split of bias_pieces.py "zshape": the amplitude
              (redshift-evolution) error of each snapshot applied at every k,
              "z|amplitude", and the scale dependence left over by spectrum
              and range of k, "GI|0.06|0.3" and so on; each piece's figure of
              bias (fob) and its signed shift (mu: Omega_m, S8 in sigma)
    largest   the largest piece's figure of bias, and largest_shape that of
              the scale-dependence pieces alone (the amplitude piece left out)
    remainder what the response base misses (snapshot interpolation)
    plots/cancel_scan     NLA and TATT on the stitched truth (S1)
    plots/cancel_signed   the same, signed shifts per piece
    plots/cancel_scan_lin as cancel_scan, on the linear-limit truth
    plots/hm_cancel       every model on the stitched and linear-limit truths
    plots/cut_probes      separate cuts: cosmic shear (stitched, E minus B,
                          linear limit) and 2x2pt (stitched, linear limit)

    python make_cancel_figures.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
K0 = ["0.2", "0.25", "0.3", "0.35", "0.4", "0.5", "0.6"]
PIECES = ["z|amplitude", "GI|0|0.06", "GI|0.06|0.3", "GI|0.3|inf", "II|0|0.06", "II|0.06|0.3", "II|0.3|inf"]
MODELS = ["NLA", "TATT", "NLAa", "TATTa", "NLAf", "TATTf"]


def lines(R, M, probes, models, truths):
    out = {}
    for p in probes:
        for m in models:
            for tr in truths:
                rows = [R.get(f"{p}+cmb|{m}|{k}|{tr}") for k in K0]
                maps = [M.get(f"{p}+cmb|{m}|{k}|{tr}") for k in K0]
                out[f"{p}|{m}|{tr}"] = dict(
                    total=[r["flatprior"]["fob_map"] if r else None for r in maps],
                    mu_map=[r["flatprior"]["mu_map"] if r else None for r in maps],
                    linear=[r["direct"]["fob"] if r else None for r in rows],
                    remainder=[r["remainder"]["fob"] if r else None for r in rows],
                    largest=[r["largest_zshape_fob"] if r else None for r in rows],
                    largest_shape=[max(v["fob"] for q, v in r["zshape"].items() if q != "z|amplitude")
                                   if r else None for r in rows],
                    chi2=[r["chi2_min"] if r else None for r in maps],
                    pieces={q: dict(fob=[r["zshape"][q]["fob"] if r else None for r in rows],
                                    mu=[r["zshape"][q]["mu"] if r else None for r in rows]) for q in PIECES})
    return out


def main():
    R = json.load(open(os.path.join(HERE, "bias_pieces.json")))["runs"]
    M = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
    about = f"Producer: make_cancel_figures.py from bias_pieces.json and fisher_minimize.json.\n{__doc__}"
    common = dict(about=about, K0=[float(k) for k in K0], threshold=0.3, pieces=PIECES)
    figs = {
        "cancel_scan": dict(common, probes=["cs", "3x2"], models=["NLA", "TATT"], truths=["S1"],
                            lines=lines(R, M, ["cs", "3x2"], ["NLA", "TATT"], ["S1"])),
        "hm_cancel": dict(common, probes=["cs", "3x2"], models=MODELS, truths=["S1", "lin"],
                          lines=lines(R, M, ["cs", "3x2"], MODELS, ["S1", "lin"])),
    }
    figs["cancel_signed"] = dict(figs["cancel_scan"])
    rows = [["cs", "S1"], ["cs", "EB"], ["cs", "lin"], ["2x2", "S1"], ["2x2", "lin"]]
    L = {}
    for p, tr in rows:
        L.update(lines(R, M, [p], MODELS, [tr]))
    figs["cut_probes"] = dict(common, rows=rows, models=MODELS, lines=L)
    figs["cancel_scan_lin"] = dict(common, probes=["cs", "3x2"], models=["NLA", "TATT"], truths=["lin"],
                                   lines=lines(R, M, ["cs", "3x2"], ["NLA", "TATT"], ["lin"]))
    for name, d in figs.items():
        os.makedirs(os.path.join(HERE, "plots", name), exist_ok=True)
        json.dump(d, open(os.path.join(HERE, "plots", name, "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
