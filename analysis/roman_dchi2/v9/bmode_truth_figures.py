#!/usr/bin/env python3
"""Figure data of the B-mode truth check (bmode_truth_3x2.py, fisher_minimize.py).

bmode_truth_dv: the change of xi_+- when the close-pair II term N_s is kept
    with its B-mode partner (truth NB against S1), split into the E part
    (N_s back in P_EE) and the B part (P_BB in C_BB), in units of the
    diagonal error, for three auto pairs; EB - S1 for scale; the angles cut
    at K0 = 0.2.
bmode_truth_fob: the FoB of every model at the recommended 3x2pt cut
    (xi_+- K0 = 0.2, gamma_t 0.5 of sources behind lenses, w 0.2, CMB
    priors) and of NLA and TATT for cosmic shear alone at K0 = 0.35, on the
    truths S1, EB, NB and NBall.

    python bmode_truth_figures.py        (any Python with numpy)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
OUT = os.path.join(ROOT, "notes", "ia_error_budget", "plots")
CV = os.path.join(HERE, "..", "..", "cut_validation")
NXI = 1080
PAIRS = ((0, 0), (3, 3), (7, 7))
TRUTHS = ("S1", "EB", "NB", "NBall")
MODELS = ("NLA", "TATT", "NLAa", "TATTa", "NLAf", "TATTf")


def pair_index(i, j):
    return sum(8 - a for a in range(i)) + (j - i)


def main():
    c = np.load(os.path.join(HERE, "ia_base_3x2_common.npz"))
    sig = np.sqrt(np.diag(c["cov"]))[:NXI]
    keep = c["xi|0.2"]
    nb = np.load(os.path.join(HERE, "ia_base_3x2_NB.npz"))
    ref = {t: np.load(os.path.join(HERE, f"ia_base_3x2_{t}.npz"))["xi_ia"][:NXI] for t in ("S1", "EB")}
    e = nb["xi_E"][:NXI] - ref["S1"]
    b = nb["xi_B"][:NXI]
    tot = nb["xi_ia"][:NXI] - ref["S1"]
    eb = ref["EB"] - ref["S1"]
    edges = np.logspace(np.log10(2.5), np.log10(250.0), 16)
    theta = np.sqrt(edges[1:]*edges[:-1])
    panels = []
    for half, name in ((0, "xi_+"), (1, "xi_-")):
        for i, j in PAIRS:
            s = slice(half*540 + 15*pair_index(i, j), half*540 + 15*pair_index(i, j) + 15)
            panels.append(dict(xi=name, pair=[i + 1, j + 1], kept=keep[s].tolist(),
                               E=(e[s]/sig[s]).tolist(), B=(b[s]/sig[s]).tolist(),
                               total=(tot[s]/sig[s]).tolist(), EB_minus_S1=(eb[s]/sig[s]).tolist()))
    bt = json.load(open(os.path.join(HERE, "bmode_truth.json")))
    os.makedirs(os.path.join(OUT, "bmode_truth_dv"), exist_ok=True)
    json.dump(dict(about="Producer: analysis/roman_dchi2/v9/bmode_truth_figures.py from ia_base_3x2_{S1,EB,NB}.npz "
                   "(bmode_truth_3x2.py). Change of the IA part of xi_+- in units of the diagonal error of the "
                   "Gaussian 3x2pt covariance: E = N_s kept in P_EE (stitched as S1), B = the B-mode partner "
                   "P_BB, total = NB - S1, EB_minus_S1 for scale. kept: the angles of the xi_+- cut K0 = 0.2. "
                   "theta: bin centres [arcmin].",
                   theta=theta.tolist(), panels=panels, dchi2=bt["truths"]),
              open(os.path.join(OUT, "bmode_truth_dv", "data.json"), "w"), indent=1)

    runs = json.load(open(os.path.join(CV, "fisher_minimize.json")))["runs"]
    rows = []
    for probe, K0, gt, models in (("3x2+cmb", 0.2, "|gt0.5", MODELS), ("cs+cmb", 0.35, "", ("NLA", "TATT"))):
        for m in models:
            r = {}
            for t in TRUTHS:
                k = f"{probe}|{m}|{K0:g}|{t}{gt}"
                if k in runs:
                    x = runs[k]
                    r[t] = dict(fob=x["fob"], mu=x["mu"], chi2=x["chi2_min"], at_bound=x.get("at_bound"),
                                best={n: v for n, v in x["best"].items() if n.startswith(("roman_A", "roman_BTA", "hm"))})
            rows.append(dict(probe=probe, K0=K0, model=m, truths=r))
    os.makedirs(os.path.join(OUT, "bmode_truth_fob"), exist_ok=True)
    json.dump(dict(about="Producer: analysis/roman_dchi2/v9/bmode_truth_figures.py from cut_validation/"
                   "fisher_minimize.json. FoB (minimizer, every parameter free, CMB priors) of each IA model on "
                   "the truths S1, EB (N_s or P_BB subtracted, no B modes) and NB, NBall (N_s kept with its "
                   "B-mode partner; stitched as S1, or at every k). 3x2: xi_+- K0 = 0.2, gamma_t 0.5 of sources "
                   "behind lenses, w 0.2. cs: xi_+- alone, K0 = 0.35.", threshold=0.3, rows=rows),
              open(os.path.join(OUT, "bmode_truth_fob", "data.json"), "w"), indent=1)
    for r in rows:
        print(r["probe"], r["K0"], r["model"], {t: round(v["fob"], 3) for t, v in r["truths"].items()})


if __name__ == "__main__":
    main()
