#!/usr/bin/env python3
"""Figure data of the galaxy-bias section of the IA note (ia_error_budget.tex).

Writes, under notes/ia_error_budget/plots/:
  gbias_bins/data.json     the piece of the galaxy-bias figure of bias from
                           one lens bin's gamma_t or w rows, that bin's cut
                           raised alone (all others at 0.1), linear and
                           one-loop bias (gbias_perbin.json, "single");
  gbias_configs/data.json  figure of bias and sigma(S8) of the named cut
                           configurations (gbias_configs.json), linear and
                           one-loop, on the stitched NLA reference.
Prints the numbers quoted in the note.

    python make_gbias_figures.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "notes", "ia_error_budget", "plots"))
REF = "3x2+cmb|NLA|0.2|S1|gt0.5"
SHOW = ["baseline", "w8_015", "hiz_only", "perbin", "greedy_S1", "uniform015"]   # A to F in the note
LABEL = {"baseline": "A", "w8_015": "B", "hiz_only": "C", "perbin": "D", "greedy_S1": "E", "uniform015": "F"}
ZBIN = [0.31, 0.55, 0.75, 0.95, 1.17, 1.45, 1.85, 2.69]      # mean lens z (example1.nz)


def main():
    pb = json.load(open(os.path.join(HERE, "gbias_perbin.json")))["runs"]
    cuts = pb[f"{REF}|xi0.2|flam"]["CUTS"]
    bins = {}
    for model, tag in (("linear", f"{REF}|xi0.2|flam"), ("oneloop", f"{REF}|xi0.2|flam|oneloop")):
        s = pb[tag]["single"]
        for probe in ("w", "gt"):
            for i in range(1, 9):
                v = [s[f"{probe}|{i}|{k:g}"]["piece"] for k in cuts]
                if any(x is None for x in v):
                    continue
                bins[f"{model}|{probe}|{i}"] = v
    about = ("Producer: analysis/cut_validation/galaxy_bias/make_gbias_figures.py from gbias_perbin.json "
             f"(reference {REF}: NLA, stitched truth, xi_+- at 0.2, CMB prior). Piece of the galaxy-bias "
             "figure of bias from the rows of one lens bin and one probe, that bin's cut raised alone, all "
             "other cuts at 0.1.")
    os.makedirs(os.path.join(PLOTS, "gbias_bins"), exist_ok=True)
    json.dump(dict(about=about, cuts=cuts, zbin=ZBIN, lines=bins),
              open(os.path.join(PLOTS, "gbias_bins", "data.json"), "w"), indent=1)
    cf = json.load(open(os.path.join(HERE, "gbias_configs.json")))
    pts = {}
    for model, tag in (("linear", REF), ("oneloop", f"{REF}|oneloop")):
        for name in SHOW:
            r = cf["runs"][tag][name]
            pts[f"{model}|{name}"] = dict(gb=r["gb"], total=r["total"], sigma_S8=r["sigma_S8"],
                                          max_piece=r["max_piece"])
    os.makedirs(os.path.join(PLOTS, "gbias_configs"), exist_ok=True)
    json.dump(dict(about="Producer: analysis/cut_validation/galaxy_bias/make_gbias_figures.py from "
                   f"gbias_configs.json (reference {REF}). Figure of bias of the galaxy bias alone (gb) and "
                   "of IA plus galaxy bias (total), and sigma(S8), per cut configuration.",
                   configs={n: cf["configs"][n] for n in SHOW}, label=LABEL, points=pts),
              open(os.path.join(PLOTS, "gbias_configs", "data.json"), "w"), indent=1)
    for k, v in pts.items():
        print(k, {q: round(x, 4) for q, x in v.items()})


if __name__ == "__main__":
    main()
