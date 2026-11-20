#!/usr/bin/env python3
"""The freer redshift evolutions (notes/ia_zevo) against NLA and TATT.

For every (probe, model, truth, K0) of jobs_zevo.txt and the matching NLA and
TATT runs: the MAP figure of bias (flat-prior sigma), the redshift-amplitude
piece and the largest piece of the bias split, sigma(S8), the extra
parameters and any parameter at a bound (within 2% of its prior range,
recomputed from the best fit as make_combo_figure.at_bound; b_TA at 0
allowed), and the reliable cut of the note: the largest K0 at which the
total MAP figure of bias and every piece of the bias split are below 0.3,
there and at every smaller K0 with a physical fit (unphysical fits are
skipped). Writes zevo_summary.json and prints the tables.

    python zevo_summary.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "fisher_minimize.json")))["runs"]
P = json.load(open(os.path.join(HERE, "bias_pieces.json")))["runs"]
CUTS = {"cs": ["0.2", "0.3", "0.4", "0.5"], "3x2": ["0.2", "0.25", "0.3", "0.4", "0.5", "0.6"]}
BOUNDS = {"roman_A1_1": (-5, 5), "roman_A1_2": (-5, 5), "roman_A2_1": (-5, 5), "roman_A2_2": (-5, 5),
          "roman_BTA_1": (0, 2), "zevo_deta": (-5, 5), "zevo_d2": (-0.9, 3), "zevo_d3": (-0.9, 3),
          **{f"roman_B1_{i}": (0.4, 4.0) for i in range(1, 9)}}
TOL = 0.3


def at_bound(best, tol=0.02):
    out = [p for p, (lo, hi) in BOUNDS.items() if p in best and min(best[p] - lo, hi - best[p]) < tol*(hi - lo)]
    return [p for p in out if not (p == "roman_BTA_1" and best[p] < 1)]


def reliable(rows, ks):
    """The largest K0 passing, with every smaller physical K0 passing; None if the first fails."""
    cut = None
    for k, e in zip(ks, rows):
        if e is None or e["bound"]:
            continue
        if e["fob"] < TOL and e["largest"] is not None and e["largest"] < TOL:
            cut = k
        else:
            break
    return cut
FAMILY = {"NLA": ["NLA", "NLAbk", "NLAhz"], "TATT": ["TATT", "TATTbk", "TATThz"]}
EXTRA = ("zevo_deta", "zevo_d2", "zevo_d3")


def key(probe, m, k, tr):
    return f"{probe}+cmb|{m}|{k}|{tr}" + ("|gt0.5" if probe == "3x2" else "")


def entry(k):
    r, p = R.get(k), P.get(k)
    if r is None:
        return None
    bound = at_bound(r["best"])
    return dict(fob=r["flatprior"]["fob_map"], sigS8=r["sigma_flatprior"][1], bound=bound,
                extra={q: r["best"][q] for q in EXTRA if q in r["best"]},
                eta1=r["best"]["roman_A1_2"], A1=r["best"]["roman_A1_1"], chi2=r["chi2_min"],
                amp=p["zshape"]["z|amplitude"]["fob"] if p else None,
                largest=p["largest_zshape_fob"] if p else None)


out = {}
for probe, ks in CUTS.items():
    for fam, models in FAMILY.items():
        for tr in ("S1", "EB", "lin"):
            print(f"\n{probe} {fam} {tr}: FoB MAP / amplitude piece / sigma(S8) [extra params] per K0 {ks}")
            for m in models:
                row = [entry(key(probe, m, k, tr)) for k in ks]
                out[f"{probe}|{m}|{tr}"] = dict(K0=ks, rows=row, reliable=reliable(row, ks))
                cells = []
                for e in row:
                    if e is None:
                        cells.append("        --          ")
                        continue
                    ex = ",".join(f"{v:+.2f}" for v in e["extra"].values())
                    amp = f"{e['amp']:.2f}" if e["amp"] is not None else " -- "
                    cells.append(f"{e['fob']:.2f}/{amp}/{e['sigS8']:.4f}" + (f"[{ex}]" if ex else "")
                                 + ("!" if e["bound"] else ""))
                print(f"  {m:7s} " + "  ".join(cells))
json.dump(dict(about=__doc__, runs=out), open(os.path.join(HERE, "zevo_summary.json"), "w"), indent=1)
print("\n! = a parameter at a prior bound")
print("\nreliable cut (sigma(S8) there):")
for probe in CUTS:
    for fam, models in FAMILY.items():
        for m in models:
            cells = []
            for tr in ("S1", "EB", "lin"):
                o = out[f"{probe}|{m}|{tr}"]
                c = o["reliable"]
                s8 = o["rows"][o["K0"].index(c)]["sigS8"] if c else None
                cells.append(f"{tr} {c or 'none':>4} ({s8:.4f})" if c else f"{tr} none         ")
            print(f"  {probe:3s} {m:7s} " + "   ".join(cells))
