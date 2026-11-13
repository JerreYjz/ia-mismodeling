#!/usr/bin/env python3
"""Roman FoB table for the overnight report: NLA / TATT (native, same
pipeline), F21, F21 red/blue, red NLA + blue TT (+ b_TA) + satellites."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "report")
rows = []
J = []
for f, ren in (("halo_roman_example1.json", {"halo": "F21"}), ("halo_f21x_roman_example1_rb.json", {"halo_rb": "F21_rb"}),
               ("halo_hyb_roman_example1.json", {"NLA_nat": "NLA", "TATT_nat": "TATT", "hyb0": "hyb0", "hyb": "hyb"})):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        for r in json.load(open(p)):
            if r["model"] in ren:
                J.append(dict(r, m=ren[r["model"]]))
COLS = [("NLA", "NLA"), ("TATT", "TATT"), ("F21", "F21"), ("F21_rb", "F21 red/blue"), ("hyb0", "red NLA + blue TT + sat."),
        ("hyb", r"+ $b_{\rm TA}$")]
MASKS = [f"K0/D {k:g}" for k in (0.25, 0.3, 0.35, 0.4, 0.5)] + [f"const {k:g}" for k in (0.3, 0.4)]


def get(m, mask, case):
    r = [x for x in J if x["m"] == m and x["mask"] == mask and x["case"] == case]
    return r[0] if r else None


lines = ["\\begin{tabular}{l " + "c"*len(COLS) + "}\\toprule", "mask & " + " & ".join(c[1] for c in COLS) + "\\\\ \\midrule"]
for mask in MASKS:
    cells = []
    for m, _ in COLS:
        a, b = get(m, mask, "A"), get(m, mask, "B")
        fa = f"{a['FoB']:.2f}" if a else "n/a"
        fb = f"{b['FoB']:.2f}" if b else "n/a"
        cells.append(f"{fa} / {fb}")
    lab = mask.replace("K0/D", "$K_0/D$,").replace("const", r"$k_{\rm UV}$ =")
    lines.append(f"{lab} & " + " & ".join(cells) + "\\\\")
lines.append("\\bottomrule\\end{tabular}")
open(os.path.join(REP, "tab_on_roman.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
for m in ("hyb", "hyb0"):
    for x in J:
        if x["m"] == m:
            print(m, x["mask"], x["case"], "a1h", x["ia"][-2], "eta1h", x["ia"][-1], "FoB", round(x["FoB"], 3))
