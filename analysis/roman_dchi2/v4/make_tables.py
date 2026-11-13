#!/usr/bin/env python3
"""Table bodies for report/strategies.tex and the numbers quoted in its text.
Reads derived.json, sim_strategies.json, roman_theory.json, fisher_white.json,
v3/roman_results.csv, v3/mask_scan.json. Writes report/tab_calib.tex,
tab_cons.tex, tab_theory.tex, and prints the quoted numbers."""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "report")
V3 = os.path.join(HERE, "..", "v3")
D = json.load(open(os.path.join(HERE, "derived.json")))
sim = json.load(open(os.path.join(HERE, "sim_strategies.json")))
kg = np.array(sim["kgrid"])
zs = np.array(sim["zs"])
core = np.array(sim["core"])
NAMES = {"GoF": "GoF $p$-rule", "naive-ref": "$a_1$ stability, naive (vs 0.09)", "naive-seq": "$a_1$ stability, naive sequential",
         "nested": "stability, nested test"}


def steps(lo, hi):
    return np.log(np.asarray(hi)/np.asarray(lo))/np.log(kg[1]/kg[0])


# ---------------- calibration and noise ----------------
lines = [r"\begin{tabular}{l rrr rrr}\toprule",
         r" & \multicolumn{3}{c}{NLA} & \multicolumn{3}{c}{TATT}\\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"rule & false-cut rate & any core & 16 to 84\% & false-cut rate & any core & 16 to 84\%\\ \midrule"]
for rule in ("GoF", "naive-ref", "naive-seq", "nested"):
    row = [NAMES[rule]]
    for mdl in ("NLA", "TATT"):
        c = D["calib"][mdl][rule]
        n = D["noise"][mdl][rule]
        lo, hi = np.median(n["p16"]), np.median(n["p84"])
        fmt = lambda v: "$<$0.09" if v < 0.08 else f"{v:.2f}"
        row += [f"{100*c['false_rate_per_snapshot']:.1f}\\%", f"{100*c['false_rate_any_core']:.0f}\\%", f"{fmt(lo)} to {fmt(hi)}"]
    lines.append(" & ".join(row) + r"\\")
lines.append(r"\midrule")
row = ["GoF, core combined"]
for mdl in ("NLA", "TATT"):
    q = D["noise"][mdl]["comb"]
    fmt = lambda v: "$<$0.09" if v < 0.08 else f"{v:.2f}"
    row += [f"{100*D['calib'][mdl]['comb_false_rate']:.1f}\\%", "", f"{fmt(q[0])} to {fmt(q[2])}"]
lines.append(" & ".join(row) + r"\\")
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "tab_calib.tex"), "w").write("\n".join(lines))
print("\n".join(lines))
for mdl in ("NLA", "TATT"):
    for rule in ("GoF", "nested"):
        n = D["noise"][mdl][rule]
        print(mdl, rule, "median cut per core z:", np.round(n["p50"], 3), " 16%:", np.round(n["p16"], 3), " 84%:", np.round(n["p84"], 3),
              " measured:", np.round(n["measured"], 3))

# ---------------- consistency ----------------
lines = [r"\begin{tabular}{llrrrrr}\toprule",
         r"model & masks & $\delta S_8$ small & $\delta S_8$ large & change & noise of change & ratio\\ \midrule"]
for x in D["consistency"]:
    if x["case"] != "A":
        continue
    lines.append(f"{x['model']} & {x['small'].split()[-1]} $\\to$ {x['large'].split()[-1]} & {x['dS8_small']:+.4f} & {x['dS8_large']:+.4f} & "
                 f"{x['change']:+.4f} & {x['sigma_diff']:.4f} & {x['ratio']:+.2f}\\\\")
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "tab_cons.tex"), "w").write("\n".join(lines))

# ---------------- theory error ----------------
th = json.load(open(os.path.join(HERE, "roman_theory.json")))
lab = {"n1:0.3": "$K_0/D$ 0.3", "n1:0.4": "$K_0/D$ 0.4", "n1:0.5": "$K_0/D$ 0.5", "scan:flat_0.5": "constant 0.5"}
lines = [r"\begin{tabular}{llrrrrrr}\toprule",
         r" & & \multicolumn{3}{c}{statistical $C$} & \multicolumn{3}{c}{$C + C_{\rm th}$}\\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"model & mask & FoB & $\sigma(S_8)$ & $\chi^2$ & FoB & $\sigma(S_8)$ & $|\delta S_8|/\sigma_{\rm stat}$\\ \midrule"]
for mdl in ("NLA", "TATT"):
    for key in lab:
        a = [x for x in th if x["model"] == mdl and x["mask"] == key and x["cov"] == "stat"]
        b = [x for x in th if x["model"] == mdl and x["mask"] == key and x["cov"] == "stat+theory"]
        if not a or not b:
            continue
        a, b = a[0], b[0]
        lines.append(f"{mdl} & {lab[key]} & {a['FoB']:.2f} & {a['sS8']:.4f} & {a['chi2']:.2f} & {b['FoB']:.2f} & {b['sS8']:.4f} & {abs(b['dS8'])/a['sS8']:.2f}\\\\")
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "tab_theory.tex"), "w").write("\n".join(lines))
print("\n".join(lines))

# ---------------- B1 interpolation ----------------
rr = [r for r in csv.DictReader(open(os.path.join(V3, "roman_results.csv"))) if r["shape"] == "TATT_GoF"]
fac = np.array([float(r["factor"]) for r in rr])
k1 = np.array([float(r["kuv_z1"]) for r in rr])
for mdl in ("NLA", "TATT"):
    d = np.array([float(r[f"dchi2_{mdl}"]) for r in rr])
    for thr in (1.0, 0.5):
        i = np.where(d >= thr)[0][0]
        t = np.log(thr/d[i - 1])/np.log(d[i]/d[i - 1])
        kx = np.exp(np.log(k1[i - 1]) + t*np.log(k1[i]/k1[i - 1]))
        print(f"B1 {mdl}: dchi2_Roman = {thr} at k_UV(z=1) = {kx:.3f} (TATT-shaped mask, factor {fac[i-1] + t*(fac[i]-fac[i-1]):.2f})")

# ---------------- white nuisance cost ----------------
fw = json.load(open(os.path.join(HERE, "fisher_white.json")))
for key in ("n1:0.25", "n1:0.3", "n1:0.35", "n1:0.4", "n1:0.5", "scan:flat_0.3", "scan:flat_0.5"):
    r = fw[key]
    print(key, " ".join(f"{m}{c}: {r[f'{m}_{c}_nw']['sS8']:.4f}->{r[f'{m}_{c}_w']['sS8']:.4f} ({100*(r[f'{m}_{c}_w']['sS8']/r[f'{m}_{c}_nw']['sS8']-1):+.0f}%)"
                          for m in ("NLA", "TATT") for c in ("A", "B")))

# ---------------- tolerance numbers ----------------
rs = json.load(open(os.path.join(HERE, "roman_strategies.json")))
for which in ("GI", "II"):
    rows = [x for x in rs["tolerance"]["n1:0.3"]["TATT"] if x["shape"] == "step" and x["which"] == which]
    print("tolerance K0=0.3", which, [(round(x["kx"], 2), round(x["eps_dchi1"], 2)) for x in rows])
for mdl in ("NLA", "TATT"):
    for fam, t in D["tradeoff"][mdl].items():
        print(mdl, fam, "sS8 at FoB=0.3:", t["sS8_at_FoB03"], "K0:", t["K0_at_FoB03"])

# ---------------- split verification ----------------
vs = json.load(open(os.path.join(HERE, "verify_split.json")))
sp = json.load(open(os.path.join(HERE, "split_pm.json")))
lines = [r"\begin{tabular}{lcc r ccc cc}\toprule",
         r"model & $K_0^+$ & $K_0^-$ & pts & FoB A & FoB B & $\chi^2$ A & $\sigma(S_8)$ A & linear FoB\\ \midrule"]
for mdl, kp, km in sorted({(x["model"], x["K0p"], x["K0m"]) for x in vs}, key=lambda t: (t[0], t[1], t[2])):
    a = [x for x in vs if (x["model"], x["K0p"], x["K0m"], x["case"]) == (mdl, kp, km, "A")]
    b = [x for x in vs if (x["model"], x["K0p"], x["K0m"], x["case"]) == (mdl, kp, km, "B")]
    if not a:
        continue
    a = a[0]
    lin = [x for x in sp if (x["model"], x["K0p"], x["K0m"]) == (mdl, kp, km)][0]["FoB"]
    fa = f"{a['FoB']:.2f}" + (f" $\\pm$ {a['FoB_sig']:.2f}" if a.get("FoB_sig") else "")
    fb = f"{b[0]['FoB']:.2f}" if b else "n/a"
    lines.append(f"{mdl} & {kp:g} & {'none' if km == 0 else f'{km:g}'} & {a['npts']} & {fa} & {fb} & {a['chi2']:.2f} & {a['sS8']:.4f} & {lin:.2f}\\\\")
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "tab_split.tex"), "w").write("\n".join(lines))
print("\n".join(lines))
