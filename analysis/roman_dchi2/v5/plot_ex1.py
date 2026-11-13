#!/usr/bin/env python3
"""Gaussian covariance versus the production covariance example1_cov:
full-fit FoB against K0 for the 1/D and constant families (v3/mask_scan.json
vs v3/mask_scan_ex1_{D1,flat}.json), and sigma(S8) from the linear scans.
Writes fig_ex1_fob (220 dpi PNG + PDF) and ex1_table.tex / ex1_split.tex."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..", "v3")
V4 = os.path.join(HERE, "..", "v4")
REP = os.path.join(HERE, "..", "report")
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
G = json.load(open(os.path.join(V3, "mask_scan.json")))
E = []
for f in ("mask_scan_ex1_D1.json", "mask_scan_ex1_flat.json"):
    p = os.path.join(V3, f)
    if os.path.exists(p):
        E += json.load(open(p))
LE = json.load(open(os.path.join(HERE, "linear_example1.json")))
LG = json.load(open(os.path.join(HERE, "linear_gauss.json")))


def sS8(L, fam, K0, mdl="NLA", case="A"):
    return [x for x in L if x["kind"] == "family" and x["family"] == fam and abs(x["K0"] - K0) < 1e-9][0][f"{mdl}_{case}"]["sS8"]


fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
for r, fam in enumerate(("D1", "flat")):
    for c, (key, ttl) in enumerate((("A", "case A ($\\Omega_m$, $A_s$)"), ("B", "case B (+ $n_s$, $H_0$, $\\Omega_b$)"), ("wA", "with the white II nuisance, case A"))):
        a = ax[r, c]
        for lab, res, ls, mk in (("Gaussian cov", G, ":", "o"), ("example1_cov", E, "-", "s")):
            rr = sorted([x for x in res if x["family"] == fam], key=lambda x: x["K0"])
            if not rr:
                continue
            K = [x["K0"] for x in rr]
            for mdl in ("NLA", "TATT"):
                k = f"{mdl}w_A" if key == "wA" else f"{mdl}_{key}"
                fob = np.array([x[k]["FoB"] for x in rr])
                sig = np.array([x[k].get("FoB_sig") or 0 for x in rr])
                a.errorbar(K, fob, sig if key == "A" else None, ls=ls, marker=mk, ms=5, color=COL[mdl], capsize=2,
                           mfc=COL[mdl] if lab == "example1_cov" else "none", label=f"{mdl}, {lab}")
        a.axhline(0.3, color="k", lw=0.8, ls="--")
        a.set_ylim(0, 0.8)
        a.set_xlabel(r"$K_0$ [h/Mpc]  (" + ("$k_{\\rm UV}=K_0/D(z)$" if fam == "D1" else "constant $k_{\\rm UV}=K_0$") + ")")
        a.set_ylabel(r"FoB [$\sigma$]")
        a.set_title(f"{'1/D' if fam == 'D1' else 'constant'} family, {ttl}", fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ex1_fob.png"), dpi=220)
fig.savefig(os.path.join(HERE, "fig_ex1_fob.pdf"))

# table: full fits, both covariances
lines = [r"\begin{tabular}{llr cc cc cc c}\toprule",
         r" & & & \multicolumn{2}{c}{FoB A (NLA / TATT)} & \multicolumn{2}{c}{FoB B (NLA / TATT)} & \multicolumn{2}{c}{+ white, A (NLA / TATT)} & $\sigma(S_8)$\\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"family & $K_0$ & pts & Gauss & example1 & Gauss & example1 & Gauss & example1 & Gauss $\to$ example1\\ \midrule"]
for fam in ("D1", "flat"):
    for K0 in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        g = [x for x in G if x["family"] == fam and abs(x["K0"] - K0) < 1e-9]
        e = [x for x in E if x["family"] == fam and abs(x["K0"] - K0) < 1e-9]
        if not g or not e:
            continue
        g, e = g[0], e[0]
        f = lambda x, k: f"{x[f'NLA{k}']['FoB']:.2f} / {x[f'TATT{k}']['FoB']:.2f}"
        lines.append(f"{'$1/D$' if fam == 'D1' else 'constant'} & {K0:g} & {e['npts']} & {f(g, '_A')} & {f(e, '_A')} & {f(g, '_B')} & {f(e, '_B')} & "
                     f"{f(g, 'w_A')} & {f(e, 'w_A')} & {sS8(LG, fam, K0):.4f} $\\to$ {sS8(LE, fam, K0):.4f}\\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule\end{tabular}"
open(os.path.join(REP, "tab_ex1.tex"), "w").write("\n".join(lines))
print("\n".join(lines))

vs = os.path.join(V4, "verify_split_ex1.json")
if os.path.exists(vs):
    v = json.load(open(vs))
    lines = [r"\begin{tabular}{lcc r cc c c}\toprule",
             r"model & $K_0^+$ & $K_0^-$ & pts & FoB A & FoB B & $\chi^2$ A & $\sigma(S_8)$ A\\ \midrule"]
    for x in v:
        if x["case"] != "A":
            continue
        b = [y for y in v if (y["model"], y["K0p"], y["K0m"], y["case"]) == (x["model"], x["K0p"], x["K0m"], "B")]
        kms = "none" if x["K0m"] == 0 else f"{x['K0m']:g}"
        if b:
            lines.append(f"{x['model']} & {x['K0p']:g} & {kms} & {x['npts']} & "
                         f"{x['FoB']:.2f} $\\pm$ {x['FoB_sig']:.2f} & {b[0]['FoB']:.2f} & {x['chi2']:.2f} & {x['sS8']:.4f}\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(os.path.join(REP, "tab_ex1_split.tex"), "w").write("\n".join(lines))
    print("\n".join(lines))
