#!/usr/bin/env python3
"""Figure and LaTeX tables for the HM (halo model with full 2-halo physics)
Roman scale cuts: hm_scan_ex1_{D1,flat}.json -> fig_hm_fob, report/tab_hm_*.tex,
and the FLAMINGO goodness-of-fit cuts (f21v_hyb.npz = HM, f21v_NLA, f21v_TATT)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2 as cd

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "report")
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8", "HM": "#2a9d55"}
LAB = {"NLA": "NLA", "TATT": "TATT", "HM": "halo model (full 2-halo)"}
J = {f: (json.load(open(os.path.join(HERE, f"hm_scan_ex1_{f}.json"))) if os.path.exists(os.path.join(HERE, f"hm_scan_ex1_{f}.json")) else [])
     for f in ("D1", "flat")}

fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
CC = {"A": "#2a9d55", "B": "#1b4d89"}
for i, fam in enumerate(("D1", "flat")):
    rows = sorted(J[fam], key=lambda r: r["K0"])
    a = ax[i]
    for case in ("A", "B"):
        pts = [(r["K0"], r[f"HM_{case}"]["FoB"], r[f"HM_{case}"].get("FoB_sig") or 0) for r in rows if f"HM_{case}" in r]
        if not pts:
            continue
        K, F, S = map(np.array, zip(*pts))
        a.plot(K, F, "o-", color=CC[case], lw=1.8, label=f"case {case}" + (r" ($\Omega_m$, $A_s$, IA free)" if case == "A" else r" (+ $n_s$, $H_0$, $\Omega_b$ with priors)"))
        a.fill_between(K, F - S, F + S, color=CC[case], alpha=0.15, lw=0)
    a.axhline(0.3, color="k", lw=1, ls="--")
    a.set_title(r"$k_{\rm UV}(z) = K_0/D(z)$" if fam == "D1" else r"$k_{\rm UV} = K_0$ (constant)", fontsize=11)
    a.set_xlabel(r"$K_0$ [$h\,$Mpc$^{-1}$]")
    a.grid(alpha=0.3)
    a.legend(fontsize=8.5, loc="upper left")
ax[0].set_ylabel("figure of bias in $(\\Omega_m, S_8)$")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_hm_fob.png"), dpi=220)
fig.savefig(os.path.join(HERE, "fig_hm_fob.pdf"))

# Roman tables: halo model only, cases B and A side by side
for fam in ("D1", "flat"):
    rows = sorted(J[fam], key=lambda r: r["K0"])
    L = ["\\begin{tabular}{c ccc r cccc cccc}\\toprule",
         " & \\multicolumn{3}{c}{$k_{\\rm UV}$ [$h\\,$Mpc$^{-1}$]} & & \\multicolumn{4}{c}{case B} & \\multicolumn{4}{c}{case A}\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}",
         "$K_0$ & $z=0.5$ & $z=1$ & $z=2$ & $N_{\\rm pts}$ & $\\sigma(S_8)$ & $\\delta\\Omega_m/\\sigma$ & $\\delta S_8/\\sigma$ & FoB & "
         "$\\sigma(S_8)$ & $\\delta\\Omega_m/\\sigma$ & $\\delta S_8/\\sigma$ & FoB\\\\ \\midrule"]
    for r in rows:
        c = [f"{r['K0']:g}", f"{r['kuv_z05']:.2f}", f"{r['kuv_z1']:.2f}", f"{r['kuv_z2']:.2f}", str(r["npts"])]
        for case in ("B", "A"):
            q = r.get(f"HM_{case}")
            c += ([f"{q['sS8']:.4f}", f"${q['dOm']:+.2f}$", f"${q['dS8']:+.2f}$", f"${q['FoB']:.2f}\\pm{(q.get('FoB_sig') or 0):.2f}$"]
                  if q else ["n/a"]*4)
        L.append(" & ".join(c) + "\\\\")
    L.append("\\bottomrule\\end{tabular}")
    open(os.path.join(REP, f"tab_hm_{fam}.tex"), "w").write("\n".join(L) + "\n")

# FLAMINGO goodness of fit (HM = hyb variant of f21_variants.py)
import simfit as SF   # noqa: E402
core = SF.core
out = []
for n, lab in (("hyb", "halo model"),):
    d = np.load(os.path.join(HERE, f"f21v_{n}.npz"))
    C, nu = d[f"{n}_chi2"], d[f"{n}_nu"]
    P = np.where(nu > 0, cd.sf(C, np.maximum(nu, 1)), 1.0)
    cuts = np.array([SF.cut_rule(P[s] >= 0.05) for s in range(SF.nsn)])[core]
    out.append((lab, cuts))
L = ["\\begin{tabular}{l " + "c"*int(core.sum()) + "}\\toprule", " & " + " & ".join(f"$z={z:g}$" for z in SF.zs[core]) + "\\\\ \\midrule"]
for lab, cuts in out:
    L.append(lab + " & " + " & ".join("$<$0.09" if not np.isfinite(c) else f"{c:.2f}" for c in cuts) + "\\\\")
L.append("\\bottomrule\\end{tabular}")
open(os.path.join(REP, "tab_hm_gof.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
for fam in ("D1", "flat"):
    for r in sorted(J[fam], key=lambda r: r["K0"]):
        print(fam, r["K0"], " ".join(f"{m}_{c} {r[f'{m}_{c}']['FoB']:.2f}" for c in "BA" for m in ("NLA", "TATT", "HM") if f"{m}_{c}" in r),
              " HM a1h", [round(r[f"HM_{c}"]["ia"][5], 3) for c in "BA" if f"HM_{c}" in r])
