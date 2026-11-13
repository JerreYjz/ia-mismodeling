#!/usr/bin/env python3
"""Figures and tables for the overnight halo-model report (f21_variants.py
outputs; Roman json files). 220 dpi PNG + PDF; k horizontal everywhere."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f21_variants as FV          # noqa: E402
SF = FV.SF
REP = os.path.join(HERE, "..", "report")
NAMES = ["NLA", "F21", "F21_ext", "F21_win", "F21_rb", "F21_all", "hyb0", "hyb", "hyb_ext", "TATT"]
LAB = {"NLA": "NLA", "TATT": "TATT", "F21": "F21 (as published)", "F21_ext": r"F21 + $b$, $c_{\rm sat}$, $\alpha$",
       "F21_win": "F21 + free windows", "F21_rb": "F21 red/blue (no cross term)", "F21_all": "F21 all freedoms",
       "hyb0": "red NLA + blue TT + sat.", "hyb": r"red NLA + blue TT + $b_{\rm TA}$ + sat.", "hyb_ext": "hyb + sat. grid + windows"}
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8", "F21": "#2a9d55", "F21_ext": "#7fbf7b", "F21_win": "#1b7837",
       "F21_rb": "#e08a1e", "F21_all": "#b35806", "hyb0": "#8073ac", "hyb": "#542788", "hyb_ext": "#c2a5cf"}
R = {n: dict(np.load(os.path.join(HERE, f"f21v_{n}.npz"))) for n in NAMES if os.path.exists(os.path.join(HERE, f"f21v_{n}.npz"))}
NAMES = [n for n in NAMES if n in R]
from scipy.stats import chi2 as _cd
for _n in NAMES:
    # k_max with nu <= 0 cannot be tested: count as a pass (the saved cuts
    # treated them as failures, which wrongly cut grid-profiled models at 0.09)
    _C, _nu = R[_n][f"{_n}_chi2"], R[_n][f"{_n}_nu"]
    _P = np.where(_nu > 0, _cd.sf(_C, np.maximum(_nu, 1)), 1.0)
    R[_n][f"{_n}_p"] = _P
    R[_n][f"{_n}_cut"] = np.array([SF.cut_rule(_P[s] >= 0.05) for s in range(SF.nsn)])
kg, core, zs = SF.kg, SF.core, SF.zs


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def kax(a, lo=0.08, hi=3.3):
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(ticker.FixedLocator([0.1, 0.2, 0.3, 0.5, 1, 2, 3]))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.grid(alpha=0.3, which="both")


# 1. combined chi2/nu vs k_max and probe shares
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for n in NAMES:
    c = R[n][f"{n}_chi2"][core].sum(0)
    nu = R[n][f"{n}_nu"][core].sum(0)
    ax[0].plot(kg, c/nu, "o-", ms=3, color=COL[n], label=LAB[n], ls="--" if n in ("NLA", "TATT") else "-")
from scipy.stats import chi2 as cd
nu_ref = R["NLA"]["NLA_nu"][core].sum(0)
ax[0].plot(kg, cd.ppf(0.95, nu_ref)/nu_ref, "k:", lw=1, label=r"$p = 0.05$")
ax[0].set_yscale("log")
kax(ax[0], 0.08, 3.3)
ax[0].set_xlabel(r"$k_{\max}$ [$h\,$Mpc$^{-1}$]")
ax[0].set_ylabel(r"$\chi^2/\nu$, core snapshots combined")
ax[0].legend(fontsize=7.5, ncol=2)
j = int(np.argmin(np.abs(kg - 0.52)))
w = 0.8/len(NAMES)
for i, n in enumerate(NAMES):
    cp = R[n][f"{n}_chi2p"][core, j].sum(0)
    nu = R[n][f"{n}_nu"][core, j].sum()
    bottom = 0
    for ip, (pn, hatch) in enumerate((("hE", ""), ("EE", "//"), ("BB", ".."))):
        ax[1].bar(i, cp[ip]/nu, bottom=bottom, color=COL[n], hatch=hatch, edgecolor="k", lw=0.5, alpha=0.85 if ip == 0 else 0.55)
        bottom += cp[ip]/nu
ax[1].set_xticks(range(len(NAMES)))
ax[1].set_xticklabels([LAB[n] for n in NAMES], rotation=40, ha="right", fontsize=7.5)
ax[1].set_ylabel(r"$\chi^2/\nu$ at $k_{\max} = 0.52$, split by probe")
from matplotlib.patches import Patch
ax[1].legend(handles=[Patch(facecolor="w", edgecolor="k", hatch=h, label=l) for h, l in (("", r"$P_{hE}$"), ("//", r"$P_{EE}$"), ("..", r"$P_{BB}$"))], fontsize=8)
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "fig_on_chi2")

# 2. cuts (k horizontal, z vertical)
fig, a = plt.subplots(figsize=(8.5, 5.5))
zc = zs[core]
for i, n in enumerate(NAMES):
    c = R[n][f"{n}_cut"][core]
    cc = np.where(np.isfinite(c), c, 0.075)
    a.plot(cc*(1 + 0.012*(i - len(NAMES)/2)), zc, "o", color=COL[n], ms=6, label=LAB[n],
           mfc="none" if n.startswith("hyb") else COL[n])
a.set_xscale("log")
a.set_xlim(0.07, 0.5)
a.xaxis.set_major_locator(ticker.FixedLocator([0.075, 0.09, 0.12, 0.16, 0.22, 0.29, 0.39]))
a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "<0.09" if x < 0.08 else f"{x:g}"))
a.xaxis.set_minor_locator(ticker.NullLocator())
a.set_xlabel(r"goodness-of-fit cut $k_{\max}$ [$h\,$Mpc$^{-1}$] (two consecutive $p < 0.05$)")
a.set_ylabel("z")
a.grid(alpha=0.3)
a.legend(fontsize=7.5, loc="upper right")
save(fig, "fig_on_cuts")

# 3. spectra at one snapshot, fits to k <= 0.52 (plain spectra, residual / sigma)
for zsel in (0.3, 1.0):
    s = int(np.argmin(np.abs(zs - zsel)))
    j = int(np.argmin(np.abs(kg - 0.52)))
    m = SF.kd >= 0.05
    k = SF.kd[m]
    sh = ["NLA", "F21", "F21_rb", "hyb0", "hyb", "TATT"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    for n in sh:
        fac = FV.VARIANTS[n][0](s)
        pred, _, _ = fac[int(R[n][f"{n}_grid"][s, j])]
        x = R[n][f"{n}_x"][s, j]
        x = x[np.isfinite(x)]
        p = pred(x)[:, m]
        for i in range(3):
            sg = -1 if i == 0 else 1
            ax[0, i].plot(k, sg*p[i], color=COL[n], lw=1.6, label=f"{LAB[n]} ($\\chi^2/\\nu$ = {R[n][f'{n}_chi2'][s, j]/R[n][f'{n}_nu'][s, j]:.1f})")
            ax[1, i].plot(k, (p[i] - SF.y[s][i, m])/SF.err[s][i, m], color=COL[n], lw=1.3)
    lab = (r"$-P_{hE}$", r"$P_{EE}$", r"$P_{BB}$")
    for i in range(3):
        sg = -1 if i == 0 else 1
        ax[0, i].errorbar(k, sg*SF.y[s][i, m], SF.err[s][i, m], fmt="o", ms=3, color="k", lw=0.8, zorder=5, label="FLAMINGO")
        if i > 0:
            ax[0, i].axhline(SF.Ns[s], color="0.4", ls="--", lw=1)
        ax[0, i].set_yscale("log")
        ax[0, i].set_ylabel(lab[i] + r" [$(h^{-1}$Mpc$)^3$]")
        ax[0, i].axvspan(kg[j], 3.3, color="0.93", zorder=0)
        ax[1, i].axvspan(kg[j], 3.3, color="0.93", zorder=0)
        ax[1, i].axhline(0, color="k", lw=0.6)
        ax[1, i].fill_between([0.05, 4], -1, 1, color="0.85", lw=0)
        ax[1, i].set_ylim(-30, 30)
        ax[1, i].set_ylabel(r"(model $-$ data)/$\sigma$")
        ax[1, i].set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        kax(ax[0, i], 0.045, 3.3)
        kax(ax[1, i], 0.045, 3.3)
    ax[0, 0].set_ylim(0.5*np.nanmin(-SF.y[s][0, m]), 1.6*np.nanmax(-SF.y[s][0, m]))
    if True:
        ax[0, 2].set_ylim(0.6*SF.Ns[s], 1.3*np.nanmax(SF.y[s][2, m] + SF.err[s][2, m]))
    ax[0, 0].legend(fontsize=7.5, loc="lower left")
    ax[0, 1].set_title(f"z = {zs[s]:g}, fits to $k \\leq {kg[j]:.2f}\\,h\\,$Mpc$^{{-1}}$ (grey: not fitted)")
    fig.tight_layout()
    save(fig, f"fig_on_spectra_z{zs[s]:g}".replace(".", "p"))

# 4. table body of cuts and chi2/nu at 0.29, 0.52, 1.0 (core combined)
rows = []
for n in NAMES:
    c = R[n][f"{n}_cut"][core]
    js = [int(np.argmin(np.abs(kg - kk))) for kk in (0.29, 0.52, 1.0)]
    cv = [R[n][f"{n}_chi2"][core, jj].sum()/R[n][f"{n}_nu"][core, jj].sum() for jj in js]
    sh = R[n][f"{n}_chi2p"][core, js[1]].sum(0)/R[n][f"{n}_chi2"][core, js[1]].sum()
    cs = " & ".join("$<$0.09" if not np.isfinite(v) else f"{v:.2f}" for v in c)
    rows.append(f"{LAB[n]} & {int(R[n][f'{n}_npar'])} & {cs} & " + " & ".join(f"{v:.1f}" for v in cv) +
                f" & {sh[0]:.2f}/{sh[1]:.2f}/{sh[2]:.2f}\\\\")
hdr = ("\\begin{tabular}{l c " + "c"*int(core.sum()) + " ccc c}\\toprule\n"
       " & & \\multicolumn{" + str(int(core.sum())) + "}{c}{goodness-of-fit cut $k_{\\max}$ [$h\\,$Mpc$^{-1}$] at $z =$} & "
       "\\multicolumn{3}{c}{$\\chi^2/\\nu$ (core) at $k_{\\max} =$} & share at 0.52\\\\\n"
       "model & $n_{\\rm par}$ & " + " & ".join(f"{z:g}" for z in zc) + " & 0.29 & 0.52 & 1.0 & $hE$/$EE$/$BB$\\\\ \\midrule\n")
open(os.path.join(REP, "tab_on_variants.tex"), "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print("figures and table written")
