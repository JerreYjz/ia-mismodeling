#!/usr/bin/env python3
"""FLAMINGO spectra against the halo model (TATT 2-halo + satellite 1-halo)
and TATT, fitted to k <= 0.29 h/Mpc, at z = 0.3, 1, 2. Plain spectra (shape
noise included in EE, BB) and (model - data)/sigma. -> fig_hm_spectra."""
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
KFIT = 0.29
MODELS = (("TATT", "TATT", "#1f5fa8", "--"), ("hyb", "halo model", "#2a9d55", "-"))
R = {n: dict(np.load(os.path.join(HERE, f"f21v_{n}.npz"))) for n, *_ in MODELS}
j = int(np.argmin(np.abs(SF.kg - KFIT)))
m = SF.kd >= 0.05
k = SF.kd[m]
ZS = (0.3, 1.0, 2.0)
fig = plt.figure(figsize=(16, 15))
gs = fig.add_gridspec(2*len(ZS), 3, height_ratios=[3, 1.3]*len(ZS), hspace=0.12, wspace=0.25)
lab = (r"$-P_{hE}$", r"$P_{EE}$", r"$P_{BB}$")
for iz, zsel in enumerate(ZS):
    s = int(np.argmin(np.abs(SF.zs - zsel)))
    for i in range(3):
        a = fig.add_subplot(gs[2*iz, i])
        ar = fig.add_subplot(gs[2*iz + 1, i], sharex=a)
        sg = -1 if i == 0 else 1
        a.errorbar(k, sg*SF.y[s][i, m], SF.err[s][i, m], fmt="o", ms=3, color="k", lw=0.8, zorder=5, label="FLAMINGO")
        for n, ln, col, ls in MODELS:
            pred, _, _ = FV.VARIANTS[n][0](s)[int(R[n][f"{n}_grid"][s, j])]
            x = R[n][f"{n}_x"][s, j]
            x = x[np.isfinite(x)]
            p = pred(x)[:, m]
            c2 = R[n][f"{n}_chi2"][s, j]/R[n][f"{n}_nu"][s, j]
            a.plot(k, sg*p[i], color=col, ls=ls, lw=1.8, label=f"{ln} ($\\chi^2/\\nu$ = {c2:.2f})")
            ar.plot(k, (p[i] - SF.y[s][i, m])/SF.err[s][i, m], color=col, ls=ls, lw=1.4)
        if i > 0:
            a.axhline(SF.Ns[s], color="0.5", ls=":", lw=1, label=r"$N_s$")
        a.set_yscale("log")
        a.set_ylabel(lab[i] + r" [$(h^{-1}$Mpc$)^3$]")
        yy = sg*SF.y[s][i, m]
        lo = 0.6*min(np.nanmin(yy[yy > 0]), SF.Ns[s] if i else np.inf)
        a.set_ylim(lo, 1.5*np.nanmax(yy + SF.err[s][i, m]))
        for ax_ in (a, ar):
            ax_.axvspan(SF.kg[j], 3.3, color="0.93", zorder=0)
            ax_.set_xscale("log")
            ax_.set_xlim(0.045, 3.3)
            ax_.xaxis.set_major_locator(ticker.FixedLocator([0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]))
            ax_.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax_.xaxis.set_minor_locator(ticker.NullLocator())
            ax_.grid(alpha=0.3)
        plt.setp(a.get_xticklabels(), visible=False)
        ar.axhline(0, color="k", lw=0.6)
        ar.fill_between([0.04, 4], -1, 1, color="0.82", lw=0)
        ar.set_ylim(-12, 12)
        ar.set_ylabel(r"$\Delta/\sigma$")
        if iz == len(ZS) - 1:
            ar.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        if i == 1:
            a.set_title(f"z = {SF.zs[s]:g}, fits to $k \\leq {SF.kg[j]:.2f}\\,h\\,$Mpc$^{{-1}}$ (grey: not fitted)", fontsize=10)
        if i == 0:
            a.legend(fontsize=8, loc="lower left")
fig.savefig(os.path.join(HERE, "fig_hm_spectra.png"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_hm_spectra.pdf"), bbox_inches="tight")
for n, *_ in MODELS:
    print(n, "params at 0.29 (z=0.3,1,2):", [np.round(R[n][f"{n}_x"][int(np.argmin(np.abs(SF.zs - z))), j][:5], 3).tolist() for z in ZS])
