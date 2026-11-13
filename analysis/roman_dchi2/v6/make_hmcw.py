#!/usr/bin/env python3
"""Figures for the central 1-halo term (HMc_lin of halo_central_fit.py)
against TATT and HM (f21_variants "hyb"): chi2/nu vs k_max (summed over the
core snapshots and per snapshot), goodness-of-fit cuts (k horizontal, z
vertical), and plain spectra at z = 0.3 and 1 with the 2-halo / central
1-halo split. 220 dpi PNG + PDF."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "_plot"]
import halo_central_fit as HF   # noqa: E402
try:
    os.remove(os.path.join(HERE, "halo_central_fit_log__plot.txt"))
except OSError:
    pass
FV, SF = HF.FV, HF.SF
kg, core, zs = SF.kg, SF.core, SF.zs
MOD = {"TATT": ("TATT", "#9fb8d8"), "TATT_W": ("TATT + white E=B (W)", "#1f5fa8"),
       "HMc_lin": ("HMc", "#9fd3b5"), "HMc_W": ("HMc + white E=B (W)", "#2a9d55")}
SPEC = ("TATT_W", "HMc_W")
D = {n: np.load(os.path.join(HERE, f"f21v_{n}.npz")) for n in MOD}


def kax(a, lo, hi, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3)):
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(ticker.FixedLocator(list(ticks)))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.grid(alpha=0.3, which="both")


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# 1. goodness of fit
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
for n, (lab, col) in MOD.items():
    C, NU = D[n][f"{n}_chi2"], D[n][f"{n}_nu"]
    ax[0].plot(kg, C[core].sum(0)/NU[core].sum(0), "o-", color=col, ms=4, label=lab)
    for s in np.where(core)[0]:
        ax[1].plot(kg, C[s]/NU[s], "-", color=col, lw=0.9, alpha=0.75)
    cuts = D[n][f"{n}_cut"][core]
    off = {"TATT": -0.06, "TATT_W": -0.02, "HMc_lin": 0.02, "HMc_W": 0.06}[n]
    ax[2].plot(cuts, zs[core] + off, "o-", color=col, ms=6, label=lab)
for a in ax[:2]:
    a.set_yscale("log")
    kax(a, 0.14, 3.3, (0.16, 0.22, 0.3, 0.5, 1, 2, 3))
    a.set_xlabel(r"$k_{\max}$ [$h\,$Mpc$^{-1}$]")
    a.axhline(1, color="k", lw=0.6, ls=":")
    a.set_ylim(0.5, 200)
ax[0].set_ylabel(r"$\chi^2_{\min}/\nu$, summed over $z = 0.2$ to 2")
ax[1].set_ylabel(r"$\chi^2_{\min}/\nu$ per snapshot")
ax[0].legend(fontsize=9)
kax(ax[2], 0.14, 0.8, (0.16, 0.22, 0.29, 0.39, 0.52, 0.7))
ax[2].set_xlabel(r"goodness-of-fit cut $k_{\max}$ [$h\,$Mpc$^{-1}$]")
ax[2].set_ylabel("z")
ax[2].legend(fontsize=8, loc="upper left")
fig.tight_layout()
save(fig, "fig_hmcw_gof")

# 2. spectra: TATT and HMc fitted to k <= 0.94, HMc 2-halo / central 1-halo split
KFIT = 0.94
for zsel in (0.3, 1.0):
    s = int(np.argmin(np.abs(zs - zsel)))
    j = int(np.argmin(np.abs(kg - KFIT)))
    m = SF.kd >= 0.05
    k = SF.kd[m]
    xt = D["TATT_W"]["TATT_W_x"][s, j, :4]
    tt = HF.m_hmcw(s, "TATT")[0](xt)[:, m]
    xc = D["HMc_W"]["HMc_W_x"][s, j, :6]
    pred = HF.m_hmcw(s)[0]
    hc = pred(xc)[:, m]
    xc0 = xc.copy()
    xc0[4] = 0.0
    h2 = pred(xc0)[:, m]                        # without the central 1-halo term
    c1 = (h2 - hc)[0]                           # = b(k) a_c Pc (positive)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    names = (r"$-P_{hE}$", r"$P_{EE}$", r"$P_{BB}$")
    ct = D["TATT_W"]["TATT_W_chi2"][s, j]/D["TATT_W"]["TATT_W_nu"][s, j]
    cc = D["HMc_W"]["HMc_W_chi2"][s, j]/D["HMc_W"]["HMc_W_nu"][s, j]
    for i in range(3):
        sg = -1 if i == 0 else 1
        a = ax[0, i]
        a.errorbar(k, sg*SF.y[s][i, m], SF.err[s][i, m], fmt="o", ms=3, color="k", lw=0.8, zorder=5,
                   label="FLAMINGO" if i == 0 else None)
        a.plot(k, sg*tt[i], color=MOD["TATT_W"][1], lw=1.6,
               label=(f"TATT + W: $A_1$={xt[0]:.2f}, $A_2$={xt[1]:.2f}, $b_{{TA}}$={xt[2]:.2f}, $W/N_s$={xt[3]/SF.Ns[s]:.2f}; $\\chi^2/\\nu$={ct:.1f}") if i == 0 else None)
        a.plot(k, sg*hc[i], color=MOD["HMc_W"][1], lw=1.8,
               label=(f"HMc + W: $A_1$={xc[0]:.2f}, $A_2$={xc[1]:.2f}, $b_{{TA}}$={xc[2]:.2f}, $a_c$={xc[4]:.1f}, $W/N_s$={xc[5]/SF.Ns[s]:.2f}; "
                      f"$\\chi^2/\\nu$={cc:.1f}") if i == 0 else None)
        if i == 0:
            a.plot(k, -h2[0], color=MOD["HMc_W"][1], lw=1.0, ls="--", label="HMc 2-halo")
            a.plot(k, c1, color="#e08a1e", lw=1.2, ls="--", label="HMc central 1-halo")
        else:
            a.axhline(SF.Ns[s], color="0.6", ls=":", lw=1, label="$N_s$" if i == 2 else None)
        a.axvline(KFIT, color="0.4", lw=0.8, ls=":")
        a.set_yscale("log")
        a.set_ylabel(names[i] + r" [$(h^{-1}$Mpc$)^3$]")
        for mod, pp in (("TATT_W", tt), ("HMc_W", hc)):
            ax[1, i].plot(k, (pp[i] - SF.y[s][i, m])/SF.err[s][i, m], color=MOD[mod][1], lw=1.3)
        ax[1, i].axhline(0, color="k", lw=0.6)
        ax[1, i].fill_between([0.04, 4], -1, 1, color="0.88", lw=0)
        ax[1, i].set_ylim(-30, 30)
        ax[1, i].axvline(KFIT, color="0.4", lw=0.8, ls=":")
        ax[1, i].set_ylabel(r"(model $-$ data)/$\sigma$")
        ax[1, i].set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        kax(ax[0, i], 0.045, 3.3)
        kax(ax[1, i], 0.045, 3.3)
    ax[0, 0].set_ylim(0.3*np.nanmin(-SF.y[s][0, m]), 1.6*np.nanmax(-SF.y[s][0, m]))
    ax[0, 1].set_ylim(0.5*SF.Ns[s], 1.6*np.nanmax(SF.y[s][1, m]))
    ax[0, 2].set_ylim(0.6*SF.Ns[s], 1.4*np.nanmax(SF.y[s][2, m] + SF.err[s][2, m]))
    ax[0, 0].legend(fontsize=7.2, loc="lower left")
    ax[0, 1].set_title(f"z = {zs[s]:g}: fits to $k \\leq {kg[j]:.2f}$ (dotted line)")
    fig.tight_layout()
    save(fig, f"fig_hmcw_spectra_z{zs[s]:g}".replace(".", "p"))

for n in MOD:
    C, NU = D[n][f"{n}_chi2"], D[n][f"{n}_nu"]
    print(f"{n:8s} cuts {np.round(D[n][f'{n}_cut'][core], 2)}  summed chi2/nu @0.39,0.52,0.94,3.04: " +
          " ".join(f"{C[core, j].sum()/NU[core, j].sum():6.2f}" for j in (5, 6, 8, 12)))
