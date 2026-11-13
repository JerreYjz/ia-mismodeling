#!/usr/bin/env python3
"""FLAMINGO figures for the halo model with the full 2-halo physics (HM =
variant "hyb" of f21_variants.py): chi2/nu against k_max per snapshot, the
goodness-of-fit cuts, and the spectra with the 2-halo / 1-halo split.
Plain spectra (no ratios); 220 dpi PNG + PDF."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats import chi2 as cd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f21_variants as FV     # noqa: E402
SF = FV.SF
d = np.load(os.path.join(HERE, "f21v_hyb.npz"))
C, NU, X = d["hyb_chi2"], d["hyb_nu"], d["hyb_x"]
kg, core, zs = SF.kg, SF.core, SF.zs
P = np.where(NU > 0, cd.sf(C, np.maximum(NU, 1)), 1.0)
cuts = np.array([SF.cut_rule(P[s] >= 0.05) for s in range(SF.nsn)])


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


# 1. chi2/nu vs k_max and the cuts
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
cmap = plt.get_cmap("viridis")
sel = np.where(core)[0]
for i, s in enumerate(sel):
    col = cmap(i/(len(sel) - 1))
    ax[0].plot(kg, C[s]/NU[s], "o-", ms=3, color=col, label=f"z = {zs[s]:g}")
    ax[0].plot(kg, cd.ppf(0.95, NU[s])/NU[s], ":", color=col, lw=0.8)
ax[0].set_yscale("log")
kax(ax[0], 0.08, 3.3, (0.1, 0.2, 0.3, 0.5, 1, 2, 3))
ax[0].set_xlabel(r"$k_{\max}$ [$h\,$Mpc$^{-1}$]")
ax[0].set_ylabel(r"$\chi^2_{\min}/\nu$ (dotted: $p = 0.05$)")
ax[0].legend(fontsize=7.5, ncol=2)
a = ax[1]
a.plot(cuts[sel], zs[sel], "o-", color="#2a9d55", ms=7)
kax(a, 0.08, 0.5, (0.09, 0.12, 0.16, 0.22, 0.29, 0.39))
a.set_xlabel(r"goodness-of-fit cut $k_{\max}$ [$h\,$Mpc$^{-1}$]")
a.set_ylabel("z")
fig.tight_layout()
save(fig, "fig_hm_gof")

# 2. spectra with the 2-halo / 1-halo split, fits to the snapshot's cut and to k <= 1
for zsel in (0.3, 1.0):
    s = int(np.argmin(np.abs(zs - zsel)))
    m = SF.kd >= 0.05
    k = SF.kd[m]
    Sd, Si = FV.sat_tables(s)
    f1, f2 = FV.win1(4.0), FV.win2(6.0)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.2), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    for kfit, ls in ((cuts[s], "-"), (3.04, "--")):
        j = int(np.argmin(np.abs(kg - kfit)))
        a1, a2, bt, a1h = X[s, j, :4]
        pr = SF.model.predict(zs[s], SF.kd, a1, a2, bt, 1.0, K=SF.KER[s], PNL=SF.Pnl[s], Ns=0.0, bk=SF.b[s])
        sd, si = FV.interp_a(Sd, a1h), FV.interp_a(Si, a1h)
        two = np.array([SF.b[s]*pr["dE"]*f2, pr["EE"]*f2, pr["BB"]*f2])[:, m]
        one = np.array([-SF.b[s]*a1h*f1*sd, a1h**2*f1*si, 0*SF.kd])[:, m]
        tot = two + one + np.array([0.0, SF.Ns[s], SF.Ns[s]])[:, None]
        lab = (f"fit to $k\\leq{kg[j]:.2f}$: $A_1$={a1:.2f}, $A_2$={a2:.2f}, $b_{{TA}}$={bt:.2f}, $a_{{1h}}$={a1h:.3f}; "
               f"$\\chi^2/\\nu$={C[s, j]/NU[s, j]:.2f}")
        for i in range(3):
            sg = -1 if i == 0 else 1
            ax[0, i].plot(k, sg*tot[i], color="#2a9d55", ls=ls, lw=1.8, label=lab if i == 0 else None)
            ax[0, i].plot(k, sg*two[i] + (SF.Ns[s] if i else 0), color="0.45", ls=ls, lw=1.0,
                          label=("2-halo (+ $N_s$)" if ls == "-" else None) if i == 1 else None)
            if i < 2:
                ax[0, i].plot(k, np.abs(one[i]), color="#e08a1e", ls=ls, lw=1.0,
                              label=("1-halo (satellites)" if ls == "-" else None) if i == 1 else None)
            ax[1, i].plot(k, (tot[i] - SF.y[s][i, m])/SF.err[s][i, m], color="#2a9d55", ls=ls, lw=1.3)
    names = (r"$-P_{hE}$", r"$P_{EE}$", r"$P_{BB}$")
    for i in range(3):
        sg = -1 if i == 0 else 1
        ax[0, i].errorbar(k, sg*SF.y[s][i, m], SF.err[s][i, m], fmt="o", ms=3, color="k", lw=0.8, zorder=5, label="FLAMINGO" if i == 0 else None)
        if i:
            ax[0, i].axhline(SF.Ns[s], color="0.6", ls=":", lw=1)
        ax[0, i].set_yscale("log")
        ax[0, i].set_ylabel(names[i] + r" [$(h^{-1}$Mpc$)^3$]")
        ax[1, i].axhline(0, color="k", lw=0.6)
        ax[1, i].fill_between([0.04, 4], -1, 1, color="0.88", lw=0)
        ax[1, i].set_ylim(-25, 25)
        ax[1, i].set_ylabel(r"(model $-$ data)/$\sigma$")
        ax[1, i].set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
        kax(ax[0, i], 0.045, 3.3)
        kax(ax[1, i], 0.045, 3.3)
    ax[0, 0].set_ylim(0.5*np.nanmin(-SF.y[s][0, m]), 1.6*np.nanmax(-SF.y[s][0, m]))
    ax[0, 1].set_ylim(1e-3, 1.6*np.nanmax(SF.y[s][1, m]))
    ax[0, 2].set_ylim(0.6*SF.Ns[s], 1.4*np.nanmax(SF.y[s][2, m] + SF.err[s][2, m]))
    ax[0, 0].legend(fontsize=7.2, loc="lower left")
    ax[0, 1].legend(fontsize=8, loc="upper right")
    ax[0, 1].set_title(f"z = {zs[s]:g}: halo model fitted to its cut (solid) and to $k\\leq3$ (dashed)")
    fig.tight_layout()
    save(fig, f"fig_hm_spectra_z{zs[s]:g}".replace(".", "p"))
print("cuts:", dict(zip([f"{z:g}" for z in zs[sel]], cuts[sel])))
for s in sel:
    j = int(np.argmin(np.abs(kg - cuts[s])))
    print(f"z {zs[s]:g}: at cut {cuts[s]:.2f}: A1 {X[s, j, 0]:.2f} A2 {X[s, j, 1]:.2f} bTA {X[s, j, 2]:.3f} a1h {X[s, j, 3]:.3g}")
