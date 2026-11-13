#!/usr/bin/env python3
"""One FLAMINGO snapshot against the NLA, TATT and halo-model best fits.

Columns: fits to k <= 0.29 (TATT GoF cut region) and to k <= 3.04 (all scales).
Rows: P_dE/P_NL, residual; (P_EE - N_s)/P_NL, residual; P_BB - N_s.
Data are the cosmolike-convention spectra of v2/fits.npz (as all fits).
Usage: plot_one_snap.py [z]   (default z = 1)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v3"))
import halo_sim as HS            # noqa: E402
import simfit as SF              # noqa: E402

Z = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
KFITS = (0.291, 3.04)
KMIN = 0.05
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8", "halo": "#2a9d55", "strict": "#e08a1e"}
LS = {"NLA": "-", "TATT": "-", "halo": "-", "strict": "--"}

v3 = SF.v3
tab = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
hs = np.load(os.path.join(HERE, "halo_sim.npz"))
s = int(np.argmin(np.abs(SF.zs - Z)))
kd, kg = SF.kd, SF.kg
m = kd >= KMIN
k = kd[m]
y, e = SF.y[s][:, m], SF.err[s][:, m]
Ns, b, P = SF.Ns[s], SF.b[s][m], SF.Pnl[s][m]
hpred = HS.model_factory(tab, s, nowin=False)
import halo_strict as HX  # noqa: E402
stab = dict(np.load(os.path.join(HERE, "halo_strict_tables.npz")))
ss = np.load(os.path.join(HERE, "halo_strict_sim.npz"))
spred = HX.model_factory(stab, s)


def obs(pr):
    """[hE, EE, BB] observed -> (P_dE/P_NL, (P_EE - N_s)/P_NL, P_BB - N_s)."""
    return pr[0]/(b*P), (pr[1] - Ns)/P, pr[2] - Ns


fig, AX = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
TT = (r"$-P_{hE}(k)$  [$(h^{-1}$Mpc$)^3$]", r"$P_{EE}(k)$  [$(h^{-1}$Mpc$)^3$]", r"$P_{BB}(k)$  [$(h^{-1}$Mpc$)^3$]")
for c, kf in enumerate(KFITS):
    j = int(np.argmin(np.abs(kg - kf)))
    fits = {}
    for name in ("NLA", "TATT"):
        x = v3[f"{name}_x"][s, j]
        fits[name] = (SF.pred(s, *x)[:, m], x)
    xh = hs["halo_x"][s, j]
    fits["halo"] = (hpred(*xh)[:, m], xh)
    xs = ss["strict_x"][s, j]
    fits["strict"] = (spred(*xs)[:, m], xs)
    chi = {"NLA": v3["NLA_chi2"][s, j]/v3["NLA_dof"][s, j], "TATT": v3["TATT_chi2"][s, j]/v3["TATT_dof"][s, j],
           "halo": hs["halo_chi2"][s, j]/hs["halo_dof"][s, j], "strict": ss["strict_chi2"][s, j]/ss["strict_dof"][s, j]}
    lab = {"NLA": f"NLA ($A_1$={fits['NLA'][1][0]:.2f})",
           "TATT": f"TATT ($A_1$={fits['TATT'][1][0]:.2f}, $A_2$={fits['TATT'][1][1]:.2f}, $b_{{TA}}$={fits['TATT'][1][2]:.2f})",
           "halo": f"halo F21 ($a_1$={xh[0]:.2f}, $a_{{1h}}$={xh[1]:.3f})",
           "strict": f"halo strict ($a_1$={xs[0]:.2f}, $a_{{1h}}$={xs[1]:.3f})"}
    for i in range(3):
        a = AX[i, c]
        sg = -1 if i == 0 else 1
        a.errorbar(k, sg*y[i], e[i], fmt="o", ms=3.5, color="k", lw=0.9, label="FLAMINGO", zorder=5)
        for name in ("NLA", "TATT", "halo", "strict"):
            a.plot(k, sg*fits[name][0][i], color=COL[name], lw=1.8, ls=LS[name],
                   label=f"{lab[name]}, $\\chi^2/\\nu$={chi[name]:.2f}")
        if i > 0:
            a.axhline(Ns, color="0.4", ls="--", lw=1, label=r"$N_s$ (B plateau)" if i == 1 else None)
        a.set_yscale("log")
        a.set_ylabel(TT[i])
        a.axvspan(kg[j], 3.3, color="0.93", zorder=0)
        a.grid(alpha=0.3, which="both")
        if i == 0:
            a.set_title(f"$z={SF.zs[s]:g}$, fit to $k\\leq{kg[j]:.2f}\\,h\\,$Mpc$^{{-1}}$ (grey: not fitted)")
            a.legend(fontsize=7.5, loc="lower left")
        if i == 1:
            hh, ll = a.get_legend_handles_labels()
            a.legend([hh[ll.index(r"$N_s$ (B plateau)")]], [r"$N_s$ (B plateau)"], fontsize=8, loc="upper right")
    AX[0, c].set_ylim(0.5*np.nanmin(-y[0]), 1.6*np.nanmax(-y[0]))
    for i in (1, 2):
        lo = 0.8*min(np.nanmin(y[1]), np.nanmin(y[2]))
        hi = 1.25*max(np.nanmax(y[1] + e[1]), np.nanmax(y[2] + e[2]))
        AX[i, c].set_ylim(lo, hi)
    AX[2, c].set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
for a in AX.ravel():
    a.set_xscale("log")
    a.set_xlim(0.045, 3.3)
    a.xaxis.set_major_locator(ticker.FixedLocator([0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
fig.tight_layout()
name = f"fig_one_snap_z{SF.zs[s]:g}".replace(".", "p")
fig.savefig(os.path.join(HERE, name + ".png"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight")
print(name, "saved")
