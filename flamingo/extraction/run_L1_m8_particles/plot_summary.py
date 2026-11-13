#!/usr/bin/env python3
"""Summary figures of the FLAMINGO L1_m8 IA spectra (TNG shape definition) against TNG300.

All spectra in the TNG / sim_IA.c conventions (k-perp, shear units, NGP window on EE; folds 1/4/16
stitched; P_dE ~= P_hE / b(k)), from plot_stitch_flamingo.sim_flamingo().

  summary_spectra.png : -P_dE and P_EE per finished redshift, with the NLA fit, and TNG300
                        stitched at the nearest z
  summary_anla.png    : a_nla(z) from the low-k NLA fit, FLAMINGO vs TNG300
"""
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_stitch_flamingo import COSMO, sim_flamingo, tng_curve  # noqa: E402
from stitch import camb_power, fit_lowk_amplitude  # noqa: E402

SNAPS = [78, 76, 74, 72, 70, 68, 66, 63, 58, 48, 38, 18, 14]
SETS = {"all": ("results", "FLAMINGO L1_m8, unreduced tensor of all bound stars (TNG definition)", "#2a78d6", "o")}
INK, INK2, GRID, TNGC = "#0b0b0b", "#52514e", "#e6e5e0", "#1baf7a"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

SNAPS = [sn for sn in SNAPS if os.path.isfile(os.path.join(HERE, "results", "PS", f"Phh_sn{sn}_nfold16.dat"))]
if not SNAPS:
    sys.exit("no finished snapshot yet")
zs = {sn: round(float(h5py.File(os.path.join(HERE, "results", "Galaxies", f"galaxies_{sn}.hdf5"), "r")
                      .attrs["Redshift"]), 3) for sn in SNAPS}
Pnl, Plin = camb_power(COSMO, list(zs.values()), kmax=30.0)

NCOL = min(5, len(SNAPS))  # panels wrap into blocks of NCOL redshifts (dE row + EE row per block)
NBLK = -(-len(SNAPS) // NCOL)
fig, axes = plt.subplots(2 * NBLK, NCOL, figsize=(3.1 * NCOL + 0.8, 7.2 * NBLK), facecolor="#fcfcfb",
                         sharex=True, squeeze=False)
for j in range(len(SNAPS), NBLK * NCOL):
    axes[2 * (j // NCOL), j % NCOL].set_visible(False)
    axes[2 * (j // NCOL) + 1, j % NCOL].set_visible(False)


def ax(i, j):
    return axes[2 * (j // NCOL) + i, j % NCOL]


anla = {s: [] for s in SETS}
for j, sn in enumerate(SNAPS):
    z = zs[sn]
    tng, ztng = tng_curve(z)
    for which, (res, label, col, mk) in SETS.items():
        if not os.path.isfile(os.path.join(HERE, res, "PS", f"Phh_sn{sn}_nfold16.dat")):
            continue
        k, pde, sde, pee, see = sim_flamingo(res, sn, z, Pnl)[0]
        A, sA = fit_lowk_amplitude(k, pde, sde, Pnl(k, z), 0.05, 0.3)
        norm = -0.0134 * COSMO.Om / float(COSMO.growth(z)[0])
        anla[which].append((z, A / norm, abs(sA / norm)))
        kk = np.logspace(-2, np.log10(0.6), 100)
        for i, (P, S, sign) in enumerate(((pde, sde, -1), (pee, see, 1))):
            a = ax(i, j)
            a.errorbar(k, sign * P, S, fmt=mk, ms=3, color=col, lw=0.9, capsize=0, label=label, zorder=3)
            nla = (A * Pnl(kk, z)) if i == 0 else (A ** 2 * Pnl(kk, z))
            a.plot(kk, sign * nla, color=col, ls="--", lw=1.1, alpha=0.9)
    for i in range(2):
        a = ax(i, j)
        if tng is not None:
            a.plot(tng[:, 0], (-tng[:, 4]) if i == 0 else tng[:, 8], color=TNGC, lw=1.6,
                   label="TNG300 stitched (nearest z)")
        a.set_xscale("log")
        a.set_xlim(8e-3, 20)
        a.set_yscale("log")
        a.grid(True, color=GRID, lw=0.5)
    ax(0, j).set_title(f"z = {z:g}  (snapshot {sn})", loc="left", color=INK)
    ax(1, j).set_xlabel("k  [h/Mpc]")
    ax(1, j).xaxis.set_tick_params(labelbottom=True)
for b in range(NBLK):
    axes[2 * b, 0].set_ylabel(r"$-P_{\delta E}$  [(Mpc/h)$^3$]")
    axes[2 * b + 1, 0].set_ylabel(r"$P_{EE}$  [(Mpc/h)$^3$]")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h[:2], l[:2], loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
fig.text(0.01, 0.005, r"$k_\perp$ (μ = 0), shear units (1/2R per shape), NGP window on EE; "
         r"$P_{\delta E} \approx P_{hE}/b(k)$; dashed: NLA fitted to $P_{\delta E}$ at k = 0.05-0.3",
         color=INK2, fontsize=7.5)
fig.tight_layout(rect=(0, 0.02 / NBLK, 1, 1 - 0.05 / NBLK))
fig.savefig(os.path.join(HERE, "results", "plots", "summary_spectra.png"), dpi=130)

# a_nla(z)
fig, a = plt.subplots(figsize=(6.4, 4.2), facecolor="#fcfcfb")
tz = np.loadtxt("/Users/yijiezhu/Downloads/ia_halofit_v2/stitch_demo/stitch_summary.txt", usecols=(1, 2, 3))
tz = tz[tz[:, 0] <= max(zs.values()) + 0.05]
a.errorbar(tz[:, 0], tz[:, 1], tz[:, 2], fmt="-", color=TNGC, lw=1.6, label="TNG300 (stitch_summary.txt)")
for which, (res, label, col, mk) in SETS.items():
    if anla[which]:
        d = np.array(sorted(anla[which]))
        a.errorbar(d[:, 0], d[:, 1], d[:, 2], fmt=mk + "-", ms=5, color=col, lw=1.3, capsize=0, label=label)
a.axhline(0, color=INK2, lw=0.7)
a.set_ylim(0, None)
a.set_xlabel("redshift")
a.set_ylabel(r"$a_{\rm NLA}$  (C$_1\rho_{\rm cr}$ = 0.0134, shear, $k_\perp$)")
a.set_title("NLA amplitude, low-k fit (k = 0.05-0.3 h/Mpc)", loc="left", color=INK, fontsize=9.5)
a.grid(True, color=GRID, lw=0.5)
a.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "plots", "summary_anla.png"), dpi=140)
for which in SETS:
    for z, v, e in sorted(anla[which]):
        print(f"{which:4s} z={z:4.2f}  a_nla = {v:+.3f} +- {e:.3f}")
print("wrote results/plots/summary_spectra.png, summary_anla.png")
