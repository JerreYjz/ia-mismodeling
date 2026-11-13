#!/usr/bin/env python3
"""k_max summary figures (FLAMINGO side and the translation to Roman).

fig_kmax_gof      goodness of fit of NLA / TATT to the measured FLAMINGO
                  spectra (P_hE, P_EE, P_BB, 0.05 <= k <= k_max) vs k_max:
                  per snapshot, and combined (all z; z = 0.5-2, where Roman
                  looks).
fig_kmax_dchi2    Delta chi2_FL (best fit vs noise-free FLAMINGO, FLAMINGO
                  errors, in-fit k <= k_max) vs k_max, and the Roman
                  Delta chi2 (cosmolike, E modes, production masks) of the
                  same fits, and their ratio.
fig_kmax_weight   Roman Delta chi2 per unit FLAMINGO chi2 placed at one k
                  node (summed over snapshots and over P_dE, P_EE), per mask:
                  where Roman actually looks.
Tables: analysis/kmax_summary.csv
"""
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cosmolike_driver import Cosmolike   # noqa: E402
from cosmolike_xi_ia import KMIN          # noqa: E402

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
MASKS = (0.25, 0.35, 0.5)
LS = {0.25: ":", 0.35: "-", 0.5: "--"}


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def main():
    data = sys.argv[1]
    f = np.load("../v2/fits.npz")
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    zs, kg, kd = f["zs"], f["kgrid"], f["k"][0]
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    import pandas as pd
    roman = pd.read_csv("analysis/dchi2_masks.csv")
    sel_mid = (zs >= 0.5) & (zs <= 2.0)
    cmap = cm.viridis

    # ---------------- GoF ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, mdl in enumerate(("NLA", "TATT")):
        for s, z in enumerate(zs):
            ax[i].plot(kg, f[f"{mdl}_p_raw"][s], "-o", ms=3, lw=1.2, color=cmap(s/12), label=f"z={z:g}")
        ax[i].axhline(0.05, color="k", lw=0.8, ls="--")
        ax[i].set_title(f"{mdl}: p-value per snapshot")
    ax[0].legend(fontsize=7, ncol=2, loc="lower left")
    comb = {}
    for mdl in ("NLA", "TATT"):
        c2, dof = f[f"{mdl}_chi2_raw"], f[f"{mdl}_dof"]
        for lab, sel, ls in (("all z", np.ones(13, bool), "-"), ("z = 0.5-2", sel_mid, "--")):
            p = chi2_dist.sf(c2[sel].sum(0), dof[sel].sum(0))
            comb[(mdl, lab)] = (c2[sel].sum(0)/dof[sel].sum(0), p)
            ax[2].plot(kg, p, ls, color=COL[mdl], marker="o", ms=3, label=f"{mdl}, {lab}")
    ax[2].axhline(0.05, color="k", lw=0.8, ls="--")
    ax[2].set_title("combined p-value")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_ylim(1e-6, 1.5)
        a.set_xlabel(r"$k_{\max}$ of the fit [h/Mpc]")
        a.set_xticks([0.12, 0.2, 0.3, 0.5, 0.7, 1, 2, 3])
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        a.xaxis.set_minor_formatter(ticker.NullFormatter())
        a.grid(alpha=0.3, which="both")
    ax[0].set_ylabel("p-value (chi2 of the fit to the measured spectra)")
    fig.tight_layout()
    save(fig, "analysis/fig_kmax_gof")

    # ---------------- Delta chi2_FL and Roman ----------------
    dfl = {}
    for mdl in ("NLA", "TATT"):
        d = f[f"{mdl}_dFL"]                            # (13, jfit, jK)
        infit = np.array([d[:, j, j] for j in range(len(kg))]).T   # (13, jfit)
        dfl[(mdl, "all z")] = infit.sum(0)
        dfl[(mdl, "z = 0.5-2")] = infit[sel_mid].sum(0)
    rom = {}
    for mdl in ("NLA", "TATT"):
        rows = roman[roman["model"] == mdl]
        kk = rows["kmax"].values
        for K0 in MASKS:
            rom[(mdl, K0)] = np.interp(kg, kk, rows[f"n1_{K0:g}"].values)
        rom[(mdl, "all")] = np.interp(kg, kk, rows["all"].values)
        rom[(mdl, "matched")] = np.array([rows[f"n0_{k:.3f}"].values[np.argmin(np.abs(kk - k))] for k in kg])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for mdl in ("NLA", "TATT"):
        ax[0].plot(kg, dfl[(mdl, "all z")], "-o", ms=3, color=COL[mdl], label=f"{mdl}, all z")
        ax[0].plot(kg, dfl[(mdl, "z = 0.5-2")], "--o", ms=3, color=COL[mdl], label=f"{mdl}, z = 0.5-2")
        for K0 in MASKS:
            ax[1].plot(kg, rom[(mdl, K0)], LS[K0], marker="o", ms=3, color=COL[mdl], label=f"{mdl}, mask {K0}/D(z)")
            ax[2].plot(kg, rom[(mdl, K0)]/dfl[(mdl, "all z")], LS[K0], marker="o", ms=3, color=COL[mdl])
        ax[1].plot(kg, rom[(mdl, "matched")], "-.", marker="s", ms=3, color=COL[mdl], alpha=0.6,
                   label=f"{mdl}, const-k mask at k_max")
    ax[0].set_ylabel(r"$\Delta\chi^2_{\rm FL}$ (in fit, 13 snapshots, FLAMINGO errors)")
    ax[0].set_title("FLAMINGO: best fit vs noise-free spectra")
    ax[1].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ ($\xi_\pm$, IA amplitudes from the FLAMINGO fit)")
    ax[1].set_title("Roman: same fits, E modes, cosmolike")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[2].set_ylabel(r"$\Delta\chi^2_{\rm Roman}\,/\,\Delta\chi^2_{\rm FL}$")
    ax[2].set_title("translation ratio (depends on the residual, not a constant)")
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(r"$k_{\max}$ of the FLAMINGO fit [h/Mpc]")
        a.set_xticks([0.12, 0.2, 0.3, 0.5, 0.7, 1, 2, 3])
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        a.xaxis.set_minor_formatter(ticker.NullFormatter())
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "analysis/fig_kmax_dchi2")

    # ---------------- where Roman looks: per-k weight ----------------
    J = np.load("analysis/J_cells.npy")
    cl = Cosmolike(os.path.join(HERE, "lib"), data)
    h = cl.h
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    bk, Pf, err = f["b"], f["PNL"], f["err"]
    cells, sig = [], []
    for s in range(13):
        for x in range(2):
            conv = PNLd[s]/(bk[s]*Pf[s]) if x == 0 else PNLd[s]/Pf[s]
            for i in nodes:
                cells.append((s, x, i))
                sig.append(err[s, x, i]*conv[i])
    sig = np.array(sig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    wk = {}
    for K0 in MASKS:
        m = M1[K1.index(K0)]
        Ci = np.linalg.inv(cov[np.ix_(m, m)])
        Jm = J[m]*sig[None, :]
        diagA = np.einsum("ij,ij->j", Jm, Ci @ Jm)          # Roman chi2 per unit FLAMINGO chi2, per cell
        w = np.zeros(len(nodes))
        wz = np.zeros(13)
        for c, (s, x, i) in enumerate(cells):
            w[list(nodes).index(i)] += diagA[c]
            wz[s] += diagA[c]
        wk[K0] = (w, wz)
        ax[0].plot(kd[nodes], w, LS[K0], marker="o", ms=3, color="k", label=f"mask {K0}/D(z)")
        ax[1].plot(zs, wz, LS[K0], marker="o", ms=3, color="k", label=f"mask {K0}/D(z)")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("FLAMINGO k node [h/Mpc]")
    ax[0].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ per unit $\chi^2_{\rm FL}$ (sum over z, $P_{\delta E}$ + $P_{EE}$)")
    ax[1].set_xlabel("snapshot z")
    ax[1].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ per unit $\chi^2_{\rm FL}$ (sum over k)")
    for a in ax:
        a.set_yscale("log")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    ax[0].set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax[0].set_title("a 1-sigma FLAMINGO residual at one k: what Roman sees")
    fig.tight_layout()
    save(fig, "analysis/fig_kmax_weight")

    # ---------------- table ----------------
    with open("analysis/kmax_summary.csv", "w") as fh:
        fh.write("model,kmax,chi2dof_all,p_all,chi2dof_mid,p_mid,dchi2FL_all,dchi2FL_mid,"
                 + ",".join(f"roman_{K0:g}D" for K0 in MASKS) + ",roman_all,roman_constk_matched\n")
        for mdl in ("NLA", "TATT"):
            for j, k in enumerate(kg):
                fh.write(f"{mdl},{k:.3f},{comb[(mdl, 'all z')][0][j]:.3f},{comb[(mdl, 'all z')][1][j]:.3g},"
                         f"{comb[(mdl, 'z = 0.5-2')][0][j]:.3f},{comb[(mdl, 'z = 0.5-2')][1][j]:.3g},"
                         f"{dfl[(mdl, 'all z')][j]:.2f},{dfl[(mdl, 'z = 0.5-2')][j]:.2f},"
                         + ",".join(f"{rom[(mdl, K0)][j]:.3f}" for K0 in MASKS)
                         + f",{rom[(mdl, 'all')][j]:.3f},{rom[(mdl, 'matched')][j]:.3f}\n")
    with open("analysis/kmax_weight.csv", "w") as fh:
        fh.write("k," + ",".join(f"w_{K0:g}D" for K0 in MASKS) + "\n")
        for n, i in enumerate(nodes):
            fh.write(f"{kd[i]:.4f}," + ",".join(f"{wk[K0][0][n]:.3e}" for K0 in MASKS) + "\n")
        fh.write("z," + ",".join(f"wz_{K0:g}D" for K0 in MASKS) + "\n")
        for s in range(13):
            fh.write(f"{zs[s]:g}," + ",".join(f"{wk[K0][1][s]:.3e}" for K0 in MASKS) + "\n")
    print(open("analysis/kmax_summary.csv").read())
    print(open("analysis/kmax_weight.csv").read())


if __name__ == "__main__":
    main()
