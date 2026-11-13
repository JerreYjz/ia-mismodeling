#!/usr/bin/env python3
"""fig_white_ii: FLAMINGO's white II excess, its Roman signature, and the FoB
decomposition (single tidal field / plus white excess / full)."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


def main():
    f = np.load("../v2/fits.npz")
    zs, kd = f["zs"], f["k"][0]
    y, err, b, Pf, Ns = f["y"], f["err"], f["b"], f["PNL"], f["Ns_true"]
    cov = np.load("analysis/cov_gauss_d3a.npy")
    sd = np.sqrt(np.diag(cov))
    W = np.load("run/xi_white_only.npz")["ratio_ratio"]
    dec = json.load(open("analysis/bias_white.json"))
    v2 = json.load(open("analysis/bias_v2.json"))["fits"]
    X = (y[:, 1] - Ns[:, None]) - (y[:, 0]/b)**2/Pf
    dX = np.sqrt(err[:, 1]**2 + (2*(y[:, 0]/b)*err[:, 0]/b/Pf)**2)
    sel = (kd >= 0.05) & (kd <= 0.5)
    Xw = np.array([np.sum(X[s, sel]/dX[s, sel]**2)/np.sum(1/dX[s, sel]**2) for s in range(13)])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    show = [2, 5, 7, 8, 9, 10]
    for s in show:
        c = cm.viridis(s/12)
        m = (kd >= 0.05) & (kd <= 1.0)
        ax[0].errorbar(kd[m], X[s, m]/Ns[s], dX[s, m]/Ns[s], fmt="o", ms=3, color=c, capsize=2, label=f"z = {zs[s]:g}")
        ax[0].axhline(Xw[s]/Ns[s], color=c, lw=0.8, ls="--")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("k [h/Mpc]")
    ax[0].set_ylabel(r"$[(P_{EE}-N_s) - P_{\delta E}^2/P_{\rm NL}]\,/\,N_s$")
    ax[0].set_title("II excess over the GI-implied II, in units of the\nwhite level $N_s$ (from $P_{BB}$); dashed: per-snapshot mean")
    ax[0].set_xticks([0.05, 0.1, 0.2, 0.5, 1])
    ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)

    TH = np.sqrt(np.geomspace(2.5, 250, 16)[:-1]*np.geomspace(2.5, 250, 16)[1:])
    r = (W/sd).reshape(2, 36, 15)
    PAIRS = [(i, j) for i in range(8) for j in range(i, 8)]
    for ip, (i, j) in enumerate(PAIRS):
        if i == j:
            ax[1].plot(TH, np.abs(r[1, ip]), "-", color=cm.viridis(i/7), lw=1.2, label=f"$\\xi_-$ ({i},{i})")
    ax[1].plot(TH, np.median(np.abs(r[0]), axis=0), "k:", label=r"$\xi_+$ (median, all pairs)")
    ax[1].plot(TH, np.abs(r[1, 8, 0])*(TH/TH[0])**-2, "k--", lw=0.8, label=r"$\propto\theta^{-2}$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-4, 20)
    ax[1].set_xlabel(r"$\theta$ [arcmin]")
    ax[1].set_ylabel(r"$|\xi^{\rm white\,II}| / \sigma_{\rm Roman}$")
    ax[1].set_title("Roman signature of the white II excess alone\n(cosmolike; a constant $C_\\ell$ gives $\\xi_- = 2C/\\pi\\theta^2$, $\\xi_+ = 0$)")
    ax[1].set_xticks([3, 10, 30, 100, 250])
    ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax[1].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3, which="both")

    masks = [0.25, 0.35, 0.5]
    lab = {"GI_II1": "single tidal field: FLAMINGO GI, II = GI$^2$/$P_{\\rm NL}$",
           "GI_II1_W": "+ white II excess (exactly white)",
           "GI_II1_Wk": "full FLAMINGO (production)"}
    mk = {"GI_II1": "o", "GI_II1_W": "s", "GI_II1_Wk": "^"}
    ls = {"GI_II1": ":", "GI_II1_W": "--", "GI_II1_Wk": "-"}
    for mdl in ("NLA", "TATT"):
        for var in ("GI_II1", "GI_II1_W", "GI_II1_Wk"):
            fob = [[d for d in dec if d["model"] == mdl and d["variant"] == var and d["mask"] == K0][0]["flat"]["FoB"] for K0 in masks]
            ax[2].plot(masks, fob, ls[var], marker=mk[var], ms=5, color=COL[mdl], label=f"{mdl}: {lab[var]}")
    ax[2].axhline(0.3, color="k", lw=0.8, ls="--")
    ax[2].set_xlabel(r"mask $K_0$: $k_{\rm UV} = K_0/D(z)$ [h/Mpc]")
    ax[2].set_ylabel(r"figure of bias in $(\Omega_m, S_8)$ [$\sigma$]")
    ax[2].set_title("FoB decomposition (case A)")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("analysis/fig_white_ii.png", dpi=220)
    fig.savefig("analysis/fig_white_ii.pdf")


if __name__ == "__main__":
    main()
