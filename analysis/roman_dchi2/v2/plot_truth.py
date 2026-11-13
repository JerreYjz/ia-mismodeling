#!/usr/bin/env python3
"""FLAMINGO data vs the smoothed truth, per snapshot: ratios P_dE/P_NL,
(P_EE - Ns)/P_NL, (P_BB - Ns)/P_NL with error bars, plus the NLA and TATT
truth fits at a few k_max (one panel per snapshot and probe)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "fits.npz")
zs, kg = d["zs"], d["kgrid"]
lab = [r"$P_{\delta E}/P_{NL}$", r"$(P_{EE}-N_s)/P_{NL}$", r"$(P_{BB}-N_s)/P_{NL}$"]
picks = [1, 3, 5, 7]
cm = plt.cm.viridis
for pi, p in enumerate(lab):
    fig, axs = plt.subplots(3, 5, figsize=(22, 12))
    for ax, i in zip(axs.flat, range(len(zs))):
        k = d["k"][i]
        n = d["b"][i]*d["PNL"][i] if pi == 0 else d["PNL"][i]
        off = 0 if pi == 0 else d["Ns_true"][i]
        y = (d["y"][i, pi] - off)/n
        e = d["err"][i, pi]/n
        m = k >= 0.05
        ax.errorbar(k[m], y[m], e[m], fmt="o", ms=3, color="k", label="FLAMINGO")
        ax.plot(k, d["Rtrue"][i, pi], "k-", lw=2, alpha=0.5, label="smoothed truth")
        for j in picks:
            ax.plot(k, d["NLA_R"][i, j, pi], "--", color=cm(j/len(kg)), lw=1.2,
                    label=f"NLA fit k<={kg[j]:.2f}")
            ax.plot(k, d["TATT_R"][i, j, pi], "-", color=cm(j/len(kg)), lw=1.2,
                    label=f"TATT fit k<={kg[j]:.2f}")
        ax.set_xscale("log")
        ax.set_xlim(0.04, 20)
        good = m & (e < 0.5*np.abs(y) + 1e-6)
        if good.any():
            lo, hi = np.percentile(y[good], [0, 100])
            pad = 0.3*(hi - lo) + 1e-4
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"z = {zs[i]:.2f}")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_ylabel(p)
    axs.flat[0].legend(fontsize=6)
    for ax in axs.flat[len(zs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"truth_{['dE', 'EE', 'BB'][pi]}.png", dpi=220)
    fig.savefig(f"truth_{['dE', 'EE', 'BB'][pi]}.pdf")
    plt.close(fig)
