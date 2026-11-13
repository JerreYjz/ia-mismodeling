"""IA amplitudes against k_max, one panel per snapshot, from fits to the raw
FLAMINGO spectra (fit_kmax_scan.py kmax_scan.csv).
Usage: python plot_A_vs_kmax.py kmax_scan.csv out_prefix"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, pre = sys.argv[1], sys.argv[2]
R = list(csv.DictReader(open(path)))
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
OFF = {"NLA": 0.97, "TATT": 1.03}
zs = sorted({(float(r["z"]), int(r["sn"])) for r in R})
for par, models in (("a1", ["NLA", "TATT"]), ("a2", ["TATT"]), ("bTA", ["TATT"])):
    fig, axs = plt.subplots(4, 4, figsize=(17, 13.5))
    for ax, (z, sn) in zip(axs.flat, zs):
        for m in models:
            rows = sorted([r for r in R if int(r["sn"]) == sn and r["model"] == m],
                          key=lambda r: float(r["kmax"]))
            k = np.array([float(r["kmax"]) for r in rows])
            v = np.array([float(r[par]) for r in rows])
            e = np.array([float(r[par + "_err"]) for r in rows])
            ax.errorbar(k*OFF[m], v, e, fmt="o-", ms=3.5, lw=1, capsize=2, color=COL[m], label=m)
        ax.set_xscale("log")
        ax.set_title(f"z = {z:.2f}  (sn{sn})", fontsize=10)
        ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$", fontsize=9)
        ax.set_ylabel(par, fontsize=9)
    axs.flat[0].legend(fontsize=8)
    for ax in axs.flat[len(zs):]:
        ax.axis("off")
    fig.suptitle(f"{par} vs k_max, fits to the FLAMINGO spectra only (P_hE, P_EE, P_BB; "
                 "k_min = 0.03; errors: mode scatter + 2% floor)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{pre}_{par}.png", dpi=105)
    print("wrote", f"{pre}_{par}.png")
