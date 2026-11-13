"""Overlay the trend-onset k_max(z) of several trend_cut.py runs."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
sys.path.insert(0, ".")
from fit_stitched_kmax import KT

runs = [a.split(":") for a in sys.argv[2:]]
out = sys.argv[1]
OM = 0.306
E = lambda a: np.sqrt(OM/a**3 + 1 - OM)
g = lambda a: 2.5*OM*E(a)*quad(lambda x: 1/(x*E(x))**3, 1e-6, a)[0]
zz = np.linspace(0, 4, 100); Dz = np.array([g(1/(1 + x))/g(1) for x in zz])
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
cols = ["#264653", "#e76f51", "#2a9d8f", "#8338ec"]
for ax, m in zip(axs, ("NLA", "TATT")):
    for c, (path, lab) in zip(cols, runs):
        R = sorted(csv.DictReader(open(path)), key=lambda r: float(r["z"]))
        z = np.array([float(r["z"]) for r in R]); k = np.array([float(r[f"k_{m}"]) for r in R])
        nn = np.array([r[f"kind_{m}"] == "none" for r in R])
        ax.plot(z, k, "-o", ms=4, color=c, label=lab)
        ax.plot(z[nn], k[nn], "^", ms=9, mfc="none", color=c)
    zs = sorted(KT, key=lambda s: -s)
    ax.plot(z, [KT[int(r["sn"])] for r in R], "-", lw=1, color="0.6", label="stitch k_t")
    ax.plot(zz, 0.35/Dz, ":", color="0.4", label="0.35/D(z)")
    ax.set_yscale("log"); ax.set_xlabel("z"); ax.set_title(f"{m}: a1 leaves its plateau")
    ax.legend(fontsize=8)
axs[0].set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
fig.text(0.5, 0.01, "open triangles: stable to the end of the k grid (lower bound)", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)
