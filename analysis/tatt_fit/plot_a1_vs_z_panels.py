"""a1 vs z with error bars, one panel per k_max, NLA and TATT overlaid.
Usage: python plot_a1_vs_z_panels.py fits.csv out_prefix "title"
Writes <out_prefix>_a1.png (a1) and <out_prefix>_frac.png (% change from the
lowest k_max, error = sigma(a1)/a1_ref)."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, pre, title = sys.argv[1], sys.argv[2], sys.argv[3]
R = list(csv.DictReader(open(path)))
ks = sorted({round(float(r["kmax"]), 4) for r in R})
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
OFF = {"NLA": -0.025, "TATT": 0.025}

def series(m, kk):
    rows = sorted([r for r in R if r["model"] == m and abs(float(r["kmax"]) - kk) < 1e-3],
                  key=lambda r: float(r["z"]))
    return (np.array([float(r["z"]) for r in rows]), np.array([float(r["a1"]) for r in rows]),
            np.array([float(r["a1_err"]) for r in rows]))

ref = {m: series(m, ks[0]) for m in COL}
nc = 4
nr = int(np.ceil(len(ks)/nc))
for kind in ("a1", "frac"):
    fig, axs = plt.subplots(nr, nc, figsize=(4.4*nc, 3.3*nr), sharex=True,
                            sharey=(kind == "frac"))
    axs = np.atleast_2d(axs)
    for i, kk in enumerate(ks):
        ax = axs.flat[i]
        for m in COL:
            z, a, e = series(m, kk)
            if kind == "a1":
                ax.errorbar(z + OFF[m], a, e, fmt="o-", ms=3.5, lw=1, capsize=2,
                            color=COL[m], label=m)
            else:
                zr, ar, _ = ref[m]
                a0 = np.array([ar[np.argmin(abs(zr - x))] for x in z])
                ax.errorbar(z + OFF[m], 100*(a/a0 - 1), 100*e/a0, fmt="o-", ms=3.5, lw=1,
                            capsize=2, color=COL[m], label=m)
                ax.axhline(0, color="k", lw=0.7)
                ax.axhspan(-1, 1, color="0.85", zorder=0)
        ax.set_title(f"k_max = {kk:.2f} h/Mpc", fontsize=10)
        if i % nc == 0:
            ax.set_ylabel("a1" if kind == "a1" else f"a1 / a1(k_max={ks[0]:.2f}) - 1 [%]")
        if i >= len(ks) - nc:
            ax.set_xlabel("z")
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(ks), nr*nc):
        axs.flat[j].axis("off")
    if kind == "frac":
        lim = 0
        for m in COL:
            for kk in ks:
                z, a, e = series(m, kk)
                zr, ar, _ = ref[m]
                a0 = np.array([ar[np.argmin(abs(zr - x))] for x in z])
                lim = max(lim, np.max(np.abs(100*(a/a0 - 1)) + 100*e/a0))
        for ax in axs.flat:
            ax.set_ylim(-min(lim, 60), min(lim, 60))
    fig.suptitle(title + ("" if kind == "a1" else "   (grey band: +-1%)"), fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{pre}_{kind}.png", dpi=110)
    print("wrote", f"{pre}_{kind}.png")
