"""a1 against z for every k_max, NLA and TATT, from a fits csv
(fit_stitched_kmax.py fits.csv or fit_kmax_scan.py kmax_scan.csv)."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
models = sys.argv[4].split(",") if len(sys.argv) > 4 else ["NLA", "TATT"]
R = list(csv.DictReader(open(path)))
ks = sorted({round(float(r["kmax"]), 4) for r in R})
cmap = plt.get_cmap("plasma")
fig, axs = plt.subplots(2, len(models), figsize=(6.5*len(models), 8.5), sharex=True,
                        gridspec_kw=dict(height_ratios=[1.6, 1]))
for j, m in enumerate(models):
    rm = [r for r in R if r["model"] == m]
    zs = sorted({float(r["z"]) for r in rm})
    ref = {}
    for i, kk in enumerate(ks):
        rows = sorted([r for r in rm if abs(float(r["kmax"]) - kk) < 1e-3], key=lambda r: float(r["z"]))
        if not rows:
            continue
        z = np.array([float(r["z"]) for r in rows])
        a = np.array([float(r["a1"]) for r in rows])
        e = np.array([float(r["a1_err"]) for r in rows])
        c = cmap(i/(len(ks) - 1))
        dz = (i - len(ks)/2)*0.006
        axs[0, j].errorbar(z + dz, a, e, fmt="o-", ms=3, lw=1, color=c, elinewidth=0.8,
                           label=f"k_max = {kk:.2f}")
        if not ref:
            ref = dict(zip(z, a))
        axs[1, j].plot(z, 100*(a/np.array([ref[x] for x in z]) - 1), "o-", ms=3, lw=1, color=c)
    axs[0, j].set_title(f"{m}: a1 vs z, one line per k_max")
    axs[0, j].set_ylabel(r"$a_1$  (cosmolike, $C_1\rho_{\rm cr} = 0.01389$)")
    axs[1, j].set_ylabel(f"a1 / a1(k_max = {ks[0]:.2f}) - 1  [%]")
    axs[1, j].axhline(0, color="k", lw=0.8)
    axs[1, j].set_xlabel("z")
    axs[0, j].legend(fontsize=6.5, ncol=2, loc="upper left")
fig.suptitle(title, fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
