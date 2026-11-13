#!/usr/bin/env python3
"""Compare k_UV / D(z) with the TATT k_max(z) from the FLAMINGO fits.

Reads kmax_cut.csv from fit_kmax_scan.py runs (one per error floor) and plots
the per-snapshot TATT k_max, its running median, k_UV/D(z) for the chosen
k_UV, and the best fit of the form k0/D(z)^n (free n, and n = 1). Snapshots
where TATT passes to the end of the k grid ("none") are lower bounds: shown
as triangles, left out of the fits.

    python plot_kuv_vs_tatt.py kmax_scan/kmax_cut.csv:2% \\
        kmax_scan_sys5/kmax_cut.csv:5% --kuv 0.35 --out kuv_vs_tatt.png
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

OM = 0.306


def growth(z):
    E = lambda a: np.sqrt(OM/a**3 + 1 - OM)
    g = lambda a: 2.5*OM*E(a)*quad(lambda x: 1/(x*E(x))**3, 1e-6, a)[0]
    return np.array([g(1/(1 + zz))/g(1.0) for zz in np.atleast_1d(z)])


def read(path):
    rows = list(csv.DictReader(open(path)))
    z = np.array([float(r["z"]) for r in rows])
    o = np.argsort(z)
    get = lambda key: np.array([r[key] for r in rows])[o]
    return dict(z=z[o], k=get("k_TATT").astype(float), why=get("why_TATT"),
                med=get("k_smooth_TATT").astype(float),
                kn=get("k_NLA").astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="kmax_cut.csv[:label]")
    ap.add_argument("--kuv", type=float, nargs="+", default=[0.35])
    ap.add_argument("--zmin-fit", type=float, default=0.0)
    ap.add_argument("--out", default="kuv_vs_tatt.png")
    a = ap.parse_args()

    zz = np.linspace(0, 4, 200)
    Dzz = growth(zz)
    cols = ["#1f5fa8", "#7b2cbf", "#2a9d8f"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
    for i, run in enumerate(a.runs):
        path, _, lab = run.partition(":")
        d = read(path)
        lab = lab or path
        c = cols[i % len(cols)]
        D = growth(d["z"])
        lower = d["why"] == "none"
        fit = (~lower) & (d["why"] != "first") & (d["z"] >= a.zmin_fit - 1e-9)
        ax[0].plot(d["z"][~lower], d["k"][~lower], "o", color=c, alpha=0.6,
                   label=f"TATT k_max, {lab} floor")
        ax[0].plot(d["z"][lower], d["k"][lower], "^", color=c, alpha=0.6)
        ax[0].plot(d["z"], d["med"], "-", color=c, lw=2,
                   label=f"running median ({lab})")
        x, y = -np.log(D[fit]), np.log(d["k"][fit])
        n, lk0 = np.polyfit(x, y, 1)
        k1 = np.exp(np.mean(y - x))                   # n = 1: k0/D(z)
        sc_free = np.std(y - (lk0 + n*x))
        sc_one = np.std(y - (np.log(k1) + x))
        ax[0].plot(zz, np.exp(lk0)*Dzz**(-n), "--", color=c, lw=1,
                   label=f"fit {np.exp(lk0):.2f} D^-{n:.2f} (rms {sc_free:.2f} in ln k)")
        ax[0].plot(zz, k1/Dzz, ":", color=c, lw=1.3,
                   label=f"fit {k1:.2f}/D(z) (rms {sc_one:.2f} in ln k)")
        for kuv in a.kuv:
            r = d["med"]/(kuv/D)
            ax[1].plot(d["z"], r, "-o", ms=3, color=c,
                       label=f"median {lab} / ({kuv:g}/D)")
        print(f"[{lab}] TATT k_max = {np.exp(lk0):.3f} D^-{n:.2f} (rms {sc_free:.2f}); "
              f"n=1: {k1:.3f}/D (rms {sc_one:.2f}); {fit.sum()} snapshots in fit, "
              f"{lower.sum()} lower bounds")
        for kuv in a.kuv:
            r = d["med"]/(kuv/D)
            print("   median/(%g/D): " % kuv + " ".join(
                f"{zi:.2f}:{ri:.2f}" for zi, ri in zip(d["z"], r)))
        if i == 0:
            ax[0].plot(d["z"], d["kn"], "s", ms=4, color="#d1495b", alpha=0.45,
                       label=f"NLA k_max ({lab}), for reference")
    for j, kuv in enumerate(a.kuv):
        ax[0].plot(zz, kuv/Dzz, "-", color="k", lw=1.6 if j == 0 else 1,
                   alpha=1 if j == 0 else 0.5, label=f"k_UV/D(z), k_UV = {kuv:g}")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    ax[0].legend(fontsize=7, ncol=2, loc="upper left")
    ax[0].set_title(r"TATT $k_{\rm max}$ from FLAMINGO vs $k_{\rm UV}/D(z)$"
                    r"  ($\triangle$: passes to the end of the grid)", fontsize=10)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel(r"$k_{\rm max}^{\rm TATT}\,/\,[k_{\rm UV}/D(z)]$")
    ax[1].set_xlabel("z")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
