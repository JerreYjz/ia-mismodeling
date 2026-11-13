#!/usr/bin/env python3
"""k_max from the A1-stability criterion, from a fit_kmax_scan.py run.

The cut is the first k_max where a1 leaves the n-sigma band of the reference
fit (k <= k_ref), interpolated in ln k between grid points and counted only
if the next k_max is outside the band too. Uses the fits in kmax_scan.csv,
so no refitting; run fit_kmax_scan.py with --with-branch to include TATT-I.

    python plot_a1_criterion.py kmax_scan/kmax_scan.csv --nsig 1 --out a1_crit

Writes a1_crit/a1_criterion.pdf (summary page, then one page per snapshot),
a1_crit/a1_summary.png and a1_crit/kmax_z_a1_<model>.txt (running median
over 3 snapshots, input for make_uv_scale_cuts.py --kz-file).
"""
import argparse
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8", "TATT-I": "#2a9d8f"}
LABEL = {"NLA": "NLA", "TATT": "TATT (global best fit)",
         "TATT-I": "TATT-I (branch I, a2 <= 0)"}


def load(path):
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[(int(r["sn"]), r["model"])].append(
            dict(z=float(r["z"]), k=float(r["kmax"]), a1=float(r["a1"]),
                 e=float(r["a1_err"])))
    return rows


def a1_cut(rs, kref, nsig):
    """(k_cut, a1_ref, sig_ref); k_cut = inf if a1 never leaves the band."""
    k = np.array([r["k"] for r in rs])
    a = np.array([r["a1"] for r in rs])
    i0 = int(np.argmin(np.abs(k - kref)))
    a0, s0 = a[i0], rs[i0]["e"]
    dev = np.abs(a - a0)/s0
    for i in range(i0 + 1, len(k)):
        if dev[i] > nsig and (i + 1 == len(k) or dev[i + 1] > nsig):
            t = (nsig - dev[i - 1])/(dev[i] - dev[i - 1])
            return float(np.exp(np.log(k[i - 1]) + np.clip(t, 0, 1)
                                * (np.log(k[i]) - np.log(k[i - 1])))), a0, s0
    return np.inf, a0, s0


def running_median(z, kc, kcap, w=3):
    o = np.argsort(z)
    lk = np.log(np.minimum(kc[o], kcap))     # "never" -> end of the grid
    out = np.empty(len(z))
    for i in range(len(z)):
        out[o[i]] = np.exp(np.median(lk[max(0, i - w//2):i + w//2 + 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--nsig", type=float, default=1.0)
    ap.add_argument("--kref", type=float, default=0.16)
    ap.add_argument("--out", default="a1_crit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = load(a.csv)
    models = [m for m in ("NLA", "TATT", "TATT-I") if any(k[1] == m for k in rows)]
    sns = sorted({k[0] for k in rows}, key=lambda s: rows[(s, models[0])][0]["z"])
    z = np.array([rows[(s, models[0])][0]["z"] for s in sns])
    kcap = max(r["k"] for rs in rows.values() for r in rs)
    cut = {m: np.array([a1_cut(rows[(s, m)], a.kref, a.nsig)[0] for s in sns])
           for m in models}
    med = {m: running_median(z, cut[m], kcap) for m in models}

    print(f"k_max where a1 leaves the {a.nsig:g} sigma band of the k <= {a.kref} fit")
    print("   z   " + "".join(f"{m:>10s}" for m in models))
    for i, s in enumerate(sns):
        print(f"{z[i]:5.2f}  " + "".join(
            f"{cut[m][i]:10.2f}" if np.isfinite(cut[m][i]) else f"{'never':>10s}"
            for m in models))

    pdf = PdfPages(os.path.join(a.out, "a1_criterion.pdf"))
    # --- summary --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    o = np.argsort(z)
    for m in models:
        kc = cut[m]
        fin = np.isfinite(kc)
        ax.plot(z[fin], kc[fin], "o", color=COL[m], alpha=0.55)
        ax.plot(z[~fin], np.full((~fin).sum(), kcap), "^", color=COL[m], alpha=0.55)
        ax.plot(z[o], med[m][o], "-", lw=2, color=COL[m], label=LABEL[m])
    zz = np.linspace(0, z.max(), 200)
    ax.axhline(kcap, color="0.8", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    ax.set_title(rf"$a_1$ leaves the {a.nsig:g}$\sigma$ band of its "
                 rf"$k\leq{a.kref}$ value (lines: running median of 3; "
                 r"$\triangle$: never, within grid)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "a1_summary.png"), dpi=130)
    pdf.savefig(fig)
    plt.close(fig)

    # --- one page per snapshot -------------------------------------------
    for i, s in enumerate(sns):
        fig, axs = plt.subplots(1, len(models), figsize=(5*len(models), 3.8),
                                sharex=True)
        for axm, m in zip(np.atleast_1d(axs), models):
            rs = rows[(s, m)]
            k = np.array([r["k"] for r in rs])
            v = np.array([r["a1"] for r in rs])
            e = np.array([r["e"] for r in rs])
            kc, a0, s0 = a1_cut(rs, a.kref, a.nsig)
            axm.axhspan(a0 - a.nsig*s0, a0 + a.nsig*s0, color=COL[m], alpha=0.15,
                        label=rf"{a.nsig:g}$\sigma$ band at $k\leq{a.kref}$")
            inside = np.abs(v - a0) <= a.nsig*s0
            axm.errorbar(k, v, e, fmt="none", ecolor=COL[m], alpha=0.6)
            axm.plot(k[inside], v[inside], "o", color=COL[m])
            axm.plot(k[~inside], v[~inside], "o", mfc="white", color=COL[m])
            if np.isfinite(kc):
                axm.axvline(kc, color=COL[m], ls="--", lw=1.2,
                            label=f"k_max = {kc:.2f}")
            axm.axvline(a.kref, color="0.5", ls=":", lw=1)
            axm.set_xscale("log")
            axm.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
            axm.set_title(LABEL[m] + (f": k_max {kc:.2f}" if np.isfinite(kc)
                                      else ": never leaves"), fontsize=10)
            axm.legend(fontsize=7, loc="best")
        np.atleast_1d(axs)[0].set_ylabel(r"$a_1$")
        fig.suptitle(f"sn{s}  z = {z[i]:.2f}   (open: outside the band)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()

    for m in models:
        np.savetxt(os.path.join(a.out, f"kmax_z_a1_{m.lower()}.txt"),
                   np.column_stack([z[o], med[m][o]]), fmt="%.3f %.4f",
                   header=f"z  k_max [h/Mpc]: a1 {a.nsig:g} sigma criterion, "
                          "running median of 3 snapshots")
    print(f"wrote {a.out}/a1_criterion.pdf, a1_summary.png, kmax_z_a1_<model>.txt")


if __name__ == "__main__":
    main()
