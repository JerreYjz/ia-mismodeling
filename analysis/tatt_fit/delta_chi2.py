#!/usr/bin/env python3
"""Delta chi2 diagnostics vs k_max, per snapshot, from gof_per_snapshot.py's gof.csv.

1. Model comparison: dchi2 = chi2_NLA - chi2_TATT at the same k_max (TATT has
   2 more parameters; under "NLA is enough" dchi2 ~ chi2_2, so dchi2 > 5.99
   means the data prefer TATT at 95%).
2. Incremental: dchi2_i = chi2(k_i) - chi2(k_{i-1}) for each model, the chi2
   added by the new k bin. For nested fits of a valid model it follows chi2
   with dn dof (dn = data points added: 1 for P_hE, 2 for P_hE + P_EE); the
   95% line is chi2_isf(0.05, dn).
Cuts: the last k_max before the statistic crosses its 95% line at two
consecutive k_max.

    python delta_chi2.py gof/gof.csv --out gof
"""
import argparse
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
LS = {"P_hE": "-", "P_hE + P_EE": "--"}
NPROBE = {"P_hE": 1, "P_hE + P_EE": 2}


def first_cross(k, v, thr):
    """Last k before v > thr at two consecutive points (None: never)."""
    for i in range(len(v)):
        nxt = v[i + 1] if i + 1 < len(v) else v[i]
        if v[i] > thr[i] and nxt > (thr[i + 1] if i + 1 < len(v) else thr[i]):
            return k[i - 1] if i > 0 else np.nan
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gof_csv")
    ap.add_argument("--out", default="gof")
    a = ap.parse_args()
    R = collections.defaultdict(dict)
    zz = {}
    for r in csv.DictReader(open(a.gof_csv)):
        sn = int(r["sn"])
        zz[sn] = float(r["z"])
        R[(sn, r["model"], r["probes"])][round(float(r["kmax"]), 4)] = (float(r["chi2"]), int(r["dof"]))
    sns = sorted(zz, key=zz.get)
    probes = list(NPROBE)

    # 1. NLA - TATT
    cut_cmp = {}
    fig, axs = plt.subplots(4, 4, figsize=(17, 13.5))
    for ax, sn in zip(axs.flat, sns):
        for pl in probes:
            n_, t_ = R[(sn, "NLA", pl)], R[(sn, "TATT", pl)]
            ks = sorted(set(n_) & set(t_))
            d = np.array([n_[k][0] - t_[k][0] for k in ks])
            ax.plot(ks, np.maximum(d, 1e-2), LS[pl], marker="o", ms=3, color="k", lw=1.2,
                    label=pl)
            c = first_cross(ks, d, [5.99]*len(ks))
            cut_cmp[(sn, pl)] = c
            if c is not None and np.isfinite(c):
                ax.axvline(c, color="#d1495b", ls=LS[pl], lw=0.9)
        ax.axhline(5.99, color="#d1495b", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e4)
        ax.set_title(f"z = {zz[sn]:.2f}  (sn{sn})", fontsize=10)
        ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$", fontsize=9)
        ax.set_ylabel(r"$\chi^2_{\rm NLA}-\chi^2_{\rm TATT}$", fontsize=9)
    axs.flat[0].legend(fontsize=8)
    for ax in axs.flat[len(sns):]:
        ax.axis("off")
    fig.suptitle("dchi2 = chi2_NLA - chi2_TATT vs k_max (red line: 5.99, 95% for 2 extra "
                 "parameters; vertical: last k_max before the data prefer TATT)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "dchi2_nla_vs_tatt.png"), dpi=105)
    plt.close(fig)

    # 2. incremental
    cut_inc = {}
    fig, axs = plt.subplots(4, 4, figsize=(17, 13.5))
    for ax, sn in zip(axs.flat, sns):
        for pl in probes:
            for m in ("NLA", "TATT"):
                rows = R[(sn, m, pl)]
                ks = sorted(rows)
                c2 = np.array([rows[k][0] for k in ks])
                dof = np.array([rows[k][1] for k in ks])
                inc = np.diff(c2)
                dn = np.maximum(np.diff(dof), 1)
                thr = chi2_dist.isf(0.05, dn)
                kk = ks[1:]
                ax.plot(kk, np.maximum(inc, 1e-2), LS[pl], marker="o", ms=3, color=COL[m],
                        lw=1.2, label=f"{m}, {pl}")
                c = first_cross(kk, inc, thr)
                if c is not None and np.isfinite(c) and c == kk[0]:
                    c = ks[0]
                cut_inc[(sn, m, pl)] = c
                if c is not None and np.isfinite(c):
                    ax.axvline(c, color=COL[m], ls=LS[pl], lw=0.8, alpha=0.7)
        ax.axhline(chi2_dist.isf(0.05, 1), color="0.4", lw=0.8)
        ax.axhline(chi2_dist.isf(0.05, 2), color="0.4", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e4)
        ax.set_title(f"z = {zz[sn]:.2f}  (sn{sn})", fontsize=10)
        ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$", fontsize=9)
        ax.set_ylabel(r"$\chi^2(k_i)-\chi^2(k_{i-1})$", fontsize=9)
    axs.flat[0].legend(fontsize=7)
    for ax in axs.flat[len(sns):]:
        ax.axis("off")
    fig.suptitle("chi2 added by each new k bin (grey: 95% for 1 (solid) and 2 (dashed) added "
                 "points; vertical: last k_max before two consecutive exceedances)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "dchi2_incremental.png"), dpi=105)
    plt.close(fig)

    f = lambda v: "  none" if v is None else ("   nan" if not np.isfinite(v) else f"{v:6.2f}")
    print("last k_max before the data prefer TATT (chi2_NLA - chi2_TATT > 5.99), and before the")
    print("added chi2 per new bin exceeds its 95% line twice in a row  [h/Mpc]")
    print("   z  | NLA-TATT: P_hE  P_hE+EE | incremental NLA: P_hE  P_hE+EE | TATT: P_hE  P_hE+EE")
    with open(os.path.join(a.out, "dchi2_cuts.csv"), "w") as fh:
        fh.write("sn,z,cmp_hE,cmp_hEEE,inc_NLA_hE,inc_NLA_hEEE,inc_TATT_hE,inc_TATT_hEEE\n")
        for sn in sns:
            vals = [cut_cmp[(sn, "P_hE")], cut_cmp[(sn, "P_hE + P_EE")],
                    cut_inc[(sn, "NLA", "P_hE")], cut_inc[(sn, "NLA", "P_hE + P_EE")],
                    cut_inc[(sn, "TATT", "P_hE")], cut_inc[(sn, "TATT", "P_hE + P_EE")]]
            print(f"{zz[sn]:5.2f} |      {f(vals[0])}  {f(vals[1])}  |            {f(vals[2])}  "
                  f"{f(vals[3])}  |      {f(vals[4])}  {f(vals[5])}")
            fh.write(f"{sn},{zz[sn]}," + ",".join("none" if v is None else f"{v:.4f}" for v in vals) + "\n")
    print(f"wrote {a.out}/dchi2_nla_vs_tatt.png, dchi2_incremental.png, dchi2_cuts.csv")


if __name__ == "__main__":
    main()
