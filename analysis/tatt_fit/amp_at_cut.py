#!/usr/bin/env python3
"""Per-snapshot IA amplitudes at each snapshot's own scale cut.

Reads gof.csv and gof_cuts.csv of gof_per_snapshot.py (per-snapshot fits,
amplitudes only: NLA a1; TATT a1, a2, b_TA; no redshift power law) and
reports, for each snapshot, the amplitudes and errors at the last k_max
before the fit fails (Delta chi2 above the 95% value of chi2_dof at two
consecutive k_max). Snapshots that never fail use the last k_max (lower
bound, '+'); a snapshot that fails from the first k_max uses the first
(flagged '!').

    python amp_at_cut.py gof_flerr --probes "P_hE + P_EE"

Writes <dir>/amp_at_cut.csv, A_z_<model>.txt (z, a1[, a2, b_TA] at the cut)
and amp_<par>_vs_kmax.png (one panel per snapshot, cut marked).
"""
import argparse
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
C1_COSMOLIKE, C1_REPORT = 0.01389, 0.0134


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--probes", default="P_hE + P_EE", choices=["P_hE", "P_hE + P_EE"])
    a = ap.parse_args()
    rows = collections.defaultdict(list)
    zz = {}
    for r in csv.DictReader(open(os.path.join(a.dir, "gof.csv"))):
        if r["probes"] != a.probes:
            continue
        sn = int(r["sn"])
        zz[sn] = float(r["z"])
        rows[(sn, r["model"])].append({k: (float(v) if k not in ("model", "probes") else v)
                                       for k, v in r.items()})
    cuts = {}
    for r in csv.DictReader(open(os.path.join(a.dir, "gof_cuts.csv"))):
        for m in ("NLA", "TATT"):
            cuts[(int(r["sn"]), m)] = r[f"{m} {a.probes}"]
    sns = sorted(zz, key=zz.get)

    out = {}
    for sn in sns:
        for m in ("NLA", "TATT"):
            rs = sorted(rows[(sn, m)], key=lambda r: r["kmax"])
            c = cuts[(sn, m)]
            if c == "none":
                r, flag = rs[-1], "+"
            elif c == "nan":
                r, flag = rs[0], "!"
            else:
                r = min(rs, key=lambda x: abs(x["kmax"] - float(c)))
                flag = ""
            out[(sn, m)] = (r, flag)

    print(f"Amplitudes at each snapshot's cut ({a.probes}; cosmolike units, C1 rho_cr = "
          f"{C1_COSMOLIKE}; aNLA in the notes' units = a1 x {C1_COSMOLIKE/C1_REPORT:.4f})")
    print("   z   | NLA: k_max   a1              | TATT: k_max   a1              a2               b_TA")
    with open(os.path.join(a.dir, "amp_at_cut.csv"), "w") as fh:
        fh.write("sn,z,model,kmax,flag,a1,a1_err,a2,a2_err,bTA,bTA_err\n")
        for sn in sns:
            rn, fn = out[(sn, "NLA")]
            rt, ft = out[(sn, "TATT")]
            print(f"{zz[sn]:5.2f}  |  {rn['kmax']:5.2f}{fn:1s}  {rn['a1']:6.3f} +- {rn['a1_err']:.3f}  |"
                  f"  {rt['kmax']:5.2f}{ft:1s}  {rt['a1']:6.3f} +- {rt['a1_err']:.3f}  "
                  f"{rt['a2']:+6.3f} +- {rt['a2_err']:.3f}  {rt['bTA']:+6.3f} +- {rt['bTA_err']:.3f}")
            for m, (r, f) in (("NLA", (rn, fn)), ("TATT", (rt, ft))):
                fh.write(f"{sn},{zz[sn]},{m},{r['kmax']:.4f},{f},{r['a1']:.5f},{r['a1_err']:.5f},"
                         f"{r['a2']:.5f},{r['a2_err']:.5f},{r['bTA']:.5f},{r['bTA_err']:.5f}\n")
    for m, cols in (("NLA", ["a1", "a1_err"]),
                    ("TATT", ["a1", "a1_err", "a2", "a2_err", "bTA", "bTA_err"])):
        arr = [[zz[sn], out[(sn, m)][0]["kmax"]] + [out[(sn, m)][0][c] for c in cols] for sn in sns]
        np.savetxt(os.path.join(a.dir, f"A_z_{m.lower()}.txt"), arr, fmt="%.4f",
                   header="z k_max " + " ".join(cols) + f"   ({a.probes}, at each snapshot's cut)")

    for par, models in (("a1", ["NLA", "TATT"]), ("a2", ["TATT"]), ("bTA", ["TATT"])):
        fig, axs = plt.subplots(4, 4, figsize=(17, 13.5))
        for ax, sn in zip(axs.flat, sns):
            for m in models:
                rs = sorted(rows[(sn, m)], key=lambda r: r["kmax"])
                k = np.array([r["kmax"] for r in rs])
                v = np.array([r[par] for r in rs])
                e = np.array([r[par + "_err"] for r in rs])
                ax.errorbar(k*(0.98 if m == "NLA" else 1.02), v, e, fmt="o-", ms=3.5, lw=1,
                            capsize=2, color=COL[m], label=m)
                r, f = out[(sn, m)]
                ax.axvline(r["kmax"], color=COL[m], ls="--", lw=1.2)
                ax.plot(r["kmax"], r[par], "*", ms=13, color=COL[m], mec="k")
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$", fontsize=9)
            ax.set_ylabel(par, fontsize=9)
            ax.set_title(f"z = {zz[sn]:.2f}  (sn{sn})", fontsize=10)
            # zoom on the values up to a bit beyond the cut
            vals = []
            for m in models:
                rs = [r for r in rows[(sn, m)] if r["kmax"] <= 1.6*out[(sn, m)][0]["kmax"]]
                vals += [r[par] for r in rs]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = 0.25*(hi - lo) + 0.02*abs(hi) + 1e-3
                ax.set_ylim(lo - pad, hi + pad)
        axs.flat[0].legend(fontsize=8)
        for ax in axs.flat[len(sns):]:
            ax.axis("off")
        fig.suptitle(f"{par} vs k_max, per-snapshot amplitude fits to FLAMINGO ({a.probes}); "
                     "dashed + star: the snapshot's cut (Delta chi2 test)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(a.dir, f"amp_{par}_vs_kmax.png"), dpi=105)
        plt.close(fig)
    print(f"wrote {a.dir}/amp_at_cut.csv, A_z_nla.txt, A_z_tatt.txt, amp_<par>_vs_kmax.png")


if __name__ == "__main__":
    main()
