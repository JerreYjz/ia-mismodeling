#!/usr/bin/env python3
"""Goodness of fit of NLA and TATT to FLAMINGO against k_max, per snapshot.

Per snapshot, free parameters (NLA: a1; TATT: a1, a2, b_TA; plus one white
shape-noise residual Ns when P_EE is fitted), fitted to the raw FLAMINGO
spectra over k_min <= k <= k_max for every k_max on the data's k grid.
Probe sets: P_hE alone, and P_hE + P_EE. Errors: P_hE analytic Gaussian,
(P_hE0^2 + P_hh P_EE0)/N_modes; P_EE measured mode scatter; both (+)
--sys-frac. TATT: best of a grid of starts plus the previous k_max's fit.

    python gof_per_snapshot.py --ps-dir PS --out gof

Writes gof_pvalue.png, gof_chi2dof.png (one panel per snapshot, x = k_max),
gof.csv and gof_cuts.csv (last k_max before p < --p-min at two consecutive
k_max, per model and probe set).
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_tatt_flamingo as F  # noqa: E402

FREE = {"NLA": ["a1"], "TATT": ["a1", "a2", "bTA"]}
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
PROBES = {"P_hE": ["hE"], "P_hE + P_EE": ["hE", "EE"]}
LS = {"P_hE": "-", "P_hE + P_EE": "--"}


def load(model, sn, a):
    z, e_rms = F.SNAPSHOTS[sn][:2]
    d = F.load_snapshot(a.ps_dir, sn, 1)
    k = d["k"]
    PNL = model.Pnl(z, k)
    s1 = 1.0/(2.0*(1.0 - e_rms**2))
    _, PhE0, _, Nm = F._load(a.ps_dir, "PhE0", sn, 1)
    _, Phh, _, _ = F._load(a.ps_dir, "Phh", sn, 1)
    _, PEE0, _, _ = F._load(a.ps_dir, "PEE0", sn, 1)
    sg = 1.5*s1*np.sqrt(np.clip((PhE0**2 + Phh*PEE0)/Nm, 0, None))
    shE = np.where(sg > 0, sg, d["hE"][1]) if a.err == "gauss" else d["hE"][1]
    err = {"hE": np.sqrt(shE**2 + (a.sys_frac*d["hE"][0])**2),
           "EE": np.sqrt(d["EE"][1]**2 + (a.sys_frac*d["EE"][0])**2)}
    return dict(sn=sn, z=z, k=k, PNL=PNL, K=model.kernels(z, k), bk=F.scale_bias(d, PNL),
                y={p: d[p][0] for p in ("hE", "EE")}, err=err)


def fit(model, s, mdl, probes, kmax, a, warm):
    k = s["k"]
    m = (k >= a.kmin) & (k <= kmax*(1 + 1e-9))
    names = FREE[mdl] + (["Ns"] if "EE" in probes else [])
    a1g = F.SNAPSHOTS[s["sn"]][4]*F.C1RHOCRIT_REPORT/F.C1RHOCRIT

    def resid(x):
        th = dict(a1=a1g, a2=0.0, bTA=0.0, Ns=0.0)
        th.update(zip(names, x))
        pr = model.predict(s["z"], k, th["a1"], th["a2"], th["bTA"], 1.0, K=s["K"],
                           PNL=s["PNL"], Ns=th["Ns"], bk=s["bk"])
        return np.concatenate([((s["y"][p] - pr[p])/s["err"][p])[m] for p in probes])

    starts = []
    base = {"a1": a1g, "a2": 0.0, "bTA": 0.0, "Ns": 0.1}
    if mdl == "TATT":
        for a2 in (-2.0, 0.0, 2.0):
            for b in (-1.0, 0.0, 1.0):
                starts.append([a1g, a2, b] + ([0.1] if "Ns" in names else []))
    else:
        starts.append([base[n] for n in names])
    if warm is not None:
        starts.append(list(warm))
    lo = np.array([{"a1": -60, "a2": -100, "bTA": -20, "Ns": -50}[n] for n in names], float)
    hi = -lo
    best = None
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi),
                          x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    c2 = float(best.fun @ best.fun)
    dof = best.fun.size - len(names)
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    return dict(chi2=c2, dof=dof, p=float(chi2_dist.sf(c2, dof)) if dof > 0 else np.nan,
                x=best.x, th=dict(zip(names, best.x)),
                err=dict(zip(names, np.sqrt(np.abs(np.diag(cov))))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="gof")
    ap.add_argument("--kmin", type=float, default=0.05)
    ap.add_argument("--kmax-hi", type=float, default=1.8)
    ap.add_argument("--sys-frac", type=float, default=0.02)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--err", choices=["gauss", "mode"], default="gauss",
                    help="P_hE error: analytic Gaussian or FLAMINGO's measured error bars")
    ap.add_argument("--halofit", default="takahashi")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = F.TATTModel(halofit=a.halofit)
    sns = sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    res = {}
    fh = open(os.path.join(a.out, "gof.csv"), "w")
    fh.write("sn,z,model,probes,kmax,chi2,dof,p,a1,a1_err,a2,a2_err,bTA,bTA_err\n")
    for sn in sns:
        s = load(model, sn, a)
        k = s["k"]
        for pl, probes in PROBES.items():
            for mdl in ("NLA", "TATT"):
                npar = len(FREE[mdl]) + ("EE" in probes)
                rows, warm = [], None
                for km in k[k <= a.kmax_hi]:
                    nd = int(((k >= a.kmin) & (k <= km*(1 + 1e-9))).sum())*len(probes)
                    if nd - npar < 1:
                        continue
                    f = fit(model, s, mdl, probes, km, a, warm)
                    warm = f["x"]
                    rows.append((float(km), f))
                    th, er = f["th"], f["err"]
                    fh.write(f"{sn},{s['z']},{mdl},{pl},{km:.4f},{f['chi2']:.3f},{f['dof']},"
                             f"{f['p']:.3e},{th.get('a1', np.nan):.4f},{er.get('a1', 0):.4f},"
                             f"{th.get('a2', 0):.4f},{er.get('a2', 0):.4f},"
                             f"{th.get('bTA', 0):.4f},{er.get('bTA', 0):.4f}\n")
                res[(sn, mdl, pl)] = rows
        print(f"sn{sn} z={s['z']:.2f} done", flush=True)
    fh.close()

    # cuts: last k_max before the first p < p_min
    cuts = {}
    with open(os.path.join(a.out, "gof_cuts.csv"), "w") as fc:
        fc.write("sn,z," + ",".join(f"{m} {p}" for p in PROBES for m in ("NLA", "TATT")) + "\n")
        for sn in sns:
            z = F.SNAPSHOTS[sn][0]
            line = []
            for pl in PROBES:
                for mdl in ("NLA", "TATT"):
                    rows = res[(sn, mdl, pl)]
                    kc = None
                    for i, (km, f) in enumerate(rows):
                        nxt = rows[i + 1][1]["p"] if i + 1 < len(rows) else f["p"]
                        if f["p"] < a.p_min and nxt < a.p_min:   # two in a row
                            kc = rows[i - 1][0] if i > 0 else np.nan
                            break
                    cuts[(sn, mdl, pl)] = kc
                    line.append("none" if kc is None else f"{kc:.3f}")
            fc.write(f"{sn},{z}," + ",".join(line) + "\n")

    for what in ("dchi2", "pvalue", "chi2dof"):
        fig, axs = plt.subplots(4, 4, figsize=(17, 13.5))
        for ax, sn in zip(axs.flat, sns):
            z = F.SNAPSHOTS[sn][0]
            for pl in PROBES:
                for mdl in ("NLA", "TATT"):
                    rows = res[(sn, mdl, pl)]
                    km = np.array([r[0] for r in rows])
                    if what == "dchi2":
                        v = np.array([r[1]["chi2"] for r in rows])
                    elif what == "pvalue":
                        v = np.array([max(r[1]["p"], 1e-15) for r in rows])
                    else:
                        v = np.array([r[1]["chi2"]/r[1]["dof"] for r in rows])
                    ax.plot(km, v, LS[pl], marker="o", ms=3, color=COL[mdl], lw=1.3,
                            label=f"{mdl}, {pl}")
                    kc = cuts[(sn, mdl, pl)]
                    if kc is not None and np.isfinite(kc):
                        ax.axvline(kc, color=COL[mdl], ls=LS[pl], lw=0.7, alpha=0.6)
            if what == "dchi2":
                for pl in PROBES:
                    rows = res[(sn, "NLA", pl)]
                    km = np.array([r[0] for r in rows])
                    dof = np.array([r[1]["dof"] for r in rows])
                    ax.plot(km, dof, LS[pl], color="0.55", lw=1)
                    ax.plot(km, chi2_dist.isf(a.p_min, dof), LS[pl], color="0.8", lw=1)
                ax.set_yscale("log")
                ax.set_ylim(0.3, 3e4)
                ax.set_ylabel(r"$\Delta\chi^2$ to FLAMINGO", fontsize=9)
            elif what == "pvalue":
                ax.axhline(a.p_min, color="k", lw=0.8)
                ax.set_yscale("log")
                ax.set_ylim(1e-12, 2)
                ax.set_ylabel("p-value of chi2_min", fontsize=9)
            else:
                ax.axhline(1, color="k", lw=0.8)
                ax.set_yscale("log")
                ax.set_ylim(0.2, 30)
                ax.set_ylabel("chi2 / dof", fontsize=9)
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$", fontsize=9)
            ax.set_title(f"z = {z:.2f}  (sn{sn})", fontsize=10)
        axs.flat[0].legend(fontsize=7)
        for ax in axs.flat[len(sns):]:
            ax.axis("off")
        title = {"dchi2": "dchi2 = sum (P_model - P_FLAMINGO)^2 / sigma^2 of the best fit (grey: dof, "
                          "light grey: 95% of chi2_dof)\n",
                 "pvalue": "p-value", "chi2dof": "chi2/dof"}[what] + \
            " NLA (red), TATT (blue); solid: P_hE, dashed: " \
            f"P_hE + P_EE; vertical lines: last k_max before p < {a.p_min}"
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"gof_{what}.png"), dpi=105)
        plt.close(fig)

    print("\nlast k_max before p < %.2f  [h/Mpc]  (none: never fails within the grid)" % a.p_min)
    print("   z    NLA P_hE  TATT P_hE   NLA P_hE+P_EE  TATT P_hE+P_EE")
    for sn in sns:
        z = F.SNAPSHOTS[sn][0]
        f = lambda v: "  none" if v is None else f"{v:6.2f}"
        print(f"{z:5.2f}   {f(cuts[(sn, 'NLA', 'P_hE')])}    {f(cuts[(sn, 'TATT', 'P_hE')])}"
              f"        {f(cuts[(sn, 'NLA', 'P_hE + P_EE')])}         {f(cuts[(sn, 'TATT', 'P_hE + P_EE')])}")
    print(f"wrote {a.out}/gof_dchi2.png, gof_pvalue.png, gof_chi2dof.png, gof.csv, gof_cuts.csv")


if __name__ == "__main__":
    sys.exit(main())
