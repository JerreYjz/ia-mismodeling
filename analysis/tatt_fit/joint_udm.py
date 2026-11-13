#!/usr/bin/env python3
"""Scale cuts for NLA and TATT fitted jointly to FLAMINGO with the cosmolike
redshift evolution (the parameterization of the chains).

    a1(z) = A1 x^eta1,   a2(z) = A2 x^eta2,   b_TA = const,   x = (1+z)/1.62
    NLA: A1, eta1.   TATT: A1, eta1, A2, eta2, b_TA.

All snapshots in [--zmin, --zmax] are fitted at once to the raw FLAMINGO
P_hE (= b(k) P_dE, b(k) from the data) and, with --probes hE EE (default),
P_EE (+ one white shape-noise residual Ns per snapshot, a nuisance). Each
snapshot uses k_min <= k <= k_max(z) = k0 D(z)^-n. For fixed n, k0 is
scanned upward; at each k0 the global parameters theta and their Laplace
covariance C (Ns marginalized) are recorded.

Overfitting: Q_UDM = (th1 - th2)^T (C1 - C2)^+ (th1 - th2) for every pair
k0_1 < k0_2 (Saraivanov, Rubira & Miranda 2026, arXiv:2611.03556), chi2 with
d = rank(C1 - C2) dof (eigenvalues > 1e-8 of the largest), as N_sigma; the
cut is the largest k0_2 with no pair below it above --nsig (2). For TATT
Q_UDM uses (A1, eta1, A2, b_TA): eta2 is undefined while A2 ~ 0 (at low k0
it runs to its bounds with huge errors) and is treated as a nuisance. The
scan starts once the joint fit has >= --dof-per-par dof per global
parameter, and k0 values that select exactly the same bins as the previous
one are skipped.
Goodness of fit: joint chi2_min, p < --p-min ends the range.

Errors: P_hE analytic Gaussian (P_hE0^2 + P_hh P_EE0)/N_modes (as
fit_linear_udm.py), P_EE the measured mode scatter; both (+) --sys-frac.

    python joint_udm.py --ps-dir PS --out joint
    python joint_udm.py --ps-dir PS --out joint_bta --bta-positive
    python joint_udm.py --ps-dir PS --out joint_wide --zmin 0.2 --zmax 3

Writes joint.pdf (summary, then per n: N_sigma matrices, p-values,
parameters vs k0), joint_scan.csv and joint_cuts.csv.
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_tatt_flamingo as F  # noqa: E402

GLOB = {"NLA": ["A1", "eta1"], "TATT": ["A1", "eta1", "A2", "eta2", "bTA"]}
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


def load(model, a):
    out = []
    for sn in sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0]):
        z, e_rms = F.SNAPSHOTS[sn][:2]
        if not (a.zmin - 1e-9 <= z <= a.zmax + 1e-9):
            continue
        d = F.load_snapshot(a.ps_dir, sn, 1)
        k = d["k"]
        PNL = model.Pnl(z, k)
        s1 = 1.0/(2.0*(1.0 - e_rms**2))
        _, PhE0, _, Nm = F._load(a.ps_dir, "PhE0", sn, 1)
        _, Phh, _, _ = F._load(a.ps_dir, "Phh", sn, 1)
        _, PEE0, _, _ = F._load(a.ps_dir, "PEE0", sn, 1)
        sg = 1.5*s1*np.sqrt(np.clip((PhE0**2 + Phh*PEE0)/Nm, 0, None))
        shE = np.where(sg > 0, sg, d["hE"][1])
        err = {"hE": np.sqrt(shE**2 + (a.sys_frac*d["hE"][0])**2),
               "EE": np.sqrt(d["EE"][1]**2 + (a.sys_frac*d["EE"][0])**2)}
        out.append(dict(sn=sn, z=z, D=model.D(z), k=k, PNL=PNL, K=model.kernels(z, k),
                        bk=F.scale_bias(d, PNL), y={p: d[p][0] for p in ("hE", "EE")},
                        err=err))
    return out


def fit(model, snaps, mdl, k0, n, a, starts):
    names = GLOB[mdl]
    use_ee = "EE" in a.probes
    nsn = len(snaps)
    masks = [(s["k"] >= a.kmin) & (s["k"] <= k0*s["D"]**(-n)*(1 + 1e-9)) for s in snaps]

    def unpack(x):
        g = dict(zip(names, x[:len(names)]))
        ns = x[len(names):] if use_ee else np.zeros(nsn)
        return g, ns

    def resid(x):
        g, ns = unpack(x)
        out = []
        for s, m, nsi in zip(snaps, masks, ns):
            if not m.any():
                continue
            xx = (1 + s["z"])/F.ONEPLUSZ0
            a1 = g["A1"]*xx**g["eta1"]
            a2 = g["A2"]*xx**g["eta2"] if "A2" in g else 0.0
            bta = g.get("bTA", 0.0)
            pr = model.predict(s["z"], s["k"], a1, a2, bta, 1.0, K=s["K"], PNL=s["PNL"],
                               Ns=nsi, bk=s["bk"])
            for p in a.probes:
                out.append(((s["y"][p] - pr[p])/s["err"][p])[m])
        return np.concatenate(out)

    lo = [-60, -10] + ([-100, -10, (0.0 if a.bta_positive else -20)] if mdl == "TATT" else [])
    hi = [60, 10] + ([100, 10, 20] if mdl == "TATT" else [])
    lo += [-50]*(nsn if use_ee else 0)
    hi += [50]*(nsn if use_ee else 0)
    lo, hi = np.array(lo, float), np.array(hi, float)
    best = None
    for x0 in starts:
        x0 = np.clip(np.array(x0, float), lo + 1e-6, hi - 1e-6)
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    cov = np.linalg.pinv(J.T @ J)
    ng = len(names)
    c2 = float(best.fun @ best.fun)
    nd = best.fun.size
    dof = nd - len(best.x)
    return dict(k0=float(k0), x=best.x, th=best.x[:ng], C=cov[:ng, :ng], chi2=c2, nd=nd,
                dof=dof, p=float(chi2_dist.sf(c2, dof)) if dof > 0 else 1.0,
                nk=[int(m.sum()) for m in masks])


def nsig(f1, f2, rtol=1e-8):
    dC = f1["C"] - f2["C"]
    dC = 0.5*(dC + dC.T)
    w, U = np.linalg.eigh(dC)
    keep = w > rtol*max(w.max(), 1e-300)
    if not keep.any():
        return 0.0
    dt = U.T @ (f1["th"] - f2["th"])
    Q = float(np.sum(dt[keep]**2/w[keep]))
    sf = chi2_dist.sf(Q, int(keep.sum()))
    return float(norm.isf(sf/2)) if sf > 0 else 40.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="joint")
    ap.add_argument("--zmin", type=float, default=0.5)
    ap.add_argument("--zmax", type=float, default=1.5)
    ap.add_argument("--probes", nargs="+", default=["hE", "EE"], choices=["hE", "EE"])
    ap.add_argument("--kmin", type=float, default=0.05)
    ap.add_argument("--powers", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--k0", type=float, nargs="+",
                    default=list(np.round(np.geomspace(0.08, 1.75, 40), 4)))
    ap.add_argument("--sys-frac", type=float, default=0.02)
    ap.add_argument("--nsig", type=float, default=2.0)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--bta-positive", action="store_true")
    ap.add_argument("--dof-per-par", type=float, default=3.0)
    ap.add_argument("--halofit", default="takahashi")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(a.out, exist_ok=True)
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = F.TATTModel(halofit=a.halofit)
    snaps = load(model, a)
    print(f"snapshots: {[round(s['z'], 2) for s in snaps]}, probes {a.probes}"
          f"{', b_TA >= 0' if a.bta_positive else ''}", flush=True)
    nsn = len(snaps)
    use_ee = "EE" in a.probes

    scan = {}
    csv = open(os.path.join(a.out, "joint_scan.csv"), "w")
    csv.write("model,n,k0,chi2,dof,p," + ",".join(f"{p},{p}_err" for p in GLOB["TATT"]) + "\n")
    for mdl in ("NLA", "TATT"):
        for n in a.powers:
            fits, warm = [], None
            last_nk = None
            for k0 in a.k0:
                kz = [k0*s["D"]**(-n) for s in snaps]
                nk = tuple(int(((s["k"] >= a.kmin) & (s["k"] <= kk*(1 + 1e-9))).sum())
                           for s, kk in zip(snaps, kz))
                if nk == last_nk:
                    continue
                ndat = sum(nk)*len(a.probes)
                npar = len(GLOB[mdl]) + (nsn if use_ee else 0)
                if ndat - npar < a.dof_per_par*len(GLOB[mdl]) or min(nk) < 2:
                    continue
                last_nk = nk
                base = [3.8, 0.6] + ([-1.0, -0.7, 0.2] if mdl == "TATT" else [])
                ns0 = [0.1]*nsn if use_ee else []
                starts = [base + ns0]
                if mdl == "TATT":
                    for a2s in (-1.0, 1.0):
                        for bs in ((0.3, 1.0) if a.bta_positive else (-0.3, 0.3, 1.0)):
                            starts.append([3.8, 0.6, a2s, 0.0, bs] + ns0)
                if warm is not None:
                    starts.append(list(warm))
                f = fit(model, snaps, mdl, k0, n, a, starts)
                warm = f["x"]
                fits.append(f)
                e = np.sqrt(np.diag(f["C"]))
                vals = dict(zip(GLOB[mdl], zip(f["th"], e)))
                csv.write(f"{mdl},{n},{k0},{f['chi2']:.2f},{f['dof']},{f['p']:.3e}," + ",".join(
                    f"{vals[p][0]:.5f},{vals[p][1]:.5f}" if p in vals else "nan,nan"
                    for p in GLOB["TATT"]) + "\n")
            m = len(fits)
            N = np.full((m, m), np.nan)
            qi = [GLOB[mdl].index(p) for p in GLOB[mdl] if p != "eta2"]
            for i in range(m):
                for j in range(i + 1, m):
                    f1 = dict(th=fits[i]["th"][qi], C=fits[i]["C"][np.ix_(qi, qi)])
                    f2 = dict(th=fits[j]["th"][qi], C=fits[j]["C"][np.ix_(qi, qi)])
                    N[i, j] = nsig(f1, f2)
            k0s = np.array([f["k0"] for f in fits])
            ku, kind_u = k0s[-1], "none"
            for j in range(1, m):
                if np.nanmax(N[:j + 1, :j + 1]) > a.nsig:
                    ku, kind_u = k0s[j - 1], "cut"
                    break
            kg, kind_g = k0s[-1], "none"
            for j in range(m):
                if fits[j]["p"] < a.p_min:
                    kg, kind_g = (k0s[j - 1] if j > 0 else k0s[0]), "cut"
                    break
            scan[(mdl, n)] = dict(fits=fits, N=N, k0=k0s, udm=(ku, kind_u), gof=(kg, kind_g))
            print(f"{mdl:4s} n={n:3.1f}: k0(overfitting) = {ku:.3f}{'+' if kind_u == 'none' else ''}"
                  f"   k0(gof) = {kg:.3f}{'+' if kind_g == 'none' else ''}   "
                  f"[k_max at z = {snaps[0]['z']:.2f}..{snaps[-1]['z']:.2f}: "
                  f"{ku*snaps[0]['D']**(-n):.2f}..{ku*snaps[-1]['D']**(-n):.2f}]", flush=True)
    csv.close()
    with open(os.path.join(a.out, "joint_cuts.csv"), "w") as fh:
        fh.write("model,n,k0_udm,kind_udm,k0_gof,kind_gof\n")
        for (mdl, n), r in scan.items():
            fh.write(f"{mdl},{n},{r['udm'][0]},{r['udm'][1]},{r['gof'][0]},{r['gof'][1]}\n")

    pdf = PdfPages(os.path.join(a.out, "joint.pdf"))
    # summary: cuts vs n
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for mdl in ("NLA", "TATT"):
        for c, mk in (("udm", "o-"), ("gof", "s--")):
            kk = [scan[(mdl, n)][c][0] for n in a.powers]
            ax.plot(a.powers, kk, mk, color=COL[mdl],
                    label=f"{mdl}: {'overfitting (Q_UDM)' if c == 'udm' else 'goodness of fit'}")
            for n, v in zip(a.powers, kk):
                if scan[(mdl, n)][c][1] == "none":
                    ax.plot(n, v, "^", ms=10, mfc="none", color="k")
    ax.set_yscale("log")
    ax.set_xlabel("n in k_max(z) = k0 D(z)^-n")
    ax.set_ylabel("k0 at the cut [h/Mpc]")
    ax.set_title(f"Joint fit, z = {snaps[0]['z']:.2f}-{snaps[-1]['z']:.2f}, probes {'+'.join(a.probes)}"
                 f"{', b_TA >= 0' if a.bta_positive else ''}\n(triangles: no cut within the k0 range)",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "joint_summary.png"), dpi=130)
    pdf.savefig(fig)
    plt.close(fig)
    for n in a.powers:
        fig = plt.figure(figsize=(18, 8.5))
        gs = fig.add_gridspec(2, 4)
        for j, mdl in enumerate(("NLA", "TATT")):
            r = scan[(mdl, n)]
            ax = fig.add_subplot(gs[0, j])
            im = ax.imshow(np.clip(r["N"], 0, 5), origin="lower", cmap="magma_r", vmin=0, vmax=5)
            lbl = [f"{x:.2f}" for x in r["k0"]]
            ax.set_xticks(range(len(lbl)))
            ax.set_xticklabels(lbl, rotation=90, fontsize=7)
            ax.set_yticks(range(len(lbl)))
            ax.set_yticklabels(lbl, fontsize=7)
            for i in range(len(lbl)):
                for jj in range(i + 1, len(lbl)):
                    v = r["N"][i, jj]
                    ax.text(jj, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                            color="w" if v > 2.5 else "k")
            ax.set_xlabel("k0 (updated fit)")
            ax.set_ylabel("k0")
            ax.set_title(f"{mdl}: N_sigma(Q_UDM), cut k0 = {r['udm'][0]:.2f}"
                         f"{'+' if r['udm'][1] == 'none' else ''}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax = fig.add_subplot(gs[0, 2])
        for mdl in ("NLA", "TATT"):
            r = scan[(mdl, n)]
            ax.plot(r["k0"], [max(f["p"], 1e-12) for f in r["fits"]], "-o", ms=3, color=COL[mdl],
                    label=f"{mdl}: cut k0 = {r['gof'][0]:.2f}")
            ax.plot(r["k0"], [f["chi2"]/max(f["dof"], 1) for f in r["fits"]], ":", color=COL[mdl])
        ax.axhline(a.p_min, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 20)
        ax.set_xlabel("k0 [h/Mpc]")
        ax.set_title("p-value (solid) and chi2/dof (dotted)", fontsize=9)
        ax.legend(fontsize=7)
        for j, p in enumerate(GLOB["TATT"]):
            ax = fig.add_subplot(gs[1, j]) if j < 4 else fig.add_subplot(gs[0, 3])
            for mdl in ("NLA", "TATT"):
                if p not in GLOB[mdl]:
                    continue
                r = scan[(mdl, n)]
                i = GLOB[mdl].index(p)
                v = np.array([f["th"][i] for f in r["fits"]])
                e = np.array([np.sqrt(f["C"][i, i]) for f in r["fits"]])
                ax.errorbar(r["k0"]*(0.98 if mdl == "NLA" else 1.02), v, e, fmt="o-", ms=3.5,
                            lw=1, capsize=2, color=COL[mdl], label=mdl)
                ax.axvline(r["udm"][0], color=COL[mdl], ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("k0 [h/Mpc]")
            ax.set_title(f"{p} vs k0 (dashed: overfitting cut)", fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle(f"k_max(z) = k0 D(z)^-{n:g};  snapshots z = {snaps[0]['z']:.2f}-{snaps[-1]['z']:.2f};"
                     f"  probes {'+'.join(a.probes)}{';  b_TA >= 0' if a.bta_positive else ''}",
                     fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print(f"wrote {a.out}/joint.pdf, joint_summary.png, joint_scan.csv, joint_cuts.csv")


if __name__ == "__main__":
    sys.exit(main())
