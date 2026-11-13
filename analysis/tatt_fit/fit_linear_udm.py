#!/usr/bin/env python3
"""Scale cuts from linear fits of FLAMINGO P_hE: Q_UDM overfitting + goodness of fit.

P_hE = b(k) P_dE, with the cosmolike TATT P_dE written as a model LINEAR in
its parameters,

    P_hE = -(Om c1rhocrit / D) b(k) [ a1 P_NL + (a1 b_TA) K_dE - 5 (a2/D) K_mix ],

so with theta = (a1, a1 b_TA, a2) (TATT) or theta = (a1) (NLA) every fit is
weighted linear least squares: one solution (no near-degenerate branches),
exact Gaussian covariance C = (X^T W X)^-1. b(k) = sqrt(P_hh/P_mm), K from
FAST-PT, P_NL from CAMB, all as in fit_tatt_flamingo.py. Data: the raw
FLAMINGO P_hE (no stitch), k >= --kmin (0.05: the two lowest bins of the
681 Mpc/h box have only tens of modes). Errors (--err gauss, default): the
analytic Gaussian error of the monopole,
    sigma^2 = (P_hE0^2 + P_hh,tot P_EE0,tot) / N_modes,
with the raw (noise-including) P_hh and P_EE monopoles, converted like P_hE;
the measured mode scatter (--err mode) is noisy where there are few modes
and ~25% below this. A fractional floor --sys-frac is added in quadrature.
Each model's k_max scan starts once its fit has >= --min-dof dof.

1. Overfitting (Saraivanov, Rubira & Miranda 2026, arXiv:2611.03556, Sec. V C):
   for every pair k1 < k2 of k_max values,
       Q_UDM = (th1 - th2)^T (C1 - C2)^+ (th1 - th2),
   chi2-distributed with d = rank(C1 - C2) dof; p = CDF, N_sigma = sqrt(2)
   erfinv(p). Significant drift: N_sigma > --nsig (2). k_max(UDM) = the
   largest k2 for which no pair (k1 < k2' <= k2) exceeds it.
2. Goodness of fit: chi2_min and its p-value; k_max(gof) = the last k_max
   before p < --p-min. The fractional residual of the best fit,
   max |P_data/P_model - 1| over bins significant at > 2 sigma, is also
   recorded (k_max(acc): the last k_max with it below --eps).
3. Redshift dependence: ln k_max = ln k0 - n ln D(z) fitted to the
   per-snapshot cuts (n = 1 is k_UV/D(z), n = 0 a constant comoving k).

    python fit_linear_udm.py --ps-dir PS --out udm

Writes udm.pdf (summary, then one page per snapshot: N_sigma matrices,
p-values, parameters vs k_max), udm_summary.png, fits.csv, cuts.csv.
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import erfinv
from scipy.stats import chi2 as chi2_dist, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_tatt_flamingo as F  # noqa: E402

PARS = {"NLA": ["a1"], "TATT": ["a1", "a1bTA", "a2"]}
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


def design(model, d, z, bias="scale"):
    k = d["k"]
    D = model.D(z)
    PNL = model.Pnl(z, k)
    K = model.kernels(z, k)
    b = F.scale_bias(d, PNL)
    if bias == "const":   # large-scale linear bias, no scale dependence
        b = np.full_like(b, np.mean(b[(k > 0.05) & (k < 0.15)]))
    c = F.COSMO["Om"]*F.C1RHOCRIT/D
    X = np.column_stack([-c*b*PNL, -c*b*K["dE"], 5*c/D*b*K["mix"]])
    return X, D


def linfit(X, y, s):
    W = 1/s**2
    A = X.T @ (X*W[:, None])
    C = np.linalg.inv(A)
    th = C @ (X.T @ (W*y))
    r = (y - X @ th)/s
    return th, C, float(r @ r)


def nsig_udm(th1, C1, th2, C2, rtol=1e-8):
    dC = C1 - C2
    dC = 0.5*(dC + dC.T)
    w, U = np.linalg.eigh(dC)
    keep = w > rtol*max(w.max(), 1e-300)
    d = int(keep.sum())
    if d == 0:
        return 0.0, 0
    dt = U.T @ (th1 - th2)
    Q = float(np.sum(dt[keep]**2/w[keep]))
    # N_sigma = sqrt(2) erfinv(CDF); use the survival function for accuracy
    sf = chi2_dist.sf(Q, d)
    ns = float(norm.isf(sf/2)) if sf > 0 else 40.0
    return ns, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="udm")
    ap.add_argument("--kmin", type=float, default=0.05)
    ap.add_argument("--err", choices=["gauss", "mode"], default="gauss")
    ap.add_argument("--min-dof", type=int, default=2)
    ap.add_argument("--bias", choices=["scale", "const"], default="scale",
                    help="b(k) from the data (default) or the k < 0.15 linear bias")
    ap.add_argument("--kmax-lo", type=float, default=0.08)
    ap.add_argument("--kmax-hi", type=float, default=1.8)
    ap.add_argument("--sys-frac", type=float, default=0.02)
    ap.add_argument("--err-scale", type=float, default=1.0)
    ap.add_argument("--nsig", type=float, default=2.0)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--halofit", default="takahashi")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(a.out, exist_ok=True)
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = F.TATTModel(halofit=a.halofit)
    sns = sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])

    fcsv = open(os.path.join(a.out, "fits.csv"), "w")
    fcsv.write("sn,z,model,kmax,ndata,chi2,p_gof,max_frac_resid,"
               + ",".join(f"{p},{p}_err" for p in PARS["TATT"]) + "\n")
    pages, table = [], []
    for sn in sns:
        z = F.SNAPSHOTS[sn][0]
        d = F.load_snapshot(a.ps_dir, sn, 1)
        k = d["k"]
        y, sm = d["hE"]
        if a.err == "gauss":
            e_rms = F.SNAPSHOTS[sn][1]
            s1 = 1.0/(2.0*(1.0 - e_rms**2))
            _, PhE0, _, Nm = F._load(a.ps_dir, "PhE0", sn, 1)
            _, Phh, _, _ = F._load(a.ps_dir, "Phh", sn, 1)
            _, PEE0, _, _ = F._load(a.ps_dir, "PEE0", sn, 1)
            v = (PhE0**2 + Phh*PEE0)/Nm
            sg = 1.5*s1*np.sqrt(np.clip(v, 0, None))
            # a negative noisy monopole (lowest bins): keep the measured scatter
            sm = np.where(sg > 0, sg, sm)
        s = np.sqrt((a.err_scale*sm)**2 + (a.sys_frac*y)**2)
        X, D = design(model, d, z, a.bias)
        kg = k[(k >= a.kmax_lo) & (k <= a.kmax_hi)]
        res = {}
        for m in ("NLA", "TATT"):
            cols = [0] if m == "NLA" else [0, 1, 2]
            fits = []
            for km in kg:
                if int(((k >= a.kmin) & (k <= km*(1 + 1e-9))).sum()) - len(cols) < a.min_dof:
                    continue
                sel = (k >= a.kmin) & (k <= km*(1 + 1e-9))
                th, C, c2 = linfit(X[sel][:, cols], y[sel], s[sel])
                dof = int(sel.sum()) - len(cols)
                mod = X[sel][:, cols] @ th
                fr = y[sel]/mod - 1
                sig = np.abs(y[sel] - mod) > 2*s[sel]
                mfr = float(np.max(np.abs(fr[sig]))) if sig.any() else 0.0
                fits.append(dict(kmax=float(km), th=th, C=C, chi2=c2, dof=dof,
                                 p=float(chi2_dist.sf(c2, dof)) if dof > 0 else 1.0,
                                 mfr=mfr, n=int(sel.sum())))
                e = np.sqrt(np.diag(C))
                vals = list(th) + [np.nan]*(3 - len(th))
                errs = list(e) + [np.nan]*(3 - len(e))
                fcsv.write(f"{sn},{z},{m},{km:.4f},{int(sel.sum())},{c2:.3f},"
                           f"{fits[-1]['p']:.3e},{mfr:.4f}," +
                           ",".join(f"{v:.5f},{er:.5f}" for v, er in zip(vals, errs)) + "\n")
            kgm = np.array([f["kmax"] for f in fits])
            n = len(fits)
            N = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(i + 1, n):
                    N[i, j], _ = nsig_udm(fits[i]["th"], fits[i]["C"],
                                          fits[j]["th"], fits[j]["C"])
            # k_max(UDM): largest k2 with no pair below it above the threshold
            ku = kgm[-1]
            kind_u = "none"
            for j in range(1, n):
                if np.nanmax(N[:j + 1, :j + 1]) > a.nsig:
                    ku, kind_u = kgm[j - 1], "cut"
                    break
            kgof, kind_g = kgm[-1], "none"
            for j in range(n):
                if fits[j]["p"] < a.p_min:
                    kgof, kind_g = (kgm[j - 1] if j > 0 else kgm[0]), "cut"
                    break
            kacc, kind_a = kgm[-1], "none"
            for j in range(n):
                if fits[j]["mfr"] > a.eps:
                    kacc, kind_a = (kgm[j - 1] if j > 0 else kgm[0]), "cut"
                    break
            res[m] = dict(fits=fits, N=N, k=kgm, cut={"udm": (ku, kind_u), "gof": (kgof, kind_g),
                                               "acc": (kacc, kind_a)})
        table.append(dict(sn=sn, z=z, D=D, res=res))
        pages.append((sn, z, kg, res))
        f = lambda c: f"{c[0]:.2f}{'+' if c[1] == 'none' else ' '}"
        print(f"sn{sn:<3d} z={z:4.2f}  NLA: udm {f(res['NLA']['cut']['udm'])} gof "
              f"{f(res['NLA']['cut']['gof'])} acc {f(res['NLA']['cut']['acc'])}   TATT: udm "
              f"{f(res['TATT']['cut']['udm'])} gof {f(res['TATT']['cut']['gof'])} acc "
              f"{f(res['TATT']['cut']['acc'])}", flush=True)
    fcsv.close()

    # --- redshift dependence ------------------------------------------------
    D = np.array([t["D"] for t in table])
    z = np.array([t["z"] for t in table])
    fitres = {}
    with open(os.path.join(a.out, "cuts.csv"), "w") as fh:
        fh.write("sn,z,D," + ",".join(f"{m}_{c},{m}_{c}_kind" for m in ("NLA", "TATT")
                                      for c in ("udm", "gof", "acc")) + "\n")
        for t in table:
            fh.write(f"{t['sn']},{t['z']},{t['D']:.4f}," + ",".join(
                f"{t['res'][m]['cut'][c][0]:.4f},{t['res'][m]['cut'][c][1]}"
                for m in ("NLA", "TATT") for c in ("udm", "gof", "acc")) + "\n")
    print("\nln k_max = ln k0 - n ln D(z), fitted to the per-snapshot cuts "
          "(snapshots without a cut, '+', left out):")
    for m in ("NLA", "TATT"):
        for c in ("udm", "gof", "acc", "both"):
            if c == "both":
                kc = np.array([min(t["res"][m]["cut"]["udm"][0], t["res"][m]["cut"]["gof"][0])
                               for t in table])
                ok = np.array([t["res"][m]["cut"]["udm"][1] == "cut" or
                               t["res"][m]["cut"]["gof"][1] == "cut" for t in table])
            else:
                kc = np.array([t["res"][m]["cut"][c][0] for t in table])
                ok = np.array([t["res"][m]["cut"][c][1] == "cut" for t in table])
            if ok.sum() >= 3:
                x, yv = -np.log(D[ok]), np.log(kc[ok])
                A = np.vstack([np.ones(ok.sum()), x]).T
                coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
                rr = yv - A @ coef
                s2 = np.sum(rr**2)/max(ok.sum() - 2, 1)
                se = np.sqrt(np.diag(s2*np.linalg.inv(A.T @ A)))
                fitres[(m, c)] = (np.exp(coef[0]), coef[1], se[1], ok, kc)
                print(f"  {m:4s} {c:4s}: k0 = {np.exp(coef[0]):.3f} h/Mpc, n = {coef[1]:+.2f} "
                      f"+- {se[1]:.2f}  ({ok.sum()} snapshots, rms {np.sqrt(np.mean(rr**2)):.2f} in ln k)")
            else:
                fitres[(m, c)] = (np.nan, np.nan, np.nan, ok, kc)
                print(f"  {m:4s} {c:4s}: fewer than 3 snapshots with a cut")

    # --- plots ----------------------------------------------------------------
    pdf = PdfPages(os.path.join(a.out, "udm.pdf"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    mk = {"udm": "o", "gof": "s", "acc": "^"}
    lab = {"udm": "overfitting (Q_UDM, 2 sigma)", "gof": f"goodness of fit (p < {a.p_min})",
           "acc": f"accuracy ({100*a.eps:.0f}%)"}
    xx = np.linspace(1.0, 1/D.min(), 50)
    for ax, m in zip(axs, ("NLA", "TATT")):
        for c in ("udm", "gof", "acc"):
            k0, n, ne, ok, kc = fitres[(m, c)]
            ax.plot(1/D[ok], kc[ok], mk[c], color=COL[m], alpha=0.7, label=lab[c])
            ax.plot(1/D[~ok], kc[~ok], mk[c], mfc="none", color=COL[m], alpha=0.5)
            if np.isfinite(n):
                ax.plot(xx, k0*xx**n, "-" if c == "udm" else (":" if c == "acc" else "--"),
                        color=COL[m], lw=1.2,
                        label=f"  fit: {k0:.2f} D^-n, n = {n:+.2f} +- {ne:.2f}")
        ax.plot(xx, 0.35*xx, "-", color="0.7", lw=1, label="0.35/D (n = 1)")
        for zi, di in zip(z, D):
            ax.annotate(f"{zi:g}", (1/di, 0.02), xycoords=("data", "axes fraction"),
                        fontsize=6, color="0.4", ha="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("1/D(z)   (numbers: z of each snapshot)")
        ax.set_title(f"{m}: k_max per snapshot vs 1/D (slope = n); open: no cut within the grid",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "udm_summary.png"), dpi=130)
    pdf.savefig(fig)
    plt.close(fig)

    for sn, zz, kg, res in pages:
        fig = plt.figure(figsize=(18, 8.5))
        gs = fig.add_gridspec(2, 4)
        for j, m in enumerate(("NLA", "TATT")):
            ax = fig.add_subplot(gs[0, j])
            N = res[m]["N"]
            im = ax.imshow(np.clip(N, 0, 5), origin="lower", cmap="magma_r", vmin=0, vmax=5)
            n = len(res[m]["k"])
            lbl = [f"{x:.2f}" for x in res[m]["k"]]
            ax.set_xticks(range(n))
            ax.set_xticklabels(lbl, rotation=90, fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(lbl, fontsize=7)
            ax.set_xlabel("k2 (updated fit)")
            ax.set_ylabel("k1")
            for i in range(n):
                for jj in range(i + 1, n):
                    if np.isfinite(N[i, jj]):
                        ax.text(jj, i, f"{N[i, jj]:.1f}", ha="center", va="center", fontsize=5.5,
                                color="w" if N[i, jj] > 2.5 else "k")
            ku = res[m]["cut"]["udm"]
            ax.set_title(f"{m}: N_sigma(Q_UDM); k_max = {ku[0]:.2f}"
                         f"{' (no drift)' if ku[1] == 'none' else ''}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax = fig.add_subplot(gs[0, 2])
        for m in ("NLA", "TATT"):
            fits = res[m]["fits"]
            ax.plot(res[m]["k"], [max(f["p"], 1e-12) for f in fits], "-o", ms=3, color=COL[m],
                    label=f"{m}: k_max(gof) = {res[m]['cut']['gof'][0]:.2f}")
        ax.axhline(a.p_min, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1.5)
        ax.set_title("goodness of fit: p-value of chi2_min", fontsize=9)
        ax.set_xlabel(r"$k_{\rm max}$")
        ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[0, 3])
        for m in ("NLA", "TATT"):
            fits = res[m]["fits"]
            ax.plot(res[m]["k"], [100*f["mfr"] for f in fits], "-o", ms=3, color=COL[m],
                    label=f"{m}: k_max(acc) = {res[m]['cut']['acc'][0]:.2f}")
        ax.axhline(100*a.eps, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title("max |P_data/P_model - 1| over bins off by > 2 sigma [%]", fontsize=9)
        ax.set_xlabel(r"$k_{\rm max}$")
        ax.legend(fontsize=7)
        for j, (p, name) in enumerate(zip(PARS["TATT"], ["a1", "a1 b_TA", "a2"])):
            ax = fig.add_subplot(gs[1, j])
            for m in ("NLA", "TATT"):
                if p not in PARS[m]:
                    continue
                i = PARS[m].index(p)
                fits = res[m]["fits"]
                v = np.array([f["th"][i] for f in fits])
                e = np.array([np.sqrt(f["C"][i, i]) for f in fits])
                ax.errorbar(res[m]["k"]*(0.98 if m == "NLA" else 1.02), v, e, fmt="o-", ms=3.5, lw=1,
                            capsize=2, color=COL[m], label=m)
                ax.axvline(res[m]["cut"]["udm"][0], color=COL[m], ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
            ax.set_title(f"{name} vs k_max (dashed: overfitting cut)", fontsize=9)
            ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[1, 3])
        k = None
        ax.axis("off")
        ax.text(0.0, 0.9, f"sn{sn}, z = {zz:.2f}", fontsize=13, transform=ax.transAxes)
        y0 = 0.75
        for m in ("NLA", "TATT"):
            c = res[m]["cut"]
            ax.text(0.0, y0, f"{m}:  overfitting {c['udm'][0]:.2f}{'+' if c['udm'][1] == 'none' else ''}"
                    f"   gof {c['gof'][0]:.2f}{'+' if c['gof'][1] == 'none' else ''}"
                    f"   accuracy {c['acc'][0]:.2f}{'+' if c['acc'][1] == 'none' else ''}",
                    fontsize=10, color=COL[m], transform=ax.transAxes)
            y0 -= 0.12
        ax.text(0.0, 0.35, "'+': no cut within the k_max grid\nP_hE only, raw FLAMINGO, "
                f"k_min = {a.kmin}, errors: {a.err} (+) {100*a.sys_frac:.0f}%",
                fontsize=8, transform=ax.transAxes)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(os.path.join(a.out, f"udm_sn{sn}.png"), dpi=95)
        plt.close(fig)
    pdf.close()
    print(f"wrote {a.out}/udm.pdf (summary first), udm_summary.png, udm_sn*.png, fits.csv, cuts.csv")


if __name__ == "__main__":
    sys.exit(main())
