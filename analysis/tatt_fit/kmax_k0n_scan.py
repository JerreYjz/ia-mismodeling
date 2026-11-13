#!/usr/bin/env python3
"""k_max(z) = k0 D(z)^-n from all FLAMINGO snapshots at once.

Builds on fit_stitched_kmax.py (same NLA + FLAMINGO stitched spectra and
covariance). Three changes against the per-snapshot cuts:

1. Soft cut. A bin at k enters the fit with weight
       t(k) = 1 / (1 + exp[(ln k - ln k_max) / s_t]),   s_t = --taper (0.15)
   by inflating its variance, C_eff = C + diag[C_ii (1/t - 1)]; bins with
   t < 1e-6 are dropped. Every diagnostic is then a smooth function of
   k_max, evaluated on a fine k_max grid (--nk points), not on the data's
   1.34-spaced k bins. Effective dof = sum(t) - N_par.

2. Three diagnostics per snapshot and model, as functions of k_max:
     A1   : |a1(k_max) - a1_ref| / a1_ref, a1_ref from the hard fit to
            k <= k_ref (--kref): fractional drift of the IA amplitude;
     pred : dchi2 of the k <= k_ref band, k_max fit against the k_ref fit
            (large scales); under the null <= chi2 with N_par dof;
     gof  : chi2_min and dof_eff of the k_max fit.

3. Joint (k0, n) scan. For each (k0, n), every snapshot s is cut at
   k_s = k0 D_s^-n (clamped to the k_max grid). Each snapshot's diagnostic
   is divided by its own threshold,
     A1   : drift_s / --a1-frac (0.01),
     pred : dchi2_s / chi2_isf(--p-min, N_par),
     gof  : -log10 p_s / -log10(--p-min),
   and (k0, n) passes if the rms of these over the snapshots is <= 1. The
   rms weights every snapshot equally and does not gain statistical power
   as snapshots are added (a summed chi2 would), and it penalizes a single
   failing snapshot more than a mean would.
   k0*(n) = the largest k0 below the first k0 that fails persistently.

The power n is chosen by collapse: for several levels L of each
diagnostic, find the k_max at which each snapshot first reaches L
(continuous crossing); if k_max(z) = k0 D^-n, then ln k_s(L) is linear in
-ln D_s with slope n. n is the error-weighted mean of the per-level
slopes; the final cut is k0*(n) D(z)^-n. k0*(0) (comoving) and k0*(1)
(k_UV/D(z)) are printed too.

    python kmax_k0n_scan.py --ps-dir PS --out k0n_scan

Writes k0n_scan.pdf (summary, k0(n), collapse, then one page per
snapshot), k0n_summary.png, curves.npz, k0n_result.csv and
kmax_z_<model>_<crit>.txt (fitted k0 D^-n at the snapshot redshifts).
"""
import argparse
import os
import sys
import types

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_tatt_flamingo as F          # noqa: E402
import fit_stitched_kmax as S          # noqa: E402

MODELS = ["NLA", "TATT"]
CRITS = ["A1", "pred", "gof"]
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
CLAB = {"A1": "a1 drift", "pred": "large-scale dchi2", "gof": "goodness of fit"}


# ---------------------------------------------------------------------------
def soft_fit(model, pre, St, free, kmax, args, base, starts_extra=None, grid=None):
    kv = St["kv"]
    t = 1.0/(1.0 + np.exp((np.log(kv) - np.log(kmax))/args.taper))
    sel = np.where((kv >= args.kmin) & (t > 1e-6))[0]
    C = St["C"][np.ix_(sel, sel)].copy()
    C[np.diag_indices_from(C)] += np.diag(C)*(1.0/t[sel] - 1.0)
    L = cholesky(C, lower=True)
    Wt = lambda v: solve_triangular(L, v, lower=True)
    yw = Wt(St["y"][sel])

    def theta(x):
        th = dict(base)
        th.update(zip(free, x))
        return th

    def resid(x):
        mv, _ = S.model_vector(model, pre, theta(x), args.probes)
        return yw - Wt(mv[sel])

    lo = np.array([S.LO[n] for n in free])
    hi = np.array([S.HI[n] for n in free])
    starts = [[]]
    for n in free:
        starts = [s + [v] for s in starts for v in (grid or {}).get(n, [base[n]])]
    for w in starts_extra or []:
        starts.append([w[n] for n in free])
    best = None
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                          x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    chi2 = float(best.fun @ best.fun)
    dof = float(np.sum(t[sel])) - len(free)
    return dict(theta=theta(best.x), cov=cov, free=list(free),
                err={n: float(np.sqrt(abs(cov[i, i]))) for i, n in enumerate(free)},
                chi2=chi2, dof=dof)


def first_cross(k, g, level, above=True):
    """First k where g crosses level for good (next grid point too),
    interpolated in ln k; nan if it never does."""
    f = g >= level if above else g <= level
    for i in range(1, len(g)):
        if f[i] and (i + 1 == len(g) or f[i + 1]):
            a, b = g[i - 1], g[i]
            tt = np.clip((level - a)/(b - a), 0, 1) if b != a else 1.0
            return float(np.exp(np.log(k[i - 1]) + tt*(np.log(k[i]) - np.log(k[i - 1]))))
    return np.nan


def interp_ln(kgrid, arr, kq):
    return np.interp(np.log(kq), np.log(kgrid), arr)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="k0n_scan")
    ap.add_argument("--snapshots", type=int, nargs="*", default=None)
    ap.add_argument("--probes", nargs="+", default=["hE", "EE", "BB"])
    ap.add_argument("--kmin", type=float, default=0.01)
    ap.add_argument("--kref", type=float, default=0.067)
    ap.add_argument("--kmax-lo", type=float, default=0.08)
    ap.add_argument("--kmax-hi", type=float, default=1.7)
    ap.add_argument("--nk", type=int, default=33)
    ap.add_argument("--taper", type=float, default=0.15)
    ap.add_argument("--sys-frac", type=float, default=0.02)
    ap.add_argument("--err-scale", type=float, default=1.0)
    ap.add_argument("--a1-frac", type=float, default=0.01)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--halofit", default="takahashi")
    args = ap.parse_args()
    sargs = types.SimpleNamespace(probes=args.probes, sys_frac=args.sys_frac,
                                  err_scale=args.err_scale, kmin=args.kmin,
                                  kt_scale=1.0, width=S.SIGMA_W)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(args.out, exist_ok=True)
    sns = args.snapshots or sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = F.TATTModel(halofit=args.halofit)
    KG = np.geomspace(args.kmax_lo, args.kmax_hi, args.nk)

    cur = {m: {c: [] for c in ("drift", "dchi2", "chi2", "dof", "a1", "a1e", "a2", "a2e")}
           for m in MODELS}
    zs, Ds = [], []
    for sn in sns:
        d = F.load_snapshot(args.ps_dir, sn, 1)
        z, k = d["z"], d["k"]
        PNL = model.Pnl(z, k)
        pre = dict(z=z, k=k, D=model.D(z), PNL=PNL, K=model.kernels(z, k),
                   bk=F.scale_bias(d, PNL), model=model)
        nf = S.nla_fit(model, d, pre, sargs)
        St = S.build_stitched(model, d, pre, nf, sargs)
        base = dict(a1=nf["a1"], a2=0.0, bTA=0.0, Ns=nf["Ns"])
        zs.append(z)
        Ds.append(pre["D"])
        for m in MODELS:
            ref = S.fit(model, pre, St, S.FREE[m], args.kref, sargs, base,
                        grid=S.GRID.get(m))
            c_ref = S.band_chi2(model, pre, St, ref["theta"], args.kref, sargs)
            warm = ref["theta"]
            out = {c: [] for c in cur[m]}
            for km in KG:
                r = soft_fit(model, pre, St, S.FREE[m], km, args, base, [warm],
                             S.GRID.get(m))
                warm = r["theta"]
                q = max(S.band_chi2(model, pre, St, r["theta"], args.kref, sargs) - c_ref, 0)
                out["drift"].append(abs(r["theta"]["a1"]/ref["theta"]["a1"] - 1))
                out["dchi2"].append(q)
                out["chi2"].append(r["chi2"])
                out["dof"].append(r["dof"])
                out["a1"].append(r["theta"]["a1"])
                out["a1e"].append(r["err"]["a1"])
                out["a2"].append(r["theta"]["a2"])
                out["a2e"].append(r["err"].get("a2", 0.0))
            for c in out:
                cur[m][c].append(np.array(out[c]))
        print(f"sn{sn:<3d} z={z:4.2f} done", flush=True)
    zs, Ds = np.array(zs), np.array(Ds)
    for m in MODELS:
        for c in cur[m]:
            cur[m][c] = np.array(cur[m][c])          # [snap, kgrid]
    np.savez(os.path.join(args.out, "curves.npz"), kgrid=KG, z=zs, D=Ds,
             **{f"{m}_{c}": cur[m][c] for m in MODELS for c in cur[m]})

    # --- per-snapshot continuous crossings at the main thresholds -----------
    npar = {m: len(S.FREE[m]) for m in MODELS}

    def stat(m, c):
        """per-snapshot statistic [snap, k] and its main threshold (fail if >=)."""
        if c == "A1":
            return cur[m]["drift"], args.a1_frac
        if c == "pred":
            return cur[m]["dchi2"], chi2_dist.isf(args.p_min, npar[m])
        # gof: -log10 p, threshold -log10 p_min
        p = chi2_dist.sf(cur[m]["chi2"], np.maximum(cur[m]["dof"], 1))
        return -np.log10(np.maximum(p, 1e-300)), -np.log10(args.p_min)

    LEVELS = {"A1": [0.5, 1.0, 2.0], "pred": [0.5, 1.0, 2.0], "gof": [0.5, 1.0, 2.0]}
    cross = {}
    collapse = {}
    for m in MODELS:
        for c in CRITS:
            g, thr = stat(m, c)
            cross[(m, c)] = np.array([first_cross(KG, g[i], thr) for i in range(len(zs))])
            slopes, wts = [], []
            for lv in LEVELS[c]:
                kc = np.array([first_cross(KG, g[i], thr*lv) for i in range(len(zs))])
                ok = np.isfinite(kc)
                if ok.sum() >= 4:
                    x, y = -np.log(Ds[ok]), np.log(kc[ok])
                    A = np.vstack([np.ones(ok.sum()), x]).T
                    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
                    yr = y - A @ coef
                    s2 = np.sum(yr**2)/max(ok.sum() - 2, 1)
                    se = np.sqrt(s2*np.linalg.inv(A.T @ A)[1, 1])
                    slopes.append(coef[1])
                    wts.append(1/max(se, 1e-3)**2)
            if slopes:
                n = float(np.average(slopes, weights=wts))
                ne = float(1/np.sqrt(np.sum(wts)))
            else:
                n, ne = np.nan, np.nan
            collapse[(m, c)] = (n, ne, slopes)

    # --- joint (k0, n) scan ------------------------------------------------
    NG = np.round(np.linspace(-0.6, 1.6, 45), 3)
    K0 = np.geomspace(0.04, 3.0, 400)

    def pooled_pass(m, c, ks):
        g, thr = stat(m, c)
        v = np.array([interp_ln(KG, g[i], ks[i]) for i in range(len(zs))])/thr
        return np.sqrt(np.mean(np.maximum(v, 0)**2)) <= 1.0

    k0n = {}
    for m in MODELS:
        for c in CRITS:
            res = []
            for n in NG:
                kbest, flag = np.nan, "none"
                fails = 0
                for j, k0 in enumerate(K0):
                    ks = np.clip(k0*Ds**(-n), KG[0], KG[-1])
                    if np.all(ks >= KG[-1]):
                        flag = "grid"
                        break
                    if pooled_pass(m, c, ks):
                        kbest, fails = k0, 0
                    else:
                        fails += 1
                        if fails >= 4:
                            flag = "cut"
                            break
                res.append((kbest, flag))
            k0n[(m, c)] = res

    def k0_at(m, c, n):
        r = k0n[(m, c)]
        kk = np.array([x[0] for x in r])
        i = int(np.argmin(np.abs(NG - n)))
        return kk[i], r[i][1]

    # --- results -------------------------------------------------------------
    print(f"\nk_max(z) = k0 D(z)^-n; pooled over {len(zs)} snapshots")
    print(f"{'model':5s} {'criterion':18s} {'n (collapse)':>14s} {'k0*(n)':>8s} "
          f"{'k0*(0)':>8s} {'k0*(1)':>8s}")
    rows = []
    for m in MODELS:
        for c in CRITS:
            n, ne, sl = collapse[(m, c)]
            kn, fn = k0_at(m, c, n) if np.isfinite(n) else (np.nan, "-")
            k00, f0 = k0_at(m, c, 0.0)
            k01, f1 = k0_at(m, c, 1.0)
            fmt = lambda v, f: (f"{v:8.3f}" if np.isfinite(v) else "     nan") + ("+" if f == "grid" else " ")
            print(f"{m:5s} {CLAB[c]:18s} {n:7.2f} +- {ne:4.2f} {fmt(kn, fn)}{fmt(k00, f0)}{fmt(k01, f1)}"
                  f"   per-level n: {', '.join(f'{x:.2f}' for x in sl)}")
            rows.append((m, c, n, ne, kn, fn, k00, k01))
    with open(os.path.join(args.out, "k0n_result.csv"), "w") as fh:
        fh.write("model,criterion,n,n_err,k0_n,flag,k0_n0,k0_n1\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    o = np.argsort(zs)
    for (m, c, n, ne, kn, fn, k00, k01) in rows:
        if np.isfinite(n) and np.isfinite(kn):
            np.savetxt(os.path.join(args.out, f"kmax_z_{m.lower()}_{c}.txt"),
                       np.column_stack([zs[o], kn*Ds[o]**(-n)]), fmt="%.3f %.4f",
                       header=f"z  k_max [h/Mpc] = {kn:.4f} D(z)^({-n:+.3f}) ({m}, {c})")

    # --- plots ---------------------------------------------------------------
    pdf = PdfPages(os.path.join(args.out, "k0n_scan.pdf"))
    from scipy.integrate import quad
    Om = F.COSMO["Om"]
    E = lambda a: np.sqrt(Om/a**3 + 1 - Om)
    gg = lambda a: 2.5*Om*E(a)*quad(lambda x: 1/(x*E(x))**3, 1e-6, a)[0]
    zz = np.linspace(0, zs.max(), 120)
    Dz = np.array([gg(1/(1 + x))/gg(1) for x in zz])
    mk = {"A1": "o", "pred": "s", "gof": "^"}
    ls = {"A1": "-", "pred": "--", "gof": ":"}

    # 1. summary
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, m in zip(axs, MODELS):
        for c in CRITS:
            kc = cross[(m, c)]
            ax.plot(zs, kc, mk[c], color=COL[m], alpha=0.45, ms=5)
            n = collapse[(m, c)][0]
            kn, fn = k0_at(m, c, n) if np.isfinite(n) else (np.nan, "-")
            if np.isfinite(kn):
                ax.plot(zz, kn*Dz**(-n), ls[c], color=COL[m], lw=2,
                        label=f"{CLAB[c]}: {kn:.2f} D(z)^({-n:+.2f}){' (lower bound)' if fn == 'grid' else ''}")
        ax.plot(zz, 0.35/Dz, "-", color="0.75", lw=1, label="0.35/D(z)")
        ax.set_yscale("log")
        ax.set_xlabel("z")
        ax.set_title(f"{m}: joint k0 D(z)^-n (lines) and per-snapshot crossings (points)",
                     fontsize=9)
        ax.legend(fontsize=7)
    axs[0].set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "k0n_summary.png"), dpi=130)
    pdf.savefig(fig)
    plt.close(fig)

    # 2. k0*(n)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, m in zip(axs, MODELS):
        for c in CRITS:
            kk = np.array([x[0] for x in k0n[(m, c)]])
            fl = np.array([x[1] == "grid" for x in k0n[(m, c)]])
            ax.plot(NG, kk, ls[c], color=COL[m], lw=1.8, label=CLAB[c])
            ax.plot(NG[fl], kk[fl], "^", ms=4, mfc="none", color=COL[m])
            n = collapse[(m, c)][0]
            if np.isfinite(n):
                ax.axvline(n, color=COL[m], ls=ls[c], lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.axvline(1, color="0.8", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_title(f"{m}: largest k0 passing the pooled test, vs n (vertical: collapse n; "
                     r"$\triangle$: k grid end)", fontsize=8)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("k0*(n)  [h/Mpc]")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # 3. collapse: statistic vs x = k D^n
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5))
    for i, m in enumerate(MODELS):
        for j, c in enumerate(CRITS):
            ax = axs[i, j]
            g, thr = stat(m, c)
            n = collapse[(m, c)][0]
            nn = n if np.isfinite(n) else 0.0
            for s in range(len(zs)):
                ax.plot(KG*Ds[s]**nn, g[s]/thr, "-", color=cmap(zs[s]/zs.max()), lw=1.2)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(0, 20)
            ax.set_xlabel(rf"$k_{{\rm max}}\,D(z)^{{{nn:.2f}}}$")
            ax.set_title(f"{m}, {CLAB[c]} / threshold (n = {nn:.2f})", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, zs.max()))
    fig.subplots_adjust(hspace=0.42, wspace=0.25, right=0.88)
    fig.colorbar(sm, ax=axs, label="z", shrink=0.6)
    fig.suptitle("Collapse: each snapshot's diagnostic against the rescaled k_max", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)

    # 4. per snapshot
    for s in o:
        fig, axs = plt.subplots(1, 5, figsize=(20, 3.6))
        for m in MODELS:
            for j, c in enumerate(CRITS):
                g, thr = stat(m, c)
                axs[j].plot(KG, g[s]/thr, "-", color=COL[m], lw=1.6, label=m)
                if np.isfinite(cross[(m, c)][s]):
                    axs[j].axvline(cross[(m, c)][s], color=COL[m], ls="--", lw=1)
            axs[3].fill_between(KG, cur[m]["a1"][s] - cur[m]["a1e"][s],
                                cur[m]["a1"][s] + cur[m]["a1e"][s], color=COL[m], alpha=0.15)
            axs[3].plot(KG, cur[m]["a1"][s], "-", color=COL[m], label=m)
        axs[4].fill_between(KG, cur["TATT"]["a2"][s] - cur["TATT"]["a2e"][s],
                            cur["TATT"]["a2"][s] + cur["TATT"]["a2e"][s], color=COL["TATT"],
                            alpha=0.15)
        axs[4].plot(KG, cur["TATT"]["a2"][s], "-", color=COL["TATT"])
        for j, c in enumerate(CRITS):
            axs[j].axhline(1, color="k", lw=0.8)
            axs[j].set_yscale("symlog", linthresh=0.1)
            axs[j].set_ylim(0, 20)
            axs[j].set_title(f"{CLAB[c]} / threshold", fontsize=9)
        axs[3].set_title("a1 (soft cut)", fontsize=9)
        axs[4].set_title("TATT a2 (soft cut)", fontsize=9)
        for ax in axs:
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
        axs[0].legend(fontsize=8)
        fig.suptitle(f"z = {zs[s]:.2f}   (dashed: this snapshot's crossing)", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print(f"wrote {args.out}/k0n_scan.pdf, k0n_summary.png, curves.npz, k0n_result.csv, "
          "kmax_z_<model>_<crit>.txt  ('+': k0 limited by the end of the k grid)")


if __name__ == "__main__":
    sys.exit(main())
