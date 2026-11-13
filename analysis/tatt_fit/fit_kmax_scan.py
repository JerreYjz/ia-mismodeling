#!/usr/bin/env python3
"""NLA and TATT fits to the FLAMINGO IA spectra as a function of k_max.

For every snapshot and every k_max on the data's k grid, fit separately

    NLA   : a1            (+ white shape-noise residual Ns in P_EE, P_BB)
    TATT  : a1, a2, b_TA  (+ Ns), global best fit
   [TATT-I: TATT restricted to a2 <= 0 (the fiducial's branch I), plotted
            with --with-branch only; at z <= 0.4 the data prefer a2 > 0 and
            TATT-I sits on its a2 = 0 boundary]

to P_hE, P_EE and P_BB over k_min <= k <= k_max (one k range for all probes,
physical k at the snapshot redshift). Model, data conventions, galaxy bias
(scale-dependent b(k) = sqrt(P_hh/P_mm), the Part 2 injection convention)
and errors (mode scatter plus a fractional floor) are those of
fit_tatt_flamingo.py, which this script imports.

Two calibrated tests, each a p-value:
  * goodness of fit: p_gof = P(chi2 > chi2_min) for N_data - N_free dof;
  * overfitting: does the k_max fit still describe the large scales?
        dchi2_ref = chi2_ref(theta_kmax) - chi2_ref(theta_ref),
    chi2 over the reference band k <= k_ref (--kref, 0.16 h/Mpc), where
    theta_ref is the best fit to that band alone. This tests predictions,
    not parameters: moving along a degenerate (a2, b_TA) direction, or
    jumping between the two near-degenerate TATT solutions, costs nothing,
    while high-k data dragging the model away from the large scales does.
    For a valid model dchi2_ref = dtheta^T F_ref dtheta with
    dtheta ~ N(0, C_ref - C_kmax), so its mean is <= N_par and
    p_drift = P(chi2_{N_par} > dchi2_ref) is conservative. (A test on the
    parameter shifts themselves needs the TATT covariances, which are not
    Gaussian here: the Gauss-Newton errors on a1 and a2 even grow when data
    are added, which makes C_ref - C_kmax non-positive.)
The cut: p = min(p_gof, p_drift); k_max is where log p crosses --p-min,
interpolated in ln k between grid points (the grid steps by 1.34 in k, so a
hard pass/fail on the grid would quantize the cut). A crossing counts only if
the next k_max also fails (one isolated 5% fluctuation does not end the
range). Neighbouring snapshots share initial conditions, so k_max(z) should
be smooth: the script also writes a running median over 3 neighbouring
snapshots, which removes single-snapshot outliers but keeps features shared
by several snapshots (NLA holds to higher k at z ~ 0.75-1.5, where the
tidal-torquing and density-weighting terms of P_dE nearly cancel).

    python fit_kmax_scan.py --ps-dir PS --out kmax_scan
    python fit_kmax_scan.py --ps-dir PS --sys-frac 0.05 --out kmax_scan_sys5

Writes, in --out: kmax_scan.csv (every fit), kmax_cut.csv (per-snapshot cut
and smooth fit), kmax_z_<model>.txt (z, smooth k_max: input for
make_uv_scale_cuts.py --kz-file), kmax_sn<sn>.png, kmax_summary.png and
kmax_scan_all.pdf (all pages).
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_tatt_flamingo as F  # noqa: E402

FREE = {"NLA": ["a1"], "TATT": ["a1", "a2", "bTA"], "TATT-I": ["a1", "a2", "bTA"]}
GRID = {"TATT": {"a2": [-2.0, 0.0, 2.0], "bTA": [-1.0, 0.0, 1.0]},
        "TATT-I": {"a2": [-2.0, -1.0, -0.3], "bTA": [0.0, 0.5, 1.0]}}
HI_OVERRIDE = {"TATT-I": {"a2": 0.0}}
CUT_MODELS = ["NLA", "TATT"]
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8", "TATT-I": "#8ecae6"}
LO = {"a1": -30.0, "a2": -100.0, "bTA": -20.0, "Ns": -50.0}
HI = {"a1": 60.0, "a2": 100.0, "bTA": 20.0, "Ns": 50.0}
PHYS = ["a1", "a2", "bTA"]       # parameters entering the drift test


def prepare(model, d, args):
    z, sn, k = d["z"], d["sn"], d["k"]
    PNL = model.Pnl(z, k, sn)
    err = {p: np.sqrt((args.err_scale*d[p][1])**2
                      + (args.sys_frac*np.abs(d[p][0]))**2) for p in args.probes}
    return dict(z=z, sn=sn, k=k, K=model.kernels(z, k), PNL=PNL,
                bk=F.scale_bias(d, PNL), err=err)


def fit(model, d, pre, free, kmax, args, warm=None, grid=None, hi_over=None):
    """Least-squares fit of the parameters in free over kmin <= k <= kmax."""
    k = pre["k"]
    m = (k >= args.kmin) & (k <= kmax*(1 + 1e-9))
    a_rep = F.SNAPSHOTS[pre["sn"]][4]*F.C1RHOCRIT_REPORT/F.C1RHOCRIT
    base = dict(a1=a_rep, a2=0.0, bTA=0.0, Ns=0.0)

    def theta(x):
        th = dict(base)
        th.update(zip(free, x))
        return th

    def resid(x):
        th = theta(x)
        mod = model.predict(pre["z"], k, th["a1"], th["a2"], th["bTA"], 1.0,
                            K=pre["K"], PNL=pre["PNL"], Ns=th["Ns"],
                            bk=pre["bk"])
        return np.concatenate([((d[p][0] - mod[p])/pre["err"][p])[m]
                               for p in args.probes])

    lo = np.array([LO[n] for n in free])
    hi = np.array([(hi_over or {}).get(n, HI[n]) for n in free])
    starts = [[]]
    for n in free:
        vals = (grid or {}).get(n, [base[n]])
        starts = [s + [v] for s in starts for v in vals]
    if warm is not None:
        starts.append([warm[n] for n in free])
    best = None
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6),
                          bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    cov = np.linalg.pinv(J.T @ J)
    chi2 = float(best.fun @ best.fun)
    ndata = best.fun.size
    dof = ndata - len(free)
    return dict(theta=theta(best.x), free=list(free), cov=cov,
                err={n: float(np.sqrt(abs(cov[i, i]))) for i, n in enumerate(free)},
                chi2=chi2, ndata=ndata, dof=dof,
                p_gof=float(chi2_dist.sf(chi2, dof)) if dof > 0 else 1.0,
                nk=int(m.sum()))


def chi2_band(model, d, pre, th, kmax, args):
    """chi2 of the parameters th over kmin <= k <= kmax."""
    k = pre["k"]
    m = (k >= args.kmin) & (k <= kmax*(1 + 1e-9))
    mod = model.predict(pre["z"], k, th["a1"], th["a2"], th["bTA"], 1.0,
                        K=pre["K"], PNL=pre["PNL"], Ns=th["Ns"], bk=pre["bk"])
    return float(sum(np.sum((((d[p][0] - mod[p])/pre["err"][p])[m])**2)
                     for p in args.probes))


def crossing(rows, pmin):
    """k_max where log p first crosses pmin (persistent), interpolated in
    ln k. Returns (k_cut, reason): reason gof/drift/none/first."""
    p = np.array([max(r["p"], 1e-300) for r in rows])
    k = np.array([r["kmax"] for r in rows])
    fail = p < pmin
    if fail[0] and (len(p) == 1 or fail[1]):
        return k[0], "first"
    for i in range(1, len(rows)):
        if fail[i] and (i + 1 == len(rows) or fail[i + 1]):
            j = i - 1
            while j > 0 and fail[j]:
                j -= 1
            lp0, lp1 = np.log(p[j]), np.log(p[i])
            t = (np.log(pmin) - lp0)/(lp1 - lp0) if lp1 != lp0 else 1.0
            kc = float(np.exp(np.log(k[j]) + np.clip(t, 0, 1)
                              * (np.log(k[i]) - np.log(k[j]))))
            why = "gof" if rows[i]["p_gof"] <= rows[i]["p_drift"] else "drift"
            return kc, why
    return float(k[-1]), "none"


def running_median(z, kc, ok, w=3):
    """Median of ln k_max over w neighbouring snapshots (in z order)."""
    o = np.argsort(z)
    zz, kk, oo = z[o], np.log(kc[o]), ok[o]
    out = np.full(len(z), np.nan)
    h = w//2
    for i in range(len(zz)):
        sl = [j for j in range(max(0, i - h), min(len(zz), i + h + 1)) if oo[j]]
        if sl:
            out[o[i]] = np.exp(np.median(kk[sl]))
    return out


# ---------------------------------------------------------------------------
def plot_snapshot(sn, z, res, kcut, D, path, pmin, pdf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    off = {"NLA": 0.97, "TATT": 1.0, "TATT-I": 1.03}
    for a, n in zip(ax[:3], PHYS):
        for mdl, rows in res.items():
            if n not in FREE[mdl]:
                continue
            km = np.array([r["kmax"] for r in rows])
            v = np.array([r["theta"][n] for r in rows])
            e = np.array([r["err"][n] for r in rows])
            ok = np.array([r["p"] >= pmin for r in rows])
            a.errorbar(km*off[mdl], v, e, fmt="none", ecolor=COL[mdl], alpha=0.6)
            a.plot(km[ok]*off[mdl], v[ok], "o", color=COL[mdl], label=mdl)
            a.plot(km[~ok]*off[mdl], v[~ok], "o", mfc="white", color=COL[mdl])
        a.set_title(n)
    for mdl in CUT_MODELS:
        rows = res[mdl]
        km = np.array([r["kmax"] for r in rows])
        ax[3].plot(km, [max(r["p_gof"], 1e-12) for r in rows], "-o", ms=3,
                   color=COL[mdl], label=f"{mdl} fit")
        ax[3].plot(km, [max(r["p_drift"], 1e-12) for r in rows], "--s", ms=3,
                   color=COL[mdl], label=f"{mdl} drift")
    ax[3].axhline(pmin, color="k", lw=0.8)
    ax[3].set_yscale("log")
    ax[3].set_ylim(1e-6, 1.5)
    ax[3].set_title("p-values (solid: goodness of fit, dashed: large-scale fit)")
    ax[3].legend(fontsize=7, loc="lower left")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
        for mdl, (kc, _) in kcut.items():
            a.axvline(kc, color=COL[mdl], ls="--", lw=1)
        a.axvline(0.35/D, color="0.4", ls=":", lw=1)
    ax[0].legend(fontsize=8, title="open: p < p_min", title_fontsize=7)
    fig.suptitle(f"sn{sn}  z = {z:.2f}    k_max: " + ", ".join(
        f"{m} {kcut[m][0]:.2f} ({kcut[m][1]})" for m in CUT_MODELS) +
        f" h/Mpc   (dotted: 0.35/D(z) = {0.35/D:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def plot_summary(table, smooth, model, path, pdf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.array([t["z"] for t in table])
    zz = np.linspace(0, z.max(), 200)
    o = np.argsort(z)
    Dz = np.array([model.D(x) for x in zz])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for mdl in CUT_MODELS:
        kc = np.array([t["kcut"][mdl] for t in table])
        o = np.argsort(z)
        ax[0].plot(z, kc, "o", color=COL[mdl], alpha=0.5, label=f"{mdl} per snapshot")
        ax[0].plot(z[o], smooth[mdl][o], "-", color=COL[mdl], lw=2,
                   label=f"{mdl} running median (3 snapshots)")
        ax[1].errorbar(z, [t["a1"][mdl][0] for t in table],
                       [t["a1"][mdl][1] for t in table], fmt="o", color=COL[mdl],
                       label=f"{mdl} at its k_max")
    ax[0].plot(zz, 0.35/Dz, ":", color="0.4", label="0.35/D(z)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    ax[0].set_title("k_max(z) from the FLAMINGO fits")
    ax[1].set_ylabel(r"$a_1$")
    for a in ax:
        a.set_xlabel("z")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="kmax_scan")
    ap.add_argument("--snapshots", type=int, nargs="*", default=None)
    ap.add_argument("--nfold", type=int, default=1)
    ap.add_argument("--probes", nargs="+", default=["hE", "EE", "BB"],
                    choices=["hE", "EE", "BB"])
    ap.add_argument("--kmin", type=float, default=0.03)
    ap.add_argument("--kmax-lo", type=float, default=0.08)
    ap.add_argument("--kmax-hi", type=float, default=1.8)
    ap.add_argument("--kref", type=float, default=0.16,
                    help="reference k_max of the drift test [h/Mpc]")
    ap.add_argument("--err-scale", type=float, default=1.0)
    ap.add_argument("--sys-frac", type=float, default=0.02,
                    help="fractional error floor (as fit_tatt_flamingo.py)")
    ap.add_argument("--no-shape-noise", dest="shape_noise", action="store_false")
    ap.add_argument("--halofit", default="takahashi")
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--smooth-zmin", type=float, default=0.0,
                    help="lowest snapshot z in the smooth k_max(z) fit")
    ap.add_argument("--with-branch", action="store_true",
                    help="also fit TATT-I (a2 <= 0, plotted only)")
    args = ap.parse_args()

    models = CUT_MODELS + (["TATT-I"] if args.with_branch else [])
    os.makedirs(args.out, exist_ok=True)
    sns = args.snapshots or sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = F.TATTModel(halofit=args.halofit)
    extra = ["Ns"] if args.shape_noise and ({"EE", "BB"} & set(args.probes)) else []

    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(os.path.join(args.out, "kmax_scan_all.pdf"))
    csv = open(os.path.join(args.out, "kmax_scan.csv"), "w")
    csv.write("sn,z,model,kmax,nk,ndata,dof,chi2,p_gof,dchi2_drift,p_drift,"
              "a1,a1_err,a2,a2_err,bTA,bTA_err,Ns\n")
    table, pages = [], []
    for sn in sns:
        d = F.load_snapshot(args.ps_dir, sn, args.nfold)
        pre = prepare(model, d, args)
        z, D = d["z"], model.D(d["z"])
        kgrid = d["k"][(d["k"] >= args.kmax_lo) & (d["k"] <= args.kmax_hi)]
        res = {}
        for mdl in models:
            free = FREE[mdl] + extra
            rows, warm = [], None
            for km in kgrid:
                r = fit(model, d, pre, free, km, args, warm, GRID.get(mdl),
                        HI_OVERRIDE.get(mdl))
                r["kmax"] = float(km)
                rows.append(r)
                warm = r["theta"]
            iref = int(np.argmin(np.abs(kgrid - args.kref)))
            kr = float(kgrid[iref])
            c_ref = chi2_band(model, d, pre, rows[iref]["theta"], kr, args)
            for i, r in enumerate(rows):
                if i <= iref:
                    r["dchi2"], r["p_drift"] = 0.0, 1.0
                else:
                    q = max(chi2_band(model, d, pre, r["theta"], kr, args) - c_ref, 0.0)
                    r["dchi2"] = q
                    r["p_drift"] = float(chi2_dist.sf(q, len(free)))
                r["p"] = min(r["p_gof"], r["p_drift"])
                th, e = r["theta"], r["err"]
                csv.write(f"{sn},{z},{mdl},{r['kmax']:.4f},{r['nk']},{r['ndata']},"
                          f"{r['dof']},{r['chi2']:.3f},{r['p_gof']:.3e},"
                          f"{r['dchi2']:.3f},{r['p_drift']:.3e},"
                          f"{th['a1']:.5f},{e['a1']:.5f},{th['a2']:.5f},"
                          f"{e.get('a2', 0):.5f},{th['bTA']:.5f},"
                          f"{e.get('bTA', 0):.5f},{th['Ns']:.5f}\n")
            res[mdl] = rows
        kcut = {mdl: crossing(res[mdl], args.p_min) for mdl in CUT_MODELS}

        def a1_at(mdl):
            rows = res[mdl]
            km = np.array([r["kmax"] for r in rows])
            i = int(np.argmin(np.abs(np.log(km) - np.log(kcut[mdl][0]))))
            return rows[i]["theta"]["a1"], rows[i]["err"]["a1"]
        table.append(dict(sn=sn, z=z, D=D,
                          kcut={m: kcut[m][0] for m in CUT_MODELS},
                          why={m: kcut[m][1] for m in CUT_MODELS},
                          a1={m: a1_at(m) for m in CUT_MODELS}))
        print(f"sn{sn:<3d} z={z:4.2f}  " + "  ".join(
            f"{m} {kcut[m][0]:.3f} ({kcut[m][1]:5s})" for m in CUT_MODELS) +
            f"   0.35/D = {0.35/D:.3f}", flush=True)
        pages.append((sn, z, res, kcut, D))
    csv.close()

    z = np.array([t["z"] for t in table])
    D = np.array([t["D"] for t in table])
    smooth = {}
    o = np.argsort(z)
    for mdl in CUT_MODELS:
        kc = np.array([t["kcut"][mdl] for t in table])
        # "none": passes up to the last k_max, kept as a lower bound
        ok = np.array([t["why"][mdl] in ("gof", "drift", "none") for t in table]) \
            & (z >= args.smooth_zmin - 1e-9)
        smooth[mdl] = running_median(z, kc, ok)
        np.savetxt(os.path.join(args.out, f"kmax_z_{mdl.lower()}.txt"),
                   np.column_stack([z[o], smooth[mdl][o]]), fmt="%.3f %.4f",
                   header="z  k_max [h/Mpc] (running median of 3 snapshots)")
        np.savetxt(os.path.join(args.out, f"kmax_z_{mdl.lower()}_raw.txt"),
                   np.column_stack([z[o], kc[o]])[ok[o]], fmt="%.3f %.4f",
                   header="z  k_max [h/Mpc] per snapshot (for --kz-model dpow)")
        res_ln = np.log(kc[ok]) - np.log(smooth[mdl][ok])
        print(f"{mdl:6s} running median: " + " ".join(
            f"{z[i]:.2f}:{smooth[mdl][i]:.2f}" for i in o) +
            f"   (snapshot scatter about it {np.std(res_ln):.2f} in ln k)")
    with open(os.path.join(args.out, "kmax_cut.csv"), "w") as f:
        f.write("sn,z,D," + ",".join(f"k_{m},why_{m},k_smooth_{m}" for m in CUT_MODELS)
                + "\n")
        for t in table:
            f.write(f"{t['sn']},{t['z']},{t['D']:.4f}," + ",".join(
                f"{t['kcut'][m]:.4f},{t['why'][m]},"
                f"{smooth[m][table.index(t)]:.4f}" for m in CUT_MODELS)
                + "\n")
    plot_summary(table, smooth, model, os.path.join(args.out, "kmax_summary.png"), pdf)
    for page in pages:
        plot_snapshot(*page, os.path.join(args.out, f"kmax_sn{page[0]}.png"),
                      args.p_min, pdf)
    pdf.close()
    print(f"wrote {args.out}/: kmax_scan.csv, kmax_cut.csv, kmax_z_<model>.txt, "
          "kmax_sn*.png, kmax_summary.png, kmax_scan_all.pdf (summary first)")


if __name__ == "__main__":
    sys.exit(main())
