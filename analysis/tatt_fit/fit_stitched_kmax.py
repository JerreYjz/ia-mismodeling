#!/usr/bin/env python3
"""NLA and TATT fits vs k_max to NLA + FLAMINGO stitched IA spectra.

Per snapshot:
  1. NLA fit to FLAMINGO (the recipe of the FLAMINGO notes and
     ia_halofit_v2/stitch.py --amp lowk): P_hE = b(k) P_dE with
     P_dE = -C1 P_NL, C1 = a1 Om c1rhocrit / D(z), fitted at
     k = 0.05-0.3 h/Mpc; b(k) = sqrt(P_hh/P_mm) (deterministic bias). The
     white shape-noise residual Ns of P_EE and P_BB is fitted on the same
     range with a1 fixed: the measured P_EE / NLA^2 rises from ~1.1 at
     k ~ 0.05 to ~1.5 at k ~ 0.3 and P_BB ~ 0.5 P_EE, the signature of a
     white residual; leaving Ns out of the NLA side would put a step into
     the stitched P_EE and P_BB at k_t that no model follows.
  2. Stitch, as in the notes:
         P_st(k) = [1 - w(k)] P_NLA(k) + w(k) P_FLAMINGO(k),
         w(k) = 1 / (1 + exp[-(ln k - ln k_t) / sigma_w]),
     with the notes' k_t(z) (KT below, from the figure titles) and
     sigma_w = 0.30 in ln k (measured from their w(k) curve). NLA side:
     P_hE = b P_dE, P_EE = C1^2 P_NL + Ns, P_BB = Ns.
  3. Covariance of the stitched spectra:
         C = A C_NLA A^T + diag[(1-w)^2 (f P_NLA)^2 + w^2 sigma_sim^2],
     A = (1-w) dP_NLA/d(a1, Ns): the NLA-amplitude error is fully
     correlated across k and probes; f (--sys-frac) is the fractional floor
     used for both sides, sigma_sim^2 = (mode scatter)^2 + (f P_sim)^2.
     The plotted error bars are sqrt(diag C).
  4. Fit NLA (a1, Ns) and TATT (a1, a2, b_TA, Ns) to the stitched spectra
     over k_min <= k <= k_max with the full covariance, for every k_max on
     the data's k grid; parameter errors from the Gauss-Newton covariance.

Criteria for the cut (all interpolated in ln k between grid points, and
counted only if the next k_max fails too):
  * A1 (main): a1 leaves the --nsig band of its value at k_ref
    (--kref, 0.067 h/Mpc, where w < 0.1 at every z: the NLA regime);
  * prediction: dchi2 of the k <= k_ref band, the k_max fit against the
    k_ref fit, with p = P(chi2_{N_par} > dchi2) < --p-min (degeneracies of
    a2, b_TA cost nothing here);
  * goodness of fit: P(chi2 > chi2_min) < --p-min.
Below k_t the stitched spectra are NLA by construction, so every cut is
>~ k_t: these cuts measure where NLA and TATT fail against FLAMINGO's
small scales, given NLA on large scales.

    python fit_stitched_kmax.py --ps-dir PS --out stitch_kmax
    python fit_stitched_kmax.py --ps-dir PS --probes hE --out stitch_kmax_dE

Writes stitch_kmax.pdf (summary first, then one page per snapshot),
stitch_summary.png, fits.csv, kmax_cut.csv and kmax_z_<model>_<crit>.txt
(input for make_uv_scale_cuts.py --kz-file, --kz-model dpow).
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_tatt_flamingo as F  # noqa: E402

# stitch transition k_t [h/Mpc] per snapshot, from the FLAMINGO notes
KT = {78: 0.155, 76: 0.159, 74: 0.162, 72: 0.167, 70: 0.172, 68: 0.181,
      66: 0.190, 63: 0.200, 58: 0.210, 48: 0.230, 38: 0.269, 18: 0.345,
      14: 0.447}
SIGMA_W = 0.30
FREE = {"NLA": ["a1", "Ns"], "TATT": ["a1", "a2", "bTA", "Ns"]}
GRID = {"TATT": {"a2": [-2.0, 0.0, 2.0], "bTA": [-1.0, 0.0, 1.0]}}
LO = {"a1": -30.0, "a2": -100.0, "bTA": -20.0, "Ns": -50.0}
HI = {"a1": 60.0, "a2": 100.0, "bTA": 20.0, "Ns": 50.0}
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
MODELS = ["NLA", "TATT"]
CRITS = ["A1", "pred", "gof"]
LAB = {"hE": r"$-P_{hE}$", "EE": r"$P_{EE}$", "BB": r"$P_{BB}$"}


# ---------------------------------------------------------------------------
def weight(k, kt, s=SIGMA_W):
    return 1.0/(1.0 + np.exp(-(np.log(k) - np.log(kt))/s))


def nla_fit(model, d, pre, args):
    """Step 1: a1 from P_hE at 0.05-0.3 (mode-scatter weights, as the notes),
    then Ns from P_EE and P_BB on the same range with a1 fixed."""
    k = d["k"]
    m = (k >= 0.05) & (k <= 0.3)
    D = pre["D"]
    T = -(F.COSMO["Om"]*F.C1RHOCRIT/D)*pre["bk"]*pre["PNL"]   # P_hE per a1
    P, s = d["hE"]
    w = 1.0/s[m]**2
    a1 = float(np.sum(w*P[m]*T[m])/np.sum(w*T[m]**2))
    sa1 = float(1.0/np.sqrt(np.sum(w*T[m]**2)))
    C1 = a1*F.COSMO["Om"]*F.C1RHOCRIT/D
    EE, sE = d["EE"]
    BB, sB = d["BB"]
    y = np.concatenate([EE[m] - C1**2*pre["PNL"][m], BB[m]])
    wy = 1.0/np.concatenate([sE[m], sB[m]])**2
    Ns = float(np.sum(wy*y)/np.sum(wy))
    sNs = float(1.0/np.sqrt(np.sum(wy)))
    return dict(a1=a1, sa1=sa1, Ns=Ns, sNs=sNs, C1=C1,
                a_report=a1*F.C1RHOCRIT/F.C1RHOCRIT_REPORT)


def nla_spectra(pre, nf):
    C1, P = nf["C1"], pre["PNL"]
    return {"hE": -C1*pre["bk"]*P, "EE": C1**2*P + nf["Ns"],
            "BB": np.full_like(P, nf["Ns"])}


def build_stitched(model, d, pre, nf, args):
    """Stitched spectra, the stacked vector and its covariance."""
    k = d["k"]
    w = weight(k, KT[d["sn"]]*args.kt_scale, args.width)
    N = nla_spectra(pre, nf)
    dN_da1 = {"hE": N["hE"]/nf["a1"], "EE": 2*nf["C1"]**2*pre["PNL"]/nf["a1"],
              "BB": np.zeros_like(k)}
    dN_dNs = {"hE": np.zeros_like(k), "EE": np.ones_like(k), "BB": np.ones_like(k)}
    f = args.sys_frac
    st, err_sim, y, A, dg, pk, pp = {}, {}, [], [], [], [], []
    for p in args.probes:
        Psim, ssim = d[p]
        sim_var = (args.err_scale*ssim)**2 + (f*Psim)**2
        st[p] = (1 - w)*N[p] + w*Psim
        err_sim[p] = np.sqrt(sim_var)
        y.append(st[p])
        A.append(np.column_stack([(1 - w)*dN_da1[p], (1 - w)*dN_dNs[p]]))
        dg.append((1 - w)**2*(f*N[p])**2 + w**2*sim_var)
        pk.append(k)
        pp += [p]*len(k)
    A = np.vstack(A)
    Cn = np.diag([nf["sa1"]**2, nf["sNs"]**2])
    C = A @ Cn @ A.T + np.diag(np.concatenate(dg))
    return dict(w=w, N=N, st=st, err_sim=err_sim, y=np.concatenate(y), C=C,
                kv=np.concatenate(pk), pv=np.array(pp),
                err_st={p: np.sqrt(np.diag(C))[np.array(pp) == p] for p in args.probes})


def model_vector(model, pre, th, probes):
    m = model.predict(pre["z"], pre["k"], th["a1"], th["a2"], th["bTA"], 1.0,
                      K=pre["K"], PNL=pre["PNL"], Ns=th["Ns"], bk=pre["bk"])
    return np.concatenate([m[p] for p in probes]), m


def whitener(C, sel):
    L = cholesky(C[np.ix_(sel, sel)], lower=True)
    return lambda v: solve_triangular(L, v, lower=True)


def fit(model, pre, S, free, kmax, args, base, warm=None, grid=None):
    sel = np.where((S["kv"] >= args.kmin) & (S["kv"] <= kmax*(1 + 1e-9)))[0]
    Wt = whitener(S["C"], sel)
    yw = Wt(S["y"][sel])

    def theta(x):
        th = dict(base)
        th.update(zip(free, x))
        return th

    def resid(x):
        mv, _ = model_vector(model, pre, theta(x), args.probes)
        return yw - Wt(mv[sel])

    lo = np.array([LO[n] for n in free])
    hi = np.array([HI[n] for n in free])
    starts = [[]]
    for n in free:
        starts = [s + [v] for s in starts for v in (grid or {}).get(n, [base[n]])]
    if warm is not None:
        starts.append([warm[n] for n in free])
    best = None
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9),
                          bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    cov = np.linalg.pinv(best.jac.T @ best.jac)
    chi2 = float(best.fun @ best.fun)
    dof = len(sel) - len(free)
    return dict(theta=theta(best.x), free=list(free), kmax=float(kmax),
                err={n: float(np.sqrt(abs(cov[i, i]))) for i, n in enumerate(free)},
                chi2=chi2, ndata=len(sel), dof=dof,
                p_gof=float(chi2_dist.sf(chi2, dof)) if dof > 0 else 1.0)


def band_chi2(model, pre, S, th, kref, args):
    sel = np.where((S["kv"] >= args.kmin) & (S["kv"] <= kref*(1 + 1e-9)))[0]
    Wt = whitener(S["C"], sel)
    mv, _ = model_vector(model, pre, th, args.probes)
    r = Wt(S["y"][sel] - mv[sel])
    return float(r @ r)


def crossing(k, stat, thr, above):
    """First persistent crossing of stat over thr (above=True: stat > thr
    fails), interpolated in ln k. Returns (k_cut, kind): kind 'cut',
    'none' (never fails) or 'first' (fails from the first point)."""
    k, s = np.asarray(k), np.asarray(stat, float)
    fail = s > thr if above else s < thr
    if fail[0] and (len(s) == 1 or fail[1]):
        return float(k[0]), "first"
    for i in range(1, len(s)):
        if fail[i] and (i + 1 == len(s) or fail[i + 1]):
            j = i - 1
            g = (lambda v: v) if above else (lambda v: np.log(max(v, 1e-300)))
            a, b, t0 = g(s[j]), g(s[i]), g(thr)
            t = np.clip((t0 - a)/(b - a), 0, 1) if b != a else 1.0
            return float(np.exp(np.log(k[j]) + t*(np.log(k[i]) - np.log(k[j])))), "cut"
    return float(k[-1]), "none"


# ---------------------------------------------------------------------------
def plot_snapshot(d, pre, nf, S, res, cuts, args, pdf):
    import matplotlib.pyplot as plt
    k, z, sn = d["k"], d["z"], d["sn"]
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    # --- row 1: stitched spectra and best fits at each model's A1 cut
    for j, p in enumerate(args.probes):
        ax = fig.add_subplot(gs[0, j])
        sg = -1.0 if p == "hE" else 1.0
        P, s = d[p]
        ax.errorbar(k, sg*P, S["err_sim"][p], fmt="o", ms=2.5, color="0.6",
                    alpha=0.7, label="FLAMINGO")
        ax.errorbar(k*1.02, sg*S["st"][p], S["err_st"][p], fmt="s", ms=3,
                    color="k", label="stitched (error from C)")
        ax.plot(k, sg*S["N"][p], ":", color="0.3", label="NLA fit (stitch)")
        for mdl in MODELS:
            kc = cuts[mdl]["A1"][0]
            rows = res[mdl]
            i = int(np.argmin(np.abs(np.log([r["kmax"] for r in rows]) - np.log(kc))))
            th = rows[i]["theta"]
            mm = pre["model"].predict(z, k, th["a1"], th["a2"], th["bTA"], 1.0,
                                      K=pre["K"], PNL=pre["PNL"], Ns=th["Ns"],
                                      bk=pre["bk"])
            ax.plot(k, sg*mm[p], "-", color=COL[mdl], lw=1.4,
                    label=f"{mdl} fit to k_max = {rows[i]['kmax']:.2f}")
        ax.axvline((KT[sn]*args.kt_scale), color="0.5", ls="--", lw=0.8)
        ax.set_xscale("log")
        if p == "BB":
            ax.set_yscale("symlog", linthresh=0.3*np.median(np.abs(S["st"][p])))
        else:
            ax.set_yscale("log")
            pos = sg*S["st"][p]
            ax.set_ylim(0.3*np.min(pos[pos > 0]), 3*np.max(pos))
        ax.set_xlim(0.008, 3.5)
        ax.set_title(LAB[p] + "   (dashed: k_t)")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        if j == 0:
            ax.legend(fontsize=7, loc="lower left")
    axw = fig.add_subplot(gs[0, 3])
    axw.plot(k, S["w"], "-", color="#e76f51")
    axw.axvline((KT[sn]*args.kt_scale), color="0.5", ls="--", lw=0.8)
    axw.set_xscale("log")
    axw.set_title(f"w(k): k_t = {(KT[sn]*args.kt_scale):.3f}, width {args.width} in ln k")
    axw.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    # --- row 2: a1, a2, bTA vs k_max; p-values
    ref = {m: res[m][0] for m in MODELS}
    for j, n in enumerate(["a1", "a2", "bTA"]):
        ax = fig.add_subplot(gs[1, j])
        for mdl in MODELS:
            if n not in FREE[mdl]:
                continue
            rows = res[mdl]
            km = np.array([r["kmax"] for r in rows])
            v = np.array([r["theta"][n] for r in rows])
            e = np.array([r["err"][n] for r in rows])
            off = 0.98 if mdl == "NLA" else 1.02
            if n == "a1":
                a0, s0 = ref[mdl]["theta"]["a1"], ref[mdl]["err"]["a1"]
                ax.axhspan(a0 - args.nsig*s0, a0 + args.nsig*s0, color=COL[mdl],
                           alpha=0.12)
                ax.axvline(cuts[mdl]["A1"][0], color=COL[mdl], ls="--", lw=1.2)
            ax.errorbar(km*off, v, e, fmt="o", ms=3.5, color=COL[mdl], label=mdl)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_title(n + (f"  (bands: {args.nsig:g} sigma at k_ref; dashed: A1 cut)"
                          if n == "a1" else ""), fontsize=9)
        if j == 0:
            ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 3])
    for mdl in MODELS:
        rows = res[mdl]
        km = [r["kmax"] for r in rows]
        ax.plot(km, [max(r["p_gof"], 1e-12) for r in rows], "-o", ms=3,
                color=COL[mdl], label=f"{mdl} fit")
        ax.plot(km, [max(r["p_pred"], 1e-12) for r in rows], "--s", ms=3,
                color=COL[mdl], label=f"{mdl} large scales")
    ax.axhline(args.p_min, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    ax.set_xlabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    ax.set_title("p-values", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        f"sn{sn}  z = {z:.2f}   NLA stitch amplitude a_NLA = {nf['a_report']:.3f} +- "
        f"{nf['sa1']*F.C1RHOCRIT/F.C1RHOCRIT_REPORT:.3f} (notes' units), Ns = {nf['Ns']:.3f}"
        f"     A1 cut: NLA {cuts['NLA']['A1'][0]:.2f}, TATT {cuts['TATT']['A1'][0]:.2f} h/Mpc",
        fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    fig.savefig(os.path.join(args.out, f"stitch_sn{sn}.png"), dpi=100)
    plt.close(fig)


def plot_summary(table, args, pdf):
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    z = np.array([t["z"] for t in table])
    o = np.argsort(z)
    Om = F.COSMO["Om"]
    E = lambda a: np.sqrt(Om/a**3 + 1 - Om)
    g = lambda a: 2.5*Om*E(a)*quad(lambda x: 1/(x*E(x))**3, 1e-6, a)[0]
    zz = np.linspace(0, z.max(), 120)
    Dz = np.array([g(1/(1 + x))/g(1) for x in zz])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    sty = {"A1": ("-o", 2.0), "pred": ("--s", 1.2), "gof": (":^", 1.2)}
    nm = {"A1": f"A1 leaves {args.nsig:g} sigma band", "pred": "large-scale prediction",
          "gof": "goodness of fit"}
    for ax, mdl in zip(axs, MODELS):
        for c in CRITS:
            kc = np.array([t["cut"][mdl][c][0] for t in table])
            kind = np.array([t["cut"][mdl][c][1] for t in table])
            ls, lw = sty[c]
            ax.plot(z[o], kc[o], ls, color=COL[mdl], lw=lw, ms=4,
                    alpha=1 if c == "A1" else 0.6, label=nm[c])
            nn = kind == "none"
            if nn.any():
                ax.plot(z[nn], kc[nn], "^", color="k", ms=6, mfc="none")
        ax.plot(z[o], [(KT[t["sn"]]*args.kt_scale) for t in np.array(table)[o]], "-", color="#e76f51",
                lw=1, label="stitch k_t")
        ax.plot(zz, 0.35/Dz, ":", color="0.4", label="0.35/D(z)")
        ax.set_yscale("log")
        ax.set_xlabel("z")
        ax.set_title(f"{mdl}: k_max from fits to the stitched spectra")
        ax.legend(fontsize=7)
    axs[0].set_ylabel(r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    fig.text(0.5, 0.005, "open black triangles: never fails up to the end of the "
             "k grid (lower bound)", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "stitch_summary.png"), dpi=130)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True)
    ap.add_argument("--out", default="stitch_kmax")
    ap.add_argument("--snapshots", type=int, nargs="*", default=None)
    ap.add_argument("--probes", nargs="+", default=["hE", "EE", "BB"],
                    choices=["hE", "EE", "BB"])
    ap.add_argument("--kmin", type=float, default=0.01)
    ap.add_argument("--kref", type=float, default=0.067)
    ap.add_argument("--kmax-hi", type=float, default=1.8)
    ap.add_argument("--err-scale", type=float, default=1.0)
    ap.add_argument("--sys-frac", type=float, default=0.02)
    ap.add_argument("--nsig", type=float, default=1.0)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--halofit", default="takahashi")
    ap.add_argument("--kt-scale", type=float, default=1.0,
                    help="multiply the notes' k_t (stitch-dependence test)")
    ap.add_argument("--width", type=float, default=SIGMA_W,
                    help="logistic width in ln k (notes: 0.30)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(args.out, exist_ok=True)
    sns = args.snapshots or sorted(F.SNAPSHOTS, key=lambda s: F.SNAPSHOTS[s][0])
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = F.TATTModel(halofit=args.halofit)

    fcsv = open(os.path.join(args.out, "fits.csv"), "w")
    fcsv.write("sn,z,model,kmax,ndata,dof,chi2,p_gof,p_pred,a1_dev,a1,a1_err,a2,"
               "a2_err,bTA,bTA_err,Ns,Ns_err\n")
    table, pages = [], []
    for sn in sns:
        d = F.load_snapshot(args.ps_dir, sn, 1)
        z, k = d["z"], d["k"]
        PNL = model.Pnl(z, k)
        pre = dict(z=z, k=k, D=model.D(z), PNL=PNL, K=model.kernels(z, k),
                   bk=F.scale_bias(d, PNL), model=model)
        nf = nla_fit(model, d, pre, args)
        S = build_stitched(model, d, pre, nf, args)
        base = dict(a1=nf["a1"], a2=0.0, bTA=0.0, Ns=nf["Ns"])
        kgrid = k[(k >= args.kref*(1 - 1e-6)) & (k <= args.kmax_hi)]
        res, cut = {}, {}
        for mdl in MODELS:
            rows, warm = [], None
            for km in kgrid:
                r = fit(model, pre, S, FREE[mdl], km, args, base, warm, GRID.get(mdl))
                rows.append(r)
                warm = r["theta"]
            r0 = rows[0]
            c_ref = band_chi2(model, pre, S, r0["theta"], r0["kmax"], args)
            for r in rows:
                q = max(band_chi2(model, pre, S, r["theta"], r0["kmax"], args) - c_ref, 0)
                r["p_pred"] = float(chi2_dist.sf(q, len(FREE[mdl]))) if r is not r0 else 1.0
                r["a1_dev"] = abs(r["theta"]["a1"] - r0["theta"]["a1"])/r0["err"]["a1"]
                th, e = r["theta"], r["err"]
                fcsv.write(f"{sn},{z},{mdl},{r['kmax']:.4f},{r['ndata']},{r['dof']},"
                           f"{r['chi2']:.3f},{r['p_gof']:.3e},{r['p_pred']:.3e},"
                           f"{r['a1_dev']:.3f},{th['a1']:.5f},{e['a1']:.5f},"
                           f"{th['a2']:.5f},{e.get('a2', 0):.5f},{th['bTA']:.5f},"
                           f"{e.get('bTA', 0):.5f},{th['Ns']:.5f},{e['Ns']:.5f}\n")
            km = [r["kmax"] for r in rows]
            cut[mdl] = {"A1": crossing(km, [r["a1_dev"] for r in rows], args.nsig, True),
                        "pred": crossing(km, [r["p_pred"] for r in rows], args.p_min, False),
                        "gof": crossing(km, [r["p_gof"] for r in rows], args.p_min, False)}
            res[mdl] = rows
        table.append(dict(sn=sn, z=z, cut=cut, a_nla=nf["a_report"]))
        pages.append((d, pre, nf, S, res, cut))
        print(f"sn{sn:<3d} z={z:4.2f} k_t={(KT[sn]*args.kt_scale):.3f} a_NLA={nf['a_report']:.3f} "
              f"Ns={nf['Ns']:.3f} | " + "  ".join(
                  f"{m}: " + " ".join(f"{c} {cut[m][c][0]:.2f}{'+' if cut[m][c][1] == 'none' else ''}"
                                      for c in CRITS) for m in MODELS), flush=True)
    fcsv.close()

    with open(os.path.join(args.out, "kmax_cut.csv"), "w") as f:
        f.write("sn,z,k_t," + ",".join(f"{m}_{c},{m}_{c}_kind" for m in MODELS
                                       for c in CRITS) + "\n")
        for t in table:
            f.write(f"{t['sn']},{t['z']},{(KT[t['sn']]*args.kt_scale)}," + ",".join(
                f"{t['cut'][m][c][0]:.4f},{t['cut'][m][c][1]}" for m in MODELS
                for c in CRITS) + "\n")
    zs = np.array([t["z"] for t in table])
    o = np.argsort(zs)
    for m in MODELS:
        for c in CRITS:
            rows = [(zs[i], table[i]["cut"][m][c][0]) for i in o
                    if table[i]["cut"][m][c][1] in ("cut", "none")]
            np.savetxt(os.path.join(args.out, f"kmax_z_{m.lower()}_{c}.txt"), rows,
                       fmt="%.3f %.4f", header=f"z  k_max [h/Mpc], {m}, criterion {c}")

    pdf = PdfPages(os.path.join(args.out, "stitch_kmax.pdf"))
    plot_summary(table, args, pdf)
    for pg in pages:
        plot_snapshot(*pg, args, pdf)
    pdf.close()
    print(f"wrote {args.out}/: stitch_kmax.pdf (summary first), stitch_summary.png, "
          "stitch_sn*.png, fits.csv, kmax_cut.csv, kmax_z_<model>_<crit>.txt  "
          "('+': never fails within the grid)")


if __name__ == "__main__":
    sys.exit(main())
