#!/usr/bin/env python3
"""FLAMINGO-only scale cuts, revised (v3).

Changes with respect to v2/fit_models.py
  * the white level N_s is FIXED to the B-mode (shape-noise) value, so a
    white II excess counts as model failure instead of being absorbed by a
    free nuisance (v2 fitted N_s 2-5 times the B-mode value);
  * TATT bounded to the chain priors (a2 in [-5, 5], b_TA in [0, 2]), a1 in
    [0, 20]; several starting points;
  * snapshot selection: core = 0.2 <= z <= 2 (z = 0, 0.1: 2-4 M galaxies and
    < 0.1% of Roman's weight; z = 3, 4: 0.5-1.6 M galaxies and < 1% of the
    weight);
  * four cut criteria per snapshot and model, all from the simulation:
      GoF      refit at every k_max, p(chi2, dof) of the fit to the measured
               spectra; cut = last k_max before two consecutive p < 0.05
      anchored parameters fixed by the fit to k <= 0.16 (large scales),
               cumulative p of the extrapolation to k_max; same rule
      5% / 10% anchored model vs the noise-free (smoothed) spectra:
               max_k |model / truth - 1| over k <= k_max, for GI (P_dE) and
               II (P_EE - N_s) separately; cut = last k_max within tolerance
  * Delta chi2_FL(k_max) of the refit against the noise-free spectra;
  * A1(z), A2(z), b_TA(z) with errors at k_max = 0.16 (anchor) and at the
    snapshot's own GoF cut, plus power-law fits A (1+z)/1.62)^eta.
Reads v2/fits.npz (spectra, errors, smoothed truth, b(k), P_NL, N_s).
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402

KMIN = 0.05
KANCHOR = 0.1615
CORE = (0.2, 2.0)
PROBES = ("hE", "EE", "BB")
FREE = {"NLA": ["a1"], "TATT": ["a1", "a2", "bTA"]}
LO = {"a1": 0.0, "a2": -5.0, "bTA": 0.0}
HI = {"a1": 20.0, "a2": 5.0, "bTA": 2.0}
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


def pvalue(chi2, w, npar, nprobes=None):
    """p-value of a fit chi2 whose residuals have variance w_k^2 sigma_k^2
    (stitched truth; w = 1 is the standard chi2 test). For w != 1 the
    statistic sum w^2 Z^2 is matched to c chi2_nu (Satterthwaite):
    c nu = sum w^2 - npar, 2 c^2 nu = 2 sum w^4. Returns (effective dof, p);
    p = 1 (no test possible) when sum w^2 - npar < 1."""
    nprobes = PVALUE_NPROBES if nprobes is None else nprobes
    w = np.asarray(w, float)
    if np.all(w == 1.0):
        dof = nprobes*w.size - npar
        return float(dof), float(chi2_dist.sf(chi2, dof))
    s2, s4 = nprobes*np.sum(w**2), nprobes*np.sum(w**4)
    mu = s2 - npar
    if mu < 1.0:
        return np.nan, 1.0
    c = s4/mu
    nu = mu**2/s4
    return float(mu), float(chi2_dist.sf(chi2/c, nu))


PVALUE_NPROBES = 3


def cut_rule(kg, ok):
    """Last k_max before two consecutive failures (ok: boolean per k_max)."""
    for j in range(len(kg) - 1):
        if not ok[j] and not ok[j + 1]:
            return kg[j - 1] if j > 0 else np.nan
    return kg[-1] if ok[-1] else kg[-2]


def main(truth=None):
    """truth (optional): dict with y, T (observed units, [nsn, 3, nk]), Ns_EE
    (white level of P_EE per snapshot), w (stitch weight [nsn, nk]; the
    effective dof is sum w^2), probes (indices into hE, EE, BB), outdir,
    label. Default: the production analysis (measured data, spline truth,
    N_s = B plateau, three probes, w = 1)."""
    f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
    zs, sns, kd = f["zs"], f["sns"], f["k"][0]
    y, err, T, b, Pnl, Ns = f["y"], f["err"], f["T"], f["b"], f["PNL"], f["Ns_true"]
    nsn = len(zs)
    truth = truth or {}
    y = truth.get("y", y)
    T = truth.get("T", T)
    err = truth.get("err", err)
    NsE = truth.get("Ns_EE", Ns)
    W = truth.get("w", np.ones((nsn, kd.size)))
    PI = list(truth.get("probes", [0, 1, 2]))
    global PVALUE_NPROBES
    PVALUE_NPROBES = len(PI)
    outdir = truth.get("outdir", HERE)
    label = truth.get("label", "")
    os.makedirs(outdir, exist_ok=True)
    NSADD = [np.zeros(nsn), NsE, Ns]                 # white level added to hE, EE, BB
    core = (zs >= CORE[0]) & (zs <= CORE[1])
    kg = kd[(kd >= 0.08) & (kd <= 3.1)]           # k_max grid: the data nodes
    model = F.TATTModel(halofit="takahashi")
    out = dict(zs=zs, sns=sns, kgrid=kg, core=core)

    def pred(s, K, a1, a2, bTA):
        pr = model.predict(zs[s], kd, a1, a2, bTA, 1.0, K=K, PNL=Pnl[s], Ns=0.0, bk=b[s])
        return np.array([pr[p] + NSADD[i][s] for i, p in enumerate(PROBES)])   # observed units, N_s included

    def fit(s, K, mdl, kmax):
        m = (kd >= KMIN) & (kd <= kmax*(1 + 1e-9))
        names = FREE[mdl]

        def th(x):
            d = dict(a1=0.0, a2=0.0, bTA=0.0)
            d.update(zip(names, x))
            return d

        def resid(x):
            t = th(x)
            return ((y[s] - pred(s, K, t["a1"], t["a2"], t["bTA"]))/err[s])[PI][:, m].ravel()
        starts = [[4.0]] if mdl == "NLA" else [[4.0, 0.0, 0.1], [4.0, -1.0, 0.5], [4.0, 1.0, 0.5], [4.0, 0.0, 1.0]]
        lo = np.array([LO[n] for n in names])
        hi = np.array([HI[n] for n in names])
        best = None
        for x0 in starts:
            r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
        c2 = float(best.fun @ best.fun)
        dof, pval = pvalue(c2, W[s][m], len(names))
        cov = np.linalg.pinv(best.jac.T @ best.jac)
        return dict(x=best.x, th=th(best.x), chi2=c2, dof=dof, p=pval,
                    err=np.sqrt(np.abs(np.diag(cov))), m=m)

    res = {mdl: dict(chi2=np.zeros((nsn, len(kg))), dof=np.zeros((nsn, len(kg))), p=np.zeros((nsn, len(kg))),
                     chi2_anch=np.zeros((nsn, len(kg))), p_anch=np.zeros((nsn, len(kg))),
                     dev_GI=np.zeros((nsn, len(kg))), dev_II=np.zeros((nsn, len(kg))),
                     dFL=np.zeros((nsn, len(kg))), x=np.zeros((nsn, len(kg), 3)), e=np.zeros((nsn, len(kg), 3)),
                     x_anch=np.zeros((nsn, 3)), e_anch=np.zeros((nsn, 3)), ratio_anch=np.zeros((nsn, 2, len(kd))))
           for mdl in FREE}
    for s in range(nsn):
        K = model.kernels(zs[s], kd)
        for mdl in FREE:
            R = res[mdl]
            ra = fit(s, K, mdl, KANCHOR)
            xa = ra["th"]
            R["x_anch"][s] = [xa["a1"], xa["a2"], xa["bTA"]]
            R["e_anch"][s, :len(FREE[mdl])] = ra["err"]
            pa = pred(s, K, xa["a1"], xa["a2"], xa["bTA"])
            r_all = (((y[s] - pa)/err[s])**2)[PI]                  # [nprobes, nk]
            dGI = pa[0]/T[s, 0] - 1                                  # hE: b P_dE
            dII = (pa[1] - NsE[s])/(T[s, 1] - NsE[s]) - 1
            R["ratio_anch"][s] = [dGI, dII]
            for j, kmax in enumerate(kg):
                r = fit(s, K, mdl, kmax)
                R["chi2"][s, j], R["dof"][s, j], R["p"][s, j] = r["chi2"], r["dof"], r["p"]
                R["x"][s, j] = [r["th"]["a1"], r["th"]["a2"], r["th"]["bTA"]]
                R["e"][s, j, :len(FREE[mdl])] = r["err"]
                m = r["m"]
                pr = pred(s, K, r["th"]["a1"], r["th"]["a2"], r["th"]["bTA"])
                R["dFL"][s, j] = float(np.sum((((T[s] - pr)/err[s])**2)[PI][:, m]))
                c = float(r_all[:, m].sum())
                R["chi2_anch"][s, j] = c
                R["p_anch"][s, j] = pvalue(c, W[s][m], 0)[1]
                R["dev_GI"][s, j] = float(np.max(np.abs(dGI[m])))
                R["dev_II"][s, j] = float(np.max(np.abs(dII[m])))
        print(f"z {zs[s]:4.2f}: NLA p(0.16/0.22/0.29) {res['NLA']['p'][s, 2]:.2f}/{res['NLA']['p'][s, 3]:.2f}/"
              f"{res['NLA']['p'][s, 4]:.3f}  TATT {res['TATT']['p'][s, 2]:.2f}/{res['TATT']['p'][s, 3]:.2f}/"
              f"{res['TATT']['p'][s, 4]:.3f}", flush=True)

    # ---------------- cuts per snapshot ----------------
    cuts = {}
    for mdl in FREE:
        R = res[mdl]
        cuts[mdl] = {
            "GoF": np.array([cut_rule(kg, R["p"][s] >= 0.05) for s in range(nsn)]),
            "anchored": np.array([cut_rule(kg, R["p_anch"][s] >= 0.05) for s in range(nsn)]),
            "5%": np.array([cut_rule(kg, (R["dev_GI"][s] <= 0.05) & (R["dev_II"][s] <= 0.05)) for s in range(nsn)]),
            "10%": np.array([cut_rule(kg, (R["dev_GI"][s] <= 0.10) & (R["dev_II"][s] <= 0.10)) for s in range(nsn)]),
            "5%_GI": np.array([cut_rule(kg, R["dev_GI"][s] <= 0.05) for s in range(nsn)]),
            "5%_II": np.array([cut_rule(kg, R["dev_II"][s] <= 0.05) for s in range(nsn)]),
        }
    # combined GoF over the core snapshots
    comb = {}
    dofa_c = len(PI)*np.array([np.sum((W[core]**2)[:, (kd >= KMIN) & (kd <= k*(1 + 1e-9))]) for k in kg])
    for mdl in FREE:
        R = res[mdl]
        ok = np.isfinite(R["dof"][core])
        dof_c = np.nansum(R["dof"][core], 0)
        npar_c = len(FREE[mdl])*core.sum()
        p_c = np.array([pvalue(np.where(ok, R["chi2"][core], 0.0).sum(0)[j], W[core][:, (kd >= KMIN) & (kd <= k*(1 + 1e-9))].ravel(), npar_c)[1]
                        for j, k in enumerate(kg)])
        comb[mdl] = dict(p=p_c,
                         p_anch=np.array([pvalue(R["chi2_anch"][core].sum(0)[j], W[core][:, (kd >= KMIN) & (kd <= k*(1 + 1e-9))].ravel(), 0)[1] for j, k in enumerate(kg)]),
                         dFL=R["dFL"][core].sum(0))
    with open(os.path.join(outdir, "cuts.csv"), "w") as fh:
        fh.write("z,core," + ",".join(f"{mdl}_{c}" for mdl in FREE for c in cuts[mdl]) + "\n")
        for s in range(nsn):
            fh.write(f"{zs[s]:g},{int(core[s])}," + ",".join(f"{cuts[mdl][c][s]:.3f}" for mdl in FREE for c in cuts[mdl]) + "\n")
    with open(os.path.join(outdir, "combined.csv"), "w") as fh:
        fh.write("kmax," + ",".join(f"{mdl}_p,{mdl}_p_anch,{mdl}_dFL" for mdl in FREE) + "\n")
        for j, k in enumerate(kg):
            fh.write(f"{k:.3f}," + ",".join(f"{comb[mdl]['p'][j]:.3g},{comb[mdl]['p_anch'][j]:.3g},{comb[mdl]['dFL'][j]:.1f}" for mdl in FREE) + "\n")

    # ---------------- amplitudes A(z) ----------------
    x = (1 + zs)/1.62
    pl = lambda xx, A, eta: A*xx**eta
    amp = {}
    for mdl in FREE:
        R = res[mdl]
        jg = [int(np.argmin(np.abs(kg - c))) if np.isfinite(c) else 0 for c in cuts[mdl]["GoF"]]
        at_cut = np.array([R["x"][s, jg[s]] for s in range(nsn)])
        e_cut = np.array([R["e"][s, jg[s]] for s in range(nsn)])
        amp[mdl] = dict(anchor=R["x_anch"], e_anchor=R["e_anch"], cut=at_cut, e_cut=e_cut)
        for lab, A, E in (("anchor", R["x_anch"], R["e_anch"]), ("cut", at_cut, e_cut)):
            for i, n in enumerate(FREE[mdl]):
                if n == "bTA":
                    continue
                sel = core & (E[:, i] > 0)
                try:
                    popt, pcov = curve_fit(pl, x[sel], A[sel, i], sigma=E[sel, i], p0=[3.5, 0.7], absolute_sigma=True)
                    amp[mdl][f"pl_{n}_{lab}"] = (popt.tolist(), np.sqrt(np.diag(pcov)).tolist())
                except Exception:
                    amp[mdl][f"pl_{n}_{lab}"] = None
    with open(os.path.join(outdir, "amplitudes.csv"), "w") as fh:
        fh.write("z,core,NLA_a1_anchor,err,NLA_a1_cut,err,NLA_kcut,TATT_a1_anchor,err,TATT_a2_anchor,err,TATT_bTA_anchor,err,"
                 "TATT_a1_cut,err,TATT_a2_cut,err,TATT_bTA_cut,err,TATT_kcut\n")
        for s in range(nsn):
            A, B_ = amp["NLA"], amp["TATT"]
            fh.write(f"{zs[s]:g},{int(core[s])},{A['anchor'][s, 0]:.3f},{A['e_anchor'][s, 0]:.3f},{A['cut'][s, 0]:.3f},{A['e_cut'][s, 0]:.3f},{cuts['NLA']['GoF'][s]:.3f},"
                     + ",".join(f"{B_['anchor'][s, i]:.3f},{B_['e_anchor'][s, i]:.3f}" for i in range(3)) + ","
                     + ",".join(f"{B_['cut'][s, i]:.3f},{B_['e_cut'][s, i]:.3f}" for i in range(3)) + f",{cuts['TATT']['GoF'][s]:.3f}\n")
    summary = dict(cuts={m: {c: v.tolist() for c, v in cuts[m].items()} for m in cuts},
                   powerlaws={m: {k: v for k, v in amp[m].items() if k.startswith("pl_")} for m in amp},
                   core=core.tolist(), zs=zs.tolist())
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    np.savez(os.path.join(outdir, "v3.npz"), **out, **{f"{m}_{k}": v for m in res for k, v in res[m].items()},
             **{f"cut_{m}_{c}": v for m in cuts for c, v in cuts[m].items()},
             **{f"comb_{m}_{k}": v for m in comb for k, v in comb[m].items()})

    # ---------------- figures ----------------
    def save(fig, name):
        fig.savefig(os.path.join(outdir, name + ".png"), dpi=220)
        fig.savefig(os.path.join(outdir, name + ".pdf"))
        plt.close(fig)

    def kaxis(a, ticks=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3)):
        a.set_xscale("log")
        a.set_xticks(ticks)
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        a.xaxis.set_minor_formatter(ticker.NullFormatter())
        a.grid(alpha=0.3, which="both")

    # GoF: refit and anchored p-values
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    for i, mdl in enumerate(FREE):
        for row, key, lab in ((0, "p", "refit at each $k_{\\max}$"), (1, "p_anch", "anchored at $k \\leq 0.16$")):
            for s in range(nsn):
                c = cm.viridis(s/12) if core[s] else "0.75"
                ax[row, i].plot(kg, res[mdl][key][s], "-o", ms=3, lw=1.2 if core[s] else 0.8, color=c,
                                label=f"z={zs[s]:g}" + ("" if core[s] else " (excl.)"))
            ax[row, i].axhline(0.05, color="k", lw=0.8, ls="--")
            ax[row, i].set_title(f"{mdl}, {lab}, $N_s$ fixed{label}")
        for row, key in ((0, "p"), (1, "p_anch")):
            ax[row, 2].plot(kg, comb[mdl][key], "-o", ms=3, color=COL[mdl], label=mdl)
    ax[0, 2].set_title("combined over the core snapshots (0.2 <= z <= 2), refit")
    ax[1, 2].set_title("combined, anchored")
    for a in ax.ravel():
        kaxis(a)
        a.set_yscale("log")
        a.set_ylim(1e-6, 1.5)
        a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
        a.axhline(0.05, color="k", lw=0.8, ls="--")
    ax[0, 0].set_ylabel("p-value")
    ax[1, 0].set_ylabel("p-value")
    ax[0, 0].legend(fontsize=6, ncol=2)
    ax[0, 2].legend(fontsize=8)
    ax[1, 2].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_v3_gof")

    # fractional deviation of the anchored model from the noise-free spectra
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.6))
    mk = (kd >= KMIN) & (kd <= 3.1)
    for i, mdl in enumerate(FREE):
        for row, (idx, lab) in enumerate(((0, "GI: $b\\,P_{\\delta E}$"), (1, "II: $P_{EE} - N_s$"))):
            for s in range(nsn):
                c = cm.viridis(s/12) if core[s] else "0.75"
                ax[row, i].plot(kd[mk], 100*res[mdl]["ratio_anch"][s, idx][mk], "-o", ms=3, lw=1.2 if core[s] else 0.8,
                                color=c, label=f"z={zs[s]:g}" + ("" if core[s] else " (excl.)"))
            for lev in (5, 10):
                ax[row, i].axhline(lev, color="k", lw=0.7, ls="--")
                ax[row, i].axhline(-lev, color="k", lw=0.7, ls="--")
            ax[row, i].set_title(f"{mdl} anchored at $k \\leq 0.16$: {lab}")
            ax[row, i].set_ylim(-60, 60)
            ax[row, i].set_ylabel("model / noise-free truth - 1  [%]")
    for a in ax.ravel():
        kaxis(a, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3))
        a.set_xlabel("k [h/Mpc]")
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "fig_v3_deviation")

    # cuts: k_max on the horizontal axis, z on the vertical
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    mks = {"GoF": "o", "anchored": "s", "5%": "^", "10%": "v"}
    for i, mdl in enumerate(FREE):
        for c, m in mks.items():
            v = cuts[mdl][c]
            ax[i].plot(v[core], zs[core], m, ms=7, color=COL[mdl], alpha=0.9 if c in ("GoF", "5%") else 0.5, label=c)
            ax[i].plot(v[~core], zs[~core], m, ms=7, mfc="none", color="0.5", alpha=0.6)
        ax[i].set_title(f"{mdl}: per-snapshot $k_{{\\max}}$ (open grey: excluded snapshots)")
        ax[i].set_xlabel(r"$k_{\max}$ [h/Mpc]")
        kaxis(ax[i], ticks=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3))
        ax[i].legend(fontsize=8)
    ax[0].set_ylabel("snapshot z")
    fig.tight_layout()
    save(fig, "fig_v3_cuts")

    # amplitudes: A vs k_max per snapshot (refit), and A(z) at the anchor (A on x, z on y)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for s in range(nsn):
        c = cm.viridis(s/12) if core[s] else "0.75"
        ax[0].plot(kg, res["NLA"]["x"][s, :, 0], "-o", ms=3, color=c, label=f"z={zs[s]:g}")
        ax[1].plot(kg, res["TATT"]["x"][s, :, 0], "-o", ms=3, color=c)
        ax[2].plot(kg, res["TATT"]["x"][s, :, 1], "-o", ms=3, color=c)
    ax[0].set_title("NLA $a_1$ vs $k_{\\max}$ (refit, $N_s$ fixed)")
    ax[1].set_title("TATT $a_1$ vs $k_{\\max}$")
    ax[2].set_title("TATT $a_2$ vs $k_{\\max}$")
    for a in ax:
        kaxis(a)
        a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "fig_v3_amp_vs_kmax")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    zz = np.linspace(0, 4, 100)
    for i, (mdl, idx, lab) in enumerate((("NLA", 0, "$A_1$"), ("TATT", 0, "$A_1$"), ("TATT", 1, "$A_2$"))):
        A, E = amp[mdl]["anchor"][:, idx], amp[mdl]["e_anchor"][:, idx]
        ax[i].errorbar(A[core], zs[core], xerr=E[core], fmt="o", color=COL[mdl], capsize=3, label="core snapshots, $k \\leq 0.16$")
        ax[i].errorbar(A[~core], zs[~core], xerr=E[~core], fmt="o", mfc="none", color="0.5", capsize=3, label="excluded")
        Ac, Ec = amp[mdl]["cut"][:, idx], amp[mdl]["e_cut"][:, idx]
        ax[i].errorbar(Ac[core], zs[core] + 0.03, xerr=Ec[core], fmt="s", mfc="none", color=COL[mdl], capsize=2, alpha=0.6, label="at the GoF cut")
        key = f"pl_{FREE[mdl][idx]}_anchor"
        if amp[mdl].get(key):
            (A0, eta), (eA, ee) = amp[mdl][key]
            ax[i].plot(pl((1 + zz)/1.62, A0, eta), zz, "-", color=COL[mdl], lw=1,
                       label=f"power law: {A0:.2f}$\\pm${eA:.2f} ((1+z)/1.62)^({eta:.2f}$\\pm${ee:.2f})")
        if idx == 0:
            ax[i].plot(pl((1 + zz)/1.62, 3.5, 0.68), zz, "k:", lw=1, label="Roman fit 3.5 ((1+z)/1.62)^0.68")
        ax[i].set_xlabel(f"{mdl} {lab}")
        ax[i].set_title(f"{mdl} {lab}(z)")
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=7)
    ax[0].set_ylabel("snapshot z")
    fig.tight_layout()
    save(fig, "fig_v3_amplitudes")

    # Delta chi2_FL vs k_max (core sum), fixed N_s
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for mdl in FREE:
        ax.plot(kg, comb[mdl]["dFL"], "-o", ms=4, color=COL[mdl], label=f"{mdl}, core snapshots")
    kaxis(ax)
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_{\max}$ [h/Mpc]")
    ax.set_ylabel(r"$\Delta\chi^2_{\rm FL}$ (refit vs noise-free spectra, $N_s$ fixed)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_v3_dchi2fl")
    print(open(os.path.join(outdir, "cuts.csv")).read())
    print(open(os.path.join(outdir, "combined.csv")).read())
    print(json.dumps(summary["powerlaws"], indent=1))


if __name__ == "__main__":
    main()
