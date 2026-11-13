#!/usr/bin/env python3
"""FLAMINGO versus the NLA / TATT best fits, and the goodness of fit as chi2/dof.

Inputs: v2/fits.npz (measured spectra y, errors, smoothed truth T, b(k),
P_NL, N_s = B plateau) and v3/v3.npz (best-fit parameters NLA_x / TATT_x
[snapshot, k_max, (a1, a2, bTA)], chi2 and dof per k_max, the p-value cuts).
The model prediction is the one used in the fits (fit_tatt_flamingo.TATTModel
with the FLAMINGO P_NL and b(k); N_s fixed to the B plateau).

Figures (220 dpi PNG + PDF)
  fig_cmp_GI_k<kmax>, fig_cmp_II_k<kmax>: top row the spectra ratio
      (P_dE / P_NL = P_hE / (b P_NL); (P_EE - N_s) / P_NL) for the selected
      snapshots with the measured points, the smoothed truth and the two
      best fits to k <= kmax; bottom row the fractional residual at the
      measured points, 100 (model / measured - 1), with the measured
      error bars (100 sigma / |measured|); the smoothed truth is shown the
      same way as a thin grey line
  fig_cmp_BB_k<kmax>: the B-mode spectrum (TATT predicts a B mode, NLA none)
  fig_chi2dof: chi2/dof against k_max per core snapshot, with the p = 0.05
      equivalent threshold chi2_0.95(nu)/nu
  fig_chi2dof_comb: combined (core snapshots) chi2/dof
  fig_chi2dof_cuts: cuts from the p rule and from fixed chi2/dof thresholds
Tables: chi2dof.csv (per snapshot and k_max), chi2dof_cuts.csv.
"""
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402

KMIN = 0.05
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
ZSEL = (0.2, 0.5, 0.75, 1.0, 1.5, 2.0)
KFITS = (0.161, 0.291, 0.523)
THRS = (1.5, 2.0, 3.0)
NPAR = {"NLA": 1, "TATT": 3}

f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
v = np.load(os.path.join(HERE, "v3.npz"))
zs, kd = f["zs"], f["k"][0]
y, err, T, b, Pnl, Ns = f["y"], f["err"], f["T"], f["b"], f["PNL"], f["Ns_true"]
kg, core = v["kgrid"], v["core"]
model = F.TATTModel(halofit="takahashi")
NSADD = [np.zeros(len(zs)), Ns, Ns]
PROBES = ("hE", "EE", "BB")


def pred(s, a1, a2, bTA):
    pr = model.predict(zs[s], kd, a1, a2, bTA, 1.0, PNL=Pnl[s], Ns=0.0, bk=b[s])
    return np.array([pr[p] + NSADD[i][s] for i, p in enumerate(PROBES)])


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=220)
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    plt.close(fig)


def kaxis(a, lo=0.045, hi=3.3):
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(ticker.FixedLocator([0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.grid(alpha=0.3, which="both")


SEL = [int(np.argmin(np.abs(zs - z))) for z in ZSEL]
KIDX = {kf: int(np.argmin(np.abs(kg - kf))) for kf in KFITS}
kmask = kd >= KMIN

# ---------------- stacked comparison + fractional difference ----------------
for kf, j in KIDX.items():
    tag = f"k{kg[j]:.2f}".replace(".", "p")
    for probe, pi in (("GI", 0), ("II", 1)):
        fig, ax = plt.subplots(2, len(SEL), figsize=(3.3*len(SEL), 7.2), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.05, wspace=0.28))
        for c, s in enumerate(SEL):
            norm = b[s]*Pnl[s] if pi == 0 else Pnl[s]
            off = 0.0 if pi == 0 else Ns[s]
            sgn = -1.0 if pi == 0 else 1.0
            a0, a1_ = ax[0, c], ax[1, c]
            dat = sgn*(y[s, pi] - off)/norm
            de = err[s, pi]/norm
            tru = sgn*(T[s, pi] - off)/norm
            a0.errorbar(kd[kmask], dat[kmask], de[kmask], fmt="o", ms=3, color="0.25", lw=0.9, capsize=0, label="FLAMINGO (measured)")
            a0.plot(kd[kmask], tru[kmask], "-", color="0.55", lw=1.2, label="FLAMINGO (smoothed truth)")
            a1_.plot(kd[kmask], 100*(tru/dat - 1)[kmask], "-", color="0.6", lw=0.9, label="smoothed truth / measured $-$ 1")
            for mdl in ("NLA", "TATT"):
                x = v[f"{mdl}_x"][s, j]
                p = pred(s, *x)
                mod = sgn*(p[pi] - off)/norm
                lab = f"{mdl}: $A_1$ = {x[0]:.2f}" + ("" if mdl == "NLA" else f", $A_2$ = {x[1]:.2f}, $b_{{TA}}$ = {x[2]:.2f}")
                a0.plot(kd[kmask], mod[kmask], "-", color=COL[mdl], lw=1.6, label=lab)
                ksh = kd*(0.985 if mdl == "NLA" else 1.015)         # small offset so the two error bars do not overlap
                a1_.errorbar(ksh[kmask], 100*(mod/dat - 1)[kmask], 100*(de/np.abs(dat))[kmask], fmt="o", ms=3, color=COL[mdl], lw=1.0, capsize=1.5, label=f"{mdl} / measured $-$ 1")
                a1_.plot(ksh[kmask], 100*(mod/dat - 1)[kmask], "-", color=COL[mdl], lw=0.7, alpha=0.6)
            for a in (a0, a1_):
                a.axvline(kg[j], color="k", lw=0.8, ls="--")
                a.axvspan(kg[j], 10, color="0.93", zorder=0)
                kaxis(a)
            a1_.axhline(0, color="k", lw=0.6)
            a1_.set_ylim(-60, 60)
            a1_.axhline(10, color="0.5", lw=0.6, ls=":")
            a1_.axhline(-10, color="0.5", lw=0.6, ls=":")
            a0.set_title(f"z = {zs[s]:g}", fontsize=10)
            a0.legend(fontsize=5.6, loc="best")
            a1_.set_xlabel(r"$k$ [h/Mpc]")
            if pi == 0:
                a0.set_yscale("log")
                a0.set_ylim(0.5*np.min(np.abs(tru[kmask & (kd < 3.3)])), 1.6*np.max(np.abs(tru[kmask & (kd < 3.3)])))
            else:
                a0.set_yscale("log")
                tt = tru[kmask & (kd < 3.3)]
                a0.set_ylim(0.5*np.min(tt[tt > 0]), 1.8*np.max(tt))
            if c == 0:
                a0.set_ylabel(r"$-P_{\delta E}/P_{\rm NL} = -P_{hE}/(b\,P_{\rm NL})$" if pi == 0 else r"$(P_{EE}-N_s)/P_{\rm NL}$")
                a1_.set_ylabel("model / measured $-$ 1  [%]")
                a1_.legend(fontsize=6, loc="upper left")
        fig.suptitle(f"{'GI (P_dE)' if pi == 0 else 'II (P_EE minus N_s)'}: FLAMINGO versus the NLA and TATT best fits to $k \\leq$ {kg[j]:.2f} h/Mpc "
                     f"(shaded: beyond the fit range)", fontsize=11, y=0.995)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.99)
        save(fig, f"fig_cmp_{probe}_{tag}")
    # B modes
    fig, ax = plt.subplots(1, len(SEL), figsize=(3.3*len(SEL), 3.6), sharex=True)
    for c, s in enumerate(SEL):
        a = ax[c]
        norm = Pnl[s]
        a.errorbar(kd[kmask], ((y[s, 2] - Ns[s])/norm)[kmask], (err[s, 2]/norm)[kmask], fmt="o", ms=3, color="0.25", lw=0.9, capsize=0, label="FLAMINGO (measured)")
        a.plot(kd[kmask], ((T[s, 2] - Ns[s])/norm)[kmask], "-", color="0.55", lw=1.2, label="smoothed truth")
        for mdl in ("NLA", "TATT"):
            x = v[f"{mdl}_x"][s, j]
            p = pred(s, *x)
            a.plot(kd[kmask], ((p[2] - Ns[s])/norm)[kmask], "-", color=COL[mdl], lw=1.6, label=mdl)
        a.axvline(kg[j], color="k", lw=0.8, ls="--")
        a.axvspan(kg[j], 10, color="0.93", zorder=0)
        kaxis(a)
        a.set_yscale("symlog", linthresh=1e-4)
        a.set_ylim(-5e-4, 1e-2)
        a.axhline(0, color="k", lw=0.6)
        a.set_title(f"z = {zs[s]:g}", fontsize=10)
        a.set_xlabel(r"$k$ [h/Mpc]")
        if c == 0:
            a.set_ylabel(r"$(P_{BB}-N_s)/P_{\rm NL}$")
            a.legend(fontsize=6)
    fig.suptitle(f"B modes: FLAMINGO versus the best fits to $k \\leq$ {kg[j]:.2f} h/Mpc ($N_s$ subtracted; NLA predicts none)", fontsize=11)
    fig.tight_layout()
    save(fig, f"fig_cmp_BB_{tag}")

# ---------------- chi2/dof ----------------
r = {mdl: v[f"{mdl}_chi2"]/v[f"{mdl}_dof"] for mdl in NPAR}
thr05 = {mdl: chi2_dist.isf(0.05, v[f"{mdl}_dof"][0])/v[f"{mdl}_dof"][0] for mdl in NPAR}   # dof identical across snapshots
cz = np.where(core)[0]
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
for i, mdl in enumerate(NPAR):
    a = ax[i]
    for n, s in enumerate(cz):
        a.plot(kg, r[mdl][s], "-o", ms=3.5, lw=1.2, color=cmap(n/(len(cz) - 1)), label=f"z = {zs[s]:g}")
    a.plot(kg, thr05[mdl], "k--", lw=1.4, label=r"$p = 0.05$ equivalent: $\chi^2_{0.95}(\nu)/\nu$")
    for t, ls in zip(THRS, (":", "-.", (0, (1, 1)))):
        a.axhline(t, color="0.4", lw=0.9, ls=ls, label=rf"$\chi^2/\nu$ = {t:g}")
    kaxis(a, 0.08, 3.3)
    a.set_yscale("log")
    a.set_ylim(0.2, 300)
    a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
    a.set_title(f"{mdl} ({NPAR[mdl]} free parameter{'s' if NPAR[mdl] > 1 else ''}; $\\nu = 3N_k - {NPAR[mdl]}$)", fontsize=10.5)
    a.legend(fontsize=7, ncol=2)
ax[0].set_ylabel(r"$\chi^2_{\min}(k_{\max})/\nu$")
fig.suptitle(r"Goodness of fit as $\chi^2/\nu$: refit at every $k_{\max}$, three probes ($P_{hE}$, $P_{EE}$, $P_{BB}$), $k \geq 0.05$, $N_s$ fixed to the B plateau", fontsize=11)
fig.tight_layout()
save(fig, "fig_chi2dof")

# combined
fig, ax = plt.subplots(figsize=(6.4, 4.6))
nu_c = {}
for mdl in NPAR:
    chi_c = v[f"{mdl}_chi2"][core].sum(0)
    nu_c[mdl] = v[f"{mdl}_dof"][core].sum(0)
    ax.plot(kg, chi_c/nu_c[mdl], "-o", ms=4, color=COL[mdl], label=f"{mdl}: $\\chi^2/\\nu$ (9 core snapshots)")
    ax.plot(kg, chi2_dist.isf(0.05, nu_c[mdl])/nu_c[mdl], "--", color=COL[mdl], lw=1, label=f"{mdl}: $p = 0.05$ equivalent")
for t, ls in zip(THRS, (":", "-.", (0, (1, 1)))):
    ax.axhline(t, color="0.4", lw=0.9, ls=ls, label=rf"$\chi^2/\nu$ = {t:g}")
kaxis(ax, 0.08, 3.3)
ax.set_yscale("log")
ax.set_ylim(0.3, 300)
ax.set_xlabel(r"$k_{\max}$ [h/Mpc]")
ax.set_ylabel(r"$\sum_z \chi^2_{\min} / \sum_z \nu$")
ax.set_title(r"Combined $\chi^2/\nu$ over $0.2 \leq z \leq 2$", fontsize=10.5)
ax.legend(fontsize=7)
fig.tight_layout()
save(fig, "fig_chi2dof_comb")


def cut_rule(ok):
    for j in range(len(kg) - 1):
        if not ok[j] and not ok[j + 1]:
            return kg[j - 1] if j > 0 else np.nan
    return kg[-1] if ok[-1] else kg[-2]


cuts = {}
for mdl in NPAR:
    cuts[(mdl, "p")] = v[f"cut_{mdl}_GoF"]
    for t in THRS:
        cuts[(mdl, t)] = np.array([cut_rule(r[mdl][s] <= t) for s in range(len(zs))])
fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
MK = {"p": "o", 1.5: "s", 2.0: "^", 3.0: "D"}
for i, mdl in enumerate(NPAR):
    a = ax[i]
    for key, mk in MK.items():
        c = cuts[(mdl, key)].copy()
        below = np.isnan(c)
        c[below] = 0.075                                   # the first two k_max already fail: cut below the grid
        lab = "p rule: two consecutive $p < 0.05$" if key == "p" else rf"two consecutive $\chi^2/\nu > {key:g}$"
        a.plot(c[core], zs[core], marker=mk, ms=7 if key == "p" else 5.5, ls="-" if key == "p" else ":", color=COL[mdl] if key == "p" else "0.3", mfc="none" if key != "p" else COL[mdl], label=lab)
    for key in MK:
        for s in np.where(np.isnan(cuts[(mdl, key)]) & core)[0]:
            a.annotate("", (0.062, zs[s]), (0.075, zs[s]), arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    kaxis(a, 0.055, 0.8)
    a.set_xlabel(r"$k_{\max}$ cut [h/Mpc]  (arrow: below the grid, $< 0.09$)")
    a.set_title(f"{mdl}", fontsize=10.5)
    a.legend(fontsize=7.5)
ax[0].set_ylabel("z")
fig.suptitle(r"Scale cuts from the p rule and from fixed $\chi^2/\nu$ thresholds (core snapshots)", fontsize=11)
fig.tight_layout()
save(fig, "fig_chi2dof_cuts")

with open(os.path.join(HERE, "chi2dof.csv"), "w") as fh:
    fh.write("model,z," + ",".join(f"k{k:.3f}" for k in kg) + "\n")
    fh.write("dof,NLA," + ",".join(f"{d:.0f}" for d in v["NLA_dof"][0]) + "\n")
    fh.write("dof,TATT," + ",".join(f"{d:.0f}" for d in v["TATT_dof"][0]) + "\n")
    fh.write("thr_p0.05,NLA," + ",".join(f"{t:.2f}" for t in thr05["NLA"]) + "\n")
    fh.write("thr_p0.05,TATT," + ",".join(f"{t:.2f}" for t in thr05["TATT"]) + "\n")
    for mdl in NPAR:
        for s in cz:
            fh.write(f"{mdl},{zs[s]:g}," + ",".join(f"{x:.2f}" for x in r[mdl][s]) + "\n")
        fh.write(f"{mdl},combined," + ",".join(f"{x:.2f}" for x in v[f"{mdl}_chi2"][core].sum(0)/nu_c[mdl]) + "\n")
with open(os.path.join(HERE, "chi2dof_cuts.csv"), "w") as fh:
    fh.write("z," + ",".join(f"{m}_{k if k == 'p' else f'chi2dof{k:g}'}" for m in NPAR for k in MK) + ",NLA_chi2dof_at_pcut,NLA_chi2dof_next,TATT_chi2dof_at_pcut,TATT_chi2dof_next\n")
    for s in cz:
        row = [f"{zs[s]:g}"] + [f"{cuts[(m, k)][s]:.3f}" for m in NPAR for k in MK]
        for mdl in NPAR:
            j = int(np.argmin(np.abs(kg - cuts[(mdl, 'p')][s])))
            row += [f"{r[mdl][s, j]:.2f}", f"{r[mdl][s, j + 1]:.2f}"]
        fh.write(",".join(row) + "\n")
print(open(os.path.join(HERE, "chi2dof.csv")).read())
print(open(os.path.join(HERE, "chi2dof_cuts.csv")).read())
