#!/usr/bin/env python3
"""Figures and derived tables for the scale-cut strategy document.

Inputs: v3/v3.npz, v3/chi2dof_cuts.csv, v3/mask_scan.json, v3/roman_test.json,
v3/roman_results.csv, v3/knl.csv, v3s/S2/v3.npz, v4/sim_strategies.json,
v4/roman_strategies.json (+ roman_theory.json).
Outputs (v4/): fig_strat_cuts, fig_strat_volume, fig_strat_noise,
fig_strat_tradeoff, fig_strat_tolerance (220 dpi PNG + PDF); derived.json;
tab_*.tex table bodies.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..", "v3")
sys.path.insert(0, HERE)
import simfit as SF  # noqa: E402

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
v3 = np.load(os.path.join(V3, "v3.npz"))
S2 = np.load(os.path.join(HERE, "..", "v3s", "S2", "v3.npz"))
zs, kg, core = v3["zs"], v3["kgrid"], v3["core"]
CZ = np.where(core)[0]
sim = json.load(open(os.path.join(HERE, "sim_strategies.json")))
rs = json.load(open(os.path.join(HERE, "roman_strategies.json")))
ms = json.load(open(os.path.join(V3, "mask_scan.json")))
rt = json.load(open(os.path.join(V3, "roman_test.json")))
knl = list(csv.DictReader(open(os.path.join(V3, "knl.csv"))))
c2c = list(csv.DictReader(open(os.path.join(V3, "chi2dof_cuts.csv"))))
DER = {}


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=220)
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    plt.close(fig)


def kaxis(a, lo, hi, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3)):
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(ticker.FixedLocator([t for t in ticks if lo <= t <= hi]))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.grid(alpha=0.3)


def Dz(z):
    """Linear growth (D3A) interpolated from knl.csv."""
    zz = np.array([float(r["z"]) for r in knl])
    dd = np.array([float(r["D"]) for r in knl])
    return np.interp(z, zz, dd)


# ======================================================================
# 1. cut map
# ======================================================================
zf = np.linspace(0.15, 2.1, 100)
fig, ax = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
crit = [  # label, source, marker, colour
    ("GoF, $p<0.05$ (two consecutive)", "gof", "o", None),
    (r"GoF, $\chi^2/\nu>2$", "c2", "s", "0.25"),
    ("parameter stability (nested test)", "nested", "D", "#6a4c93"),
    ("$A_1$ stability, naive 1$\\sigma$ (vs $k=0.09$)", "naive-ref", "v", "#8d99ae"),
    ("anchored extrapolation, $p<0.05$", "anch", "^", "#2a9d8f"),
    ("10% accuracy (GI and II)", "10", "P", "#e9c46a"),
    ("GoF on the stitched truth (S2)", "S2", "*", "#f4a261"),
]
for c, mdl in enumerate(("NLA", "TATT")):
    a = ax[c]
    for i, (lab, src, mk, colr) in enumerate(crit):
        if src == "gof":
            v = v3[f"cut_{mdl}_GoF"]
        elif src == "c2":
            v = np.full(len(zs), np.nan)
            for r in c2c:
                s = int(np.argmin(np.abs(zs - float(r["z"]))))
                x = r[f"{mdl}_chi2dof2"]
                v[s] = float(x) if x != "nan" else np.nan
        elif src in ("nested", "naive-ref"):
            v = np.array(sim["measured"][mdl][src], float)
        elif src == "anch":
            v = v3[f"cut_{mdl}_anchored"]
        elif src == "10":
            v = v3[f"cut_{mdl}_10%"]
        else:
            v = S2[f"cut_{mdl}_GoF"]
        dz = (i - 3)*0.022
        vv = np.where(np.isnan(v), 0.072, v)
        a.plot(vv[CZ], zs[CZ] + dz, ls="none", marker=mk, ms=7 if src == "gof" else 6, mfc=COL[mdl] if src == "gof" else "none",
               mec=COL[mdl] if colr is None else colr, mew=1.4, label=lab)
    for K0, ls in ((0.25, ":"), (0.3, "--")):
        a.plot(K0/Dz(zf), zf, color="k", lw=1.1, ls=ls, label=f"Roman FoB $\\approx$ 0.3 (production truth): $k_{{\\rm UV}}={K0}/D(z)$" if K0 == 0.3 else f"$k_{{\\rm UV}}={K0}/D(z)$ (noise margin)")
    a.axvline(0.3, color="0.45", lw=1.1, ls="-.", label="Roman FoB $\\approx$ 0.3, constant cut 0.3")
    a.plot([float(r["k_NL"]) for r in knl], [float(r["z"]) for r in knl], color="0.6", lw=2.2, alpha=0.6, label=r"$k_{\rm NL}(z)$, $\Delta^2_{\rm lin}=1$")
    kaxis(a, 0.065, 1.2, (0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1))
    a.set_ylim(0.1, 2.15)
    a.set_xlabel(r"$k_{\max}$ or $k_{\rm UV}(z)$ [h/Mpc]  (marker at 0.072: cut below the grid)")
    a.set_title(f"{mdl}: cuts by criterion (production truth unless stated)", fontsize=10.5)
ax[0].set_ylabel("z  (markers offset slightly in z for legibility)")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
save(fig, "fig_strat_cuts")

# ======================================================================
# 2. simulation volume
# ======================================================================
f = np.array(sim["volume"]["f"])
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
vol = {}
for c, mdl in enumerate(("NLA", "TATT")):
    a = ax[c]
    chi_meas = v3[f"{mdl}_chi2"]
    nu = v3[f"{mdl}_dof"]
    lam = np.maximum(chi_meas - nu, 0.0)
    nuc = nu[CZ].sum(0)
    lamc = np.maximum(chi_meas[CZ].sum(0) - nuc, 0.0)
    cut_l = np.array([SF.cut_rule(chi2_dist.sf(nuc + fv*lamc, nuc) >= 0.05) for fv in f])
    cut_T = np.array(sim["volume"][mdl]["comb"])
    per = np.array([[SF.cut_rule(chi2_dist.sf(nu[s] + fv*lam[s], nu[s]) >= 0.05) for fv in f] for s in CZ])
    vol[mdl] = dict(f=f.tolist(), comb_lambda=cut_l.tolist(), comb_T=cut_T.tolist(),
                    per_median=np.nanmedian(np.where(np.isnan(per), 0.072, per), 0).tolist())
    lo = np.nanmin(np.where(np.isnan(per), 0.072, per), 0)
    hi = np.nanmax(np.where(np.isnan(per), 0.072, per), 0)
    a.fill_betweenx(f, lo, hi, step="mid", color=COL[mdl], alpha=0.15, lw=0, label="range over the 9 core snapshots (non-centrality estimate)")
    a.step(np.where(np.isnan(cut_l), 0.072, cut_l), f, where="mid", color=COL[mdl], lw=2.2, label=r"core combined, non-centrality $\hat\lambda=\chi^2_{\rm meas}-\nu$")
    a.step(np.where(np.isnan(cut_T), 0.072, cut_T), f, where="mid", color="k", lw=1.2, ls="--", label=r"core combined, Asimov on the smoothed truth $T$")
    a.axhline(1.0, color="0.3", lw=0.8)
    a.axhspan(0.02, 0.04, color="#2a9d8f", alpha=0.25, lw=0, label=r"$\Delta\chi^2_{\rm Roman}/\Delta\chi^2_{\rm FL}$ of the actual residuals (0.02 to 0.04)")
    a.set_yscale("log")
    a.set_ylim(f[0], f[-1])
    kaxis(a, 0.065, 1.2, (0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1))
    a.set_xlabel(r"GoF cut $k_{\max}$ [h/Mpc]  (0.072: below the grid)")
    a.set_title(f"{mdl}: the significance cut moves with the simulation volume", fontsize=10.5)
    a.legend(fontsize=6.8, loc="upper right")
ax[0].set_ylabel(r"volume factor $f$ (errors $\sigma/\sqrt{f}$; FLAMINGO L1000 = 1)")
fig.tight_layout()
save(fig, "fig_strat_volume")
DER["volume"] = vol

# ======================================================================
# 3. noise scatter and calibration
# ======================================================================
RULES = ["GoF", "naive-ref", "naive-seq", "nested"]
noise_tab = {}
calib_tab = {}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for c, mdl in enumerate(("NLA", "TATT")):
    a = ax[c]
    arr = np.array(sim["realisations"]["noise"][mdl]["cuts"], float)        # [R, nsn, rule]
    arrc = np.array(sim["realisations"]["calib"][mdl]["cuts"], float)
    combn = np.array(sim["realisations"]["noise"][mdl]["comb"], float)
    combc = np.array(sim["realisations"]["calib"][mdl]["comb"], float)
    noise_tab[mdl] = {}
    calib_tab[mdl] = {}
    for j, rule in enumerate(RULES):
        x = np.where(np.isnan(arr[:, :, j]), 0.072, arr[:, :, j])
        q = np.percentile(x[:, CZ], [16, 50, 84], axis=0)
        meas = np.array(sim["measured"][mdl][rule], float)[CZ]
        noise_tab[mdl][rule] = dict(p16=q[0].tolist(), p50=q[1].tolist(), p84=q[2].tolist(), measured=meas.tolist())
        xc = np.where(np.isnan(arrc[:, :, j]), 0.072, arrc[:, :, j])[:, CZ]
        false = xc < 2.0                                   # two consecutive failures before k = 2.27
        calib_tab[mdl][rule] = dict(false_rate_per_snapshot=float(false.mean()),
                                    false_rate_any_core=float(false.any(1).mean()),
                                    median_false_cut=float(np.median(xc[false])) if false.any() else None)
        if rule in ("GoF", "nested"):
            dz = -0.03 if rule == "GoF" else 0.03
            mk = "o" if rule == "GoF" else "D"
            a.errorbar(q[1], zs[CZ] + dz, xerr=[q[1] - q[0], q[2] - q[1]], fmt=mk, ms=6, mfc="none", color=COL[mdl] if rule == "GoF" else "#6a4c93",
                       capsize=2, lw=1.2, label=f"{'GoF p-rule' if rule == 'GoF' else 'parameter stability (nested)'}: median and 16 to 84%")
            a.plot(np.where(np.isnan(meas), 0.072, meas), zs[CZ] + dz, ls="none", marker=mk, ms=6, color=COL[mdl] if rule == "GoF" else "#6a4c93",
                   label=f"measured spectra ({'GoF' if rule == 'GoF' else 'stability'})")
    xcn = np.where(np.isnan(combn), 0.072, combn)
    noise_tab[mdl]["comb"] = np.percentile(xcn, [16, 50, 84]).tolist()
    calib_tab[mdl]["comb_false_rate"] = float((np.where(np.isnan(combc), 0.072, combc) < 2.0).mean())
    kaxis(a, 0.065, 1.2, (0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1))
    a.set_xlabel(r"cut $k_{\max}$ [h/Mpc]  (0.072: below the grid)")
    a.set_title(f"{mdl}: scatter of the simulation cuts over {sim['N_REAL']} noise realisations", fontsize=10.5)
    a.legend(fontsize=7)
ax[0].set_ylabel("z")
fig.tight_layout()
save(fig, "fig_strat_noise")
DER["noise"] = noise_tab
DER["calib"] = calib_tab

# ======================================================================
# 4. bias-precision trade-off
# ======================================================================
F = rs["fisher"]
FAM = {"D1": (r"$K_0/D(z)$", "-", "o"), "D23": (r"$K_0/D(z)^{2/3}$", "--", "s"), "flat": (r"$K_0$ constant", ":", "^")}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
trade = {}
for c, mdl in enumerate(("NLA", "TATT")):
    a = ax[c]
    sg = np.geomspace(0.0025, 0.03, 50)
    for dS in (0.001, 0.002, 0.003):
        a.plot(sg, dS/sg, color="0.7", lw=0.9, ls="-")
        a.text(dS/0.74, 0.75, f"$|\\delta S_8|$={dS:g}", fontsize=7, color="0.4", ha="right", va="bottom")
    trade[mdl] = {}
    for fam, (lab, ls, mk) in FAM.items():
        r = sorted([x for x in ms if x["family"] == fam], key=lambda x: x["K0"])
        sS8 = np.array([F[f"scan:{fam}_{x['K0']:g}"][f"{mdl}_A"]["sS8"] for x in r])
        fob = np.array([x[f"{mdl}_A"]["FoB"] for x in r])
        fsig = np.array([x[f"{mdl}_A"]["FoB_sig"] or 0 for x in r])
        dS8abs = np.array([x[f"{mdl}_A"]["dS8"] for x in r])*sS8
        a.errorbar(sS8, fob, fsig, ls=ls, marker=mk, ms=5, color=COL[mdl], capsize=2, lw=1.3, label=lab)
        for x, s_, fb in zip(r, sS8, fob):
            a.annotate(f"{x['K0']:g}", (s_, fb), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="0.35")
        # sigma(S8) at the FoB = 0.3 crossing (linear interpolation in log sigma, first crossing)
        cross = None
        for i in range(len(r) - 1):
            if fob[i] < 0.3 <= fob[i + 1]:
                t = (0.3 - fob[i])/(fob[i + 1] - fob[i])
                cross = float(np.exp(np.log(sS8[i]) + t*(np.log(sS8[i + 1]) - np.log(sS8[i]))))
                K0x = r[i]["K0"] + t*(r[i + 1]["K0"] - r[i]["K0"])
                break
        trade[mdl][fam] = dict(K0=[x["K0"] for x in r], sS8=sS8.tolist(), FoB=fob.tolist(), dS8abs=dS8abs.tolist(),
                               sS8_at_FoB03=cross, K0_at_FoB03=K0x if cross else None)
        ax[2].plot(sS8, np.abs(dS8abs)*1e3, ls=ls, marker=mk, ms=5, color=COL[mdl], lw=1.2, label=f"{mdl}, {lab}")
    a.axhline(0.3, color="k", lw=0.8, ls="--")
    a.set_xscale("log")
    a.set_xlim(0.03, 0.003)
    a.xaxis.set_major_locator(ticker.FixedLocator([0.003, 0.004, 0.006, 0.01, 0.02, 0.03]))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.set_ylim(0, 0.8)
    a.set_xlabel(r"$\sigma(S_8)$ of the mask (Fisher, case A; smaller to the right)")
    a.set_ylabel(r"FoB [$\sigma$]")
    a.set_title(f"{mdl}: FoB against precision (labels: $K_0$); grey: constant absolute bias", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left")
    a.grid(alpha=0.3, which="both")
a = ax[2]
a.set_xscale("log")
a.set_xlim(0.03, 0.003)
a.xaxis.set_major_locator(ticker.FixedLocator([0.003, 0.004, 0.006, 0.01, 0.02, 0.03]))
a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
a.xaxis.set_minor_locator(ticker.NullLocator())
a.set_xlabel(r"$\sigma(S_8)$ (case A)")
a.set_ylabel(r"$|\delta S_8|\times10^3$ (absolute bias)")
a.set_title(r"The absolute $S_8$ bias is nearly independent of the cut", fontsize=10)
a.set_ylim(0, 3.2)
a.grid(alpha=0.3, which="both")
a.legend(fontsize=6.5, ncol=2)
fig.tight_layout()
save(fig, "fig_strat_tradeoff")
DER["tradeoff"] = trade

# ======================================================================
# 5. required accuracy vs achieved
# ======================================================================
tol = rs["tolerance"]
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
# achieved: anchored fits (k <= 0.16), mean and max over core of |M/T - 1| per k node
ach = {}
for mdl in ("NLA", "TATT"):
    dev = []
    for s in CZ:
        o = SF.fit(s, SF.y[s], mdl, 0.1615)
        M = SF.pred(s, *o["x"])
        dev.append([np.abs(M[0]/SF.T[s, 0] - 1), np.abs((M[1] - SF.Ns[s])/(SF.T[s, 1] - SF.Ns[s]) - 1)])
    dev = np.array(dev)
    ach[mdl] = dict(mean=dev.mean(0), max=dev.max(0))
kd = SF.kd
for c, (which, xi) in enumerate((("GI", 0), ("II", 1))):
    a = ax[c]
    for key, ls in (("n1:0.2", ":"), ("n1:0.3", "-"), ("n1:0.4", "--")):
        rows = [x for x in tol[key]["TATT"] if x["shape"] == "step" and x["which"] == which]
        a.plot([x["kx"] for x in rows], [x["eps_dchi1"] for x in rows], ls=ls, color="k", lw=1.6,
               label=f"required: $\\Delta\\chi^2_{{\\rm Roman}}=1$, mask $K_0$ = {key.split(':')[1]}/D")
        if key == "n1:0.3":
            a.plot([x["kx"] for x in rows], [x["eps_FoB03"] for x in rows], ls="-", color="0.6", lw=0.9, marker=".",
                   label="required: FoB = 0.3 (linear response, IA refit), $K_0$ = 0.3/D")
    m = (kd >= 0.08) & (kd <= 1.4)
    for mdl in ("NLA", "TATT"):
        a.plot(kd[m], ach[mdl]["mean"][xi][m], "-o", ms=3.5, color=COL[mdl], lw=1.4, label=f"{mdl} anchored at $k\\leq0.16$: mean over core $z$")
        a.plot(kd[m], ach[mdl]["max"][xi][m], ":", color=COL[mdl], lw=1.2, label=f"{mdl}: max over core $z$")
    a.set_yscale("log")
    a.set_ylim(0.01, 5)
    kaxis(a, 0.08, 1.4, (0.1, 0.2, 0.3, 0.5, 0.7, 1))
    a.set_xlabel(r"$k$ [h/Mpc]  (required: a coherent fractional error $\varepsilon$ at all $k'\geq k$)")
    a.set_title(f"{which}: accuracy Roman requires against what the models achieve", fontsize=10.5)
    a.legend(fontsize=6.5, loc="upper left")
ax[0].set_ylabel(r"fractional error $\varepsilon$")
fig.tight_layout()
save(fig, "fig_strat_tolerance")
DER["achieved"] = {mdl: {k: ach[mdl][k].tolist() for k in ach[mdl]} for mdl in ach}

# ======================================================================
# 6. internal consistency between nested cuts (K0/D production family)
# ======================================================================
cons = []
for case in ("A", "B"):
    for mdl in ("NLA", "TATT"):
        r = sorted([x for x in rt if x["test"] == f"{mdl}_on_FLAMINGO" and x["case"] == case and x["mask"].startswith("K0/D")],
                   key=lambda x: float(x["mask"].split()[-1]))
        for a_, b_ in zip(r[:-1], r[1:]):
            sd = np.sqrt(max(a_["sS8"]**2 - b_["sS8"]**2, 0))
            cons.append(dict(case=case, model=mdl, small=a_["mask"], large=b_["mask"], dS8_small=a_["dS8"], dS8_large=b_["dS8"],
                             change=b_["dS8"] - a_["dS8"], sigma_diff=float(sd), ratio=float((b_["dS8"] - a_["dS8"])/sd) if sd > 0 else None))
DER["consistency"] = cons

# ======================================================================
# 7. theory-error covariance
# ======================================================================
if os.path.exists(os.path.join(HERE, "roman_theory.json")):
    th = json.load(open(os.path.join(HERE, "roman_theory.json")))
    DER["theory"] = th

json.dump(DER, open(os.path.join(HERE, "derived.json"), "w"), indent=1, default=float)
print(json.dumps({k: DER[k] for k in ("calib",)}, indent=1))
for mdl in trade:
    for fam in trade[mdl]:
        t = trade[mdl][fam]
        print(mdl, fam, "sS8 at FoB 0.3:", t["sS8_at_FoB03"], "K0:", t["K0_at_FoB03"], "abs dS8 x1e3:", np.round(np.array(t["dS8abs"])*1e3, 2))
for x in cons:
    print(x["case"], x["model"], x["small"], "->", x["large"], f"change {x['change']:+.4f} sigma_diff {x['sigma_diff']:.4f} ratio {x['ratio']:+.2f}")
for mdl in vol:
    print(mdl, "volume comb (lambda):", [(round(a, 3), b) for a, b in zip(vol[mdl]["f"][::4], vol[mdl]["comb_lambda"][::4])])

# ======================================================================
# 8. separate xi_+ / xi_- cuts (linear response, split_pm.json)
# ======================================================================
if os.path.exists(os.path.join(HERE, "split_pm.json")):
    sp = json.load(open(os.path.join(HERE, "split_pm.json")))
    KP = sorted({x["K0p"] for x in sp})
    KM = sorted({x["K0m"] for x in sp})
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    best = {}
    for c, mdl in enumerate(("NLA", "TATT")):
        a = ax[c]
        Fm = np.array([[[x for x in sp if x["model"] == mdl and x["K0p"] == kp and x["K0m"] == km][0]["FoB"] for kp in KP] for km in KM])
        Sm = np.array([[[x for x in sp if x["model"] == mdl and x["K0p"] == kp and x["K0m"] == km][0]["sS8"] for kp in KP] for km in KM])
        im = a.imshow(Fm, origin="lower", cmap="RdYlBu_r", vmin=0, vmax=0.7, aspect="auto")
        for i in range(len(KM)):
            for j in range(len(KP)):
                ok = Fm[i, j] < 0.3
                a.text(j, i, f"{Fm[i, j]:.2f}\n{Sm[i, j]*1e3:.1f}", ha="center", va="center", fontsize=7,
                       fontweight="bold" if ok else "normal", color="k")
                if KP[j] == KM[i]:
                    a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.6))
        a.contour(Fm, levels=[0.3], colors="k", linewidths=1.2, linestyles="--")
        a.set_xticks(range(len(KP)))
        a.set_xticklabels([f"{k:g}" for k in KP])
        a.set_yticks(range(len(KM)))
        a.set_yticklabels(["no $\\xi_-$" if k == 0 else f"{k:g}" for k in KM])
        a.set_xlabel(r"$K_0^{+}$ for $\xi_+$  ($k_{\rm UV}=K_0^+/D(z)$)")
        a.set_ylabel(r"$K_0^{-}$ for $\xi_-$")
        a.set_title(f"{mdl}: FoB (top) and $\\sigma(S_8)\\times10^3$ (bottom); boxed: common cut; dashed: FoB = 0.3", fontsize=9.5)
        fig.colorbar(im, ax=a, label="FoB (linear response, case A)")
        okm = Fm < 0.3
        i, j = np.unravel_index(np.argmin(np.where(okm, Sm, np.inf)), Sm.shape)
        ok25 = Fm < 0.25
        i2, j2 = np.unravel_index(np.argmin(np.where(ok25, Sm, np.inf)), Sm.shape)
        best[mdl] = dict(FoB03=dict(K0p=KP[j], K0m=KM[i], FoB=float(Fm[i, j]), sS8=float(Sm[i, j])),
                         FoB025=dict(K0p=KP[j2], K0m=KM[i2], FoB=float(Fm[i2, j2]), sS8=float(Sm[i2, j2])))
    fig.tight_layout()
    save(fig, "fig_strat_split")
    DER["split_best"] = best
    json.dump(DER, open(os.path.join(HERE, "derived.json"), "w"), indent=1, default=float)
    print("split best:", json.dumps(best, indent=1))
