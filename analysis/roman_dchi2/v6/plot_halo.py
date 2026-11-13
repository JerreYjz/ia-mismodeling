#!/usr/bin/env python3
"""Figures and tables for the halo-model scale-cut note (220 dpi PNG + PDF;
k horizontal, z vertical for cut maps)."""
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
import halo_ia as H          # noqa: E402
import simfit as SF          # noqa: E402
import halo_sim as HS        # noqa: E402

REP = os.path.join(HERE, "..", "report")
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8", "halo": "#2a9d8f", "halo_nowin": "#e9c46a", "NLA_tab": "#8d0f25"}
tab = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
hs = np.load(os.path.join(HERE, "halo_sim.npz"))
v3 = SF.v3
zs, kg, core = SF.zs, SF.kg, SF.core
CZ = np.where(core)[0]
kd = SF.kd


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=220)
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    plt.close(fig)


def kaxis(a, lo, hi, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10)):
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(ticker.FixedLocator([t for t in ticks if lo <= t <= hi]))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.grid(alpha=0.3)


# ---------------- 1. spectra: FLAMINGO vs halo model components ----------------
SEL = [5, 8, 10]
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
for c, s in enumerate(SEL):
    pred = HS.model_factory(tab, s, False)
    k_all = tab["k_all"]
    idx = np.searchsorted(k_all, kd)
    frc = tab["hod"][s, 5]
    D = SF.model.D(zs[s])
    C1 = SF.model.cosmo["Om"]*H.C1RHOCRIT/D
    P = SF.Pnl[s]
    for jk, ls in ((2, "-"), (12, "--")):
        a1, a1h = hs["halo_x"][s, jk]
        la = np.log(np.maximum(tab["a_grid"], 1e-6))
        j = int(np.clip(np.searchsorted(la, np.log(max(a1h, 1e-6))), 1, la.size - 1))
        w = np.clip((np.log(max(a1h, 1e-6)) - la[j - 1])/(la[j] - la[j - 1]), 0, 1)
        sd = tab["SdI"][s, j - 1][idx]*(1 - w) + tab["SdI"][s, j][idx]*w
        si = tab["SII"][s, j - 1][idx]*(1 - w) + tab["SII"][s, j][idx]*w
        two_gi = a1*frc*C1*P*H.f2h(kd)
        one_gi = a1h*H.f1h(kd)*sd
        two_ii = (a1*frc*C1)**2*P*H.f2h(kd)
        one_ii = a1h**2*H.f1h(kd)*si
        lab = f"fit to $k\\leq${kg[jk]:.2f}: $a_1$={a1:.2f}, $a_{{1h}}$={a1h:.3f}"
        ax[0, c].plot(kd, (two_gi + one_gi)/P, ls, color=COL["halo"], lw=1.8, label=f"halo total, {lab}")
        ax[0, c].plot(kd, two_gi/P, ls, color="0.4", lw=0.9, label="2-halo" if jk == 12 else None)
        ax[0, c].plot(kd, np.maximum(one_gi/P, 1e-8), ls, color="#e76f51", lw=0.9, label="1-halo" if jk == 12 else None)
        ax[1, c].plot(kd, (two_ii + one_ii)/P, ls, color=COL["halo"], lw=1.8)
        ax[1, c].plot(kd, two_ii/P, ls, color="0.4", lw=0.9)
        ax[1, c].plot(kd, np.maximum(one_ii/P, 1e-12), ls, color="#e76f51", lw=0.9)
    m = kd >= 0.05
    ax[0, c].errorbar(kd[m], (-SF.y[s, 0]/(SF.b[s]*P))[m], (SF.err[s, 0]/(SF.b[s]*P))[m], fmt="o", ms=3, color="k", lw=0.8, label="FLAMINGO")
    ax[1, c].errorbar(kd[m], ((SF.y[s, 1] - SF.Ns[s])/P)[m], (SF.err[s, 1]/P)[m], fmt="o", ms=3, color="k", lw=0.8)
    for r, ylab in ((0, r"$-P_{\delta E}/P_{\rm NL}$"), (1, r"$(P_{EE}-N_s)/P_{\rm NL}$")):
        a = ax[r, c]
        a.set_yscale("log")
        kaxis(a, 0.045, 18)
        if c == 0:
            a.set_ylabel(ylab)
        a.set_ylim((1e-4, 0.3) if r == 0 else (1e-7, 0.05))
    ax[0, c].set_title(f"z = {zs[s]:g}  ($f_{{rc}}$ = {frc:.2f})", fontsize=10)
    ax[1, c].set_xlabel(r"$k$ [h/Mpc]")
    ax[0, c].legend(fontsize=6.3, loc="lower left")
fig.suptitle("Halo-model IA (Fortuna et al. 2021, halo.c conventions) fitted to FLAMINGO: total, 2-halo and 1-halo terms (solid: fit to $k\\leq0.16$; dashed: to $k\\leq3.04$)", fontsize=10.5)
fig.tight_layout()
save(fig, "fig_halo_spectra")

# ---------------- 2. goodness of fit ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cmap = plt.get_cmap("viridis")
for c, var in enumerate(("halo", "halo_nowin")):
    a = ax[c]
    r = hs[f"{var}_chi2"]/hs[f"{var}_dof"]
    for n, s in enumerate(CZ):
        a.plot(kg, r[s], "-o", ms=3, lw=1.1, color=cmap(n/(len(CZ) - 1)), label=f"z = {zs[s]:g}")
    nu = hs[f"{var}_dof"][0]
    a.plot(kg, chi2_dist.isf(0.05, nu)/nu, "k--", lw=1.3, label=r"$p=0.05$: $\chi^2_{0.95}(\nu)/\nu$")
    kaxis(a, 0.08, 3.3)
    a.set_yscale("log")
    a.set_ylim(0.2, 300)
    a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
    a.set_ylabel(r"$\chi^2_{\min}/\nu$")
    a.set_title(f"{'halo model, F21 windows' if var == 'halo' else 'halo model, no windows'} ($\\nu = 3N_k - 2$)", fontsize=10)
    a.legend(fontsize=6.5, ncol=2)
a = ax[2]
for lab, ch, nu, col in (("NLA", v3["NLA_chi2"], v3["NLA_dof"], COL["NLA"]), ("TATT", v3["TATT_chi2"], v3["TATT_dof"], COL["TATT"]),
                         ("halo (F21)", hs["halo_chi2"], hs["halo_dof"], COL["halo"]), ("halo (no windows)", hs["halo_nowin_chi2"], hs["halo_nowin_dof"], COL["halo_nowin"])):
    p = chi2_dist.sf(ch[CZ].sum(0), nu[CZ].sum(0))
    a.plot(kg, np.maximum(p, 1e-300), "-o", ms=4, color=col, label=lab)
a.axhline(0.05, color="k", lw=0.8, ls="--")
a.set_yscale("log")
a.set_ylim(1e-30, 2)
kaxis(a, 0.08, 1.0, (0.1, 0.2, 0.3, 0.5, 0.7, 1))
a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
a.set_ylabel("combined $p$ (core snapshots)")
a.set_title("Combined goodness of fit, $0.2 \\leq z \\leq 2$", fontsize=10)
a.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig_halo_gof")

# ---------------- 3. cut map ----------------
fig, a = plt.subplots(figsize=(7.5, 5.5))
for i, (lab, cuts, mk, col) in enumerate((("NLA", v3["cut_NLA_GoF"], "o", COL["NLA"]), ("TATT", v3["cut_TATT_GoF"], "s", COL["TATT"]),
                                           ("halo (F21 windows)", hs["halo_cut"], "D", COL["halo"]), ("halo (no windows)", hs["halo_nowin_cut"], "^", "#b08900"))):
    cc = np.where(np.isnan(cuts), 0.072, cuts)
    a.plot(cc[CZ], zs[CZ] + (i - 1.5)*0.025, ls="none", marker=mk, ms=7, mfc="none" if i > 1 else col, mec=col, mew=1.5, label=lab)
kaxis(a, 0.065, 0.6, (0.07, 0.1, 0.2, 0.3, 0.5))
a.set_xlabel(r"goodness-of-fit cut $k_{\max}$ [h/Mpc]  (0.072: below the grid)")
a.set_ylabel("z  (markers offset slightly)")
a.legend(fontsize=8)
a.set_title("Simulation cuts ($p<0.05$ at two consecutive $k_{\\max}$)", fontsize=10)
fig.tight_layout()
save(fig, "fig_halo_cuts")

# ---------------- 4. Roman FoB ----------------
rr = json.load(open(os.path.join(HERE, "halo_roman_example1.json")))
nat = json.load(open(os.path.join(HERE, "..", "v3", "mask_scan_ex1_D1.json"))) + json.load(open(os.path.join(HERE, "..", "v3", "mask_scan_ex1_flat.json")))
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
for r_, fam in enumerate(("D1", "flat")):
    for c, case in enumerate(("A", "B")):
        a = ax[r_, c]
        for mdl, lab, ls, mk in (("NLA_tab", "NLA (template path)", "-", "o"), ("halo", "halo model (F21 windows)", "-", "D"), ("halo_nowin", "halo model (no windows)", "--", "^")):
            x = sorted([y for y in rr if y["family"] == fam and y["model"] == mdl and y["case"] == case], key=lambda y: y["K0"])
            a.errorbar([y["K0"] for y in x], [y["FoB"] for y in x], [y["FoB_sig"] for y in x], ls=ls, marker=mk, ms=5, capsize=2, color=COL[mdl], label=lab)
        for mdl, ls in (("NLA", ":"), ("TATT", ":")):
            x = sorted([y for y in nat if y["family"] == fam], key=lambda y: y["K0"])
            a.plot([y["K0"] for y in x], [y[f"{mdl}_{case}"]["FoB"] for y in x], ls, marker="x", color=COL[mdl], label=f"{mdl}, native (full fit)")
        a.axhline(0.3, color="k", lw=0.8, ls="--")
        a.set_ylim(0, 0.8)
        a.set_xlabel(r"$K_0$ [h/Mpc]  (" + ("$k_{\\rm UV}=K_0/D(z)$" if fam == "D1" else "constant $k_{\\rm UV}$") + ")")
        a.set_ylabel(r"FoB [$\sigma$]")
        a.set_title(f"{'1/D' if fam == 'D1' else 'constant'} family, case {case}, example1_cov", fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
fig.tight_layout()
save(fig, "fig_halo_fob")

# ---------------- tables ----------------
hod = tab["hod"]
lines = [r"\begin{tabular}{rrrrrrrr}\toprule",
         r"$z$ & $1/\bar n_g$ [$(h^{-1}{\rm Mpc})^3$] & $b_g$ (FLAMINGO) & $\lg M_{\min}$ & $\lg M_1$ & $f_{\rm rc}$ & $f_{\rm sat}$ & core\\ \midrule"]
for s, z in enumerate(zs):
    lines.append(f"{z:g} & {1/hod[s,3]:.0f} & {hod[s,4]:.3f} & {hod[s,0]:.2f} & {hod[s,1]:.2f} & {hod[s,5]:.3f} & {hod[s,6]:.3f} & {'$\\checkmark$' if core[s] else ''}\\\\")
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "tab_halo_hod.tex"), "w").write("\n".join(lines))

f = lambda v: "$0.00$" if abs(v) < 0.005 else f"${v:+.2f}$"
fc = lambda v: "$<$0.09" if np.isnan(v) else f"{v:.2f}"
lines = [r"\begin{tabular}{r cccc cc}\toprule",
         r"$z$ & NLA & TATT & halo (F21) & halo (no win.) & $a_1$ (halo, $k\leq0.16$) & $a_{1h}$ (halo, $k\leq3.04$)\\ \midrule"]
for s in CZ:
    lines.append(f"{zs[s]:g} & {fc(v3['cut_NLA_GoF'][s])} & {fc(v3['cut_TATT_GoF'][s])} & {fc(hs['halo_cut'][s])} & {fc(hs['halo_nowin_cut'][s])} & {hs['halo_x'][s,2,0]:.2f} & {hs['halo_x'][s,12,1]:.3f}\\\\")
lines.append(r"\midrule")
cmb = {"NLA": 0.09, "TATT": 0.22}
lines.append(f"combined & 0.09 & 0.22 & {fc(float(hs['halo_comb_cut']))} & {fc(float(hs['halo_nowin_comb_cut']))} & & \\\\")
lines.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "tab_halo_cuts.tex"), "w").write("\n".join(lines))

for case in ("A", "B"):
    lines = [r"\begin{tabular}{c ccc r c ccc ccc}\toprule",
             r" & \multicolumn{3}{c}{$k_{\rm UV}$ [$h\,{\rm Mpc}^{-1}$]} & & & \multicolumn{3}{c}{NLA (template path)} & \multicolumn{3}{c}{halo model (F21)}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
             r"$k_0$ & $z=0.5$ & $z=1$ & $z=2$ & $N_{\rm pts}$ & $\sigma(S_8)$ & $\delta\Omega_m/\sigma$ & $\delta S_8/\sigma$ & FoB & $\delta\Omega_m/\sigma$ & $\delta S_8/\sigma$ & FoB\\ \midrule"]
    kz = {0.15: (0.19, 0.25, 0.36), 0.2: (0.26, 0.33, 0.48), 0.25: (0.32, 0.41, 0.60), 0.3: (0.39, 0.49, 0.71), 0.35: (0.45, 0.57, 0.83), 0.4: (0.52, 0.66, 0.95), 0.5: (0.65, 0.82, 1.19)}
    for K0 in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        n = [y for y in rr if y["family"] == "D1" and y["K0"] == K0 and y["model"] == "NLA_tab" and y["case"] == case][0]
        t = [y for y in rr if y["family"] == "D1" and y["K0"] == K0 and y["model"] == "halo" and y["case"] == case][0]
        lines.append(f"{K0:.2f} & {kz[K0][0]:.2f} & {kz[K0][1]:.2f} & {kz[K0][2]:.2f} & {n['npts']} & {t['sS8']:.4f} & {f(n['dOm'])} & {f(n['dS8'])} & ${n['FoB']:.2f}\\pm{n['FoB_sig']:.2f}$ & "
                     f"{f(t['dOm'])} & {f(t['dS8'])} & ${t['FoB']:.2f}\\pm{t['FoB_sig']:.2f}$\\\\")
    lines.append(r"\midrule")
    for sh in ("NLA", "TATT"):
        n = [y for y in rr if y["family"] == "GoF" and y["shape"] == sh and y["model"] == "NLA_tab" and y["case"] == case][0]
        t = [y for y in rr if y["family"] == "GoF" and y["shape"] == sh and y["model"] == "halo" and y["case"] == case][0]
        kk = {"NLA": (0.09, 0.12, 0.12), "TATT": (0.28, 0.29, 0.39)}[sh]
        sS = f"{t['sS8']:.4f}" if t['sS8'] < 0.1 else f"{t['sS8']:.2f}"
        lines.append(f"GoF$_{{\\rm {sh}}}$ & {kk[0]:.2f} & {kk[1]:.2f} & {kk[2]:.2f} & {n['npts']} & {sS} & {f(n['dOm'])} & {f(n['dS8'])} & ${n['FoB']:.2f}\\pm{n['FoB_sig']:.2f}$ & "
                     f"{f(t['dOm'])} & {f(t['dS8'])} & ${t['FoB']:.2f}\\pm{t['FoB_sig']:.2f}$\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(os.path.join(REP, f"tab_halo_roman_{case}.tex"), "w").write("\n".join(lines))
print("ok")
