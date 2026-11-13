#!/usr/bin/env python3
"""Tables and the term-anatomy figure for the HMc review (report/hmc_review.tex).
  report/tab_hmcr_gof.tex     summed chi2/nu vs k_max and GoF cuts, all variants
  report/tab_hmcr_par.tex     HMc + W parameters per snapshot at k_max = 0.52, 0.94, 3.04
  report/tab_hmcr_roman.tex   Roman FoB, TATT vs HMc, both cases and cut families
  fig_hmc_terms.png/pdf       GI pieces at z = 0.3 and 1: C1 P_NL, C1 I1m P_lin and
                              the central 1-halo term; kernel u_E(k|M)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "..", "report")
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "_review"]
import halo_central_fit as HF   # noqa: E402
import halo_central as HC       # noqa: E402
import halo_ia as H             # noqa: E402
try:
    os.remove(os.path.join(HERE, "halo_central_fit_log__review.txt"))
except OSError:
    pass
SF = HF.SF
core, zs, kg = SF.core, SF.zs, SF.kg

# ---------- GoF table ----------
VAR = [("TATT", "TATT", 3), ("hyb", "HM (TATT on $P_{\\rm NL}$ + F21 satellites)", 4),
       ("HMc_add", "HM + central 1h, $P_{\\rm NL}$ kept", 5),
       ("HMc", "HMc, II on $P_{\\rm NL}$", 5), ("HMc_lin", "\\textbf{HMc}", 5),
       ("HMc_lin_b", "HMc + $\\beta$ (mass dependence)", 6), ("HMc_lin_p", "HMc + $p$ (radial profile)", 6),
       ("HMcs_norad", "HMc + coherent satellites", 5),
       ("TATT_W", "TATT + $W$", 4), ("HMc_W", "\\textbf{HMc + $W$}", 6),
       ("TATT_nob", "TATT + $W$, constant $b_g$ (no measured $b(k)$)", 4),
       ("HMc_nob", "\\textbf{HMc + $W$, halo-model $I_g(k)$ (no measured $b(k)$)}", 5)]
JS = [int(np.argmin(np.abs(kg - k))) for k in (0.29, 0.39, 0.52, 0.94, 1.69, 3.04)]
L = ["\\begin{tabular}{l c " + "c"*len(JS) + " l}\\toprule",
     "model & $n_{\\rm par}$ & " + " & ".join(f"{kg[j]:.2f}" for j in JS) + " & cuts ($z = 0.2 \\ldots 2$)\\\\ \\midrule"]
for n, lab, npar in VAR:
    d = np.load(os.path.join(HERE, f"f21v_{n}.npz"))
    C, NU = d[f"{n}_chi2"], d[f"{n}_nu"]
    cuts = d[f"{n}_cut"][core]
    if n == "HMc_lin_b" or n == "HMc_lin_p":
        npar_s = f"{npar}"
    else:
        npar_s = str(npar)
    L.append(f"{lab} & {npar_s} & " + " & ".join(f"{C[core, j].sum()/NU[core, j].sum():.2f}" for j in JS) +
             " & " + " ".join(f"{c:.2f}".lstrip("0") for c in cuts) + "\\\\")
    if n in ("hyb", "HMc_lin", "HMcs_norad", "HMc_W"):
        L.append("\\addlinespace")
L.append("\\bottomrule\\end{tabular}")
open(os.path.join(REP, "tab_hmcr_gof.tex"), "w").write("\n".join(L) + "\n")

# ---------- parameter table (HMc + W) ----------
d = np.load(os.path.join(HERE, "f21v_HMc_W.npz"))
X, C, NU = d["HMc_W_x"], d["HMc_W_chi2"], d["HMc_W_nu"]
JP = [int(np.argmin(np.abs(kg - k))) for k in (0.52, 0.94, 3.04)]
L = ["\\begin{tabular}{c " + "ccccc"*len(JP) + "}\\toprule",
     " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{$k_{{\\max}} = {kg[j]:.2f}$}}" for j in JP) + "\\\\",
     "".join(f"\\cmidrule(lr){{{2 + 5*i}-{6 + 5*i}}}" for i in range(len(JP))),
     "$z$ & " + " & ".join(["$A_1$ & $A_2$ & $b_{\\rm TA}$ & $a_c$ & $W/N_s$"]*len(JP)) + "\\\\ \\midrule"]
for s in np.where(core)[0]:
    c = [f"{zs[s]:g}"]
    for j in JP:
        x = X[s, j]
        c += [f"{x[0]:.2f}", f"{x[1]:.2f}", f"{x[2]:.2f}", f"{x[4]:.1f}", f"{x[5]/SF.Ns[s]:.2f}"]
    L.append(" & ".join(c) + "\\\\")
L.append("\\bottomrule\\end{tabular}")
open(os.path.join(REP, "tab_hmcr_par.tex"), "w").write("\n".join(L) + "\n")

# ---------- Roman table ----------
J = {f: json.load(open(os.path.join(HERE, f"hmc_scan_ex1_{f}.json"))) for f in ("D1", "flat")}
L = ["\\begin{tabular}{c cc cc c cc cc}\\toprule",
     " & \\multicolumn{4}{c}{$k_{\\rm UV}(z) = K_0/D(z)$} & & \\multicolumn{4}{c}{$k_{\\rm UV} = K_0$}\\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){7-10}",
     " & \\multicolumn{2}{c}{case B} & \\multicolumn{2}{c}{case A} & & \\multicolumn{2}{c}{case B} & \\multicolumn{2}{c}{case A}\\\\",
     "$K_0$ & TATT & HMc & TATT & HMc & & TATT & HMc & TATT & HMc\\\\ \\midrule"]
rows = {f: {r["K0"]: r for r in J[f]} for f in J}
for K0 in sorted(rows["D1"]):
    c = [f"{K0:g}"]
    for f in ("D1", "flat"):
        r = rows[f].get(K0)
        for case in ("B", "A"):
            for mdl in ("TATT", "HMc"):
                q = r[f"{mdl}_{case}"] if r else None
                v = f"{q['FoB']:.2f}" if q else "n/a"
                if q and q["FoB"] > 0.3:
                    v = f"\\textcolor{{red!70!black}}{{{v}}}"
                c.append(v)
        if f == "D1":
            c.append("")
    L.append(" & ".join(c) + "\\\\")
L.append("\\bottomrule\\end{tabular}")
open(os.path.join(REP, "tab_hmcr_roman.tex"), "w").write("\n".join(L) + "\n")


# ---------- term anatomy figure ----------
def kax(a, lo, hi, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 10)):
    a.set_xscale("log")
    a.set_xlim(lo, hi)
    a.xaxis.set_major_locator(ticker.FixedLocator(list(ticks)))
    a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_locator(ticker.NullLocator())
    a.grid(alpha=0.3, which="both")


CT = HF.CT
k_all = CT["k_all"]
mk = (k_all >= 0.03) & (k_all <= 10)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.9))
for i, zsel in enumerate((0.3, 1.0)):
    s = int(np.argmin(np.abs(zs - zsel)))
    a = ax[i]
    C1 = SF.model.cosmo["Om"]*H.C1RHOCRIT/SF.model.D(zs[s])
    kd = SF.kd
    a.plot(kd, C1*SF.Pnl[s], color="#1f5fa8", lw=1.8, label=r"$C_1 P_{\rm NL}$ (NLA, TATT, HM)")
    a.plot(k_all[mk], C1*CT["I1m"][s][mk]*CT["Plin"][s][mk], color="#2a9d55", lw=1.8,
           label=r"$C_1 I_1(k) P_{\rm lin}$ (HMc 2-halo)")
    for ac, ls in ((4, ":"), (15, "--"), (40, "-.")):
        a.plot(k_all[mk], ac*CT["Pc_m"][s][mk]/3.0, color="#e08a1e", lw=1.3, ls=ls,
               label=f"central 1-halo $a_c P_c(k)/A_1$, $a_c = {ac}$")
    a.set_yscale("log")
    a.set_ylim(1e-3, 3e2)
    kax(a, 0.03, 10)
    a.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
    a.set_ylabel(r"$|P_{\delta E}|/A_1$ [$(h^{-1}$Mpc$)^3$]")
    a.set_title(f"GI pieces divided by $A_1$ (1-halo curves at $A_1 = 3$), z = {zs[s]:g}", fontsize=10)
    a.legend(fontsize=8, loc="lower left")
# kernel u_E for a few masses
a = ax[2]
lin = H.Linear([0.3])
hm = H.Halo(lin, 0.3, k_all[mk], nM=140, nx=10)
for lgM, col in ((12, "#9ecae1"), (13, "#4292c6"), (14, "#08519c"), (15, "#08306b")):
    iM = int(np.argmin(np.abs(hm.lnM/np.log(10) - lgM)))
    a.plot(k_all[mk], HC.u_E(k_all[mk], hm.rs[iM], hm.c[iM]), color=col, lw=1.6,
           label=f"$M = 10^{{{lgM}}}\\,h^{{-1}}M_\\odot$, $r_{{\\rm vir}}$ = {hm.rvir[iM]:.2f}")
a.set_xscale("log")
a.set_yscale("log")
a.set_ylim(1e-4, 1)
kax(a, 0.03, 10)
a.set_xlabel(r"$k$ [$h\,$Mpc$^{-1}$]")
a.set_ylabel(r"quadrupole kernel $u_E(k|M)$")
a.set_title("elliptical NFW quadrupole, z = 0.3", fontsize=10)
a.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_hmc_terms.png"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_hmc_terms.pdf"), bbox_inches="tight")
print(open(os.path.join(REP, "tab_hmcr_gof.tex")).read())
