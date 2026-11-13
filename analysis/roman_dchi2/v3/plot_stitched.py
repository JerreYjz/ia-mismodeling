#!/usr/bin/env python3
"""Figures and tables for the stitched-truth control test.
  fig_stitch_truths : production, stitched and low-k-NLA truths vs the data
  fig_stitch_fob    : FoB and chi2 vs cut for every truth variant
  stitched_test.csv
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
from fit_stitched_kmax import KT  # noqa: E402
from stitched_test import LABEL   # noqa: E402

os.chdir(HERE)
T = np.load("stitched_truths.npz")
fits = np.load("../v2/fits.npz")
v3 = np.load("v3.npz")
res = json.load(open("stitched_test.json"))
prod = json.load(open("roman_test.json"))
Mz = np.load("masks_v3.npz")
z2 = Mz["z2"]
kuv_flat = {f: float(np.exp(np.interp(1.0, z2, np.log(f*Mz["kuv_TATT_GoF"])))) for f in (1.0, 1.4, 2.0)}
kd, zs, sns = T["k"], T["zs"], T["sns"]
VCOL = {"prod": "k", "S1": "#1f5fa8", "S2": "#6aa2dc", "S3": "#2a9d8f", "S4": "#e07b39"}
VLS = {"prod": "-", "S1": "-", "S2": "--", "S3": ":", "S4": "-."}

# ---------------- truths ----------------
y, err, b, Pf, Ns = fits["y"], fits["err"], fits["b"], fits["PNL"], fits["Ns_true"]
show = [3, 7, 9, 10]            # z = 0.3, 0.75, 1.5, 2
fig, ax = plt.subplots(3, len(show), figsize=(4.2*len(show), 10), sharex=True)
for c, s in enumerate(show):
    z, sn = zs[s], int(sns[s])
    C1 = T["C1"][s]
    m = kd >= 0.05
    # data (measured, converted to ratios), normalised to the NLA side
    r_dE = -y[s, 0]/(b[s]*Pf[s])/C1
    e_dE = err[s, 0]/(b[s]*Pf[s])/C1
    r_EE = (y[s, 1] - Ns[s])/Pf[s]/C1**2
    e_EE = err[s, 1]/Pf[s]/C1**2
    for row, (idx, rd, ed, lab) in enumerate(((0, r_dE, e_dE, r"$-P_{\delta E}\,/\,(C_1 P_{\rm NL})$"),
                                              (1, r_EE, e_EE, r"$(P_{EE}-N_s^{B})\,/\,(C_1^2 P_{\rm NL})$"))):
        a = ax[row, c]
        a.errorbar(kd[m], rd[m], ed[m], fmt="o", ms=3, color="0.55", lw=0.8, label="FLAMINGO (measured)", zorder=1)
        sign = -1.0 if idx == 0 else 1.0
        norm = C1 if idx == 0 else C1**2
        for v in ("prod", "S1", "S3", "S4"):
            R = T[f"R_{v}"][s, idx]
            if idx == 1 and v == "S3":          # notes' N_s: show on the same (B-plateau) axis
                R = R + (np.load("../spectra.npz")["Ns_nla"][s] - Ns[s])/Pf[s]*T["w"][s]
            a.plot(kd, sign*R/norm, VLS[v], color=VCOL[v], lw=1.6 if v == "prod" else 1.2, label=LABEL[v])
        a.axvline(KT[sn], color="0.4", ls="--", lw=0.9)
        for mdl, col in (("NLA", "#d1495b"), ("TATT", "#1f5fa8")):
            cut = v3[f"cut_{mdl}_GoF"][s]
            if np.isfinite(cut):
                a.axvline(cut, color=col, ls=":", lw=1.2)
        a.set_xscale("log")
        a.grid(alpha=0.3)
        if c == 0:
            a.set_ylabel(lab)
        if row == 0:
            a.set_title(f"z = {z:g}  (snapshot {sn}, $k_t$ = {KT[sn]:.3f})")
            a.set_ylim(0.6, 1.6)
        else:
            a.set_ylim(0.6, 3.0)
    a = ax[2, c]
    a.plot(kd, T["w"][s], "-", color="0.3", label=r"stitch weight $w(k)$ of the simulation side")
    a.axvline(KT[sn], color="0.4", ls="--", lw=0.9, label=r"$k_t(z)$ of the notes")
    a.axvline(v3["cut_NLA_GoF"][s], color="#d1495b", ls=":", lw=1.2, label="NLA GoF cut (v3, $N_s$ fixed)")
    a.axvline(v3["cut_TATT_GoF"][s], color="#1f5fa8", ls=":", lw=1.2, label="TATT GoF cut (v3)")
    a.axvspan(0.009, 0.05, color="0.9", label="below data $k_{\\min}$")
    a.set_xscale("log")
    a.set_xlim(0.009, 1.0)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("k [h/Mpc]")
    a.grid(alpha=0.3)
    if c == 0:
        a.set_ylabel("w")
ax[0, 0].legend(fontsize=7, loc="upper left")
ax[2, 0].legend(fontsize=7, loc="center left")
fig.tight_layout()
fig.savefig("fig_stitch_truths.png", dpi=220)
fig.savefig("fig_stitch_truths.pdf")

# ---------------- FoB ----------------
def rows(src, fam, case, mdl, v=None):
    if v is None or v == "prod":
        r = [x for x in src if x["family"] == fam and x["case"] == case and x["test"] == f"{mdl}_on_FLAMINGO"]
    else:
        r = [x for x in src if x["family"] == fam and x["case"] == case and x["model"] == mdl and x["variant"] == v]
    if fam == "K0/D":
        r.sort(key=lambda x: x["K0"])
        xs = [x["K0"] for x in r]
    else:
        r = [x for x in r if x["factor"] in kuv_flat]
        r.sort(key=lambda x: x["factor"])
        xs = [kuv_flat[x["factor"]] for x in r]
    return np.array(xs), r


fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for row, mdl in enumerate(("NLA", "TATT")):
    for col, (fam, xl) in enumerate((("K0/D", r"mask $K_0$: $k_{\rm UV}=K_0/D(z)$ [h/Mpc]"),
                                     ("flat", r"flat cut: $k_{\rm UV}(z=1)$ [h/Mpc]"))):
        a = ax[row, col]
        for v in ("prod", "S1", "S2", "S3", "S4"):
            x, r = rows(prod if v == "prod" else res, fam, "A", mdl, v)
            if not len(r):
                continue
            fob = np.array([q["FoB"] for q in r])
            sig = np.array([q.get("FoB_sig", 0) for q in r])
            a.plot(x, fob, VLS[v], marker="o", ms=4, color=VCOL[v], lw=1.8 if v == "prod" else 1.3, label=LABEL[v])
            if v in ("prod", "S1"):
                a.fill_between(x, fob - sig, fob + sig, color=VCOL[v], alpha=0.12, lw=0)
        if fam == "K0/D":
            x, r = rows(res, fam, "B", mdl, "S1")
            a.plot(x, [q["FoB"] for q in r], "^", ms=5, mfc="none", color=VCOL["S1"], label="S1, case B (5 cosmological parameters)")
        a.axhline(0.3, color="k", lw=0.8, ls="--")
        a.set_ylabel(r"FoB in $(\Omega_m, S_8)$ [$\sigma$]")
        a.set_title(f"{mdl} fitted, all parameters free (case A), {fam} masks")
        a.set_xlabel(xl)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
        a.set_ylim(0, 0.7)
        if fam == "flat":
            a.set_xscale("log")
            a.set_xticks([0.3, 0.4, 0.6])
            a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a = ax[row, 2]
    for v in ("prod", "S1", "S2", "S3", "S4"):
        x, r = rows(prod if v == "prod" else res, "K0/D", "A", mdl, v)
        a.plot(x, [q["chi2_bestfit"] for q in r], VLS[v], marker="o", ms=4, color=VCOL[v], lw=1.8 if v == "prod" else 1.3, label=LABEL[v])
    a.axhline(1, color="k", lw=0.8, ls="--")
    a.set_ylabel(r"$\chi^2$ at the best fit")
    a.set_title(f"{mdl}: residual misfit, K0/D masks")
    a.set_xlabel(r"mask $K_0$ [h/Mpc]")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig("fig_stitch_fob.png", dpi=220)
fig.savefig("fig_stitch_fob.pdf")

# ---------------- shift direction ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for col, mdl in enumerate(("NLA", "TATT")):
    a = ax[col]
    for v in ("prod", "S1", "S2"):
        x, r = rows(prod if v == "prod" else res, "K0/D", "A", mdl, v)
        a.plot(x, [q["dOm"]/q["sOm"] for q in r], VLS[v], marker="o", ms=4, color=VCOL[v], lw=1.8 if v == "prod" else 1.3,
               label=r"$\delta\Omega_m/\sigma$: " + LABEL[v])
        a.plot(x, [q["dS8"]/q["sS8"] for q in r], VLS[v], marker="s", ms=4, mfc="none", color=VCOL[v], lw=1.8 if v == "prod" else 1.3,
               label=r"$\delta S_8/\sigma$: " + LABEL[v])
    a.axhline(0, color="k", lw=0.8)
    for yv in (-0.3, 0.3):
        a.axhline(yv, color="k", lw=0.8, ls="--")
    a.set_xlabel(r"mask $K_0$: $k_{\rm UV}=K_0/D(z)$ [h/Mpc]")
    a.set_ylabel(r"parameter shift [$\sigma$]")
    a.set_title(f"{mdl} fitted: direction of the shift (filled: $\\Omega_m$, open: $S_8$)")
    a.grid(alpha=0.3)
    a.legend(fontsize=6.5)
fig.tight_layout()
fig.savefig("fig_stitch_shift.png", dpi=220)
fig.savefig("fig_stitch_shift.pdf")

# ---------------- table ----------------
with open("stitched_test.csv", "w") as fh:
    fh.write("variant,family,mask,case,model,npts,dchi2_vs_prod,chi2,dOm_sig,dS8_sig,FoB,FoB_sig,at_bound\n")
    for v in prod:
        if v["test"] in ("NLA_on_FLAMINGO", "TATT_on_FLAMINGO") and (v["family"] == "K0/D" or v["factor"] in kuv_flat):
            fh.write(f"prod,{v['family']},{v['mask']},{v['case']},{v['model']},{v['npts']},0,{v['chi2_bestfit']:.3f},"
                     f"{v['dOm']/v['sOm']:+.2f},{v['dS8']/v['sS8']:+.2f},{v['FoB']:.3f},{v.get('FoB_sig', float('nan')):.3f},{'|'.join(v['at_bound'])}\n")
    for v in res:
        fh.write(f"{v['variant']},{v['family']},{v['mask']},{v['case']},{v['model']},{v['npts']},{v['dchi2_vs_prod']:.2f},{v['chi2_bestfit']:.3f},"
                 f"{v['dOm']/v['sOm']:+.2f},{v['dS8']/v['sS8']:+.2f},{v['FoB']:.3f},{v.get('FoB_sig', float('nan')):.3f},{'|'.join(v['at_bound'])}\n")
print(open("stitched_test.csv").read())
