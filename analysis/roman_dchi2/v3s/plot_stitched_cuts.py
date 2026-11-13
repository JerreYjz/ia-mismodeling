#!/usr/bin/env python3
"""Figures comparing the stitched-truth scale cuts with the production ones.
  fig_s_cuts   : per-snapshot cuts (k horizontal, z vertical): prod2, S2, S1, with k_t(z)
  fig_s_kuv_z  : K0/D(z) family against the stitched cuts and k_t(z)
  fig_s_dchi2  : Delta chi2_FL(k_max), core sum, production vs stitched
  fig_s_amp    : A1(z) (and A2) at the anchor for production and stitched
  cuts_compare.csv
    python plot_stitched_cuts.py <cocoa data dir>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cocoa"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
from cosmolike_driver import Cosmolike          # noqa: E402
from fit_stitched_kmax import KT                 # noqa: E402

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
V = {t: np.load(os.path.join(HERE, t, "v3.npz")) for t in ("prod2", "S2", "S1", "S2w")}
V["prod"] = np.load(os.path.join(HERE, "..", "v3", "v3.npz"))
zs, core, kg = V["S2"]["zs"], V["S2"]["core"], V["S2"]["kgrid"]
sns = V["S2"]["sns"]
kt = np.array([KT[int(s)] for s in sns])
os.chdir(HERE)


def kaxis(a, ticks=(0.1, 0.2, 0.3, 0.5, 0.7, 1)):
    a.set_xscale("log")
    a.set_xticks(ticks)
    a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a.grid(alpha=0.3, which="both")


# ---------------- cuts ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for i, mdl in enumerate(("NLA", "TATT")):
    a = ax[i]
    a.plot(kt, zs, "--", color="0.4", lw=1.2, label="$k_t(z)$ of the notes")
    for t, mk, mfc, dz, lab in (("prod2", "o", "none", -0.04, "production truth (P_hE + P_EE)"),
                                ("S2", "o", COL[mdl], 0.0, "stitched, notes' $N_s$ (S2)"),
                                ("S1", "s", "none", 0.04, "stitched, $N_s$ = B plateau (S1)")):
        c = V[t][f"cut_{mdl}_GoF"]
        ok = core & np.isfinite(c)
        a.plot(c[ok], zs[ok] + dz, mk, ms=8, color=COL[mdl], mfc=mfc, mew=1.6, label=f"GoF cut: {lab}")
    for t, mk, lab in (("S2", "^", "S2: 5% deviation (GI and II)"), ("S2", "v", "S2: 10% deviation")):
        c = V[t][f"cut_{mdl}_{'5%' if mk == '^' else '10%'}"]
        ok = core & np.isfinite(c)
        a.plot(c[ok], zs[ok], mk, ms=7, color="0.2", mfc="none", label=lab)
    a.set_title(f"{mdl}: per-snapshot $k_{{\\max}}$, core snapshots")
    a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
    kaxis(a, ticks=(0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7))
    a.set_xlim(0.07, 0.8)
    a.legend(fontsize=7, loc="upper left")
ax[0].set_ylabel("snapshot z")
ax[0].set_ylim(0.05, 2.25)
fig.tight_layout()
fig.savefig("fig_s_cuts.png", dpi=220)
fig.savefig("fig_s_cuts.pdf")

# ---------------- k_UV(z) ----------------
cl = Cosmolike(os.path.join("..", "cocoa", "lib"), sys.argv[1])
zz = np.linspace(0, 3, 121)
D = np.sqrt(cl.PKL.P(zz, 1e-3)/cl.PKL.P(0, 1e-3)).ravel()
M = np.load(os.path.join(HERE, "S2", "masks_v3.npz")) if os.path.exists(os.path.join(HERE, "S2", "masks_v3.npz")) else None
fig, ax = plt.subplots(figsize=(7.5, 6))
for K0, al in ((0.2, 0.45), (0.3, 0.7), (0.4, 0.95)):
    ax.plot(K0/D, zz, "-", color="0.3", alpha=al, lw=1.3, label=f"production: $K_0/D(z)$, $K_0={K0}$")
ax.plot(kt, zs, "--", color="0.4", lw=1.2, label="$k_t(z)$ of the notes (NLA below)")
for mdl, crit, mk, lab in (("NLA", "GoF", "o", "NLA GoF, stitched truth"), ("TATT", "GoF", "s", "TATT GoF, stitched truth"),
                            ("NLA", "10%", "v", "NLA 10% (GI+II), stitched"), ("TATT", "10%", "^", "TATT 10% (GI+II), stitched")):
    c = V["S2"][f"cut_{mdl}_{crit}"]
    ok = core & np.isfinite(c)
    ax.plot(c[ok], zs[ok], mk, ms=7, color=COL[mdl], mfc=COL[mdl] if crit == "GoF" else "none", mew=1.5, label=lab)
for mdl, mk in (("NLA", "o"), ("TATT", "s")):
    c = V["prod"][f"cut_{mdl}_GoF"]
    ok = core & np.isfinite(c)
    ax.plot(c[ok], zs[ok], mk, ms=5, color="0.6", mfc="none", label=f"{mdl} GoF, production truth (main PDF)")
if M is not None:
    ax.plot(M["kuv_NLA_GoF"], M["z2"], ":", color=COL["NLA"], lw=1.5, label="NLA-shaped mask used in Roman (f = 1)")
    ax.plot(M["kuv_TATT_GoF"], M["z2"], ":", color=COL["TATT"], lw=1.5, label="TATT-shaped mask used in Roman (f = 1)")
ax.axhspan(0.2, 2.0, color="0.92", zorder=0, label="core snapshots")
kaxis(ax, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1))
ax.set_xlim(0.05, 1.2)
ax.set_ylim(0, 3.6)
ax.set_xlabel(r"$k_{\rm UV}$ [h/Mpc]")
ax.set_ylabel("z")
ax.set_title("Scale cut versus redshift: $K_0/D(z)$ against the stitched-truth cuts")
ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig("fig_s_kuv_z.png", dpi=220)
fig.savefig("fig_s_kuv_z.pdf")

# ---------------- Delta chi2_FL ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for mdl in ("NLA", "TATT"):
    for t, ls, lab in (("prod2", "-", "production truth"), ("S2", "--", "stitched (S2)"), ("S1", ":", "stitched, B-plateau $N_s$ (S1)")):
        ax[0].plot(kg, V[t][f"comb_{mdl}_dFL"], ls, marker="o", ms=3, color=COL[mdl], label=f"{mdl}, {lab}")
        ax[1].plot(kg, V[t][f"comb_{mdl}_p"], ls, marker="o", ms=3, color=COL[mdl], label=f"{mdl}, {lab}")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$\Delta\chi^2_{\rm FL}$ (refit vs noise-free truth, core sum)")
ax[1].set_yscale("log")
ax[1].set_ylim(1e-8, 1.5)
ax[1].axhline(0.05, color="k", lw=0.8, ls="--")
ax[1].set_ylabel("combined GoF p-value (core snapshots)")
for a in ax:
    kaxis(a, ticks=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3))
    a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig("fig_s_dchi2.png", dpi=220)
fig.savefig("fig_s_dchi2.pdf")

# ---------------- amplitudes ----------------
import json
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
zzl = np.linspace(0, 4, 100)
pl = lambda xx, A, eta: A*xx**eta
for i, (mdl, idx, lab) in enumerate((("NLA", 0, "$A_1$"), ("TATT", 0, "$A_1$"), ("TATT", 1, "$A_2$"))):
    for t, mk, mfc, dz, tl in (("prod2", "o", "none", -0.03, "production"), ("S2", "o", COL[mdl], 0.03, "stitched (S2)")):
        A, E = V[t][f"{mdl}_x_anch"][:, idx], V[t][f"{mdl}_e_anch"][:, idx]
        ax[i].errorbar(A[core], zs[core] + dz, xerr=E[core], fmt=mk, mfc=mfc, color=COL[mdl], capsize=3, label=f"{tl}, anchor $k \\leq 0.16$")
        pw = json.load(open(os.path.join(HERE, t, "summary.json")))["powerlaws"][mdl].get(f"pl_{['a1', 'a2'][idx]}_anchor")
        if pw:
            (A0, eta), (eA, ee) = pw
            ax[i].plot(pl((1 + zzl)/1.62, A0, eta), zzl, "-" if t == "S2" else ":", color=COL[mdl], lw=1,
                       label=f"{tl}: {A0:.2f}$\\pm${eA:.2f} ((1+z)/1.62)^({eta:.2f}$\\pm${ee:.2f})")
    if idx == 0:
        ax[i].plot(pl((1 + zzl)/1.62, 3.5, 0.68), zzl, "k:", lw=1, label="Roman fit 3.5 ((1+z)/1.62)^0.68")
    ax[i].set_xlabel(f"{mdl} {lab}")
    ax[i].set_title(f"{mdl} {lab}(z), core snapshots")
    if idx == 1:
        ax[i].set_xlim(-3, 2)
    ax[i].grid(alpha=0.3)
    ax[i].legend(fontsize=7)
ax[0].set_ylabel("snapshot z")
fig.tight_layout()
fig.savefig("fig_s_amp.png", dpi=220)
fig.savefig("fig_s_amp.pdf")

# ---------------- table ----------------
with open("cuts_compare.csv", "w") as fh:
    crits = ("GoF", "anchored", "5%", "10%")
    fh.write("z,core,k_t," + ",".join(f"{t}_{m}_{c}" for t in ("prod2", "S2", "S2w", "S1") for m in ("NLA", "TATT") for c in crits) + "\n")
    for s in range(len(zs)):
        fh.write(f"{zs[s]:g},{int(core[s])},{kt[s]:.3f}," + ",".join(f"{V[t][f'cut_{m}_{c}'][s]:.3f}" for t in ("prod2", "S2", "S2w", "S1") for m in ("NLA", "TATT") for c in crits) + "\n")
print(open("cuts_compare.csv").read())
