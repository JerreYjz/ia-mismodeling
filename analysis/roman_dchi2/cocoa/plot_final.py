#!/usr/bin/env python3
"""fig_kmax_fob: FoB vs mask, production truth / with a white II template / single-tidal-field truth."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
v2 = json.load(open("analysis/bias_v2.json"))["fits"]
dec = json.load(open("analysis/bias_white.json"))
wt = json.load(open("analysis/bias_wtemplate.json"))
masks = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
m3 = [0.25, 0.35, 0.5]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for mdl in ("NLA", "TATT"):
    fob = np.array([[r for r in v2 if r["model"] == mdl and r["data"] == "E" and r["mask"] == K][0]["flat"]["FoB"] for K in masks])
    sig = np.array([[r for r in v2 if r["model"] == mdl and r["data"] == "E" and r["mask"] == K][0]["flat"]["FoB_sig"] for K in masks])
    chi = np.array([[r for r in v2 if r["model"] == mdl and r["data"] == "E" and r["mask"] == K][0]["chi2_bestfit"] for K in masks])
    ax[0].plot(masks, fob, "-o", ms=5, color=COL[mdl], label=f"{mdl}, FLAMINGO truth (production)")
    ax[0].fill_between(masks, fob - sig, fob + sig, color=COL[mdl], alpha=0.12, lw=0)
    ax[1].plot(masks, chi, "-o", ms=5, color=COL[mdl], label=f"{mdl}, FLAMINGO truth (production)")
    fw = [[r for r in wt if r["model"] == mdl + "w" and r["mask"] == K][0] for K in m3]
    ax[0].plot(m3, [r["flat"]["FoB"] for r in fw], "--s", ms=5, color=COL[mdl], label=f"{mdl} + free white II amplitude")
    ax[1].plot(m3, [r["chi2_bestfit"] for r in fw], "--s", ms=5, color=COL[mdl], label=f"{mdl} + free white II amplitude")
    fd = [[r for r in dec if r["model"] == mdl and r["variant"] == "GI_II1" and r["mask"] == K][0] for K in m3]
    ax[0].plot(m3, [r["flat"]["FoB"] for r in fd], ":^", ms=5, color=COL[mdl], label=f"{mdl}, single-tidal-field truth (II = GI$^2$/$P_{{\\rm NL}}$)")
    ax[1].plot(m3, [r["chi2_bestfit"] for r in fd], ":^", ms=5, color=COL[mdl], label=f"{mdl}, single-tidal-field truth")
ax[0].axhline(0.3, color="k", lw=0.8, ls="--")
ax[0].set_ylabel(r"figure of bias in $(\Omega_m, S_8)$ [$\sigma$]")
ax[0].set_title("Roman $\\xi_\\pm$, case A ($\\Omega_m$, $A_s$ free, IA free); band: FLAMINGO error bars")
ax[1].axhline(1, color="k", lw=0.8, ls="--")
ax[1].set_ylabel(r"$\chi^2$ at the best fit")
ax[1].set_title("residual misfit after fitting")
for a in ax:
    a.set_xlabel(r"mask $K_0$: $k_{\rm UV} = K_0/D(z)$ [h/Mpc]")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig("analysis/fig_kmax_fob.png", dpi=220); fig.savefig("analysis/fig_kmax_fob.pdf")
