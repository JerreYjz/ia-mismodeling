#!/usr/bin/env python3
"""k_UV(z): production family K0/D(z) versus the cuts measured in FLAMINGO (k horizontal, z vertical)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import ticker
sys.path.insert(0, "../cocoa"); from cosmolike_driver import Cosmolike
DATA = sys.argv[1]
cl = Cosmolike(os.path.join("..", "cocoa", "lib"), DATA)
zz = np.linspace(0, 3, 121)
D = np.sqrt(cl.PKL.P(zz, 1e-3)/cl.PKL.P(0, 1e-3)).ravel()
v = np.load("v3.npz"); zs, core = v["zs"], v["core"]
M = np.load("masks_v3.npz"); z2 = M["z2"]
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
fig, ax = plt.subplots(figsize=(7.5, 6))
for K0, a in ((0.15, 0.35), (0.2, 0.55), (0.25, 0.75), (0.3, 0.95)):
    ax.plot(K0/D, zz, "-", color="0.3", alpha=a, lw=1.3, label=f"production: $K_0/D(z)$, $K_0={K0}$")
for mdl, crit, mk, lab in (("NLA", "GoF", "o", "NLA GoF ($N_s$ fixed)"), ("TATT", "GoF", "s", "TATT GoF"),
                            ("TATT", "anchored", "^", "TATT anchored"), ("TATT", "10%", "v", "TATT 10% (GI+II)"),
                            ("NLA", "5%_GI", "D", "NLA 5% on GI"), ("TATT", "5%_GI", "d", "TATT 5% on GI")):
    c = v[f"cut_{mdl}_{crit}"]
    ok = core & np.isfinite(c)
    ax.plot(c[ok], zs[ok], mk, ms=7, color=COL[mdl], mfc=COL[mdl] if crit == "GoF" else "none", mew=1.5, label=lab)
ax.plot(M["kuv_NLA_GoF"], z2, ":", color=COL["NLA"], lw=1.5, label="NLA-shaped mask used in Roman (f = 1)")
ax.plot(M["kuv_TATT_GoF"], z2, ":", color=COL["TATT"], lw=1.5, label="TATT-shaped mask used in Roman (f = 1)")
ax.axhspan(0.2, 2.0, color="0.9", zorder=0, label="core snapshots")
ax.set_xscale("log"); ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]); ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(ticker.NullFormatter()); ax.set_xlim(0.05, 1.2); ax.set_ylim(0, 3.6)
ax.set_xlabel(r"$k_{\rm UV}$ [h/Mpc]"); ax.set_ylabel("z"); ax.grid(alpha=0.3, which="both")
ax.set_title("Scale cut versus redshift: $K_0/D(z)$ family against the FLAMINGO-measured cuts")
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 1.0), framealpha=0.9); fig.tight_layout()
fig.savefig("fig_kuv_z.png", dpi=220); fig.savefig("fig_kuv_z.pdf")
for z in (0.2, 0.5, 1.0, 1.5, 2.0):
    i = np.argmin(abs(zz - z)); print(f"z {z}: D {D[i]:.3f}; K0/D for K0=0.2: {0.2/D[i]:.3f}, 0.25: {0.25/D[i]:.3f}")
