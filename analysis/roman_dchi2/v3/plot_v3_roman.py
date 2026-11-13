#!/usr/bin/env python3
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import ticker
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
res = json.load(open("roman_results.json"))
_v3 = np.load("v3.npz")
def _rng(mdl):
    c = _v3[f"cut_{mdl}_GoF"][_v3["core"]]
    c = c[np.isfinite(c)]
    return f"{c.min():.2f}-{c.max():.2f}"
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for shape, ls, mk, lab in (("NLA_GoF", ":", "o", f"NLA-shaped cut (k {_rng('NLA')} at f=1)"), ("TATT_GoF", "-", "s", f"TATT-shaped cut (k {_rng('TATT')} at f=1)")):
    rows = [r for r in res if r["shape"] == shape]
    x = [r["kuv_z1"] for r in rows]
    for mdl in ("NLA", "TATT"):
        fob = np.array([r[f"FoB_{mdl}"] for r in rows]); sig = np.array([r[f"FoBsig_{mdl}"] for r in rows])
        ax[0].plot(x, fob, ls, marker=mk, ms=5, color=COL[mdl], label=f"{mdl}, {lab}")
        ax[0].fill_between(x, fob - sig, fob + sig, color=COL[mdl], alpha=0.1, lw=0)
        xw = [r["kuv_z1"] for r in rows if f"FoBw_{mdl}" in r]
        ax[0].plot(xw, [r[f"FoBw_{mdl}"] for r in rows if f"FoBw_{mdl}" in r], "x", ms=8, mew=2, color=COL[mdl],
                   label=f"{mdl} + white II amplitude" if shape == "TATT_GoF" else None)
        ax[1].plot(x, [r[f"chi2_{mdl}"] for r in rows], ls, marker=mk, ms=5, color=COL[mdl], label=f"{mdl}, {lab}")
        ax[2].plot(x, [r[f"dchi2_{mdl}"] for r in rows], ls, marker=mk, ms=5, color=COL[mdl], label=f"{mdl}, {lab}")
    for r in rows:
        ax[0].annotate(f"{r['npts']}", (r["kuv_z1"], r["FoB_NLA"]), textcoords="offset points", xytext=(0, -12), fontsize=6, ha="center", color="0.4")
ax[0].axhline(0.3, color="k", lw=0.8, ls="--"); ax[0].set_ylabel(r"figure of bias in $(\Omega_m, S_8)$ [$\sigma$]")
ax[0].set_title("FoB vs z-dependent cut (numbers: data points kept of 1080)")
ax[1].axhline(1, color="k", lw=0.8, ls="--"); ax[1].set_ylabel(r"$\chi^2$ at the best fit (IA + $\Omega_m$, $A_s$ refitted)")
ax[1].set_title("residual misfit after fitting in Roman")
ax[2].axhline(1, color="k", lw=0.8, ls="--"); ax[2].set_yscale("log"); ax[2].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$, model amplitudes from FLAMINGO")
ax[2].set_title("truth vs the model with the per-snapshot refit amplitudes (no Roman refit)")
for a in ax:
    a.set_xscale("log"); a.set_xticks([0.06, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]); a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    a.xaxis.set_minor_formatter(ticker.NullFormatter()); a.set_xlabel(r"$k_{\rm UV}(z=1)$ of the mask [h/Mpc]"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig_v3_roman.png", dpi=220); fig.savefig("fig_v3_roman.pdf")
with open("roman_results.csv", "w") as fh:
    fh.write("shape,factor,kuv_z0.5,kuv_z1,npts,dchi2_NLA,dchi2_TATT,chi2_NLA,FoB_NLA,FoBsig_NLA,chi2_TATT,FoB_TATT,FoBsig_TATT,FoBw_NLA,FoBw_TATT\n")
    for r in res:
        fh.write(f"{r['shape']},{r['factor']},{r['kuv_z05']:.3f},{r['kuv_z1']:.3f},{r['npts']},{r['dchi2_NLA']:.2f},{r['dchi2_TATT']:.2f},{r['chi2_NLA']:.2f},{r['FoB_NLA']:.2f},{r['FoBsig_NLA']:.2f},{r['chi2_TATT']:.2f},{r['FoB_TATT']:.2f},{r['FoBsig_TATT']:.2f},{r.get('FoBw_NLA', float('nan')):.2f},{r.get('FoBw_TATT', float('nan')):.2f}\n")
print(open("roman_results.csv").read())
