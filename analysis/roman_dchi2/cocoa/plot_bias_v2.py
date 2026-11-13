#!/usr/bin/env python3
"""Figures and table for bias_v2.json."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
STY = {"E": ("-", "o", "E modes (production)"), "EB": ("--", "s", "+ FLAMINGO B modes"), "raw": (":", "^", "raw spectra")}


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def main():
    out = json.load(open("analysis/bias_v2.json"))
    fits = out["fits"]
    masks = sorted(set(r["mask"] for r in fits))

    def get(mdl, dname, key, sub=None):
        v = []
        for K0 in masks:
            r = [r for r in fits if r["model"] == mdl and r["data"] == dname and r["mask"] == K0]
            if not r:
                v.append(np.nan)
            else:
                v.append(r[0][key] if sub is None else r[0][sub][key])
        return np.array(v)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for mdl in ("NLA", "TATT"):
        for dname, (ls, mk, lab) in STY.items():
            fob = get(mdl, dname, "FoB", "flat")
            ax[0].plot(masks, fob, ls, marker=mk, ms=4, color=COL[mdl], label=f"{mdl}, {lab}")
            if dname == "E":
                sig = get(mdl, dname, "FoB_sig", "flat")
                ax[0].fill_between(masks, fob - sig, fob + sig, color=COL[mdl], alpha=0.12, lw=0)
            ax[1].plot(masks, get(mdl, dname, "chi2_bestfit"), ls, marker=mk, ms=4, color=COL[mdl], label=f"{mdl}, {lab}")
        ax[2].plot(masks, get(mdl, "E", "dS8")/get(mdl, "E", "sS8", "flat"), "-", marker="o", ms=4, color=COL[mdl],
                   label=f"{mdl}: $\\Delta S_8/\\sigma$")
        ax[2].plot(masks, get(mdl, "E", "dOm")/get(mdl, "E", "sOm", "flat"), "--", marker="s", ms=4, color=COL[mdl],
                   label=f"{mdl}: $\\Delta\\Omega_m/\\sigma$")
    ax[0].axhline(0.3, color="k", lw=0.8, ls="--")
    ax[0].set_ylabel(r"figure of bias in $(\Omega_m, S_8)$ [$\sigma$]")
    ax[0].set_title("FoB (band: FLAMINGO error bars propagated)")
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].set_ylabel(r"$\chi^2$ at the best fit (IA + $\Omega_m$, $A_s$ free)")
    ax[1].set_title("residual Roman misfit after fitting")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel(r"1D shift [$\sigma$], E modes")
    ax[2].set_title("1D shifts")
    for a in ax:
        a.set_xlabel(r"mask $K_0$: $k_{\rm UV} = K_0 / D(z)$ [h/Mpc]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "analysis/fig_bias_v2")

    # IA parameters of the TATT fits
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for dname, (ls, mk, lab) in STY.items():
        for i, n in enumerate(["A1", "A2", "bTA"]):
            ax[i].plot(masks, [r["best"][n] for K0 in masks for r in fits
                               if r["model"] == "TATT" and r["data"] == dname and r["mask"] == K0],
                       ls, marker=mk, ms=4, color=COL["TATT"], label=f"TATT, {lab}")
        ax[0].plot(masks, [r["best"]["A1"] for K0 in masks for r in fits
                           if r["model"] == "NLA" and r["data"] == dname and r["mask"] == K0],
                   ls, marker=mk, ms=4, color=COL["NLA"], label=f"NLA, {lab}")
    for a, n in zip(ax, ["$A_1$", "$A_2$", "$b_{TA}$"]):
        a.set_ylabel(n)
        a.set_xlabel(r"mask $K_0$ [h/Mpc]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "analysis/fig_bias_v2_ia")

    with open("analysis/bias_v2.csv", "w") as fh:
        fh.write("data,mask,model,npts,chi2_bestfit,dOm,sOm,dS8,sS8,FoB_flatprior,FoB_sig_noise,FoB_reg50,A1,eta1,A2,eta2,bTA,at_bound\n")
        for r in fits:
            b = r["best"]
            fh.write(f"{r['data']},{r['mask']},{r['model']},{r['npts']},{r['chi2_bestfit']:.3f},{r['dOm']:+.5f},"
                     f"{r['flat']['sOm']:.5f},{r['dS8']:+.5f},{r['flat']['sS8']:.5f},{r['flat']['FoB']:.3f},"
                     f"{r['flat'].get('FoB_sig', np.nan):.3f},{r['reg50']['FoB']:.3f},"
                     f"{b['A1']:.3f},{b['eta1']:.3f},{b.get('A2', 0):.3f},{b.get('eta2', 0):.3f},{b.get('bTA', 0):.3f},"
                     f"{'|'.join(r['at_bound'])}\n")
    print(open("analysis/bias_v2.csv").read())
    if "exact" in out:
        print("exact fits:")
        for r in out["exact"]:
            print(f"  {r['data']:3s} {r['model']:4s} chi2 {r['chi2']:.3f} (lin {r['lin_chi2']:.3f}); "
                  f"dOm {r['dOm']:+.4f} (lin {r['lin_dOm']:+.4f}); dS8 {r['dS8']:+.4f} (lin {r['lin_dS8']:+.4f})")


if __name__ == "__main__":
    main()
