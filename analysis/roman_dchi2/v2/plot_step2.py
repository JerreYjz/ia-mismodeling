#!/usr/bin/env python3
"""Step 2 plots from roman/roman_map.npz: FLAMINGO Delta chi2 -> Roman Delta chi2.

mapping_scatter   : Delta chi2_Roman vs Delta chi2_FL for every (model, fit k_max,
                    K) with the same residual on 0.05 <= k <= K in both metrics;
                    colour = K; lines: pure-amplitude exchange rate at each K.
threshold_vs_K    : Delta chi2_FL*(K) that maps to Delta chi2_Roman = 1 (amplitude
                    rate, and the median rate of the model residuals), with each
                    model's own in-fit Delta chi2_FL(K = k_max): a model fails
                    where its curve crosses the threshold.
roman_vs_kmax     : Delta chi2_Roman (K = k_max) vs k_max, matched and with the
                    measured residual above K included (leakage past the cut).
exchange_snapshot : amplitude exchange rate per snapshot vs K.
Writes thresholds.csv. Figures: 220-dpi PNG + PDF.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
MK = {"NLA": "o", "TATT": "s"}
SP = {"real": r"Roman real ($\xi_\pm$)", "fourier": r"Roman Fourier ($C_\ell^{EE}$ bands)"}
KT = [0.12, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]


def kaxis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(KT))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f"{t:g}" for t in KT]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def crossing(x, y, thr):
    """First k where y crosses thr upward (log-interpolated); None if never."""
    for i in range(1, len(x)):
        if y[i - 1] < thr <= y[i]:
            f = (np.log(thr) - np.log(max(y[i - 1], 1e-12)))/(np.log(y[i]) - np.log(max(y[i - 1], 1e-12)))
            return float(np.exp(np.log(x[i - 1]) + f*(np.log(x[i]) - np.log(x[i - 1]))))
    return None if y[0] < thr else float(x[0])


def main():
    d = np.load(sys.argv[1] if len(sys.argv) > 1 else "roman/roman_map.npz")
    out = os.path.dirname(sys.argv[1]) if len(sys.argv) > 1 else "roman"
    kg, zs = d["kgrid"], d["zs"]
    nK = len(kg)
    cmap = plt.cm.viridis
    cK = lambda i: cmap(i/(nK - 1))
    thr = {}

    # 1. scatter
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, sp in zip(axs, SP):
        rate_amp = d[f"{sp}_AR"]/d["AFL"]
        for mdl in COL:
            x, y = d[f"{mdl}_dFL"], d[f"{mdl}_{sp}"]
            for j in range(nK):
                for i in range(nK):
                    if y[j, i] > 1e-4:
                        ax.scatter(x[j, i], y[j, i], color=cK(i), marker=MK[mdl], s=26,
                                   edgecolor="k" if i == j else "none", linewidth=0.8, zorder=3)
        xx = np.geomspace(1, 1e5, 50)
        for i in range(nK):
            if rate_amp[i] > 0:
                ax.plot(xx, rate_amp[i]*xx, color=cK(i), lw=0.9, alpha=0.7)
        ax.axhline(1, color="k", lw=1)
        ax.axhline(0.2, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(1e-4, 1e3)
        ax.set_xlabel(r"$\Delta\chi^2_{\rm FL}$ (sum over 13 snapshots, $0.05\leq k\leq K$)")
        ax.set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ (same residual, IR/UV mask at $K$)")
        ax.set_title(SP[sp])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.LogNorm(kg[0], kg[-1]))
    cb = fig.colorbar(sm, ax=axs, shrink=0.9)
    cb.set_label(r"$K\ [h\,{\rm Mpc}^{-1}]$")
    axs[0].scatter([], [], marker="o", color="0.5", label="NLA fits")
    axs[0].scatter([], [], marker="s", color="0.5", label="TATT fits")
    axs[0].scatter([], [], marker="o", color="w", edgecolor="k", label=r"$K=k_{\rm max}$")
    axs[0].plot([], [], color="0.5", label="pure amplitude change")
    axs[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Same residual (model - FLAMINGO) in two metrics: FLAMINGO error bars vs Roman covariance",
                 fontsize=12)
    save(fig, os.path.join(out, "mapping_scatter"))

    # 2. threshold vs K, with each model's in-fit Delta chi2_FL
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, sp in zip(axs, SP):
        AR, AFL = d[f"{sp}_AR"], d["AFL"]
        t_amp = np.where(AR > 0, AFL/np.where(AR > 0, AR, 1), np.nan)
        rates = []
        for i in range(nK):
            r = []
            for mdl in COL:
                x, y = d[f"{mdl}_dFL"][:, i], d[f"{mdl}_{sp}"][:, i]
                ok = (y > 1e-3) & (x > 0)
                r += list(y[ok]/x[ok])
            rates.append(np.median(r) if r else np.nan)
        t_mod = 1/np.array(rates)
        thr[sp] = (t_amp, t_mod)
        ax.plot(kg, t_amp, "k-", lw=2, label=r"threshold: pure amplitude change")
        ax.plot(kg, t_mod, "k--", lw=1.5, label="threshold: median of NLA/TATT residuals")
        for mdl in COL:
            ax.plot(kg, np.diag(d[f"{mdl}_dFL"]), "o-", color=COL[mdl], lw=1.5,
                    label=rf"{mdl}: own $\Delta\chi^2_{{\rm FL}}(K=k_{{\rm max}})$")
        ax.set_yscale("log")
        kaxis(ax)
        ax.set_xlabel(r"$K = k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$\Delta\chi^2_{\rm FL}$ (sum over 13 snapshots)")
        ax.set_title(SP[sp] + r": $\Delta\chi^2_{\rm FL}$ that gives $\Delta\chi^2_{\rm Roman}=1$")
        ax.set_ylim(5, 1e6)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, os.path.join(out, "threshold_vs_K"))

    # 3. Roman Delta chi2 vs k_max
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    cuts = {}
    for ax, sp in zip(axs, SP):
        for mdl in COL:
            y = np.diag(d[f"{mdl}_{sp}"])
            yf = np.diag(d[f"{mdl}_{sp}_full"])
            ax.plot(kg, np.maximum(y, 1e-5), "o-", color=COL[mdl], lw=1.6,
                    label=f"{mdl}: residual on 0.05 <= k <= k_max")
            ax.plot(kg, np.maximum(yf, 1e-5), "s:", color=COL[mdl], lw=1.2, ms=4,
                    label=f"{mdl}: + measured residual above k_max (to 17.7)")
            cuts[(mdl, sp)] = (crossing(kg, y, 1.0), crossing(kg, yf, 1.0))
        ax.axhline(1, color="k", lw=1)
        ax.axhline(0.2, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e3)
        kaxis(ax)
        ax.set_xlabel(r"$k_{\rm max}$ of the fit = mask $K\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$\Delta\chi^2_{\rm Roman}$")
        ax.set_title(SP[sp])
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, os.path.join(out, "roman_vs_kmax"))

    # 4. exchange rate per snapshot
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    zc = plt.cm.plasma
    for ax, sp in zip(axs, SP):
        R = d[f"{sp}_AR_snap"]/np.where(d["AFL_snap"] > 0, d["AFL_snap"], np.nan)
        for s, z in enumerate(zs):
            ax.plot(kg, R[s], "o-", ms=3, color=zc(s/len(zs)), label=f"z = {z:.2f}")
        ax.plot(kg, d[f"{sp}_AR"]/d["AFL"], "k-", lw=2.5, label="all snapshots")
        ax.set_yscale("log")
        kaxis(ax)
        ax.set_xlabel(r"$K\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$\Delta\chi^2_{\rm Roman}/\Delta\chi^2_{\rm FL}$ (amplitude change)")
        ax.set_title(SP[sp] + ": exchange rate per snapshot")
        ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    save(fig, os.path.join(out, "exchange_snapshot"))

    with open(os.path.join(out, "thresholds.csv"), "w") as fh:
        fh.write("K,AFL,AR_real,AR_fourier,thr_amp_real,thr_models_real,thr_amp_fourier,"
                 "thr_models_fourier,NLA_dFL,TATT_dFL,NLA_real,TATT_real,NLA_fourier,TATT_fourier,"
                 "NLA_real_full,TATT_real_full,NLA_fourier_full,TATT_fourier_full\n")
        for i, K in enumerate(kg):
            v = [K, d["AFL"][i], d["real_AR"][i], d["fourier_AR"][i], thr["real"][0][i],
                 thr["real"][1][i], thr["fourier"][0][i], thr["fourier"][1][i],
                 d["NLA_dFL"][i, i], d["TATT_dFL"][i, i]]
            v += [d[f"{m}_{sp}"][i, i] for sp in ("real", "fourier") for m in COL]
            v += [d[f"{m}_{sp}_full"][i, i] for sp in ("real", "fourier") for m in COL]
            fh.write(",".join(f"{x:.5g}" for x in v) + "\n")
    print("k_max where Delta chi2_Roman = 1 (matched / with leakage):")
    for (m, sp), (a, b) in cuts.items():
        print(f"  {m:4s} {sp:7s}: {a}  /  {b}")
    print(open(os.path.join(out, "thresholds.csv")).read())


if __name__ == "__main__":
    main()
