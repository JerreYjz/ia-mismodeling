#!/usr/bin/env python3
"""Step 1 plots and k_max table from fits.npz (FLAMINGO only).

Per snapshot, vs k_max:
  gof_pvalue   : p-value of the fit to the measured spectra (chi2_raw, dof)
  gof_chi2dof  : chi2_raw / dof
  dchi2_fl     : noise-free Delta chi2_FL(K = k_max) of the fit to the smoothed
                 spectra (model - FLAMINGO over 0.05 <= k <= k_max, FLAMINGO errors)
  amp_<par>    : fitted amplitudes (raw fits) with 1 sigma errors
Cut per snapshot and model: last k_max before two consecutive p < 0.05.
Writes kmax_gof.csv. Figures: 220-dpi PNG + PDF.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
PAR = {"NLA": ["a1", "Ns"], "TATT": ["a1", "a2", "bTA", "Ns"]}


KT = [0.12, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]


def kaxis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(KT))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in KT]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def cut(k, p, pmin=0.05):
    bad = p < pmin
    for i in range(len(p)):
        if bad[i] and (i + 1 == len(p) or bad[i + 1]):
            return (k[i - 1] if i > 0 else np.nan), ("" if i > 0 else "fails at first")
    return k[-1], "never fails (lower bound)"


def grid(n=13):
    fig, axs = plt.subplots(3, 5, figsize=(24, 13))
    for ax in axs.flat[n:]:
        ax.axis("off")
    return fig, axs


def main():
    d = np.load(sys.argv[1] if len(sys.argv) > 1 else "fits.npz")
    zs, kg = d["zs"], d["kgrid"]
    xl = r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$"
    cuts = {}
    for key, ylab, fn in (("p", "p-value (fit to measured spectra)", "gof_pvalue"),
                          ("c", r"$\chi^2_{\rm raw}/{\rm dof}$", "gof_chi2dof"),
                          ("l", r"$\Delta\chi^2_{\rm FL}$ (model $-$ FLAMINGO, $k\leq k_{\rm max}$)",
                           "dchi2_fl")):
        fig, axs = grid()
        for i, ax in enumerate(axs.flat[:len(zs)]):
            for m in COL:
                c2, dof, p = d[f"{m}_chi2_raw"][i], d[f"{m}_dof"][i], d[f"{m}_p_raw"][i]
                y = {"p": np.maximum(p, 1e-30), "c": c2/dof, "l": np.maximum(d[f"{m}_lam"][i], 1e-3)}[key]
                ax.plot(kg, y, "o-", color=COL[m], ms=4, label=m)
                if key == "p":
                    kc, _ = cut(kg, p)
                    cuts[(i, m)] = cut(kg, p)
                    if np.isfinite(kc):
                        ax.axvline(kc, color=COL[m], ls="--", lw=1)
            if key == "p":
                ax.axhline(0.05, color="k", lw=0.8)
                ax.set_yscale("log")
                ax.set_ylim(1e-30, 2)
            elif key == "c":
                ax.axhline(1, color="k", lw=0.8)
                ax.set_yscale("log")
            else:
                ax.set_yscale("log")
                ax.plot(kg, np.maximum(d["NLA_dof"][i], 1), ":", color="0.5", label="dof (NLA)")
            kaxis(ax)
            ax.set_xlabel(xl)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(f"z = {zs[i]:.2f}")
        axs.flat[0].legend(fontsize=8)
        fig.suptitle({"p": "Goodness of fit to FLAMINGO P_hE + P_EE + P_BB (FLAMINGO errors); "
                           "dashed: last k_max before two consecutive p < 0.05",
                      "c": "chi2 / dof of the fit to the measured FLAMINGO spectra",
                      "l": "Noise-free Delta chi2_FL: fit to the smoothed FLAMINGO spectra, "
                           "summed over 0.05 <= k <= k_max (FLAMINGO errors)"}[key], fontsize=12)
        fig.tight_layout()
        save(fig, fn)

    for par in ("a1", "a2", "bTA"):
        fig, axs = grid()
        for i, ax in enumerate(axs.flat[:len(zs)]):
            for m in COL:
                if par not in PAR[m]:
                    continue
                q = PAR[m].index(par)
                v, e = d[f"{m}_x_raw"][i][:, q], d[f"{m}_e_raw"][i][:, q]
                ax.errorbar(kg*(0.98 if m == "NLA" else 1.02), v, e, fmt="o-", ms=3.5, capsize=2,
                            color=COL[m], label=m)
                kc = cuts[(i, m)][0]
                if np.isfinite(kc):
                    ax.axvline(kc, color=COL[m], ls="--", lw=1)
            kaxis(ax)
            ax.set_xlabel(xl)
            ax.set_ylabel(par)
            ax.set_title(f"z = {zs[i]:.2f}")
        axs.flat[0].legend(fontsize=8)
        fig.suptitle(f"{par} vs k_max, fits to measured FLAMINGO spectra (1 sigma); dashed: GOF cut",
                     fontsize=12)
        fig.tight_layout()
        save(fig, f"amp_{par}")

    with open("kmax_gof.csv", "w") as fh:
        fh.write("z,kmax_NLA,note_NLA,kmax_TATT,note_TATT\n")
        print("   z   | NLA k_max (GOF) | TATT k_max (GOF)")
        for i, z in enumerate(zs):
            n, t = cuts[(i, "NLA")], cuts[(i, "TATT")]
            fh.write(f"{z},{n[0]:.3f},{n[1]},{t[0]:.3f},{t[1]}\n")
            print(f"{z:5.2f}  |  {n[0]:6.3f} {n[1]:20s} | {t[0]:6.3f} {t[1]}")


if __name__ == "__main__":
    main()
