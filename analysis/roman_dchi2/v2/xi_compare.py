#!/usr/bin/env python3
"""IA part of the Roman cosmic-shear correlation functions from FLAMINGO vs
NLA / TATT fits at different k_max.

xi_+-^IA(theta) = GI + II (E and B) for the 36 Roman source-bin pairs
(example1.nz, 15 theta bins 2.5-250 arcmin), from
  * FLAMINGO: the smoothed spectra of fits.npz (P_dE = P_hE / b, P_EE and P_BB
    minus the white shape noise), and
  * NLA, TATT: the fits to those spectra over 0.05 <= k <= k_max (fits.npz).
Same k support for all: 0.05 <= k <= 17.7 h/Mpc as measured (or as fitted
models evaluated there), constant P_X / P_NL below 0.05 (linear regime),
nothing above 17.7 (no data). No GG term (lensing of the signal cancels in
model - FLAMINGO); the Gaussian covariance (GG + FLAMINGO IA, 2415 deg^2)
only sets sigma for the comparison.

Figures (220-dpi PNG + PDF):
  xi_ia_<model>_<xi+|xi->.png : selected pairs; rows: theta*xi_IA, fractional
      difference to FLAMINGO, difference in Roman sigma; colour = fit k_max
  xi_ia_allpairs_<model>.png : (model - FLAMINGO)/sigma over the full data
      vector at each k_max
  dchi2_theta_cut.png : Delta chi2 of xi_IA(model) - xi_IA(FLAMINGO) vs a
      common theta_min, for each fit k_max (NLA, TATT)
  dchi2_vs_kmax_cuts.png : the same vs fit k_max at fixed theta_min
Table: xi_dchi2.csv.

    python xi_compare.py fits.npz example1.nz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import roman_limber as RL  # noqa: E402

FLAMINGO = dict(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, sigma8=0.807, mnu=0.06)
X = ("dE", "EE", "BB")
KMIN = 0.05
PAIRS = [(0, 0), (0, 3), (0, 7), (2, 2), (2, 6), (4, 4), (5, 7), (7, 7)]
SHOW = [0.161, 0.291, 0.523, 0.941, 1.692]
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def tables(cosmo, zs, k, R):
    """R[nsn, 3, nk] on the data grid -> callables on an extended grid:
    constant ratio below KMIN, zero above the last measured k."""
    lo = np.geomspace(1e-4, KMIN, 30, endpoint=False)
    hi = np.geomspace(k[-1]*1.02, 2e3, 20)
    m = k >= KMIN
    kk = np.concatenate([lo, k[m], hi])
    T = np.zeros((len(zs), 3, len(kk)))
    for s in range(len(zs)):
        for x in range(3):
            r = R[s, x, m]
            T[s, x] = np.concatenate([np.full(len(lo), r[0]), r, np.zeros(len(hi))])
    return {x: RL.ratio_table(zs, kk, T[:, i, :]) for i, x in enumerate(X)}


def main():
    fits, nz = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "xi"
    os.makedirs(out, exist_ok=True)
    d = np.load(fits)
    zs, kg, k = d["zs"], d["kgrid"], d["k"][0]
    nK = len(kg)
    cosmo = RL.Cosmology(**FLAMINGO, halofit="takahashi")
    S = RL.RomanShear(cosmo, nz)
    nt, npair = S.ntheta, len(S.pairs)
    off = npair*nt
    tc = np.sqrt(S.edges[1:]*S.edges[:-1])/RL.ARCMIN

    CEt, CBt = S.cl(tables(cosmo, zs, k, d["Rtrue"]), gg=False)
    xt = S.xi(CEt, CBt)
    CEg, CBg = S.cl(None)
    cov = S.gaussian_cov(CEg + CEt, CBg + CBt)
    sig = np.sqrt(np.diag(cov))
    xm = {}
    for mdl in COL:
        xm[mdl] = []
        for j in range(nK):
            CE, CB = S.cl(tables(cosmo, zs, k, d[f"{mdl}_R"][:, j]), gg=False)
            xm[mdl].append(S.xi(CE, CB))
        xm[mdl] = np.array(xm[mdl])
        print(f"{mdl}: done", flush=True)
    np.savez(os.path.join(out, "xi_ia.npz"), xi_flamingo=xt, theta=tc, kgrid=kg, cov=cov,
             **{f"xi_{m}": v for m, v in xm.items()})

    js = [int(np.argmin(np.abs(kg - s))) for s in SHOW]
    cm = plt.cm.viridis

    # 1. selected pairs
    for mdl in COL:
        for b, lab in ((0, "xi+"), (1, "xi-")):
            sym = r"\xi_+" if b == 0 else r"\xi_-"
            fig, axs = plt.subplots(3, len(PAIRS), figsize=(4.0*len(PAIRS), 10.5), sharex=True)
            for c, pr in enumerate(PAIRS):
                p = S.pairs.index(pr)
                sl = slice(b*off + p*nt, b*off + (p + 1)*nt)
                t, sg = xt[sl], sig[sl]
                ax = axs[0, c]
                ax.fill_between(tc, (t - sg)*tc, (t + sg)*tc, color="0.85", label=r"Roman $\pm1\sigma$")
                ax.plot(tc, t*tc, "k-", lw=2.4, label="FLAMINGO")
                for q, j in enumerate(js):
                    ax.plot(tc, xm[mdl][j][sl]*tc, "-", color=cm(q/(len(js) - 1)), lw=1.3,
                            label=f"{mdl}, k_max = {kg[j]:.2f}")
                ax.set_title(f"bins {pr[0] + 1}-{pr[1] + 1}  ${sym}^{{\\rm IA}}$")
                ax.set_ylabel(rf"$\theta\,{sym}^{{\rm IA}}$ [arcmin]" if c == 0 else "")
                ax = axs[1, c]
                for q, j in enumerate(js):
                    ax.plot(tc, (xm[mdl][j][sl] - t)/np.abs(t), "-", color=cm(q/(len(js) - 1)), lw=1.3)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_ylim(-1, 1)
                ax.set_ylabel(r"(model $-$ FL) / |FL|" if c == 0 else "")
                ax = axs[2, c]
                for q, j in enumerate(js):
                    ax.plot(tc, (xm[mdl][j][sl] - t)/sg, "-", color=cm(q/(len(js) - 1)), lw=1.3)
                ax.axhline(0, color="k", lw=0.8)
                for v in (-1, 1):
                    ax.axhline(v, color="k", lw=0.6, ls=":")
                ax.set_ylabel(r"(model $-$ FL) / $\sigma_{\rm Roman}$" if c == 0 else "")
                ax.set_xscale("log")
                ax.set_xlabel(r"$\theta$ [arcmin]")
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            axs[0, 0].legend(fontsize=7.5)
            fig.suptitle(f"IA part of Roman ${sym}$: FLAMINGO (black) vs {mdl} fitted to FLAMINGO "
                         f"P(k) up to k_max (colour). Spectra used to k = 17.7 h/Mpc.", fontsize=13)
            fig.tight_layout()
            save(fig, os.path.join(out, f"xi_ia_{mdl}_{lab}"))

    # 2. all pairs, residual / sigma
    for mdl in COL:
        fig, axs = plt.subplots(len(js), 1, figsize=(18, 2.7*len(js)), sharex=True)
        for ax, j in zip(axs, js):
            r = (xm[mdl][j] - xt)/sig
            ax.plot(r, color=COL[mdl], lw=0.9)
            ax.axvline(off, color="k", lw=1)
            for p in range(1, 2*npair):
                ax.axvline(p*nt, color="0.88", lw=0.4, zorder=0)
            ax.axhline(0, color="k", lw=0.6)
            for v in (-1, 1):
                ax.axhline(v, color="k", lw=0.6, ls=":")
            ax.set_ylabel(r"$\Delta\xi^{\rm IA}/\sigma$")
            ax.set_title(f"{mdl}, fit k_max = {kg[j]:.2f}:  chi2 of the difference, all 1080 points = "
                         f"{float((xm[mdl][j] - xt) @ np.linalg.solve(cov, xm[mdl][j] - xt)):.1f}",
                         fontsize=10)
        axs[-1].set_xlabel(r"data-vector index: $\xi_+$ (36 pairs x 15 $\theta$, small to large $\theta$ "
                           r"within a pair) | $\xi_-$")
        fig.tight_layout()
        save(fig, os.path.join(out, f"xi_ia_allpairs_{mdl}"))

    # 3. Delta chi2 vs common theta_min
    tmins = S.edges[:-1]/RL.ARCMIN
    th = np.tile(np.arange(nt), 2*npair)
    res = {}
    for mdl in COL:
        res[mdl] = np.zeros((nK, nt))
        for j in range(nK):
            dv = xm[mdl][j] - xt
            for i in range(nt):
                m = th >= i
                res[mdl][j, i] = float(dv[m] @ np.linalg.solve(cov[np.ix_(m, m)], dv[m]))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    for ax, mdl in zip(axs, COL):
        for j in range(nK):
            ax.plot(tmins, np.maximum(res[mdl][j], 1e-4), "o-", ms=3.5, color=cm(j/(nK - 1)),
                    label=f"k_max = {kg[j]:.2f}")
        ax.axhline(1, color="k", lw=1)
        ax.axhline(0.2, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.set_xlabel(r"$\theta_{\rm min}$ [arcmin] (same for all pairs, $\xi_+$ and $\xi_-$)")
        ax.set_title(f"{mdl}: $\\Delta\\chi^2$ of $\\xi^{{\\rm IA}}$(model) $-$ $\\xi^{{\\rm IA}}$(FLAMINGO)")
    axs[0].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$")
    axs[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, os.path.join(out, "dchi2_theta_cut"))

    fig, ax = plt.subplots(figsize=(9, 6.2))
    for i, ls in zip((0, 4, 7, 10), ("-", "--", "-.", ":")):
        for mdl in COL:
            ax.plot(kg, np.maximum(res[mdl][:, i], 1e-4), ls, marker="o", ms=3.5, color=COL[mdl],
                    label=f"{mdl}, theta_min = {tmins[i]:.1f}'")
    ax.axhline(1, color="k", lw=1)
    ax.axhline(0.2, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.12, 0.2, 0.3, 0.5, 1, 2, 3]))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"fit $k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ of $\xi^{\rm IA}$ difference")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, os.path.join(out, "dchi2_vs_kmax_cuts"))

    with open(os.path.join(out, "xi_dchi2.csv"), "w") as fh:
        fh.write("model,kmax," + ",".join(f"thmin_{t:.2f}" for t in tmins) + "\n")
        for mdl in COL:
            for j in range(nK):
                fh.write(f"{mdl},{kg[j]:.3f}," + ",".join(f"{v:.4g}" for v in res[mdl][j]) + "\n")
    print(open(os.path.join(out, "xi_dchi2.csv")).read())


if __name__ == "__main__":
    main()
