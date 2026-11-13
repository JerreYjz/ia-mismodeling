#!/usr/bin/env python3
"""Validate roman_limber.py against cocoa_roman_real products.

1. xi_+- (GG + NLA, A1 = 0.6, eta = -1.5 as EXAMPLE_EVALUATE1) vs the first
   1080 entries of data/example1.modelvector.
2. Gaussian xi_+- covariance with 10 angular bins vs the G column of
   data/cov_roman_real (a 10-bin 3x2pt matrix; xi_+ rows 0-359, xi_- 360-719).

    python validate.py --repo <cocoa_roman_real> --out validation
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roman_limber as RL  # noqa: E402

C1RHO = 0.01389


def nla_ratio(cosmo, A1, eta):
    def C1(Z):
        return A1*cosmo.Om*C1RHO/cosmo.D(Z)*((1 + Z)/1.62)**eta
    return {"dE": lambda K, Z: -C1(Z), "EE": lambda K, Z: C1(Z)**2,
            "BB": lambda K, Z: np.zeros_like(K)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="validation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nzf = os.path.join(a.repo, "data/example1.nz")
    cosmo = RL.Cosmology(h=0.6732, Om=0.3, Ob=0.04, ns=0.96605, As=2.1e-9, mnu=0.06)
    sh = RL.RomanShear(cosmo, nzf)
    rat = nla_ratio(cosmo, 0.6, -1.5)
    CE, CB = sh.cl(rat)
    xi = sh.xi(CE, CB)
    mv = np.loadtxt(os.path.join(a.repo, "data/example1.modelvector"))[:1080, 1]
    r = xi/mv
    nt, npair = 15, 36
    print("xi ratio mine/cocoa: median %.4f, 5-95%% %.4f %.4f" %
          (np.median(r), *np.percentile(r, [5, 95])))
    rp = r[:540].reshape(npair, nt)
    rm = r[540:].reshape(npair, nt)
    print("xi+ ratio per theta (median over pairs):", np.round(np.median(rp, 0), 3))
    print("xi- ratio per theta (median over pairs):", np.round(np.median(rm, 0), 3))
    tc = np.sqrt(sh.edges[1:]*sh.edges[:-1])/RL.ARCMIN

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, rr, lab in ((axs[0], rp, r"$\xi_+$"), (axs[1], rm, r"$\xi_-$")):
        for p, (i, j) in enumerate(sh.pairs):
            ax.plot(tc, rr[p], color=plt.cm.viridis(i/7), lw=0.8, alpha=0.8)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0.8, 1.2)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_ylabel(lab + " this code / example1.modelvector")
        ax.set_title(lab + ", 36 pairs, colour = bin i")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "val_xi_vs_modelvector.png"), dpi=110)
    plt.close(fig)

    # covariance, 10 bins
    sh.set_theta(10, 2.5, 250.0)
    CE0, CB0 = sh.cl(rat)
    cov = sh.gaussian_cov(CE0, CB0)
    raw = np.loadtxt(os.path.join(a.repo, "data/cov_roman_real"))
    n = int(raw[:, :2].max()) + 1
    G = np.zeros((n, n))
    ii, jj = raw[:, 0].astype(int), raw[:, 1].astype(int)
    G[ii, jj] = raw[:, 8]
    G[jj, ii] = raw[:, 8]
    Gx = G[:720, :720]
    d1, d2 = np.diag(cov), np.diag(Gx)
    rd = d1/d2
    print("cov diag ratio mine/cocoa (10 bins): median %.4f, 5-95%% %.4f %.4f" %
          (np.median(rd), *np.percentile(rd, [5, 95])))
    corr1 = cov/np.sqrt(np.outer(d1, d1))
    corr2 = Gx/np.sqrt(np.outer(d2, d2))
    print("max |corr diff| = %.3f, rms = %.4f" % (np.abs(corr1 - corr2).max(),
                                                  np.sqrt(np.mean((corr1 - corr2)**2))))
    # chi2 of the cocoa-cov noise against mine: tr and a test vector
    w1 = np.linalg.eigvalsh(np.linalg.solve(Gx, cov))
    print("eigenvalues of C_cocoa^-1 C_mine: min %.3f median %.3f max %.3f" %
          (w1.min(), np.median(w1), w1.max()))
    np.save(os.path.join(a.out, "eig_ratio.npy"), w1)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    axs[0].plot(rd, ".", ms=3)
    axs[0].axvline(360, color="k", lw=0.8)
    axs[0].set_xlabel("data-vector index (xi_+ | xi_-), 10 theta bins")
    axs[0].set_ylabel("diag: this code / cov_roman_real G")
    axs[0].set_ylim(0.5, 1.5)
    im = axs[1].imshow(corr1 - corr2, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    plt.colorbar(im, ax=axs[1])
    axs[1].set_title("correlation: this code minus cov_roman_real G")
    axs[2].hist(w1, bins=60)
    axs[2].set_xlabel(r"eigenvalues of $C_{\rm cocoa}^{-1} C_{\rm this}$")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "val_cov_vs_cocoa.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
