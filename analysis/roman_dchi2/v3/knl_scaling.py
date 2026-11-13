#!/usr/bin/env python3
"""Redshift scaling of the non-linear scale and of the measured cuts.

Definitions
  D(z)      linear growth factor, D(0) = 1 (CAMB, FLAMINGO D3A cosmology)
  k_NL(z)   Delta^2_lin(k_NL, z) = k^3 P_lin(k, z) / (2 pi^2) = 1
  n_eff(k)  d ln P_lin / d ln k at z = 0
For a power-law P_lin ~ k^n,  Delta^2 ~ D^2 k^(3+n)  =>  k_NL(z) = k_NL(0) D(z)^(-2/(3+n)).
The K_0/D(z) family of the scale-cut recipe is the n = -1 case (exponent 1);
a constant comoving k_max is the n -> infinity case (exponent 0).

Fits: for each cut criterion (core snapshots) ln k_max(z) = ln k_0 - p ln D(z),
p = 2/(3+n), least squares in ln k with sigma_lnk = half a grid step (the cuts
sit on the k_max grid, ratio 1.35); chi2 profile in n; best n and its 1-sigma
range from Delta chi2 = 1. The same fit to the exact k_NL(z) gives the n that
the real P_lin shape implies.
    python knl_scaling.py <cocoa data dir>
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cocoa"))
from cosmolike_driver import Cosmolike  # noqa: E402

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
SIG_LNK = 0.5*np.log(1.35)


def main():
    cl = Cosmolike(os.path.join(HERE, "..", "cocoa", "lib"), sys.argv[1])
    h = cl.h
    v3 = np.load(os.path.join(HERE, "v3.npz"))
    zs, core = v3["zs"], v3["core"]
    kk = np.logspace(-3, 1.5, 2000)

    def Plin(z, k):
        return np.asarray(cl.PKL.P(z, k*h)).ravel()*h**3

    def D(z):
        return float(np.sqrt(cl.PKL.P(z, 1e-3)/cl.PKL.P(0.0, 1e-3)))

    def kNL(z):
        f = lambda lk: np.log(np.exp(lk)**3*Plin(z, np.exp(lk))[0]/(2*np.pi**2))
        try:
            return float(np.exp(brentq(f, np.log(0.02), np.log(200.0))))
        except ValueError:
            return np.nan
    zz = np.linspace(0, 3, 61)
    Dz = np.array([D(z) for z in zz])
    kz = np.array([kNL(z) for z in zz])
    P0 = Plin(0.0, kk)
    neff = np.gradient(np.log(P0), np.log(kk))
    Ds = np.array([D(z) for z in zs])
    kNLs = np.array([kNL(z) for z in zs])
    neff_at = np.where(np.isfinite(kNLs), np.interp(np.log(np.where(np.isfinite(kNLs), kNLs, 1.0)), np.log(kk), neff), np.nan)

    # ---------------- fits ln k = ln k0 - p ln D ----------------
    def fit_p(zsel, ksel, sig=SIG_LNK):
        x = -np.log(np.array([D(z) for z in zsel]))
        y = np.log(ksel)
        A = np.vstack([np.ones_like(x), x]).T
        w = 1.0/sig**2
        cov = np.linalg.inv(A.T @ A*w)
        beta = cov @ (A.T @ y*w)
        chi2 = float(np.sum(((y - A @ beta)/sig)**2))
        return beta, cov, chi2

    def profile(zsel, ksel, ns):
        x = -np.log(np.array([D(z) for z in zsel]))
        y = np.log(ksel)
        out = []
        for n in ns:
            p = 2.0/(3.0 + n)
            lnk0 = np.mean(y - p*x)
            out.append(float(np.sum(((y - lnk0 - p*x)/SIG_LNK)**2)))
        return np.array(out)
    ns = np.linspace(-2.5, 30, 2000)
    rows = []
    crits = [("NLA", "GoF"), ("NLA", "10%"), ("TATT", "GoF"), ("TATT", "anchored"), ("TATT", "10%")]
    fits = {}
    for mdl, c in crits:
        cut = v3[f"cut_{mdl}_{c}"]
        ok = core & np.isfinite(cut)
        (lnk0, p), cov, chi2 = fit_p(zs[ok], cut[ok])
        n = 2.0/p - 3.0 if p > 0 else np.inf
        prof = profile(zs[ok], cut[ok], ns)
        inside = ns[prof <= prof.min() + 1.0]
        fits[(mdl, c)] = dict(k0=float(np.exp(lnk0)), p=float(p), ep=float(np.sqrt(cov[1, 1])), n=float(n),
                              n_lo=float(inside.min()), n_hi=float(inside.max()) if inside.max() < ns[-1] else np.inf,
                              chi2=chi2, N=int(ok.sum()), prof=prof,
                              chi2_n_m1=float(profile(zs[ok], cut[ok], [-1.0])[0]),
                              chi2_flat=float(np.sum(((np.log(cut[ok]) - np.mean(np.log(cut[ok])))/SIG_LNK)**2)))
        rows.append((mdl, c, fits[(mdl, c)]))
    # stitched truth cuts, for reference
    st = {}
    p_st = os.path.join(HERE, "..", "v3s", "S2", "v3.npz")
    if os.path.exists(p_st):
        vs = np.load(p_st)
        for mdl in ("NLA", "TATT"):
            cut = vs[f"cut_{mdl}_GoF"]
            ok = core & np.isfinite(cut)
            (lnk0, p), cov, chi2 = fit_p(zs[ok], cut[ok])
            st[mdl] = dict(k0=float(np.exp(lnk0)), p=float(p), ep=float(np.sqrt(cov[1, 1])), n=float(2/p - 3) if p > 0 else np.inf, chi2=chi2)
    # exact k_NL(z) over the core
    (lnk0, p), cov, chi2 = fit_p(zs[core], kNLs[core], sig=0.01)
    knl_fit = dict(k0=float(np.exp(lnk0)), p=float(p), n=float(2/p - 3), chi2=chi2)

    # ---------------- outputs ----------------
    with open(os.path.join(HERE, "knl.csv"), "w") as fh:
        fh.write("z,core,D,k_NL,n_eff_at_kNL,NLA_GoF,TATT_GoF,NLA_GoF_over_kNL,TATT_GoF_over_kNL\n")
        for s in range(len(zs)):
            fh.write(f"{zs[s]:g},{int(core[s])},{Ds[s]:.4f},{kNLs[s]:.4f},{neff_at[s]:.3f},{v3['cut_NLA_GoF'][s]:.3f},{v3['cut_TATT_GoF'][s]:.3f},"
                     f"{v3['cut_NLA_GoF'][s]/kNLs[s]:.3f},{v3['cut_TATT_GoF'][s]/kNLs[s]:.3f}\n")
    summary = dict(knl_fit=knl_fit, stitched={m: {k: v for k, v in d.items()} for m, d in st.items()},
                   fits={f"{m}_{c}": {k: v for k, v in d.items() if k != "prof"} for (m, c), d in fits.items()},
                   neff_z0={f"k={k}": float(np.interp(np.log(k), np.log(kk), neff)) for k in (0.1, 0.2, 0.3, 0.5, 1.0)})
    json.dump(summary, open(os.path.join(HERE, "knl.json"), "w"), indent=1)
    print(open(os.path.join(HERE, "knl.csv")).read())
    print(json.dumps(summary, indent=1))

    def save(fig, name):
        fig.savefig(os.path.join(HERE, name + ".png"), dpi=220)
        fig.savefig(os.path.join(HERE, name + ".pdf"))
        plt.close(fig)

    def kaxis(a, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2)):
        a.set_xscale("log")
        a.set_xticks(ticks)
        a.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        a.xaxis.set_minor_formatter(ticker.NullFormatter())
        a.grid(alpha=0.3, which="both")

    # Fig 1: k_NL(z) exact vs power-law scalings; n_eff(k)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    a = ax[0]
    a.plot(kz, zz, "k-", lw=2, label=r"exact: $\Delta^2_{\rm lin}(k_{\rm NL}, z) = 1$")
    for n, ls in ((-1, "--"), (-1.5, "-."), (-2, ":"), (0, (0, (1, 1)))):
        pexp = 2.0/(3.0 + n)
        a.plot(kz[0]*Dz**(-pexp), zz, ls=ls, color="0.4", lw=1.3, label=f"$k_{{\\rm NL}}(0)\\,D(z)^{{-2/(3+n)}}$, n = {n:g} (exponent {pexp:.2f})")
    a.plot(knl_fit["k0"]*Dz**(-knl_fit["p"]), zz, "-", color="#2a9d8f", lw=1.2,
           label=f"fit to the exact $k_{{\\rm NL}}$ over the core: exponent {knl_fit['p']:.2f} (n = {knl_fit['n']:.2f})")
    for mdl, mk in (("NLA", "o"), ("TATT", "s")):
        c = v3[f"cut_{mdl}_GoF"]
        ok = core & np.isfinite(c)
        a.plot(c[ok], zs[ok], mk, ms=7, color=COL[mdl], label=f"{mdl} GoF cut (measured)")
    a.axhspan(0.2, 2.0, color="0.92", zorder=0)
    kaxis(a)
    a.set_xlim(0.05, 3)
    a.set_ylim(0, 3)
    a.set_xlabel("k [h/Mpc]")
    a.set_ylabel("z")
    a.set_title(r"non-linear scale $k_{\rm NL}(z)$ and its power-law approximations")
    a.legend(fontsize=7, loc="lower right")
    a = ax[1]
    a.plot(kk, neff, "k-", lw=1.5)
    for s in np.where(core)[0]:
        a.plot(kNLs[s], neff_at[s], "o", color=plt.cm.viridis(s/12), ms=6, label=f"$k_{{\\rm NL}}$(z={zs[s]:g})")
    for n in (-1, -1.5, -2):
        a.axhline(n, color="0.6", lw=0.8, ls="--")
    kaxis(a, ticks=(0.01, 0.03, 0.1, 0.3, 1, 3, 10))
    a.set_xlim(0.01, 10)
    a.set_ylim(-3, 1)
    a.set_xlabel("k [h/Mpc]")
    a.set_ylabel(r"$n_{\rm eff}(k) = d\ln P_{\rm lin}/d\ln k$ at z = 0")
    a.set_title(r"effective spectral index; dots: at $k_{\rm NL}(z)$ of the core snapshots")
    a.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "fig_knl_z")

    # Fig 2: measured cuts with the fitted D^-p curves and the n = -1 family; chi2 profiles
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for i, mdl in enumerate(("NLA", "TATT")):
        a = ax[i]
        for c, mk, al in (("GoF", "o", 1.0), ("10%", "v", 0.6), ("anchored", "s", 0.6)):
            if (mdl, c) not in fits:
                continue
            cut = v3[f"cut_{mdl}_{c}"]
            ok = core & np.isfinite(cut)
            f = fits[(mdl, c)]
            a.errorbar(cut[ok], zs[ok], xerr=cut[ok]*SIG_LNK, fmt=mk, ms=7, color=COL[mdl], alpha=al, capsize=2,
                       label=f"{c}: fit exponent {f['p']:.2f} $\\pm$ {f['ep']:.2f}, n = {f['n']:.1f} [{f['n_lo']:.1f}, {f['n_hi']:.1f}], $\\chi^2$ {f['chi2']:.1f}/{f['N'] - 2}")
            a.plot(f["k0"]*Dz**(-f["p"]), zz, "-", color=COL[mdl], alpha=al, lw=1.2)
            if c == "GoF":
                k0m1 = np.exp(np.mean(np.log(cut[ok]) + np.log(np.array([D(z) for z in zs[ok]]))))
                a.plot(k0m1/Dz, zz, "--", color="0.3", lw=1.2, label=f"n = $-$1 ($K_0/D$, $K_0$ = {k0m1:.2f}): $\\chi^2$ {f['chi2_n_m1']:.1f}")
                a.plot(np.full_like(zz, np.exp(np.mean(np.log(cut[ok])))), zz, ":", color="0.3", lw=1.2, label=f"constant k: $\\chi^2$ {f['chi2_flat']:.1f}")
        a.axhspan(0.2, 2.0, color="0.92", zorder=0)
        kaxis(a, ticks=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1))
        a.set_xlim(0.05, 1.2)
        a.set_ylim(0, 2.5)
        a.set_xlabel(r"$k_{\max}$ [h/Mpc]")
        a.set_ylabel("z")
        a.set_title(f"{mdl}: measured cuts and $k_0\\,D(z)^{{-2/(3+n)}}$ fits")
        a.legend(fontsize=6.5, loc="upper left")
    a = ax[2]
    for (mdl, c), f in fits.items():
        ls = {"GoF": "-", "10%": ":", "anchored": "--"}[c]
        a.plot(ns, f["prof"] - f["prof"].min(), ls, color=COL[mdl], label=f"{mdl} {c}")
    pk = profile(zs[core], kNLs[core], ns)
    a.plot(ns, pk - pk.min(), "-", color="#2a9d8f", label=r"exact $k_{\rm NL}(z)$ (same $\sigma_{\ln k}$)")
    a.axhline(1, color="k", lw=0.8, ls="--")
    a.axvline(-1, color="0.3", lw=0.8, ls="--")
    a.text(-0.9, 8, "n = $-$1: $K_0/D(z)$", fontsize=8, rotation=90, va="bottom")
    a.set_xlim(-2.5, 12)
    a.set_ylim(0, 10)
    a.set_xlabel("n  (exponent 2/(3+n) of $1/D$)")
    a.set_ylabel(r"$\Delta\chi^2$ of the fit $k_{\max}(z) = k_0 D(z)^{-2/(3+n)}$")
    a.set_title("profile in n (core snapshots)")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
    sec = a.secondary_xaxis("top", functions=(lambda n: 2/(3 + n), lambda p: 2/np.maximum(p, 1e-6) - 3))
    sec.set_xticks([2.0, 1.0, 0.67, 0.5, 0.4, 0.33, 0.25, 0.2, 0.13])
    sec.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    sec.set_xlabel("exponent p = 2/(3+n)")
    fig.tight_layout()
    save(fig, "fig_knl_fit")


if __name__ == "__main__":
    main()
