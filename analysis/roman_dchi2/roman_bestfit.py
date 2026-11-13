#!/usr/bin/env python3
"""Roman-level best fit of NLA / TATT to the FLAMINGO xi_IA, per scale cut.

roman_dchi2.py takes the IA amplitudes from the P(k) fits (per snapshot,
k <= k_max, FLAMINGO error bars) and asks how far the resulting xi_+- sits
from FLAMINGO. Here the amplitudes are instead chosen to minimise the Roman
Delta chi2 on the masked xi_+- itself, which separates
  * an irreducible model failure (no amplitudes fit FLAMINGO's xi), from
  * noisy / degenerate amplitudes at the P(k) level (TATT a2, b_TA).
Cosmology is held fixed, so this is still an upper bound on what a
cosmology-marginalised analysis would leave.

xi_IA is a polynomial in the per-snapshot C1 = a1 Om c1rho / D,
C2 = a2 Om c1rho / D^2 and b_TA (the R_X tables are linear in z and k), so
it is expanded on 12 basis vectors per snapshot (Limber once each):
    GI: C1 [P], C1 bTA [K_dE], C2 [K_mix]
    EE: C1^2 [P], C1^2 bTA^2 [K_ta], C1^2 bTA [K_dE], C1 C2 [K_mix],
        C1 bTA C2 [K_mixEE], C2^2 [K_tt]
    BB: C1^2 bTA^2 [K_taBB], C1 bTA C2 [K_mixBB], C2^2 [K_ttB]
Parameterisations:
    NLA-free  : a1 per snapshot (13)
    TATT-free : a1, a2, b_TA per snapshot (39)
    NLA-pl    : a1 = A1 x^eta1, x = (1+z)/1.62  (cosmolike, 2)
    TATT-pl   : + a2 = A2 x^eta2, b_TA constant (5)
Masks: the IR/UV masks of roman_dchi2.py (k_UV constant comoving on the
k_max grid, k0/D(z) family, 0.35/D(z)).

    python roman_bestfit.py --spectra spectra.npz --res results_highk --nz example1.nz
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tatt_fit"))
import roman_limber as RL  # noqa: E402
import roman_dchi2 as RD  # noqa: E402
import fit_tatt_flamingo as F  # noqa: E402

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
TERMS = [("dE", "P", 1.0), ("dE", "dE", 1.0), ("dE", "mix", -5.0),
         ("EE", "P", 1.0), ("EE", "ta", 1.0), ("EE", "dE", 2.0), ("EE", "mix", -10.0),
         ("EE", "mixEE", -10.0), ("EE", "tt", 25.0),
         ("BB", "taBB", 1.0), ("BB", "mixBB", -10.0), ("BB", "ttB", 25.0)]


def coeffs(C1, C2, b):
    """Multipliers of the 12 terms (sign of P_dE = -(...) included)."""
    return np.array([-C1, -C1*b, -C2,
                     C1**2, C1**2*b**2, C1**2*b, C1*C2, C1*b*C2, C2**2,
                     C1**2*b**2, C1*b*C2, C2**2])


def build_basis(sh, model, zs, k, cache):
    if os.path.exists(cache):
        return np.load(cache)["B"]
    nsn = len(zs)
    B = np.zeros((nsn, len(TERMS), 2*len(sh.pairs)*sh.ntheta))
    zero = np.zeros((nsn, len(k)))
    for s, z in enumerate(zs):
        K = model.kernels(z, k)
        P = model.Pnl(z, k)
        for t, (X, name, f) in enumerate(TERMS):
            tab = {x: zero.copy() for x in ("dE", "EE", "BB")}
            tab[X][s] = f*(np.ones_like(k) if name == "P" else K[name]/P)
            rat = {x: RL.ratio_table(zs, k, tab[x]) for x in tab}
            CE, CB = sh.cl(rat, gg=False)
            B[s, t] = sh.xi(CE, CB)
        print(f"basis z={z:.2f}", flush=True)
    np.savez(cache, B=B)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", default="spectra.npz")
    ap.add_argument("--res", default="results_highk")
    ap.add_argument("--nz", required=True)
    a = ap.parse_args()
    S = np.load(a.spectra)
    R = np.load(os.path.join(a.res, "roman_dchi2.npz"))
    Fm = np.load(os.path.join(a.res, "family_noise.npz"))
    zs, k, kmax = S["zs"], S["k"], S["kmax"]
    cosmo = RL.Cosmology(**RD.FLAMINGO, halofit="takahashi")
    sh = RL.RomanShear(cosmo, a.nz)
    model = F.TATTModel(halofit="takahashi")
    B = build_basis(sh, model, zs, k, os.path.join(a.res, "basis.npz"))
    Dz = np.array([model.D(z) for z in zs])
    c0 = F.COSMO["Om"]*F.C1RHOCRIT
    xi_t, cov = R["xi_t"], R["cov"]
    x = (1 + zs)/1.62

    def xi_of(a1, a2, b):
        C = coeffs(c0*a1/Dz, c0*a2/Dz**2, b)          # [12, nsn]
        return np.einsum("ts,stn->n", C, B)

    # check against roman_dchi2's model xi at the P(k)-fit amplitudes
    for mdl in ("NLA", "TATT"):
        j = 4
        xm = xi_of(S[f"a1_{mdl}"][:, j], S[f"a2_{mdl}"][:, j], S[f"bTA_{mdl}"][:, j])
        ref = R[f"{mdl}_xi"][j]
        print(f"basis check {mdl}: max |dxi|/sigma = "
              f"{np.max(np.abs(xm - ref)/np.sqrt(np.diag(cov))):.2e}")

    masks = [(f"kUV={kk:.3f}", float(kk), R["masks"][i]) for i, kk in enumerate(kmax)]
    fam = [(f"k0/D={k0:.3f}", float(k0), m) for k0, m in zip(Fm["k0"], Fm["fam_masks"])
           if m.sum() > 0]
    out = {}
    for label, kk, m in masks + fam + [("0.35/D", 0.35, R["mask_035"])]:
        L = np.linalg.cholesky(cov[np.ix_(m, m)])
        W = np.linalg.inv(L)                      # whitening, computed once
        yt = xi_t[m]
        Bm = B[:, :, m]

        def xi_m(a1, a2, b):
            C = coeffs(c0*a1/Dz, c0*a2/Dz**2, b)
            return np.einsum("ts,stn->n", C, Bm)

        def resid_fn(unpack):
            return lambda p: W @ (xi_m(*unpack(p)) - yt)

        j = int(np.argmin(np.abs(kmax - min(kk, kmax[-1]))))
        nsn = len(zs)
        fits = {}
        specs = {
            "NLA-free": (lambda p: (p, np.zeros(nsn), np.zeros(nsn)),
                         [S["a1_NLA"][:, j]], 13),
            "TATT-free": (lambda p: (p[:nsn], p[nsn:2*nsn], p[2*nsn:]),
                          [np.concatenate([S["a1_TATT"][:, j], S["a2_TATT"][:, j],
                                           S["bTA_TATT"][:, j]]),
                           np.concatenate([S["a1_NLA"][:, j], np.zeros(2*nsn)])], 39),
            "NLA-pl": (lambda p: (p[0]*x**p[1], np.zeros(nsn), np.zeros(nsn)),
                       [[3.5, 1.0], [1.0, 0.0]], 2),
            "TATT-pl": (lambda p: (p[0]*x**p[1], p[2]*x**p[3], np.full(nsn, p[4])),
                        [[3.5, 1.0, 0.0, 0.0, 0.0], [3.5, 1.0, 1.0, 1.0, -0.3],
                         [3.5, 1.0, -1.0, 1.0, 0.5]], 5),
        }
        for name, (unpack, starts, npar) in specs.items():
            f = resid_fn(unpack)
            best = None
            for x0 in starts:
                r = least_squares(f, np.asarray(x0, float), x_scale="jac", max_nfev=4000)
                if best is None or r.cost < best.cost:
                    best = r
            fits[name] = dict(chi2=2*best.cost, p=best.x, n=int(m.sum()), npar=npar)
        out[label] = (kk, fits)
        print(f"{label:12s} ({int(m.sum()):3d} pts): " + "  ".join(
            f"{n} {v['chi2']:7.3f}" for n, v in fits.items()), flush=True)

    # table
    names = ["NLA-free", "TATT-free", "NLA-pl", "TATT-pl"]
    with open(os.path.join(a.res, "bestfit_dchi2.csv"), "w") as fh:
        fh.write("mask,k,npts," + ",".join(names) + ",pk_NLA,pk_TATT\n")
        for label, (kk, fits) in out.items():
            pk = ["", ""]
            if label.startswith("kUV"):
                i = int(np.argmin(np.abs(kmax - kk)))
                pk = [f"{R['NLA_diag'][i]:.4f}", f"{R['TATT_diag'][i]:.4f}"]
            elif label.startswith("k0/D"):
                i = int(np.argmin(np.abs(Fm["k0"] - kk)))
                pk = [f"{Fm['NLA'][i]:.4f}", f"{Fm['TATT'][i]:.4f}"]
            fh.write(f"{label},{kk:.4f},{fits['NLA-free']['n']}," +
                     ",".join(f"{fits[n]['chi2']:.4f}" for n in names) + "," + ",".join(pk) + "\n")
    np.save(os.path.join(a.res, "bestfit_params.npy"),
            {lab: {n: v["p"] for n, v in fits.items()} for lab, (kk, fits) in out.items()},
            allow_pickle=True)

    # plot: Delta chi2 vs cut, P(k)-fitted amplitudes vs Roman best fit
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.4))
    for ax, pre, kx, pkN, pkT, nf in (
            (axs[0], "kUV", kmax, R["NLA_diag"], R["TATT_diag"], Fm["nf_diag"]),
            (axs[1], "k0/D", Fm["k0"], Fm["NLA"], Fm["TATT"], Fm["nf_fam"])):
        labs = [lab for lab in out if lab.startswith(pre)]
        kk = np.array([out[l][0] for l in labs])
        for mdl in ("NLA", "TATT"):
            pk = pkN if mdl == "NLA" else pkT
            ok = pk > 0
            ax.plot(kx[ok], pk[ok], "o-", color=COL[mdl], lw=1.2,
                    label=f"{mdl}: P(k)-fit amplitudes (k<=k_max)")
            ax.plot(kk, [out[l][1][f"{mdl}-free"]["chi2"] for l in labs], "s--",
                    color=COL[mdl], label=f"{mdl}: Roman best fit, free per snapshot")
            ax.plot(kk, [out[l][1][f"{mdl}-pl"]["chi2"] for l in labs], "^:",
                    color=COL[mdl], label=f"{mdl}: Roman best fit, cosmolike power law")
        ok = nf.mean(0) > 0
        ax.fill_between(kx[ok], *np.percentile(nf[:, ok], [16, 84], axis=0), color="0.8",
                        label="FLAMINGO noise floor (16-84%)")
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(0.2, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 300)
        ax.set_ylabel(r"Roman $\Delta\chi^2$ (IR/UV-masked $\xi_\pm$)")
    axs[0].set_xlabel(r"$k_{\rm max}=k_{UV}$ (constant comoving) $[h\,{\rm Mpc}^{-1}]$")
    axs[1].set_xlabel(r"$k_0$ of $k_{UV}=k_0/D(z)$ $[h\,{\rm Mpc}^{-1}]$")
    axs[1].axvline(0.35, color="k", ls=":", lw=0.8)
    axs[0].legend(fontsize=7)
    fig.suptitle("Roman Delta chi2: IA amplitudes from P(k) fits vs best achievable at fixed "
                 "cosmology", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.res, "roman_bestfit_vs_pkfit.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
