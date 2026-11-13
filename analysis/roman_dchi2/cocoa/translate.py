#!/usr/bin/env python3
"""FLAMINGO Delta chi2 -> Roman Delta chi2 with cosmolike, on the production
IR/UV masks k_UV(z) = K0/D(z).

The same residual dP = model - FLAMINGO enters both metrics, restricted in
each snapshot to 0.05 <= k <= K0/D(z_s), the k range the mask keeps:
  Delta chi2_FL    = sum over 13 snapshots, P_hE + P_EE + P_BB, FLAMINGO
                     error bars, white offset of P_EE/P_BB re-minimised;
  Delta chi2_Roman = cosmolike xi_+- (sim path: P_dE, P_EE) on the
                     K0/D(z) mask, Gaussian covariance.
Residuals: every NLA and TATT fit of ../v2/fits.npz (k_max 0.12 - 3.0), and
a pure amplitude change of FLAMINGO's IA, all snapshots and one snapshot at
a time (which redshifts Roman weights).
Also, per fit: the in-fit Delta chi2_FL (k <= k_max) against the full Roman
Delta chi2 on the mask (residual at all measured k, as a real analysis sees).

Writes analysis/translate.csv, analysis/translate_snap.csv,
analysis/fig_translate*.png/pdf.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
from cosmolike_driver import Cosmolike  # noqa: E402
from cosmolike_xi_ia import write_set, KW, KMIN  # noqa: E402

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def main():
    data = sys.argv[1]
    d = np.load("../v2/fits.npz")
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    zs, sns, kd, kg = d["zs"], d["sns"], d["k"][0], d["kgrid"]
    nsn = len(zs)
    T, b, Pf, err = d["T"], d["b"], d["PNL"], d["err"]
    M1, K1 = R["masks_n1"], R["K_n1"]
    cl = Cosmolike(os.path.join(HERE, "lib"), data)
    h = cl.h
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    Dz = np.array([np.sqrt(cl.PKL.P(z, 1e-3)/cl.PKL.P(0, 1e-3)) for z in zs]).ravel()
    xi_gg = R["xi_gg"]
    tab = "run/tables_tr"

    def q(v, m):
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    def roman(PdE, PEE, name):
        """xi of residual tables (data grid, zero outside the masks) via cosmolike."""
        A = np.array([np.interp(np.log(KW), np.log(kd), p, left=0, right=0) for p in PdE])
        B = np.array([np.interp(np.log(KW), np.log(kd), p, left=0, right=0) for p in PEE])
        path = write_set(tab, name, sns, A, B)
        cl.use_sim_tables(path, sns, zs, 1)
        return cl.xi_sim() - xi_gg

    def fl(res, s, sel):
        """FLAMINGO chi2 of an observed-unit residual [3, nk], white re-minimised."""
        w = np.concatenate([1/err[s, 1][sel]**2, 1/err[s, 2][sel]**2])
        r = np.concatenate([res[1][sel], res[2][sel]])
        c = np.sum(w*r)/np.sum(w)
        return float(np.sum((res[0][sel]/err[s, 0][sel])**2) + np.sum(w*(r - c)**2)), c

    rows, amp_rows = [], []
    for iK, K0 in enumerate(K1):
        Ks = K0/Dz
        sels = [(kd >= KMIN) & (kd <= Ks[s]*(1 + 1e-9)) for s in range(nsn)]
        m = M1[iK]
        # amplitude change of FLAMINGO IA (signal parts), all snapshots and one at a time
        ampP = np.array([[T[s, 0], 2*(T[s, 1] - d["Ns_true"][s]), 2*(T[s, 2] - d["Ns_true"][s])]
                         for s in range(nsn)])
        PdE = [ampP[s, 0]/b[s]/Pf[s]*PNLd[s]*sels[s] for s in range(nsn)]
        PEE = [ampP[s, 1]/Pf[s]*PNLd[s]*sels[s] for s in range(nsn)]
        afl = [float(sum(np.sum((ampP[s, x][sels[s]]/err[s, x][sels[s]])**2) for x in range(3)))
               for s in range(nsn)]
        ar = q(roman(PdE, PEE, "amp"), m)
        ars = []
        for s in range(nsn):
            one_dE = [PdE[t] if t == s else 0*PdE[t] for t in range(nsn)]
            one_EE = [PEE[t] if t == s else 0*PEE[t] for t in range(nsn)]
            ars.append(q(roman(one_dE, one_EE, "amp1"), m))
        amp_rows.append((K0, sum(afl), ar, afl, ars))
        print(f"K0 = {K0:g}/D: amplitude: FL {sum(afl):.3e}, Roman {ar:.3e}, threshold "
              f"Delta chi2_FL* = {sum(afl)/ar:.0f}", flush=True)
        # model residuals
        for mdl in ("NLA", "TATT"):
            for j in range(len(kg)):
                x = d[f"{mdl}_x_true"][:, j]
                Mobs = d[f"{mdl}_R"][:, j]*np.array([[b[s]*Pf[s], Pf[s], Pf[s]] for s in range(nsn)])
                Mobs[:, 1] += x[:, -1][:, None]
                Mobs[:, 2] += x[:, -1][:, None]
                res = Mobs - T
                flv, PdE, PEE = 0.0, [], []
                fl_in = 0.0
                for s in range(nsn):
                    v, c = fl(res[s], s, sels[s])
                    flv += v
                    PdE.append(res[s, 0]/b[s]/Pf[s]*PNLd[s]*sels[s])
                    PEE.append((res[s, 1] - c)/Pf[s]*PNLd[s]*sels[s])
                    fl_in += fl(res[s], s, (kd >= KMIN) & (kd <= kg[j]*(1 + 1e-9)))[0]
                rm = q(roman(PdE, PEE, "res"), m)
                full = q(R[f"xi_{mdl}_{j}"] - R["xi_flam"], m)
                rows.append((mdl, kg[j], K0, flv, rm, fl_in, full))
        print(f"  models done for {K0:g}/D", flush=True)

    with open("analysis/translate.csv", "w") as fh:
        fh.write("model,kmax,K0_over_D,dchi2_FL_matched,dchi2_roman_matched,dchi2_FL_infit,"
                 "dchi2_roman_full\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.3f},{r[2]:g}," + ",".join(f"{v:.4g}" for v in r[3:]) + "\n")
    with open("analysis/translate_snap.csv", "w") as fh:
        fh.write("K0_over_D,FL_all,Roman_all,threshold_all," +
                 ",".join(f"rate_z{z:g}" for z in zs) + "\n")
        for K0, afl, ar, afls, ars in amp_rows:
            fh.write(f"{K0:g},{afl:.4g},{ar:.4g},{afl/ar:.4g}," +
                     ",".join(f"{a/f:.4g}" if f > 0 else "nan" for a, f in zip(ars, afls)) + "\n")

    # ---------------- figures ----------------
    rows = np.array([(0 if r[0] == "NLA" else 1, *r[1:]) for r in rows], float)
    cm = plt.cm.viridis
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.8))
    ax = axs[0]
    for mi, mdl in enumerate(COL):
        P = rows[rows[:, 0] == mi]
        sc = ax.scatter(P[:, 3], np.maximum(P[:, 4], 1e-4), c=np.log10(P[:, 2]), cmap="viridis",
                        marker="o" if mdl == "NLA" else "s", s=24, vmin=np.log10(K1[0]),
                        vmax=np.log10(K1[-1]), edgecolor="k", linewidth=0.3)
    xx = np.geomspace(1, 1e6, 50)
    for iK, (K0, afl, ar, _, _) in enumerate(amp_rows):
        ax.plot(xx, xx*ar/afl, color=cm(iK/(len(K1) - 1)), lw=0.9)
    ax.axhline(1, color="k", lw=1)
    ax.axhline(0.2, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e6)
    ax.set_ylim(1e-4, 1e3)
    ax.set_xlabel(r"$\Delta\chi^2_{\rm FL}$ (13 snapshots, $0.05\leq k\leq K_0/D(z)$)")
    ax.set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ (cosmolike, same residual, mask $K_0/D(z)$)")
    ax.set_title("same residual, same k range: o NLA fits, s TATT fits; lines: amplitude change")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"log10 $K_0$")
    ax = axs[1]
    thr = [afl/ar for _, afl, ar, _, _ in amp_rows]
    ax.plot(K1, thr, "k-o", lw=2, label="threshold: amplitude change")
    for mi, mdl in enumerate(COL):
        P = rows[rows[:, 0] == mi]
        med = []
        for K0 in K1:
            Q = P[(P[:, 2] == K0) & (P[:, 4] > 1e-3)]
            med.append(np.median(Q[:, 3]/Q[:, 4]) if len(Q) else np.nan)
        ax.plot(K1, med, "--o", color=COL[mdl], label=f"threshold: median of {mdl} residuals")
    ax.set_yscale("log")
    ax.set_xlabel(r"mask $K_0$ of $k_{UV}=K_0/D(z)$ $[h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta\chi^2_{\rm FL}$ giving $\Delta\chi^2_{\rm Roman}=1$")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "analysis/fig_translate")

    # in-fit FL chi2 vs full Roman chi2 on the production mask 0.35/D
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.3))
    i35 = list(K1).index(0.35)
    for mi, mdl in enumerate(COL):
        P = rows[(rows[:, 0] == mi) & (rows[:, 2] == 0.35)]
        axs[0].plot(P[:, 5], P[:, 6], "o-", color=COL[mdl], label=f"{mdl}: full residual (as in an analysis)")
        axs[0].plot(P[:, 3], P[:, 4], "s:", color=COL[mdl], label=f"{mdl}: residual kept only where the mask looks")
        for kmax, xv, yv in zip(P[:, 1], P[:, 5], P[:, 6]):
            axs[0].annotate(f"{kmax:.2f}", (xv, yv), fontsize=7, color=COL[mdl])
    axs[0].plot(xx, xx*amp_rows[i35][2]/amp_rows[i35][1], "k-", lw=0.8, label="amplitude change")
    axs[0].axhline(1, color="k", lw=1)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlim(5, 1e5)
    axs[0].set_ylim(1e-3, 300)
    axs[0].set_xlabel(r"$\Delta\chi^2_{\rm FL}$ (full: the fit's own, $k\leq k_{\rm max}$; matched: $k\leq 0.35/D(z)$)")
    axs[0].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$, mask $0.35/D(z)$")
    axs[0].set_title("labels: fit k_max")
    axs[0].legend(fontsize=8)
    T2 = np.array([[a/f if f > 0 else np.nan for a, f in zip(ars, afls)] for _, _, _, afls, ars in amp_rows])
    for s, z in enumerate(zs):
        axs[1].plot(K1, T2[:, s], "o-", ms=3, color=plt.cm.plasma(s/nsn), label=f"z = {z:g}")
    axs[1].plot(K1, [ar/afl for _, afl, ar, _, _ in amp_rows], "k-", lw=2.5, label="all snapshots")
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-8, 1e-2)
    axs[1].set_xlabel(r"mask $K_0$ of $K_0/D(z)$")
    axs[1].set_ylabel(r"$\Delta\chi^2_{\rm Roman}/\Delta\chi^2_{\rm FL}$, amplitude change of one snapshot")
    axs[1].set_title("which redshifts Roman weights (per unit FLAMINGO chi2)")
    axs[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "analysis/fig_translate_infit_snap")


if __name__ == "__main__":
    main()
