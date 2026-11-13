#!/usr/bin/env python3
"""The exact FLAMINGO -> Roman Delta chi2 translation, as a matrix.

A residual dP of the IA spectra (P_dE, P_EE; the sim path carries no B
modes) at FLAMINGO's k nodes and snapshots changes Roman xi_+- linearly:
dxi = J dP. cosmolike gives J column by column: one sim_IA table set per
cell (snapshot s, spectrum X, k node i), a unit 'hat' in ln k at that node
(the interpolation sim_IA itself uses), zero elsewhere. Then for any residual
    Delta chi2_Roman = dP^T F_R dP,   F_R = J^T C^-1 J   (C on the mask)
    Delta chi2_FL    = dP^T F_FL dP,  F_FL = diag(1/sigma^2)
(sigma: FLAMINGO error bars, sigma_dE = sigma_hE / b). In whitened units
u = dP/sigma, Delta chi2_Roman = u^T A u with A = S F_R S, S = diag(sigma):
  * eigenvalues of A = Roman Delta chi2 per unit FLAMINGO Delta chi2 along
    each residual direction; lambda_max is the worst case, so
    Delta chi2_FL < 1/lambda_max guarantees Delta chi2_Roman < 1;
  * diag(A) maps where Roman looks: Delta chi2_Roman per unit FLAMINGO chi2
    put into one cell.
Cells: k nodes 0.05 <= k <= 17.7 h/Mpc ("full"; a mask still sees residual
above its k_UV) and the subset k <= K0/D(z_s) ("matched").
Validation: dP^T F_R dP vs cosmolike for every NLA/TATT fit residual.

    python translate_matrix.py <cocoa_roman_real>/data
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
from cosmolike_driver import Cosmolike  # noqa: E402
from cosmolike_xi_ia import write_set, KW, KMIN  # noqa: E402

MASKS = (0.25, 0.35, 0.5)
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
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    cl = Cosmolike(os.path.join(HERE, "lib"), data)
    h = cl.h
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    Dz = np.array([np.sqrt(cl.PKL.P(z, 1e-3)/cl.PKL.P(0, 1e-3)) for z in zs]).ravel()
    xi_gg = R["xi_gg"]
    nodes = np.where(kd >= KMIN)[0]
    # residual units: cosmolike-P_NL-normalised P (what the tables carry);
    # FLAMINGO sigma in the same units
    conv = np.array([[PNLd[s]/(b[s]*Pf[s]), PNLd[s]/Pf[s]] for s in range(nsn)])  # obs -> table
    sig = np.array([[err[s, 0]*conv[s, 0], err[s, 1]*conv[s, 1]] for s in range(nsn)])

    cells = [(s, x, i) for s in range(nsn) for x in range(2) for i in nodes]
    print(f"{len(cells)} cells", flush=True)
    J = np.zeros((1080, len(cells)))
    for c, (s, x, i) in enumerate(cells):
        PdE = np.zeros((nsn, len(KW)))
        PEE = np.zeros((nsn, len(KW)))
        hat = np.zeros(len(kd))
        hat[i] = 1.0
        row = np.interp(np.log(KW), np.log(kd[nodes]), hat[nodes], left=0, right=0)
        (PdE if x == 0 else PEE)[s] = row
        path = write_set("run/tables_J", "cell", sns, PdE, PEE)
        cl.use_sim_tables(path, sns, zs, 1)
        J[:, c] = cl.xi_sim() - xi_gg
        if c % 100 == 0:
            print(f"  cell {c}", flush=True)
    np.save("analysis/J_cells.npy", J)
    S = np.array([sig[s, x, i] for (s, x, i) in cells])
    cs = np.array([s for (s, x, i) in cells])
    cx = np.array([x for (s, x, i) in cells])
    ck = np.array([kd[i] for (s, x, i) in cells])

    def resid_vec(mdl, j, K0=None):
        """model - FLAMINGO residual in table units on the cells (white offset of
        P_EE re-minimised over the cells used, as in the FLAMINGO chi2)."""
        x_ = d[f"{mdl}_x_true"][:, j]
        out = np.zeros(len(cells))
        for s in range(nsn):
            Mo = d[f"{mdl}_R"][s, j]*np.array([b[s]*Pf[s], Pf[s], Pf[s]])
            Mo[1] += x_[s][-1]
            Mo[2] += x_[s][-1]
            r = Mo - T[s]
            sel = (kd >= KMIN) & ((kd <= K0/Dz[s]*(1 + 1e-9)) if K0 else True)
            w = np.concatenate([1/err[s, 1][sel]**2, 1/err[s, 2][sel]**2])
            cc = np.sum(w*np.concatenate([r[1][sel], r[2][sel]]))/np.sum(w)
            for c, (ss, x, i) in enumerate(cells):
                if ss == s:
                    out[c] = (r[0][i] if x == 0 else r[1][i] - cc)*conv[s, x][i]
        return out

    summary, maps = [], {}
    for K0 in MASKS:
        m = M1[K1.index(K0)]
        Ci = np.linalg.inv(cov[np.ix_(m, m)])
        Jm = J[m]
        FR = Jm.T @ Ci @ Jm
        A = S[:, None]*FR*S[None, :]
        matched = ck <= K0/Dz[cs]*(1 + 1e-9)
        res = {}
        for lab, sel in (("full", np.ones(len(cells), bool)), ("matched", matched)):
            Asub = A[np.ix_(sel, sel)]
            ev, V = np.linalg.eigh(Asub)
            amp = np.zeros(len(cells))
            for c, (s, x, i) in enumerate(cells):
                amp[c] = (T[s, 0][i]*conv[s, 0][i] if x == 0 else 2*(T[s, 1][i] - d["Ns_true"][s])*conv[s, 1][i])/S[c]
            a = amp[sel]
            lam_amp = float(a @ Asub @ a/(a @ a))
            res[lab] = dict(lmax=float(ev[-1]), lam_amp=lam_amp, v=V[:, -1], sel=sel, ev=ev)
        maps[K0] = (A, res)
        # validation against cosmolike for the matched model residuals
        val = []
        for mdl in COL:
            for j in range(len(kg)):
                u = resid_vec(mdl, j, K0)
                pred = float(u[matched] @ FR[np.ix_(matched, matched)] @ u[matched])
                flv = float(np.sum((u[matched]/S[matched])**2))
                val.append((mdl, kg[j], flv, pred))
        summary.append((K0, res, val))
        r = res["matched"]
        print(f"K0 = {K0}/D: matched: lambda_max = {r['lmax']:.3e} (guaranteed FL threshold "
              f"{1/r['lmax']:.1f}), amplitude {r['lam_amp']:.3e} ({1/r['lam_amp']:.0f}); full: "
              f"lambda_max = {res['full']['lmax']:.3e} ({1/res['full']['lmax']:.1f})", flush=True)

    # cross-check prediction vs translate.py cosmolike numbers
    import csv
    tr = {(r["model"], round(float(r["kmax"]), 3), float(r["K0_over_D"])): r
          for r in csv.DictReader(open("analysis/translate.csv"))}
    with open("analysis/translate_matrix.csv", "w") as fh:
        fh.write("K0_over_D,lambda_max_matched,FL_guaranteed_matched,lambda_amp_matched,FL_amp_matched,"
                 "lambda_max_full,FL_guaranteed_full,lambda_amp_full\n")
        for K0, res, val in summary:
            fh.write(f"{K0},{res['matched']['lmax']:.4g},{1/res['matched']['lmax']:.4g},"
                     f"{res['matched']['lam_amp']:.4g},{1/res['matched']['lam_amp']:.4g},"
                     f"{res['full']['lmax']:.4g},{1/res['full']['lmax']:.4g},{res['full']['lam_amp']:.4g}\n")
            dev = []
            for mdl, kmax, flv, pred in val:
                ref = tr.get((mdl, round(float(kmax), 3), K0))
                if ref is not None and float(ref["dchi2_roman_matched"]) > 0.05:
                    dev.append(pred/float(ref["dchi2_roman_matched"]))
            print(f"  validation {K0}/D: matrix / cosmolike (matched model residuals): median "
                  f"{np.median(dev):.3f}, range {np.min(dev):.3f}-{np.max(dev):.3f}")

    # ---------------- figures ----------------
    fig, axs = plt.subplots(1, len(MASKS), figsize=(7*len(MASKS), 6), sharey=True)
    for ax, K0 in zip(axs, MASKS):
        A, res = maps[K0]
        dg = np.diag(A)
        for s in range(nsn):
            for x, ls in ((0, "-"), (1, ":")):
                sel = (cs == s) & (cx == x)
                ax.plot(ck[sel], np.maximum(dg[sel], 1e-12), ls, color=plt.cm.plasma(s/nsn), lw=1.4,
                        label=f"z = {zs[s]:g}" if x == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1e-1)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(r"$k$ of the FLAMINGO bin $[h\,{\rm Mpc}^{-1}]$")
        ax.set_title(f"mask {K0}/D(z): Roman Delta chi2 per unit FLAMINGO chi2 in one bin\n"
                     "solid P_dE (GI), dotted P_EE (II)")
        ax.axhline(res["matched"]["lam_amp"], color="k", lw=0.8, ls="--")
    axs[0].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ / $\Delta\chi^2_{\rm FL}$ (single bin)")
    axs[-1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "analysis/fig_sensitivity_map")

    fig, axs = plt.subplots(1, 2, figsize=(16, 6.3))
    for K0, res, val in summary:
        for mdl in COL:
            V = [(f, p) for m_, k_, f, p in val if m_ == mdl and p > 1e-4]
            if V:
                V = np.array(V)
                axs[0].scatter(V[:, 0], V[:, 1], color=COL[mdl], marker={0.25: "^", 0.35: "o", 0.5: "s"}[K0],
                               s=22, label=f"{mdl}, {K0}/D")
        lm, la = res["matched"]["lmax"], res["matched"]["lam_amp"]
        xx = np.geomspace(1, 1e6, 20)
        axs[0].plot(xx, lm*xx, "k-", lw=0.8)
        axs[0].plot(xx, la*xx, "k--", lw=0.8)
        axs[1].plot(np.arange(1, 41), res["matched"]["ev"][::-1][:40], "o-", ms=3, label=f"{K0}/D matched")
    axs[0].axhline(1, color="k", lw=1)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlim(1, 1e6)
    axs[0].set_ylim(1e-4, 1e3)
    axs[0].set_xlabel(r"$\Delta\chi^2_{\rm FL}$ of the residual (13 snapshots, $k\leq K_0/D(z)$)")
    axs[0].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ (same residual)")
    axs[0].set_title("solid: worst-case direction (lambda_max); dashed: amplitude change")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("eigenmode rank")
    axs[1].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ per unit $\Delta\chi^2_{\rm FL}$")
    axs[1].set_title("spectrum of the translation: Roman sees few FLAMINGO directions")
    axs[1].legend()
    fig.tight_layout()
    save(fig, "analysis/fig_translate_matrix")

    # worst-case residual direction (matched, 0.35/D)
    A, res = maps[0.35]
    r = res["matched"]
    v = np.zeros(len(cells))
    v[r["sel"]] = r["v"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in range(nsn):
        for x, ls in ((0, "-"), (1, ":")):
            sel = (cs == s) & (cx == x) & r["sel"]
            if sel.any() and np.max(np.abs(v[sel])) > 0.05:
                ax.plot(ck[sel], v[sel], ls, marker="o", ms=3, color=plt.cm.plasma(s/nsn),
                        label=f"z = {zs[s]:g} {'dE' if x == 0 else 'EE'}")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"residual / $\sigma_{\rm FLAMINGO}$ (unit norm)")
    ax.set_title(f"0.35/D(z): the FLAMINGO residual Roman is most sensitive to "
                 f"(lambda_max = {r['lmax']:.2e}; components > 0.05)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "analysis/fig_worst_direction")


if __name__ == "__main__":
    main()
