#!/usr/bin/env python3
"""Analysis of cosmolike_xi_ia.npz: FLAMINGO vs NLA / TATT in Roman xi_+-.

Covariance: Gaussian xi_+- (roman_limber.gaussian_cov; 2415 deg^2,
41.3/8 gal/arcmin^2 per bin, sigma_e 0.30 per component) at FLAMINGO D3A
with GG + FLAMINGO IA.

Outputs (220-dpi PNG + PDF, csv):
  dchi2_masks.csv / fig_dchi2_masks : Delta chi2 of xi_IA(model fit at k_max)
      - xi_IA(FLAMINGO) on the IR/UV masks k_UV/D(z) (n = 1, production
      family) and k_UV (n = 0), and on common theta_min cuts
  fig_xi_pairs_<xi+|xi->     : xi_IA for selected pairs, FLAMINGO vs fits
  truth_choices.csv / fig_truth_choices : how much the FLAMINGO-side choices
      (k > 17.7, sim_IA extrapolation, white level, smoothing) move xi_IA,
      and the model Delta chi2 against each choice
  mapping.csv / fig_mapping : Delta chi2_Roman of the matched residual vs
      Delta chi2_FL (same residual, 0.05 <= k <= K), cosmolike
  reconcile.csv : every earlier number next to the cosmolike value with the
      same definitions
  checks.txt : cosmolike native NLA/TATT vs the same models as tables;
      python Limber vs cosmolike; IA B-mode size
"""
import argparse
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

COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
NTH, NPAIR = 15, 36
PAIRS = [(i, j) for i in range(8) for j in range(i, 8)]


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def q(dv, cov, m=None):
    if m is None:
        m = np.ones(dv.size, bool)
    if m.sum() == 0:
        return 0.0
    return float(dv[m] @ np.linalg.solve(cov[np.ix_(m, m)], dv[m]))


def kaxis(ax, ticks=(0.12, 0.2, 0.3, 0.5, 1, 2, 3)):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run/cosmolike_xi_ia.npz")
    ap.add_argument("--fits", default="../v2/fits.npz")
    ap.add_argument("--v2", default="../v2/roman/roman_map.npz")
    ap.add_argument("--v2xi", default="../v2/xi/xi_ia.npz")
    ap.add_argument("--v1", default="../results_highk/dchi2_vs_kmax.csv")
    ap.add_argument("--nz", required=True)
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = np.load(a.run)
    d = np.load(a.fits)
    kg, zs = R["kgrid"], R["zs"]
    nK = len(kg)
    xt = R["xi_flam"]
    th = np.tile(np.arange(NTH), 2*NPAIR)
    log = open(os.path.join(a.out, "checks.txt"), "w")

    def say(s):
        print(s)
        log.write(s + "\n")

    # covariance and python Limber at D3A
    cosmo = RL.Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
    S = RL.RomanShear(cosmo, a.nz)
    kd = d["k"][0]
    Rt = d["Rtrue"].copy()                                   # dE, EE (Ns_hk out), BB
    lo = np.geomspace(1e-4, 0.05, 30, endpoint=False)
    hi = np.geomspace(kd[-1]*1.02, 2e3, 20)
    m = kd >= 0.05
    kk = np.concatenate([lo, kd[m], hi])

    def ext(Rtab, zero_hi=True, bb=True):
        out = np.zeros((len(zs), 3, len(kk)))
        for s in range(len(zs)):
            for x in range(3):
                if x == 2 and not bb:
                    continue
                r = Rtab[s, x, m]
                out[s, x] = np.concatenate([np.full(len(lo), r[0]), r,
                                            np.zeros(len(hi)) if zero_hi else np.full(len(hi), r[-1])])
        return {X: RL.ratio_table(zs, kk, out[:, i]) for i, X in enumerate(("dE", "EE", "BB"))}

    CEg, CBg = S.cl(None)
    CEt, CBt = S.cl(ext(Rt, zero_hi=False), gg=False)
    cov = S.gaussian_cov(CEg + CEt, CBg + CBt)
    sig = np.sqrt(np.diag(cov))
    np.save(os.path.join(a.out, "cov_gauss_d3a.npy"), cov)

    # ---------------- checks ----------------
    for mdl in ("NLA", "TATT"):
        nat, tab = R[f"check_{mdl}_native"], R[f"check_{mdl}_tables"]
        say(f"[check] {mdl} power law (A1 3.8, eta 0.57" + (", A2 -1.08, eta2 -0.71, bTA 0.18"
            if mdl == "TATT" else "") + f"): cosmolike native vs same model as sim tables: "
            f"Delta chi2 = {q(tab - nat, cov):.3f} (all 1080), max |diff|/sigma = "
            f"{np.max(np.abs(tab - nat)/sig):.3f}")
    CEz, CBz = S.cl(ext(Rt, zero_hi=True, bb=False), gg=False)
    xpy = S.xi(CEz, CBz)
    say(f"[check] python Limber vs cosmolike, FLAMINGO IA (dE + EE, nothing above 17.7): "
        f"Delta chi2 = {q(xpy - R['xi_flam_zero'], cov):.3f}, max |diff|/sigma = "
        f"{np.max(np.abs(xpy - R['xi_flam_zero'])/sig):.3f}")
    # B modes (python): truth and model BB
    CEb, CBb = S.cl({"dE": lambda K, Z: 0*K, "EE": lambda K, Z: 0*K,
                     "BB": ext(Rt, zero_hi=False)["BB"]}, gg=False)
    xbb = S.xi(CEb, CBb)
    say(f"[check] FLAMINGO IA B modes (python): chi2 of the BB contribution to xi = "
        f"{q(xbb, cov):.2f} (all), {q(xbb, cov, th >= 6):.2f} (theta >= 15.8')")

    # ---------------- Delta chi2 under masks ----------------
    M1, K1 = R["masks_n1"], R["K_n1"]
    M0, K0 = R["masks_n0"], R["K_n0"]
    say("[masks] kept points, k_UV/D(z): " + ", ".join(f"{k:g}: {int(mm.sum())}" for k, mm in zip(K1, M1)))
    say("[masks] kept points, k_UV const: " + ", ".join(f"{k:.2f}: {int(mm.sum())}" for k, mm in zip(K0, M0)))
    res = {}
    with open(os.path.join(a.out, "dchi2_masks.csv"), "w") as fh:
        fh.write("model,kmax,all," + ",".join(f"n1_{k:g}" for k in K1) + "," +
                 ",".join(f"n0_{k:.3f}" for k in K0) + "," +
                 ",".join(f"thmin_{S.edges[i]/RL.ARCMIN:.1f}" for i in range(NTH)) + "\n")
        for mdl in COL:
            for j in range(nK):
                dv = R[f"xi_{mdl}_{j}"] - xt
                row = [q(dv, cov)] + [q(dv, cov, mm) for mm in M1] + [q(dv, cov, mm) for mm in M0] \
                    + [q(dv, cov, th >= i) for i in range(NTH)]
                res[(mdl, j)] = np.array(row)
                fh.write(f"{mdl},{kg[j]:.3f}," + ",".join(f"{v:.4g}" for v in row) + "\n")
    n1 = slice(1, 1 + len(K1))
    n0 = slice(1 + len(K1), 1 + len(K1) + len(K0))
    tm = slice(1 + len(K1) + len(K0), None)

    # figure: Delta chi2 vs fit k_max for each mask family
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.3))
    cm = plt.cm.viridis
    for mdl, ls in (("NLA", "--"), ("TATT", "-")):
        for c, (k, idx) in enumerate(zip(K1, range(len(K1)))):
            if k not in (0.2, 0.3, 0.35, 0.4, 0.5, 0.6):
                continue
            y = [res[(mdl, j)][n1][idx] for j in range(nK)]
            axs[0].plot(kg, np.maximum(y, 1e-4), ls, marker="o", ms=3.5, color=cm(c/(len(K1) - 1)),
                        label=f"{mdl}, mask {k:g}/D(z)")
        y = [res[(mdl, j)][n0][j] for j in range(nK)]
        axs[1].plot(kg, np.maximum(y, 1e-4), ls, marker="o", ms=4, color=COL[mdl],
                    label=f"{mdl}: mask k_UV = fit k_max")
        for i, c2 in ((4, 0.2), (6, 0.45), (8, 0.7), (10, 0.9)):
            y = [res[(mdl, j)][tm][i] for j in range(nK)]
            axs[2].plot(kg, np.maximum(y, 1e-4), ls, marker="o", ms=3.5, color=cm(c2),
                        label=f"{mdl}, theta >= {S.edges[i]/RL.ARCMIN:.1f}'")
    titles = ["IR/UV masks k_UV/D(z) (production family)",
              "IR/UV mask at constant k_UV = fit k_max", "common theta_min (no IR/UV mask)"]
    for ax, t in zip(axs, titles):
        ax.axhline(1, color="k", lw=1)
        ax.axhline(0.2, color="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e3)
        kaxis(ax)
        ax.set_xlabel(r"fit $k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$ (amplitudes per snapshot)")
        ax.set_ylabel(r"$\Delta\chi^2$ of $\xi^{\rm IA}_{\rm model}-\xi^{\rm IA}_{\rm FLAMINGO}$")
        ax.set_title(t)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("cosmolike (roman_real + sim-IA): FLAMINGO and NLA/TATT IA through the same pipeline; "
                 "Gaussian covariance", fontsize=12)
    fig.tight_layout()
    save(fig, os.path.join(a.out, "fig_dchi2_masks"))

    # 2D: fit k_max x mask k_UV/D(z)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mdl in zip(axs, COL):
        G = np.array([res[(mdl, j)][n1] for j in range(nK)])
        im = ax.imshow(np.log10(np.maximum(G, 1e-3)).T, origin="lower", aspect="auto", cmap="viridis",
                       vmin=-2, vmax=3)
        ax.contour(np.log10(np.maximum(G, 1e-3)).T, levels=[np.log10(0.2), 0], colors=["w", "r"])
        for jj in range(nK):
            for ii in range(len(K1)):
                ax.text(jj, ii, f"{G[jj, ii]:.2g}", ha="center", va="center", fontsize=6.5,
                        color="w" if G[jj, ii] < 10 else "k")
        ax.set_xticks(range(nK), [f"{v:.2f}" for v in kg], rotation=60)
        ax.set_yticks(range(len(K1)), [f"{v:g}/D" for v in K1])
        ax.set_xlabel(r"fit $k_{\rm max}$")
        ax.set_ylabel(r"IR/UV mask $k_{UV}/D(z)$")
        ax.set_title(f"{mdl}: Delta chi2 (red 1, white 0.2)")
        plt.colorbar(im, ax=ax, label="log10 Delta chi2")
    fig.tight_layout()
    save(fig, os.path.join(a.out, "fig_dchi2_map_n1"))

    # ---------------- xi_IA pairs ----------------
    tc = np.asarray(R.get("theta", np.sqrt(S.edges[1:]*S.edges[:-1])/RL.ARCMIN))
    tc = np.sqrt(S.edges[1:]*S.edges[:-1])/RL.ARCMIN
    sel = [(0, 0), (0, 3), (0, 7), (2, 2), (2, 6), (4, 4), (5, 7), (7, 7)]
    js = [1, 3, 5, 7, 9]
    mshow = M1[list(K1).index(0.35)]
    for b, lab in ((0, "xi+"), (1, "xi-")):
        sym = r"\xi_+" if b == 0 else r"\xi_-"
        fig, axs = plt.subplots(3, len(sel), figsize=(4*len(sel), 10.5), sharex=True)
        for c, pr in enumerate(sel):
            p = PAIRS.index(pr)
            s_ = slice(b*NPAIR*NTH + p*NTH, b*NPAIR*NTH + (p + 1)*NTH)
            t, sg = xt[s_], sig[s_]
            cut = ~mshow[s_]
            for ax in axs[:, c]:
                if cut.any():
                    ax.axvspan(tc[0]/1.2, tc[np.where(cut)[0].max()]*1.1, color="0.92", zorder=0)
            ax = axs[0, c]
            ax.fill_between(tc, (t - sg)*tc, (t + sg)*tc, color="0.8", label=r"Roman $\pm1\sigma$")
            ax.plot(tc, t*tc, "k-", lw=2.4, label="FLAMINGO")
            for qq, j in enumerate(js):
                for mdl, ls in (("NLA", "--"), ("TATT", "-")):
                    ax.plot(tc, R[f"xi_{mdl}_{j}"][s_]*tc, ls, color=cm(qq/(len(js) - 1)), lw=1.2,
                            label=f"{mdl} k_max {kg[j]:.2f}" if c == 0 else None)
            ax.set_title(f"bins {pr[0] + 1}-{pr[1] + 1}  ${sym}^{{\\rm IA}}$")
            if c == 0:
                ax.set_ylabel(rf"$\theta\,{sym}^{{\rm IA}}$")
            for row, fn, ylab in ((1, lambda v: (v - t)/np.abs(t), "(model - FL)/|FL|"),
                                  (2, lambda v: (v - t)/sg, r"(model - FL)/$\sigma$")):
                ax = axs[row, c]
                for qq, j in enumerate(js):
                    for mdl, ls in (("NLA", "--"), ("TATT", "-")):
                        ax.plot(tc, fn(R[f"xi_{mdl}_{j}"][s_]), ls, color=cm(qq/(len(js) - 1)), lw=1.2)
                ax.axhline(0, color="k", lw=0.7)
                if row == 1:
                    ax.set_ylim(-1, 1)
                else:
                    for v in (-1, 1):
                        ax.axhline(v, color="k", lw=0.6, ls=":")
                if c == 0:
                    ax.set_ylabel(ylab)
            axs[2, c].set_xscale("log")
            axs[2, c].set_xlabel(r"$\theta$ [arcmin]")
            axs[2, c].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        axs[0, 0].legend(fontsize=6.5, ncol=2)
        fig.suptitle(f"IA part of Roman ${sym}$ from cosmolike: FLAMINGO (black) vs NLA (dashed) / TATT "
                     f"(solid) fitted to FLAMINGO P(k) up to k_max (colour). Grey: cut by the 0.35/D(z) "
                     f"IR/UV mask.", fontsize=12)
        fig.tight_layout()
        save(fig, os.path.join(a.out, f"fig_xi_pairs_{lab}"))

    # ---------------- truth choices ----------------
    variants = ["flam_zero", "flam_simIA", "flam_nsnla", "flam_nons", "flam_raw"]
    desc = {"flam_zero": "nothing above k = 17.7 (instead of P/P_NL held)",
            "flam_simIA": "only measured rows; sim_IA extrapolation (power law below, linear above)",
            "flam_nsnla": "white level of P_EE from the notes' low-k NLA fit",
            "flam_nons": "no white-noise subtraction in P_EE",
            "flam_raw": "measured spectra instead of smoothed"}
    i35 = list(K1).index(0.35)
    with open(os.path.join(a.out, "truth_choices.csv"), "w") as fh:
        fh.write("variant,description,dchi2_vs_main_all,dchi2_vs_main_mask0.35/D,"
                 "NLA_kmax0.29_mask0.35/D,TATT_kmax0.52_mask0.35/D,NLA_kmax0.29_all,TATT_kmax0.52_all\n")
        rows = [("flam", "main")] + [(v, desc[v]) for v in variants]
        for v, ds in rows:
            dv = R[f"xi_{v}"] - xt
            nla = R["xi_NLA_3"] - R[f"xi_{v}"]
            tat = R["xi_TATT_5"] - R[f"xi_{v}"]
            vals = [q(dv, cov), q(dv, cov, M1[i35]), q(nla, cov, M1[i35]), q(tat, cov, M1[i35]),
                    q(nla, cov), q(tat, cov)]
            fh.write(f"{v},{ds}," + ",".join(f"{x:.4g}" for x in vals) + "\n")
            say(f"[truth] {v:11s} vs main: all {vals[0]:9.2f}, 0.35/D mask {vals[1]:7.3f} | "
                f"NLA(0.29) {vals[2]:6.2f}/{vals[4]:8.1f}  TATT(0.52) {vals[3]:6.2f}/{vals[5]:8.1f}  "
                f"(mask / all)")

    # ---------------- mapping (matched residuals) ----------------
    V2 = np.load(a.v2)
    with open(os.path.join(a.out, "mapping.csv"), "w") as fh:
        fh.write("model,kmax,K,dchi2_FL,dchi2_roman_cosmolike,dchi2_roman_python_v2\n")
        pts = {mdl: [] for mdl in COL}
        for mdl in COL:
            for j in range(nK):
                for i in range(nK):
                    xr = R[f"xi_res_{mdl}_{j}_{i}"]
                    v = q(xr, cov, M0[i])
                    fl = V2[f"{mdl}_dFL"][j, i]
                    py = V2[f"{mdl}_real"][j, i]
                    pts[mdl].append((j, i, fl, v, py))
                    fh.write(f"{mdl},{kg[j]:.3f},{kg[i]:.3f},{fl:.4g},{v:.4g},{py:.4g}\n")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.3))
    for mdl in COL:
        P = np.array(pts[mdl])
        ok = P[:, 3] > 1e-4
        sc = axs[0].scatter(P[ok, 2], P[ok, 3], c=np.log10(kg[P[ok, 1].astype(int)]), cmap="viridis",
                            marker="o" if mdl == "NLA" else "s", s=22, vmin=np.log10(kg[0]), vmax=np.log10(kg[-1]))
        dg = P[P[:, 0] == P[:, 1]]
        axs[0].plot(dg[:, 2], np.maximum(dg[:, 3], 1e-4), "-", color=COL[mdl], lw=1.2,
                    label=f"{mdl}, K = k_max")
        axs[1].scatter(np.maximum(P[ok, 4], 1e-4), P[ok, 3], color=COL[mdl], s=12, label=mdl)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=1)
    axs[0].set_xlabel(r"$\Delta\chi^2_{\rm FL}$ (13 snapshots, $0.05\leq k\leq K$)")
    axs[0].set_ylabel(r"$\Delta\chi^2_{\rm Roman}$ (cosmolike, same residual, mask $k_{UV}=K$)")
    axs[0].legend()
    plt.colorbar(sc, ax=axs[0], label="log10 K")
    lim = [1e-4, 1e3]
    axs[1].plot(lim, lim, "k:", lw=1)
    axs[1].set_xlim(lim)
    axs[1].set_ylim(lim)
    axs[1].set_xlabel("python Limber (v2 roman_map, matched)")
    axs[1].set_ylabel("cosmolike")
    axs[1].set_title("same residual, same mask: python vs cosmolike")
    axs[1].legend()
    fig.tight_layout()
    save(fig, os.path.join(a.out, "fig_mapping"))

    # ---------------- reconcile ----------------
    v1 = {}
    for line in open(a.v1).read().splitlines()[1:]:
        f = line.split(",")
        v1[(f[0], round(float(f[1]), 3))] = (float(f[3]), float(f[5]))
    XI2 = np.load(a.v2xi)
    rows = []
    for mdl, j in (("NLA", 3), ("NLA", 4), ("TATT", 4), ("TATT", 6)):
        km = round(float(kg[j]), 3)
        dv = R[f"xi_{mdl}_{j}"] - xt
        old_v1 = v1.get((mdl, km), (np.nan, np.nan))
        res_m = q(R[f"xi_res_{mdl}_{j}_{j}"], cov, M0[j])
        xi2 = XI2[f"xi_{mdl}"][j] - XI2["xi_flamingo"]
        rows.append((mdl, kg[j],
                     old_v1[0], V2[f"{mdl}_real"][j, j], res_m,
                     V2[f"{mdl}_real_full"][j, j], q(dv, cov, M0[j]),
                     q(xi2, cov, M0[j]), q(dv, cov, th >= 6), q(xi2, cov, th >= 6),
                     q(dv, cov, M1[i35])))
    with open(os.path.join(a.out, "reconcile.csv"), "w") as fh:
        fh.write("model,kmax,v1_first_pass_mask_kmax,v2_matched_python,matched_cosmolike,"
                 "v2_full_python,full_cosmolike_mask_kmax,xi_compare_python_mask_kmax,"
                 "full_cosmolike_theta15.8,xi_compare_python_theta15.8,full_cosmolike_mask0.35/D\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.3f}," + ",".join(f"{x:.3g}" for x in r[2:]) + "\n")
    say(open(os.path.join(a.out, "reconcile.csv")).read())
    log.close()


if __name__ == "__main__":
    main()
