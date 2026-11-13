#!/usr/bin/env python3
"""Roman cosmic-shear Delta chi2 of NLA / TATT IA against FLAMINGO, vs k_max.

Inputs: spectra.npz from build_spectra.py (truth and per-snapshot model
spectra as R_X = P_X / P_NL on snapshot redshifts, model amplitudes fitted
over k <= k_max to FLAMINGO with FLAMINGO error bars).

For every model (NLA, TATT) and fit k_max:
    Delta xi = xi_IA[model(k_max)] - xi_IA[truth]          (GI + II, E and B)
    Delta chi2 = Delta xi^T C^-1 Delta xi   on
      * the full 1080-element xi_+- vector;
      * IR/UV masks (alpha = --alpha) at k_UV, constant comoving (n = 0),
        for every k_UV on the k_max grid: Delta chi2(k_max fit, k_UV mask);
        the diagonal k_UV = k_max is the self-consistent choice;
      * the IR/UV mask at k_UV = 0.35/D(z) (current Roman masks).
C: Gaussian xi_+- covariance (roman_limber.gaussian_cov) at the fiducial
GG + truth IA, 2415 deg^2, 41.3/8 gal/arcmin^2 per bin, sigma_e = 0.30 per
component. Truth variant (--truth nla | highk) picks the shape-noise
subtraction of P_EE / P_BB (see build_spectra.py).

P(k) side, from spectra.npz: sum over snapshots of the fit chi2 over
k <= k_max (P_hE + P_EE, FLAMINGO error bars) and of the same model's chi2
over 0.05 <= k <= 1.7.

    python roman_dchi2.py --spectra spectra.npz --nz example1.nz --out results
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

FLAMINGO = dict(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, sigma8=0.807, mnu=0.06)
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
X = ("dE", "EE", "BB")


def ratios(zs, k, R):
    """R[nsn, 3, nk] -> dict of callables."""
    return {x: RL.ratio_table(zs, k, R[:, i, :]) for i, x in enumerate(X)}


def chi2(dv, Cinv, m=None):
    if m is None:
        return float(dv @ Cinv @ dv)
    C = Cinv[m]
    return float(dv[m] @ C @ dv[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", default="spectra.npz")
    ap.add_argument("--nz", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--truth", choices=["nla", "highk", "bconst"], default="highk")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--area", type=float, default=2415.0)
    ap.add_argument("--nmc", type=int, default=40, help="noise-floor realizations")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = np.load(a.spectra)
    zs, k, kmax = S["zs"], S["k"], S["kmax"]
    print("CAMB ...", flush=True)
    cosmo = RL.Cosmology(**FLAMINGO, halofit="takahashi")
    sh = RL.RomanShear(cosmo, a.nz)
    nt, npair = sh.ntheta, len(sh.pairs)
    T = ratios(zs, k, S[f"truth_{a.truth}"])

    print("truth xi and covariance ...", flush=True)
    CEg, CBg = sh.cl(None)
    CEi, CBi = sh.cl(T, gg=False)
    xi_gg = sh.xi(CEg, CBg)
    xi_t = sh.xi(CEi, CBi)
    xi_tot = xi_gg + xi_t
    cov = sh.gaussian_cov(CEg + CEi, CBg + CBi, area_deg2=a.area)
    ev = np.linalg.eigvalsh(cov)
    print(f"cov eigenvalues: min {ev.min():.3e} max {ev.max():.3e}", flush=True)
    Cinv = np.linalg.inv(cov)
    sig = np.sqrt(np.diag(cov))
    snr_ia = np.sqrt(chi2(xi_t, Cinv))
    snr_tot = np.sqrt(chi2(xi_tot, Cinv))
    print(f"S/N: total {snr_tot:.1f}, IA part {snr_ia:.1f}")

    # IR/UV masks
    print("IR/UV masks ...", flush=True)
    masks = {}
    for kuv in kmax:
        CEr, CBr = sh.cl(T, kcut=lambda z, kk=kuv: np.full_like(z, kk))
        masks[float(kuv)] = RL.uv_mask(xi_tot, sh.xi(CEr, CBr), a.alpha, nt)
    CEr, CBr = sh.cl(T, kcut=lambda z: 0.35/cosmo.D(z))
    mask_035 = RL.uv_mask(xi_tot, sh.xi(CEr, CBr), a.alpha, nt)
    nkeep = {kk: int(m.sum()) for kk, m in masks.items()}
    print("kept points per k_UV:", {f"{kk:.3f}": v for kk, v in nkeep.items()},
          "0.35/D:", int(mask_035.sum()))

    # models
    res = {}
    for mdl in ("NLA", "TATT"):
        R = S[f"R_{mdl}"]                         # [nsn, nkmax, 3, nk]
        xis, full, diag, m035 = [], [], [], []
        grid = np.zeros((len(kmax), len(kmax)))
        for j, km in enumerate(kmax):
            CEm, CBm = sh.cl(ratios(zs, k, R[:, j]), gg=False)
            xm = sh.xi(CEm, CBm)
            dv = xm - xi_t
            xis.append(xm)
            full.append(chi2(dv, Cinv))
            for i, ku in enumerate(kmax):
                m = masks[float(ku)]
                grid[j, i] = float(dv[m] @ np.linalg.solve(cov[np.ix_(m, m)], dv[m]))
            diag.append(grid[j, j])
            m035.append(float(dv[mask_035] @ np.linalg.solve(cov[np.ix_(mask_035, mask_035)],
                                                            dv[mask_035])))
            print(f"{mdl} k_max={km:.3f}: dchi2 full={full[-1]:9.2f}  IR/UV(k_UV=k_max)="
                  f"{diag[-1]:8.3f}  0.35/D mask={m035[-1]:8.3f}", flush=True)
        res[mdl] = dict(xi=np.array(xis), full=np.array(full), diag=np.array(diag),
                        m035=np.array(m035), grid=grid,
                        pk_fit=S[f"chi2_{mdl}"].sum(0), pk_dof=S[f"dof_{mdl}"].sum(0),
                        pk_wide=S[f"chi2w_{mdl}"].sum(0))
    def dchi2_m(dv, m):
        return float(dv[m] @ np.linalg.solve(cov[np.ix_(m, m)], dv[m]))

    # k_UV(z) = k0 / D(z) family: masks at k0/D(z), and per-snapshot fits at the
    # largest grid k_max <= k0/D(z_snapshot) (first grid point if below it)
    print("k0/D(z) family ...", flush=True)
    Dz = np.array([cosmo.D(z) for z in zs])
    k0s = np.geomspace(0.08, 0.6, 12)
    fam = {"k0": k0s, "npts": [], "NLA": [], "TATT": [], "jsel": [], "masks": []}
    for k0 in k0s:
        CEr, CBr = sh.cl(T, kcut=lambda z, kk=k0: kk/cosmo.D(z))
        m = RL.uv_mask(xi_tot, sh.xi(CEr, CBr), a.alpha, nt)
        fam["masks"].append(m)
        fam["npts"].append(int(m.sum()))
        jsel = np.clip(np.searchsorted(kmax, k0/Dz*(1 + 1e-9), side="right") - 1, 0, None)
        fam["jsel"].append(jsel)
        for mdl in ("NLA", "TATT"):
            R = S[f"R_{mdl}"][np.arange(len(zs)), jsel]
            CEm, CBm = sh.cl(ratios(zs, k, R), gg=False)
            fam[mdl].append(dchi2_m(sh.xi(CEm, CBm) - xi_t, m))
        print(f"k0={k0:.3f} ({int(m.sum())} pts): NLA {fam['NLA'][-1]:8.3f}  "
              f"TATT {fam['TATT'][-1]:8.3f}  fit k_max per snapshot "
              f"{np.round(kmax[jsel], 2)}", flush=True)

    # noise floor: truth perturbed by FLAMINGO's 1-sigma errors
    print("noise floor ...", flush=True)
    rng = np.random.default_rng(a.seed)
    E = S["truth_err"]
    Rt = S[f"truth_{a.truth}"]
    nf = {"diag": [], "full": [], "m035": [], "fam": []}
    for r in range(a.nmc):
        Rp = Rt + rng.standard_normal(Rt.shape)*E
        CEp, CBp = sh.cl(ratios(zs, k, Rp), gg=False)
        dv = sh.xi(CEp, CBp) - xi_t
        nf["full"].append(chi2(dv, Cinv))
        nf["diag"].append([dchi2_m(dv, masks[float(kk)]) for kk in kmax])
        nf["m035"].append(dchi2_m(dv, mask_035))
        nf["fam"].append([dchi2_m(dv, m) for m in fam["masks"]])
    nf = {key: np.array(v) for key, v in nf.items()}
    print("noise floor (mean Delta chi2), IR/UV at k_UV:",
          dict(zip(np.round(kmax, 3), np.round(nf["diag"].mean(0), 3))),
          " 0.35/D:", round(float(nf["m035"].mean()), 3),
          " full:", round(float(nf["full"].mean()), 2), flush=True)
    np.savez(os.path.join(a.out, "family_noise.npz"), k0=k0s, npts=fam["npts"],
             NLA=fam["NLA"], TATT=fam["TATT"], jsel=np.array(fam["jsel"]),
             fam_masks=np.array(fam["masks"]),
             nf_diag=nf["diag"], nf_full=nf["full"], nf_m035=nf["m035"], nf_fam=nf["fam"])

    np.savez(os.path.join(a.out, "roman_dchi2.npz"), kmax=kmax, xi_gg=xi_gg, xi_t=xi_t,
             cov=cov, mask_035=mask_035,
             masks=np.array([masks[float(kk)] for kk in kmax]),
             **{f"{m}_{key}": v for m, d in res.items() for key, v in d.items()})

    # ---------------- summary table ----------------
    with open(os.path.join(a.out, "dchi2_vs_kmax.csv"), "w") as fh:
        fh.write("model,kmax,dchi2_full,dchi2_iruv_kuv_eq_kmax,npts_iruv,dchi2_mask035,"
                 "pk_chi2_fit,pk_dof,pk_chi2_wide\n")
        for mdl, d in res.items():
            for j, km in enumerate(kmax):
                fh.write(f"{mdl},{km:.4f},{d['full'][j]:.4f},{d['diag'][j]:.4f},"
                         f"{nkeep[float(km)]},{d['m035'][j]:.4f},{d['pk_fit'][j]:.2f},"
                         f"{d['pk_dof'][j]},{d['pk_wide'][j]:.2f}\n")
    with open(os.path.join(a.out, "dchi2_k0_over_D.csv"), "w") as fh:
        fh.write("k0,npts,dchi2_NLA,dchi2_TATT,noise_floor_mean,noise_floor_p84\n")
        for i, k0 in enumerate(k0s):
            fh.write(f"{k0:.4f},{fam['npts'][i]},{fam['NLA'][i]:.4f},{fam['TATT'][i]:.4f},"
                     f"{nf['fam'][:, i].mean():.4f},{np.percentile(nf['fam'][:, i], 84):.4f}\n")
    plots(a, sh, kmax, res, xi_t, xi_gg, sig, masks, mask_035, nkeep, S, zs)
    plot_family(a, kmax, res, fam, nf, mask_035)


def plot_family(a, kmax, res, fam, nf, mask_035):
    xl = r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$"
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axs[0]
    for mdl, d in res.items():
        ax.plot(kmax, d["diag"], "s-", color=COL[mdl], label=f"{mdl}")
        ax.axhline(d["m035"].min(), color=COL[mdl], ls=":", lw=1,
                   label=f"{mdl}: best over fit k_max, 0.35/D mask")
    lo, mid, hi = np.percentile(nf["diag"], [16, 50, 84], axis=0)
    ax.fill_between(kmax, lo, hi, color="0.75", alpha=0.7,
                    label="FLAMINGO noise floor (16-84%)")
    ax.plot(kmax, nf["diag"].mean(0), color="0.4", lw=1, label="noise floor, mean")
    ax.set_title(r"constant comoving: fit $k_{\rm max}$ = mask $k_{UV}$")
    ax.set_xlabel(xl)
    ax = axs[1]
    for mdl in ("NLA", "TATT"):
        ax.plot(fam["k0"], fam[mdl], "s-", color=COL[mdl], label=mdl)
    lo, hi = np.percentile(nf["fam"], [16, 84], axis=0)
    ax.fill_between(fam["k0"], lo, hi, color="0.75", alpha=0.7,
                    label="FLAMINGO noise floor (16-84%)")
    ax.plot(fam["k0"], nf["fam"].mean(0), color="0.4", lw=1)
    ax.axvline(0.35, color="k", ls=":", lw=0.8)
    ax.set_title(r"$k_{UV}(z)=k_0/D(z)$ masks; fits at $k_{\rm max}\leq k_0/D(z_{\rm snap})$")
    ax.set_xlabel(r"$k_0\ [h\,{\rm Mpc}^{-1}]$")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(0.2, color="k", lw=0.8, ls="--")
        ax.set_ylabel(r"Roman $\Delta\chi^2$ on the IR/UV-masked $\xi_\pm$")
        ax.legend(fontsize=7.5)
    for ax, npts, xs in ((axs[1], fam["npts"], fam["k0"]),):
        for x, n in zip(xs, npts):
            ax.annotate(str(n), (x, ax.get_ylim()[0]), xytext=(0, 4), textcoords="offset points",
                        fontsize=6.5, ha="center", color="0.3")
    fig.suptitle(f"Roman xi_+- Delta chi2 vs scale cut (truth: {a.truth}; numbers: points kept)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "roman_dchi2_cut_with_noise.png"), dpi=120)
    plt.close(fig)


def plots(a, sh, kmax, res, xi_t, xi_gg, sig, masks, mask_035, nkeep, S, zs):
    nt, npair = sh.ntheta, len(sh.pairs)
    xl = r"$k_{\rm max}\ [h\,{\rm Mpc}^{-1}]$"

    # 1. Roman dchi2 vs k_max
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5))
    for mdl, d in res.items():
        axs[0].plot(kmax, d["full"], "o-", color=COL[mdl], label=f"{mdl}: all 1080 points")
        axs[0].plot(kmax, d["diag"], "s--", color=COL[mdl],
                    label=f"{mdl}: IR/UV mask at $k_{{UV}}=k_{{\\rm max}}$")
        axs[0].plot(kmax, d["m035"], "^:", color=COL[mdl], label=f"{mdl}: 0.35/D(z) mask")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(0.2, color="k", lw=0.8, ls=":")
        ax.set_xlabel(xl)
    axs[0].set_ylabel(r"Roman $\Delta\chi^2$ ($\xi_\pm$, IA model $-$ FLAMINGO)")
    axs[0].legend(fontsize=7.5)
    axs[0].set_title("Roman cosmic shear, Gaussian cov")
    ax2 = axs[1]
    ax2.plot(kmax, [nkeep[float(kk)] for kk in kmax], "ko-")
    ax2.axhline(mask_035.sum(), color="0.5", ls=":", label="0.35/D(z) mask")
    ax2.set_yscale("linear")
    ax2.set_ylabel(f"points kept by the IR/UV mask (alpha={a.alpha}), of 1080")
    ax2.legend()
    ax2.set_title(r"IR/UV mask size vs $k_{UV}$ (constant comoving)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "roman_dchi2_vs_kmax.png"), dpi=120)
    plt.close(fig)

    # 2. dchi2 map (fit k_max x mask k_UV)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.3))
    for ax, mdl in zip(axs, res):
        g = np.log10(np.maximum(res[mdl]["grid"], 1e-4))
        im = ax.imshow(g.T, origin="lower", cmap="viridis", aspect="auto",
                       vmin=-3, vmax=3)
        ax.contour(g.T, levels=[np.log10(0.2), 0], colors=["w", "r"], linewidths=1.2)
        lab = [f"{v:.2f}" for v in kmax]
        ax.set_xticks(range(len(kmax)), lab, rotation=60, fontsize=8)
        ax.set_yticks(range(len(kmax)), lab, fontsize=8)
        ax.plot(range(len(kmax)), range(len(kmax)), "k--", lw=0.8)
        ax.set_xlabel(r"fit $k_{\rm max}$ (amplitudes per snapshot)")
        ax.set_ylabel(r"mask $k_{UV}$ (IR/UV, constant comoving)")
        ax.set_title(f"{mdl}: log10 Roman Delta chi2 (red: 1, white: 0.2)")
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "roman_dchi2_map.png"), dpi=120)
    plt.close(fig)

    # 3. P(k) chi2 vs k_max and the mapping
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for mdl, d in res.items():
        axs[0].plot(kmax, d["pk_fit"], "o-", color=COL[mdl], label=f"{mdl}: fit chi2, k<=k_max")
        axs[0].plot(kmax, d["pk_dof"], ":", color=COL[mdl], label=f"{mdl}: dof")
        axs[0].plot(kmax, d["pk_wide"], "s--", color=COL[mdl],
                    label=f"{mdl}: chi2 over 0.05-1.7")
        dd = d["pk_fit"]/d["pk_dof"]
        axs[1].plot(dd, d["diag"], "s-", color=COL[mdl], label=f"{mdl}, IR/UV at k_max")
        axs[1].plot(dd, d["full"], "o:", color=COL[mdl], alpha=0.6, label=f"{mdl}, all points")
        for j, km in enumerate(kmax):
            axs[1].annotate(f"{km:.2f}", (dd[j], d["diag"][j]), fontsize=6.5, color=COL[mdl])
        axs[2].plot(d["pk_wide"], d["full"], "o-", color=COL[mdl], label=mdl)
        for j, km in enumerate(kmax):
            axs[2].annotate(f"{km:.2f}", (d["pk_wide"][j], d["full"][j]), fontsize=6.5,
                            color=COL[mdl])
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(xl)
    axs[0].set_ylabel(r"$\sum_{\rm snapshots}\chi^2_{P(k)}$ (P_hE + P_EE, FLAMINGO errors)")
    axs[0].legend(fontsize=7.5)
    axs[0].set_title("P(k) level")
    for ax in axs[1:]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(0.2, color="k", lw=0.8, ls=":")
        ax.legend(fontsize=8)
    axs[1].set_xlabel(r"P(k) reduced $\chi^2$: $\sum\chi^2_{\rm fit}/\sum{\rm dof}$ over snapshots, $k\leq k_{\rm max}$")
    axs[1].axvline(1, color="0.5", lw=0.8)
    axs[1].set_ylabel(r"Roman $\Delta\chi^2$")
    axs[1].set_title("mapping (labels: k_max)")
    axs[2].set_xlabel(r"$\sum\chi^2$ over 0.05-1.7 h/Mpc (same model)")
    axs[2].set_ylabel(r"Roman $\Delta\chi^2$, all 1080 points")
    axs[2].set_title("mapping, fixed P(k) range")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "pk_vs_roman_dchi2.png"), dpi=120)
    plt.close(fig)

    # 4. Delta xi / sigma per data-vector element at a few k_max
    picks = [j for j in range(len(kmax)) if kmax[j] in kmax[[1, 3, 5, 7, 9]]]
    fig, axs = plt.subplots(len(picks), 1, figsize=(15, 2.6*len(picks)), sharex=True)
    for ax, j in zip(axs, picks):
        for mdl, d in res.items():
            ax.plot((d["xi"][j] - xi_t)/sig, color=COL[mdl], lw=0.8, label=mdl)
        ax.axvline(npair*nt, color="k", lw=0.8)
        for p in range(1, 2*npair):
            ax.axvline(p*nt, color="0.9", lw=0.4, zorder=0)
        ax.fill_between(np.arange(2*npair*nt), -3, 3, where=~masks[float(kmax[j])],
                        color="0.85", step="mid", zorder=0)
        ax.set_ylabel(r"$\Delta\xi/\sigma$")
        ax.set_title(f"fit k_max = {kmax[j]:.3f} (grey: cut by the IR/UV mask at k_UV = k_max)",
                     fontsize=9)
        lim = max(0.1, np.percentile(np.abs([(d["xi"][j] - xi_t)/sig for d in res.values()]), 99.5))
        ax.set_ylim(-1.1*lim, 1.1*lim)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel(r"data-vector index ($\xi_+$ | $\xi_-$; 36 pairs x 15 $\theta$)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "delta_xi_over_sigma.png"), dpi=110)
    plt.close(fig)

    # 5. xi_IA truth vs models for a few pairs
    tc = np.sqrt(sh.edges[1:]*sh.edges[:-1])/RL.ARCMIN
    sel_pairs = [(0, 7), (2, 5), (4, 4), (7, 7)]
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    cm = plt.cm.plasma
    for c, pr in enumerate(sel_pairs):
        p = sh.pairs.index(pr)
        for r, (blk, lab) in enumerate(((0, r"$\xi_+$"), (npair*nt, r"$\xi_-$"))):
            ax = axs[r, c]
            s = slice(blk + p*nt, blk + (p + 1)*nt)
            ax.plot(tc, -xi_t[s]*tc, "k-", lw=2.2, label="FLAMINGO (stitched)")
            ax.fill_between(tc, (-xi_t[s] - sig[s])*tc, (-xi_t[s] + sig[s])*tc, color="0.8",
                            label=r"$\pm1\sigma$ Roman")
            for j in (1, 4, 7, 9):
                for mdl, ls in (("NLA", "--"), ("TATT", "-")):
                    ax.plot(tc, -res[mdl]["xi"][j][s]*tc, ls, color=cm(j/len(kmax)), lw=1,
                            label=f"{mdl} k_max={kmax[j]:.2f}" if c == 0 and r == 0 else None)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\theta$ [arcmin]")
            ax.set_ylabel(r"$-\theta\,$" + lab + r"$_{\rm IA}$")
            ax.set_title(f"pair {pr} " + lab)
    axs[0, 0].legend(fontsize=6.5)
    fig.suptitle("IA part of xi (GI + II): FLAMINGO truth vs per-snapshot NLA (dashed) / TATT "
                 "(solid) fits; colour = fit k_max", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "xi_ia_examples.png"), dpi=110)
    plt.close(fig)

    # 6. truth vs model spectra (R_dE) at the snapshots, A vs k plots
    fig, axs = plt.subplots(3, 5, figsize=(21, 11))
    Rt = S[f"truth_{a.truth}"]
    for ax, (i, z) in zip(axs.flat, enumerate(zs)):
        ax.plot(S["k"], -Rt[i, 0], "k-", lw=2, label="truth P_dE/P_NL")
        for j in (1, 4, 7, 9):
            for mdl, ls in (("NLA", "--"), ("TATT", "-")):
                ax.plot(S["k"], -S[f"R_{mdl}"][i, j, 0], ls, color=cm(j/len(kmax)), lw=1,
                        label=f"{mdl} {kmax[j]:.2f}")
        ax.set_xscale("log")
        ax.set_title(f"z = {z:.2f}")
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
        ax.set_ylabel(r"$-P_{\delta E}/P_{NL}$")
    axs.flat[0].legend(fontsize=6)
    for ax in axs.flat[len(zs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "spectra_ratio_dE.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
