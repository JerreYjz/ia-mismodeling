#!/usr/bin/env python3
"""NLA fit + stitching of the FLAMINGO IA spectra with the TNG machinery (ia_halofit_v2/stitch.py).

For one snapshot (results/: unreduced inertia tensor of all bound stars, the TNG definition of
Galaxy_Shape_Tensor/eigen_iteration.c):

  1. stitch the folded spectra: nfold f is used for 0.25 k_Nyq(f/4) < k <= 0.25 k_Nyq(f)
     (folds 1, 4, 16; NGP window <= 5% there)
  2. convert to the TNG definitions of sim_kperp(): shear units (1/2R per shape leg), k-perp
     (dE: 1.5 x monopole; EE: hybrid mu=0 = M0 - 5/2 M2 + 27/8 M4, divided by the NGP window).
     There is no matter field, so P_dE ~= P_hE / b(k), b(k) = sqrt[(P_hh - V/N) / P_NL(k, z)]
  3. stitch_snapshot() from stitch.py with its defaults: NLA amplitude fitted to dE at
     k = 0.05-0.3, k_t(z) where P_NL/P_lin = 1.1, logistic width 0.3, large-scale smoothing on.
     CAMB P(k) uses the FLAMINGO cosmology (Omega_m, Omega_b, h from the SOAP header; sigma8,
     n_s from Schaye et al. 2023; massless neutrinos, as stitch.camb_power does) for the NLA
     template; b(k) uses FLAMINGO's measured matter P(k) (power_spectra/power_matter_*.txt)

Writes results/plots/stitch_<set>_sn<sn>.png, results/stitched_<set>/stitched_sn<sn>.dat (stitch.py
column layout) and appends to results/stitched_<set>/stitch_summary.txt. TNG300 stitched curves at
the nearest redshift (ia_halofit_v2/stitch_demo) are overlaid for reference.

usage: python plot_stitch_flamingo.py <sn> [all]
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, "/Users/yijiezhu/Downloads/ia_halofit_v2")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cosmology import Cosmology  # noqa: E402
from stitch import camb_power, kt_nonlinear, ngp_w2_avg, stitch_snapshot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TNG_DIR = "/Users/yijiezhu/Downloads/ia_halofit_v2/stitch_demo"
TNG_Z = {98: 0.01, 91: 0.10, 84: 0.20, 78: 0.30, 72: 0.40, 67: 0.50, 63: 0.60, 59: 0.70, 55: 0.82, 53: 0.89,
         50: 1.00, 47: 1.11, 42: 1.36, 40: 1.50, 37: 1.67, 33: 2.00, 28: 2.58, 25: 3.01}
L, NG, FOLDS = 681.0, 1024, (1, 4, 16)
SETS = {"all": ("results", "unreduced tensor, all bound stars (TNG definition)")}
COSMO = Cosmology(h=0.681, Om=0.304611, Ob=0.0486, ns=0.967, sigma8=0.807, linear_source="camb")


def kny(f):
    return np.pi * NG * f / L


def load(ps, name, sn, f):
    return np.atleast_2d(np.loadtxt(os.path.join(ps, f"{name}_sn{sn}_nfold{f}.dat")))


def sim_flamingo(res, sn, z, Pnl):
    """b(k) uses FLAMINGO's own measured matter P(k) (as export_cosmolike.py); Pnl is unused here."""
    from export_cosmolike import flamingo_pmm
    kp, pp, _ = flamingo_pmm(z)
    pdd = lambda kk: np.exp(np.interp(np.log(kk), np.log(kp), np.log(pp)))
    ps, gds = os.path.join(HERE, res, "PS"), os.path.join(HERE, res, "Galaxy_DS_Field")
    vals = open(os.path.join(gds, f"hinfo_fixng_ms_sub_red_sn{sn}_nfold1.dat")).read().split("\n")[1].split()
    ngal, erms = float(vals[2]), 0.5 * (float(vals[3]) + float(vals[4]))
    fsh = 1.0 / (2.0 * (1.0 - erms ** 2))
    cols = []
    for f in FOLDS:
        lo, hi = 0.25 * kny(f / 4) if f > 1 else 0.0, 0.25 * kny(f)
        d = {n: load(ps, n, sn, f) for n in ("Phh", "PhE0", "PEE0", "PEE2", "PEE4")}
        k = d["Phh"][:, 0]
        m = (k > lo) & (k <= hi)
        W2 = ngp_w2_avg(k[m], L / (f * NG))
        cols.append(np.column_stack([k[m], d["Phh"][m, 1], d["PhE0"][m, 1], d["PhE0"][m, 2],
                                     (d["PEE0"][m, 1] - 2.5 * d["PEE2"][m, 1] + 27 / 8 * d["PEE4"][m, 1]) / W2,
                                     np.sqrt(d["PEE0"][m, 2] ** 2 + (2.5 * d["PEE2"][m, 2]) ** 2
                                             + (27 / 8 * d["PEE4"][m, 2]) ** 2) / W2]))
    c = np.concatenate(cols)
    k = c[:, 0]
    b = np.sqrt(np.clip(c[:, 1] - L ** 3 / ngal, 1e-30, None) / pdd(k))
    sim = (k, 1.5 * c[:, 2] / b * fsh, 1.5 * c[:, 3] / b * fsh, c[:, 4] * fsh ** 2, c[:, 5] * fsh ** 2)
    return sim, dict(ngal=ngal, erms=erms, fsh=fsh, b=b)


def tng_curve(z):
    sn = min(TNG_Z, key=lambda s: abs(TNG_Z[s] - z))
    fn = os.path.join(TNG_DIR, f"stitched_sn{sn}.dat")
    if abs(TNG_Z[sn] - z) > 0.06 or not os.path.isfile(fn):
        return None, None
    return np.loadtxt(fn), TNG_Z[sn]


def main(sn, which):
    res, label = SETS[which]
    cat = os.path.join(HERE, "results", "Galaxies", f"galaxies_{sn}.hdf5")
    with h5py.File(cat, "r") as g:
        z = round(float(g.attrs["Redshift"]), 4)
        mmin, mmax = float(g.attrs["M_min"]), float(g.attrs["M_max"])
    Pnl, Plin = camb_power(COSMO, [z], kmax=30.0)
    sim, info = sim_flamingo(res, sn, z, Pnl)
    kz = kt_nonlinear(Pnl, Plin, z, 0.10)
    k_out = np.logspace(-4, np.log10(30.0), 400)
    r = stitch_snapshot(sn, z, sim, Pnl, COSMO, k_out, {"mode": "lowk"}, {"dE": kz, "EE": kz},
                        {"dE": 0.3, "EE": 0.3}, np.inf, (0.05, 0.3), Plin_fn=Plin,
                        smooth=dict(which=("dE", "EE"), h=0.5, k_raw=1.0, w_raw=0.3))

    out = os.path.join(HERE, "results", f"stitched_{which}")
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, f"stitched_sn{sn}.dat"),
               np.column_stack([k_out, r["w_dE"], r["nla_dE"], r["sim_dE"], r["P_dE"], r["w_EE"],
                                r["nla_EE"], r["sim_EE"], r["P_EE"], r["la_dE"], r["la_EE"]]), fmt="%.8e",
               header=(f"FLAMINGO L1_m8 sn{sn} z={z} shapes={label}  a_nla={r['a_nla']:.5g} +- {r['s_a_nla']:.3g} "
                       f"(C1rho=0.0134)  k_t={kz:.4g} width=0.3 shear_factor={info['fsh']:.4f}\n"
                       "k[h/Mpc] w_dE P_dE_NLA P_dE_sim P_dE w_EE P_EE_NLA P_EE_sim P_EE P_dE_LA P_EE_LA"))
    summ = os.path.join(out, "stitch_summary.txt")
    lines = [l for l in (open(summ).read().splitlines() if os.path.isfile(summ) else [])
             if l.startswith("#") or l.split()[0] != str(sn)]
    if not lines:
        lines = ["# sn z a_nla err lowk_ratio_dE lowk_ratio_EE k_t e_rms bias(k=0.05-0.3) ngal"]
    m = (sim[0] >= 0.05) & (sim[0] <= 0.3)
    lines.append(f"{sn:3d} {z:5.2f} {r['a_nla']:8.4f} {r['s_a_nla']:7.4f} {r['lowk_ratio_dE']:7.3f} "
                 f"{r['lowk_ratio_EE']:7.3f} {kz:6.3f} {info['erms']:6.4f} {np.mean(info['b'][m]):6.3f} {int(info['ngal'])}")
    open(summ, "w").write("\n".join([lines[0]] + sorted(lines[1:], key=lambda l: -int(l.split()[0]))) + "\n")

    # ---- plot ----
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    cF, cN, cS, cT = "#2a78d6", "#52514e", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
    fig = plt.figure(figsize=(12, 6.2), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 2, height_ratios=(4, 1), hspace=0.08)
    tng, ztng = tng_curve(z)
    for j, (name, power, sign) in enumerate((("dE", 1, -1), ("EE", 2, 1))):
        a = fig.add_subplot(gs[0, j])
        aw = fig.add_subplot(gs[1, j], sharex=a)
        kd, Pd, sd = r[f"data_{name}"]
        a.errorbar(kd, sign * Pd, sd, fmt="o", ms=3.5, color=cF, lw=1, capsize=0, zorder=3,
                   label=f"FLAMINGO ({label})")
        a.plot(k_out, sign * r[f"nla_{name}"], color=cN, ls="--", lw=1.4,
               label=f"NLA, a_nla = {r['a_nla']:.3f} ± {r['s_a_nla']:.3f}")
        a.plot(k_out, sign * r[f"la_{name}"], color=cN, ls=":", lw=1.2, label="LA (same amplitude)")
        a.plot(k_out, sign * r[f"P_{name}"], color=cS, lw=2, label="stitched", zorder=4)
        if tng is not None:
            col = 4 if name == "dE" else 8
            a.plot(tng[:, 0], sign * tng[:, col], color=cT, lw=1.4, alpha=0.9,
                   label=f"TNG300 stitched (z = {ztng:.2f})")
        a.set_xscale("log")
        a.set_xlim(5e-3, 25)
        # symlog: an anti-aligned sample (negative a_nla) gives -P_dE < 0, which a log axis hides
        inr = (k_out > 5e-3) & (k_out < 25)
        hi = np.nanmax(np.abs(np.r_[sign * Pd, sign * r[f"P_{name}"][inr]]))
        if tng is not None:
            hi = max(hi, np.nanmax(np.abs(tng[(tng[:, 0] > 5e-3) & (tng[:, 0] < 25), 4 if name == "dE" else 8])))
        st = sign * r[f"P_{name}"][inr]
        if np.all(st > 0) and np.mean(sign * Pd > 0) > 0.8:      # all positive: plain log axis
            pos = sign * Pd[sign * Pd > 0]
            a.set_yscale("log")
            a.set_ylim(max(min(pos.min(), st.min()) * 0.3, hi * 1e-6), hi * 3)
        else:                                                     # sign changes: symlog
            small = np.nanmedian(np.abs(sd[kd > 1.0])) if np.any(kd > 1.0) else hi * 1e-4
            a.set_yscale("symlog", linthresh=max(small, hi * 1e-6))
            a.set_ylim(-hi * 3, hi * 3)
            a.axhline(0, color=ink2, lw=0.8)
        a.grid(True, color=grid, lw=0.6)
        a.set_ylabel(r"$-P_{\delta E}$ [(Mpc/h)$^3$]" if name == "dE" else r"$P_{EE}$ [(Mpc/h)$^3$]")
        a.set_title(("Density–shape " if name == "dE" else "Shape–shape ") +
                    r"($k_\perp$, shear units)", color=ink, loc="left")
        a.legend(frameon=True, framealpha=0.85, edgecolor="none", facecolor="#fcfcfb", fontsize=7, loc="lower left")
        plt.setp(a.get_xticklabels(), visible=False)
        aw.plot(k_out, r[f"w_{name}"], color=cS, lw=1.5)
        aw.axvline(kz, color=ink2, lw=0.8, ls=":")
        aw.set_ylim(-0.05, 1.05)
        aw.set_ylabel("w(k)")
        aw.set_xlabel("k  [h/Mpc]")
        aw.grid(True, color=grid, lw=0.6)
    lm = np.log10(np.array([mmin, mmax]) / 0.681 * 1e10)
    fig.suptitle(f"FLAMINGO L1_m8 snap {sn}, z = {z:.2f}: {int(info['ngal'])} galaxies, "
                 f"{lm[0]:.2f} < log M* < {lm[1]:.2f}, e_rms = {info['erms']:.3f}; "
                 f"NLA fit to dE at k = 0.05-0.3, k_t = {kz:.3f} h/Mpc (P_dE ≈ P_hE / b)",
                 color=ink, fontsize=9.5, x=0.01, ha="left")
    os.makedirs(os.path.join(HERE, "results", "plots"), exist_ok=True)
    png = os.path.join(HERE, "results", "plots", f"stitch_{which}_sn{sn}.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"PLOT sn{sn} {which}: a_nla = {r['a_nla']:.4f} +- {r['s_a_nla']:.4f}, EE/NLA^2 at low k = "
          f"{r['lowk_ratio_EE']:.2f} -> {png}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "all")
