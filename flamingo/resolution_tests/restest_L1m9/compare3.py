#!/usr/bin/env python3
"""High-mass resolution test: a_nla of L1_m8, L1_m9 (same initial phases as L1_m8) and L2p8_m9 at z = 0,
with the identical SOAP tensor (BoundSubhalo/StellarInertiaTensorNoniterative, 1 R_half) and cut
(M* > 4.05e11 Msun). Adapted from ia_power_flamingo/restest_soap_m9_vs_m8/compare.py; L1_m8 and L2p8_m9
spectra are copied from that run. Writes results.json (per box: k, P_dE, sigma, a_nla, e_rms, bias)
and restest_L1m9_L1m8_L2p8m9.png (quick-look).

Same recipe as run_L1_m8_particles/plot_stitch_flamingo.py (TNG stitch.py --amp lowk):
P_dE ~= 1.5 P_hE^(0) / b(k) / (2R), fit A P_NL at k = 0.05-0.3, a_nla = A / (-C1rho Om / D).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/yijiezhu/Downloads/ia_halofit_v2")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cosmology import Cosmology  # noqa: E402
from stitch import camb_power  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
results = {}
COSMO = Cosmology(h=0.681, Om=0.304611, Ob=0.0486, ns=0.967, sigma8=0.807, linear_source="camb")
BOX = {"L1_m8": 681.0, "L1_m9": 681.0, "L2p8_m9": 1906.8}
Pnl, _ = camb_power(COSMO, [0.0], kmax=30.0)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), facecolor="#fcfcfb")
cols = {"L1_m8": "#2a78d6", "L1_m9": "#1baf7a", "L2p8_m9": "#eb6834"}
print(f"{'box':8s} {'N_gal':>9s} {'e_rms':>6s} {'bias':>6s} {'a_nla':>8s} {'err':>6s}")
for box, L in BOX.items():
    d = os.path.join(HERE, box)
    vals = open(os.path.join(d, "Galaxy_DS_Field", "hinfo_fixng_ms_sub_red_sn78_nfold1.dat")).read().split("\n")[1].split()
    ngal, erms = float(vals[2]), 0.5 * (float(vals[3]) + float(vals[4]))
    f = 1 / (2 * (1 - erms ** 2))
    hh = np.loadtxt(os.path.join(d, "PS", "Phh_sn78_nfold1.dat"))
    he = np.loadtxt(os.path.join(d, "PS", "PhE0_sn78_nfold1.dat"))
    k = hh[:, 0]
    b = np.sqrt(np.clip(hh[:, 1] - L ** 3 / ngal, 1e-30, None) / Pnl(k, 0.0))
    p, s = 1.5 * he[:, 1] / b * f, 1.5 * he[:, 2] / b * f
    m = (k >= 0.05) & (k <= 0.3)
    w = 1 / s[m] ** 2
    A = np.sum(w * p[m] * Pnl(k[m], 0.0)) / np.sum(w * Pnl(k[m], 0.0) ** 2)
    sA = 1 / np.sqrt(np.sum(w * Pnl(k[m], 0.0) ** 2))
    norm = -0.0134 * COSMO.Om
    a, sa = A / norm, sA / abs(norm)
    results[box] = {"k": k.tolist(), "pde": p.tolist(), "sigma": s.tolist(), "ngal": ngal, "erms": erms,
                    "bias": float(np.mean(b[m])), "anla": float(a), "anla_err": float(sa)}
    print(f"{box:8s} {int(ngal):9d} {erms:6.3f} {np.mean(b[m]):6.3f} {a:8.3f} {sa:6.3f}")
    kk = np.logspace(-2.3, 0.4, 200)
    ax[0].errorbar(k, -p, s, fmt="o", ms=3.5, color=cols[box], lw=1, label=f"{box}: a_nla = {a:.2f} ± {sa:.2f}")
    ax[0].plot(kk, -A * Pnl(kk, 0.0), color=cols[box], ls="--", lw=1.2)
    ax[1].errorbar(k, b, color=cols[box], fmt="o-", ms=3, lw=1.2, label=f"{box}")
for a_ in ax:
    a_.set_xscale("log"); a_.set_xlabel("k  [h/Mpc]"); a_.grid(True, color="#e6e5e0", lw=0.6)
    a_.spines[["top", "right"]].set_visible(False); a_.set_facecolor("#fcfcfb")
ax[0].set_yscale("log"); ax[0].set_ylabel(r"$-P_{\delta E}$ [(Mpc/h)$^3$] ($k_\perp$, shear)")
ax[0].set_title("Same SOAP tensor and cut; L1_m8 and L1_m9 share initial phases (dashed: NLA fit)", loc="left", fontsize=10)
ax[0].legend(frameon=False, fontsize=8)
ax[1].set_ylabel("b(k)"); ax[1].set_ylim(0, 6); ax[1].set_title("Galaxy bias", loc="left", fontsize=10)
ax[1].legend(frameon=False, fontsize=8)
fig.suptitle("FLAMINGO z = 0, M* > 4.05e11 Msun, SOAP StellarInertiaTensorNoniterative (1 R_half)", x=0.01,
             ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "restest_L1m9_L1m8_L2p8m9.png"), dpi=130)
print("wrote", os.path.join(OUT, "restest_L1m9_L1m8_L2p8m9.png"))

json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
