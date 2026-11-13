#!/usr/bin/env python3
"""Numerical comparison of the user's ia_halofit_v2 (TNG package) with v6
(halo_ia.py) at the FLAMINGO cosmology, z = 0.5, 1, 2, same mass range.

Compared, piece by piece:
  dn/dlnM, b(nu), c(M)            (both claim Tinker10 / Bhattacharya13)
  I2(k) [theirs] vs I1m(k) [v6]    (2-halo matter kernel, -> 1 at k -> 0)
  gamma_hat(k|M) shapes            (theirs: l = 2 only, x^2/(1+x)^2 weight,
                                    no core; v6: l = 2,4,6, x/(1+x)^2, halo.c)
  1-halo templates (same HOD)      (theirs normalised by nbar_s, v6 by n_g)
Writes compare_pkg.txt and fig_compare_pkg.{png,pdf}.
    python compare_pkg.py <path to ia_halofit_v2>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PKG = sys.argv[1]
sys.path.insert(0, PKG)
import halo_ia as H                  # noqa: E402
from cosmology import Cosmology      # noqa: E402  (user package)
from halo_model import HaloModel     # noqa: E402  (user package)

out = open(os.path.join(HERE, "compare_pkg.txt"), "w")


def say(s):
    print(s, flush=True)
    out.write(s + "\n")


import camb                          # noqa: E402
h = H.h
pars = camb.CAMBparams()
pars.set_cosmology(H0=H.H0, ombh2=H.OMB*h**2, omch2=(H.OMM - H.OMB - H.OMNU)*h**2, mnu=H.MNU, num_massive_neutrinos=1, tau=H.TAU)
pars.InitPower.set_params(As=H.AS, ns=H.NS)
pars.set_matter_power(redshifts=[0.0], kmax=10.0)
s8 = float(camb.get_results(pars).get_sigma8_0())
say(f"FLAMINGO D3A: Om {H.OMM}, Ob {H.OMB}, h {h}, ns {H.NS}, sigma8 (CAMB, total matter) {s8:.4f}")
cos = Cosmology(Om=H.OMM, Ob=H.OMB, h=h, ns=H.NS, sigma8=s8, linear_source="camb")
lin = H.Linear([0.5, 1.0, 2.0])
k = np.geomspace(0.02, 30, 80)
LGM = (10.0, 15.5)
hmT = HaloModel(cos, Mmin=10**LGM[0], Mmax=10**LGM[1], nM=140)
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for iz, z in enumerate((0.5, 1.0, 2.0)):
    hv = H.Halo(lin, z, k, nM=140, lgM=LGM, nx=1600)
    # --- mass function, bias, concentration on a common grid
    MT = hmT.M
    dnT = hmT.dndlnM(z)
    dnV = np.exp(np.interp(np.log(MT), np.log(hv.M), np.log(hv.dndlnM)))
    bT = hmT.bias(hmT.nu_grid(z))
    bV = np.interp(np.log(MT), np.log(hv.M), hv.b)
    cT = hmT._grid_conc(z)
    cV = np.interp(np.log(MT), np.log(hv.M), hv.c)
    sel = [np.argmin(np.abs(np.log10(MT) - lg)) for lg in (11, 12, 13, 14, 15)]
    say(f"\nz = {z}: ratios theirs/v6 at lg M = 11..15")
    say("  dn/dlnM " + " ".join(f"{dnT[i]/dnV[i]:.3f}" for i in sel))
    say("  bias    " + " ".join(f"{bT[i]/bV[i]:.3f}" for i in sel))
    say("  conc    " + " ".join(f"{cT[i]/cV[i]:.3f}" for i in sel))
    # --- 2-halo matter kernel
    I2 = hmT.I2_2h(k, z)
    wb = hv.dndlnM*hv.b*hv.M/H.RHO_CB
    from scipy.integrate import trapezoid
    I1m = trapezoid(wb[:, None]*hv.u, hv.lnM, axis=0) + (1 - trapezoid(wb, hv.lnM))
    I1n = trapezoid(wb[:, None]*hv.u, hv.lnM, axis=0)/trapezoid(wb, hv.lnM)
    ks = [np.argmin(np.abs(k - kk)) for kk in (0.1, 0.3, 1.0, 3.0)]
    say("  k = 0.1, 0.3, 1, 3:  I2 (theirs) " + " ".join(f"{I2[i]:.3f}" for i in ks) +
        " | I1m v6 (missing mass at u=1) " + " ".join(f"{I1m[i]:.3f}" for i in ks) +
        " | v6 integral / its own k->0 value (their normalisation) " + " ".join(f"{I1n[i]:.3f}" for i in ks))
    PlT = cos.P_lin(k, z)
    PlV = lin.Ptot(z, k)
    say("  P_lin theirs/v6 at k = 0.1, 0.3, 1, 3: " + " ".join(f"{PlT[i]/PlV[i]:.3f}" for i in ks))
    a = ax[0, iz]
    a.plot(k, I2, label="ia_halofit_v2: $I_2$ (/bias_norm)")
    a.plot(k, I1m, "--", label="v6: $I_{1m}$ (missing mass, $u=1$)")
    a.plot(k, I1n, ":", label="v6 integral, normalised as $I_2$")
    a.set_xscale("log")
    a.set_title(f"2-halo matter kernel, z = {z} (mass range $10^{{{LGM[0]:g}}}$ to $10^{{{LGM[1]:g}}}$)", fontsize=9)
    a.set_xlabel(r"$k$ [$h$/Mpc]")
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5)
    # --- satellite kernel shapes for three masses (normalised at low k by k^2)
    a = ax[1, iz]
    G = hv.gamma_hat_grid([0.0])[0]          # a1h -> 0: core at r_floor
    for col, lg in zip(("C0", "C1", "C2"), (12, 13, 14)):
        iM = int(np.argmin(np.abs(np.log10(hv.M) - lg)))
        gT = hmT.gamma_hat_sat(k, hv.M[iM], z, -2.0, 0.3, c=hv.c[iM])
        gV = G[iM]
        nT, nV = gT[0]/k[0]**2, gV[0]/k[0]**2
        a.plot(k, gT/nT/k**2, color=col, label=f"theirs, lg M = {lg}")
        a.plot(k, gV/nV/k**2, color=col, ls="--", label=f"v6 (halo.c), lg M = {lg}")
        kh = [np.argmin(np.abs(k - kk)) for kk in (0.3, 1.0, 3.0, 10.0)]
        say(f"  gamma_hat lg M = {lg}: ratio theirs/v6 after matching the k^2 limit, at k = 0.3, 1, 3, 10: " +
            " ".join(f"{(gT[i]/nT)/(gV[i]/nV):.3f}" for i in kh) + f"; raw amplitude ratio at low k {nT/nV:.3g}")
    a.set_xscale("log")
    a.set_title(r"satellite kernel $\hat\gamma(k|M)/k^2$, normalised at low $k$", fontsize=9)
    a.set_xlabel(r"$k$ [$h$/Mpc]")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_compare_pkg.png"), dpi=220)
fig.savefig(os.path.join(HERE, "fig_compare_pkg.pdf"))
out.close()
