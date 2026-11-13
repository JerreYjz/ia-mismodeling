#!/usr/bin/env python3
"""Halo-model galaxy bias, so the FLAMINGO P_hE fits need no measured b(k).

  I_g(k)  = int dlnM dn/dlnM b(M) [N_c + N_s u(k|M)]/n_g        (galaxy 2-halo integral, -> b_g at k -> 0)
  P_gg^HM = I_g^2 P_lin + int dn [2 N_c N_s u + N_s^2 u^2]/n_g^2   (check against FLAMINGO P_hh)
  b_HM(k) = (P_gg^HM / P_mm^HM)^(1/2), P_mm^HM = I_1^2 P_lin + int dn (M/rho)^2 u^2
The HOD is the one calibrated per snapshot to FLAMINGO's n_g and large-scale
b_g (two numbers, sample properties), as everywhere else.
Output halo_gbias_tables.npz on k_all: Ig, Pgg, Pmm, bHM [nsn, nk]."""
import os
import sys

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_ia as H            # noqa: E402
import halo_central as HC      # noqa: E402


def main():
    ST = HC.ST
    zs, k_all = ST["zs"], ST["k_all"]
    lin = H.Linear(list(zs))
    out = {n: np.zeros((len(zs), k_all.size)) for n in ("Ig", "Pgg", "Pmm", "bHM")}
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=10)
        hm.set_hod(ST["hod"][s, 0], HC.SIG_LGM, ST["hod"][s, 1], alpha=HC.ALPHA)
        w = hm.dndlnM[:, None]
        Nc, Ns = hm.Nc[:, None], hm.Ns[:, None]
        Ig = trapezoid(w*hm.b[:, None]*(Nc + Ns*hm.u), hm.lnM, axis=0)/hm.ng
        PL = ST["Plin"][s]
        Pgg = Ig**2*PL + trapezoid(w*(2*Ns*hm.u + Ns**2*hm.u**2), hm.lnM, axis=0)/hm.ng**2
        Pmm = ST["I1m"][s]**2*PL + trapezoid(w*((hm.M/H.RHO_M)[:, None]*hm.u)**2, hm.lnM, axis=0)
        out["Ig"][s], out["Pgg"][s], out["Pmm"][s], out["bHM"][s] = Ig, Pgg, Pmm, np.sqrt(Pgg/Pmm)
        print(f"z {z:4.2f}: b_g {hm.bg:.3f}  I_g(0.01) {np.interp(0.01, k_all, Ig):.3f}", flush=True)
    np.savez(os.path.join(HERE, "halo_gbias_tables.npz"), zs=zs, k_all=k_all, **out)


if __name__ == "__main__":
    main()
