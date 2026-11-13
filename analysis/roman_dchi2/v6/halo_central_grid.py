#!/usr/bin/env python3
"""Shape freedoms of the central 1-halo term (halo_central.py), on the data k:
  radial ellipticity profile  E(r) = E (r/r_vir)^p     (p > 0: rounder inside)
  mass-dependent alignment    a_c(M) = a_c (M/10^13.5)^beta
u_E with the profile: (k r_s / 2 m(c)) int_0^c (x/c)^p x^2/(1+x)^2 j_1(k r_s x) dx
(the first-order quadrupole of rho(r_e) with a radius-dependent E; the
d E/d ln r term is dropped, first order in p E). Output halo_central_grid.npz:
Pc[nsn, n_p, n_beta, nk_data] (matter version, Allgood sigma_E^2)."""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_ia as H            # noqa: E402
import halo_central as HC      # noqa: E402

PS = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
BETAS = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])


def main():
    t0 = time.time()
    ST = HC.ST
    zs, kd = ST["zs"], ST["kd"]
    lin = H.Linear(list(zs))
    Pc = np.zeros((len(zs), PS.size, BETAS.size, kd.size))
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), kd, nM=140, nx=10)
        hm.set_hod(ST["hod"][s, 0], HC.SIG_LGM, ST["hod"][s, 1], alpha=HC.ALPHA)
        se2 = HC.sigma_E2(hm.M, hm.nu, hm.lnM)[0]
        for ip, p in enumerate(PS):
            UE = np.zeros((hm.M.size, kd.size))
            for iM, (rs, c) in enumerate(zip(hm.rs, hm.c)):
                x = np.concatenate([[0.0], np.geomspace(1e-5*c, c, 6000)])
                w = (x/c)**p*x**2/(1 + x)**2 if p >= 0 else np.where(x > 0, (np.maximum(x, 1e-30)/c)**p, 0)*x**2/(1 + x)**2
                I = trapezoid(w[None, :]*HC.j1_small(np.outer(kd*rs, x)), x, axis=1)
                UE[iM] = kd*rs/(2*hm.mc[iM])*I
            for ib, be in enumerate(BETAS):
                wm = hm.dndlnM*hm.Nc/hm.ng*hm.M/H.RHO_M*se2*(hm.M/10**13.5)**be
                Pc[s, ip, ib] = 0.5*trapezoid(wm[:, None]*UE, hm.lnM, axis=0)
        print(f"z {z:4.2f} done [{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(HERE, "halo_central_grid.npz"), zs=zs, kd=kd, ps=PS, betas=BETAS, Pc=Pc)


if __name__ == "__main__":
    main()
