#!/usr/bin/env python3
"""Item 2: satellites aligned coherently with their host's major axis.

Satellite shapes gamma_s = a_s [E]_proj (E the host's shape tensor, as for
the central in halo_central.py), on top of the F21 radial term. Within a halo
the coherent part is a fixed spin-2 direction times the satellite profile,
so over random orientations E and B are equal:
  P_EE^coh = P_BB^coh = a_c a_s Pcs(k) + a_s^2 Pss(k),
    Pcs = (sigma_E^2/2) int dn 2 N_s/n_g^2 u_s(k)          (central x satellites, both orderings)
    Pss = (sigma_E^2/2) int dn N_s^2/n_g^2 u_s(k)^2         (satellite pairs, Poisson)
  P_dE^coh = -a_s Psm(k),
    Psm = (sigma_E^2/2) int dn (N_s/n_g)(M/rho_m) u_s(k) u_E(k)   (satellites x own triaxial halo)
(the central's self pair is shape noise; radial x coherent cross terms are
second order in the halo ellipticity and dropped). u_s = NFW with the DM
concentration; sigma_E^2(M) as halo_central.py (Allgood et al. 2006).
Output halo_satcoh_tables.npz on k_all: Pcs, Pss, Psm [nsn, nk].
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_ia as H            # noqa: E402
import halo_central as HC      # noqa: E402


def main():
    t0 = time.time()
    ST = HC.ST
    zs, k_all = ST["zs"], ST["k_all"]
    lin = H.Linear(list(zs))
    out = {n: np.zeros((len(zs), k_all.size)) for n in ("Pcs", "Pss", "Psm")}
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=10)
        hm.set_hod(ST["hod"][s, 0], HC.SIG_LGM, ST["hod"][s, 1], alpha=HC.ALPHA)
        se2 = HC.sigma_E2(hm.M, hm.nu, hm.lnM)[0]
        UE = np.array([HC.u_E(k_all, rs, c) for rs, c in zip(hm.rs, hm.c)])
        w = (hm.dndlnM*se2/2)[:, None]
        Ns = (hm.Ns/hm.ng)[:, None]
        out["Pcs"][s] = trapezoid(w*2*Ns/hm.ng*hm.u, hm.lnM, axis=0)
        out["Pss"][s] = trapezoid(w*Ns**2*hm.u**2, hm.lnM, axis=0)
        out["Psm"][s] = trapezoid(w*Ns*(hm.M/H.RHO_M)[:, None]*hm.u*UE, hm.lnM, axis=0)
        ik = [int(np.argmin(np.abs(k_all - kk))) for kk in (0.01, 0.3, 1.0, 3.0)]
        print(f"z {z:4.2f}: Pcs {np.array2string(out['Pcs'][s, ik], precision=3)}  Pss {np.array2string(out['Pss'][s, ik], precision=3)}"
              f"  Psm {np.array2string(out['Psm'][s, ik], precision=3)}  [{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(HERE, "halo_satcoh_tables.npz"), zs=zs, k_all=k_all, **out)


if __name__ == "__main__":
    main()
