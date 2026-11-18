#!/usr/bin/env python3
"""Red and blue galaxies in the halo model: the tables for hmc_rb_fit.py.

The FLAMINGO shape sample (all subhalos with M_* > 1e9 M_sun/h, n_g of
halo_sim.NGAL) has no colour split, so the red (quenched) fraction is taken
from halo-model fits to observed galaxies and applied to the HOD that
halo_sim.py calibrated to the FLAMINGO n_g and b_g at each snapshot.

Red fractions against halo mass M (M_200m, M_sun/h):
  centrals, model "zt" (fiducial):
      z <= 0.1   Zu & Mandelbaum 2016 (arXiv:1509.06758, Eq. 12, Table 2):
                 f_q^c = 1 - exp[-(M/10^12.20)^0.38]
      z = 0.36, 0.66, 0.88   Tinker et al. 2013 (arXiv:1308.2974, Table 2):
                 f_q^c at lg M = 10.8, 11.6, 12.4, 13.2, 14.0, linear in lg M
                 between the nodes, held flat outside;
      linear in z between these redshifts, held at z = 0.88 above it (no
      halo-mass-resolved measurement there; stated as an assumption).
  centrals, model "zm": Zu & Mandelbaum 2016 at every z (no evolution).
  satellites (both models): Zu & Mandelbaum 2016,
      f_q^s = 1 - exp[-(M/10^12.17)^0.15], at every z (Tinker et al. 2013
      find no significant evolution of the quenched satellite fraction).

Per snapshot and red-fraction model, stored:
  f_rc, f_bc, f_rs, f_bs   number fractions of red/blue centrals/satellites
                           in the sample (they add to 1);
  lgM_rc, lgM_bc           mean lg M of the halos of red and blue centrals;
  b_r, b_b                 linear bias of the red and blue galaxies;
  Pc_r, Pc_b               central 1-halo spectra (halo_central.py Pc_m)
                           restricted to red or blue centrals;
  SdI_r, SdI_b             satellite 1-halo GI (halo_ia.S_grid) of red or blue
                           satellites, on the amplitude grid A_GRID;
  SII_rr, SII_bb           satellite 1-halo II of each colour on A_GRID;
  SII_rb06                 the red-blue satellite II cross term for a blue
                           amplitude 0.6 a_1h (the ratio of the blue and red
                           satellite amplitudes of Fortuna et al. 2021,
                           Table 3: 0.0006 / 0.0010), on A_GRID in a_1h;
  SdI_b06, SII_bb06        the blue satellite terms at amplitude 0.6 a_1h;
  b_rc, b_bc, b_c          the mean linear bias of red, blue and all central
                           galaxies (number-weighted over their halos), for
                           the halo-based alignment of hmc_rb_fit.py (alignment
                           of a central = alignment of its halo, which scales
                           with the halo bias, times a colour-dependent
                           misalignment factor). `--light` adds only these
                           to an existing halo_redblue_tables.npz.
Checks printed per snapshot: n_g against halo_tables.npz, and the fractions
adding to 1; the red fraction of the whole sample against the observed range.

    ~/cocoa/Cocoa/.local/bin/python halo_redblue.py [--light]   -> halo_redblue_tables.npz
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_ia as H                                 # noqa: E402
from halo_central import u_E, sigma_E2              # noqa: E402

SIG_LGM, ALPHA = 0.25, 1.0
A_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 24)])   # as halo_sim.A_GRID
Q_BLUE = 0.6
# Zu & Mandelbaum 2016, Table 2 (fiducial): lg M_q [M_sun/h], mu
ZM_C, ZM_S = (12.20, 0.38), (12.17, 0.15)
# Tinker et al. 2013, Table 2: f_q of centrals at the halo-mass nodes
T13_LGM = np.array([10.8, 11.6, 12.4, 13.2, 14.0])
T13 = {0.36: np.array([0.05, 0.14, 0.54, 0.63, 0.77]),
       0.66: np.array([0.0, 0.07, 0.47, 0.68, 0.81]),
       0.88: np.array([0.0, 0.06, 0.43, 0.59, 0.76])}
Z_ZM = 0.1
MODELS = ("zt", "zm")


def zm16(M, lgMq, mu):
    return 1.0 - np.exp(-(M/10**lgMq)**mu)


def fq_central(M, z, model):
    if model == "zm":
        return zm16(M, *ZM_C)
    lgM = np.log10(M)
    nodes = [(Z_ZM, zm16(M, *ZM_C))] + [(zt, np.interp(lgM, T13_LGM, f)) for zt, f in T13.items()]
    zn = np.array([n[0] for n in nodes])
    if z <= zn[0]:
        return nodes[0][1]
    if z >= zn[-1]:
        return nodes[-1][1]
    j = int(np.searchsorted(zn, z))
    w = (z - zn[j - 1])/(zn[j] - zn[j - 1])
    return nodes[j - 1][1]*(1 - w) + nodes[j][1]*w


def fq_satellite(M):
    return zm16(M, *ZM_S)


def sat_terms(hm, Ns_x, G):
    """S_dI [n_a, nk] of the satellites with occupation Ns_x, gamma_hat G [n_a, nM, nk_in]."""
    w = hm.dndlnM[None, :, None]
    out = np.zeros((G.shape[0], hm.k.size))
    out[:, hm.kin] = trapezoid(w*(hm.M[None, :, None]/H.RHO_M)*hm.u[None, :, hm.kin]
                               *(Ns_x/hm.ng)[None, :, None]*np.abs(G), hm.lnM, axis=1)
    return out


def sat_ii(hm, Nx, Ny, Gx, Gy):
    w = hm.dndlnM[None, :, None]
    out = np.zeros((Gx.shape[0], hm.k.size))
    out[:, hm.kin] = trapezoid(w*((Nx*Ny)/hm.ng**2)[None, :, None]*Gx*Gy, hm.lnM, axis=1)
    return out


def central_bias(hm, z, m):
    """Mean linear bias of red, blue and all centrals at one snapshot."""
    qc = fq_central(hm.M, float(z), m)
    w = hm.dndlnM
    mean_b = lambda N: trapezoid(w*hm.b*N, hm.lnM)/trapezoid(w*N, hm.lnM)
    return mean_b(hm.Nc*qc), mean_b(hm.Nc*(1 - qc)), mean_b(hm.Nc)


def light():
    """Add b_rc, b_bc, b_c to the existing tables (no satellite grids)."""
    path = os.path.join(HERE, "halo_redblue_tables.npz")
    out = dict(np.load(path))
    tab = np.load(os.path.join(HERE, "halo_tables.npz"))
    zs, k_all, hod = tab["zs"], tab["k_all"], tab["hod"]
    assert np.allclose(out["zs"], zs)
    lin = H.Linear(list(zs))
    for m in MODELS:
        for kk in ("b_rc", "b_bc", "b_c"):
            out[f"{m}|{kk}"] = np.zeros(len(zs))
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=10)
        hm.set_hod(hod[s, 0], SIG_LGM, hod[s, 1], alpha=ALPHA)
        assert abs(hm.ng/hod[s, 3] - 1) < 1e-3, (hm.ng, hod[s, 3])
        for m in MODELS:
            b = central_bias(hm, z, m)
            for kk, v in zip(("b_rc", "b_bc", "b_c"), b):
                out[f"{m}|{kk}"][s] = v
            # the fractions of the full build must be reproduced
            qc = fq_central(hm.M, float(z), m)
            frc = trapezoid(hm.dndlnM*hm.Nc*qc, hm.lnM)/hm.ng
            assert abs(frc/out[f"{m}|f_rc"][s] - 1) < 1e-6, (frc, out[f"{m}|f_rc"][s])
        print(f"z {z:4.2f}: central bias red/blue/all (zt) {out['zt|b_rc'][s]:.2f}/{out['zt|b_bc'][s]:.2f}/"
              f"{out['zt|b_c'][s]:.2f}; galaxies b_g {hod[s, 4]:.2f}", flush=True)
    np.savez(path, **out)
    print("updated", path)


def main():
    if "--light" in sys.argv:
        return light()
    t0 = time.time()
    tab = np.load(os.path.join(HERE, "halo_tables.npz"))
    zs, k_all, hod = tab["zs"], tab["k_all"], tab["hod"]
    lin = H.Linear(list(zs))
    nsn, nk, na = len(zs), k_all.size, A_GRID.size
    out = {"zs": zs, "k_all": k_all, "a_grid": A_GRID, "q_blue": Q_BLUE, "models": np.array(MODELS)}
    keys = ("f_rc", "f_bc", "f_rs", "f_bs", "lgM_rc", "lgM_bc", "b_r", "b_b", "b_rc", "b_bc", "b_c")
    for m in MODELS:
        for kk in keys:
            out[f"{m}|{kk}"] = np.zeros(nsn)
        for kk in ("Pc_r", "Pc_b"):
            out[f"{m}|{kk}"] = np.zeros((nsn, nk))
        for kk in ("SdI_r", "SdI_b", "SII_rr", "SII_bb", "SII_rb06", "SdI_b06", "SII_bb06"):
            out[f"{m}|{kk}"] = np.zeros((nsn, na, nk))
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=1600)
        hm.set_hod(hod[s, 0], SIG_LGM, hod[s, 1], alpha=ALPHA)
        assert abs(hm.ng/hod[s, 3] - 1) < 1e-3, (hm.ng, hod[s, 3])
        G = hm.gamma_hat_grid(A_GRID)                     # [n_a, nM, nk_in]
        G06 = hm.gamma_hat_grid(Q_BLUE*A_GRID)
        UE = np.array([u_E(k_all, rs, c) for rs, c in zip(hm.rs, hm.c)])
        se2 = sigma_E2(hm.M, hm.nu, hm.lnM)[0]
        w = hm.dndlnM
        for m in MODELS:
            qc, qs = fq_central(hm.M, float(z), m), fq_satellite(hm.M)
            Nrc, Nbc, Nrs, Nbs = hm.Nc*qc, hm.Nc*(1 - qc), hm.Ns*qs, hm.Ns*(1 - qs)
            nint = lambda N: trapezoid(w*N, hm.lnM)
            fr = {kk: nint(N)/hm.ng for kk, N in (("f_rc", Nrc), ("f_bc", Nbc), ("f_rs", Nrs), ("f_bs", Nbs))}
            assert abs(sum(fr.values()) - 1) < 1e-6
            for kk, v in fr.items():
                out[f"{m}|{kk}"][s] = v
            out[f"{m}|lgM_rc"][s] = trapezoid(w*Nrc*hm.lnM, hm.lnM)/nint(Nrc)/np.log(10)
            out[f"{m}|lgM_bc"][s] = trapezoid(w*Nbc*hm.lnM, hm.lnM)/nint(Nbc)/np.log(10)
            out[f"{m}|b_r"][s] = trapezoid(w*hm.b*(Nrc + Nrs), hm.lnM)/nint(Nrc + Nrs)
            out[f"{m}|b_b"][s] = trapezoid(w*hm.b*(Nbc + Nbs), hm.lnM)/nint(Nbc + Nbs)
            for kk, v in zip(("b_rc", "b_bc", "b_c"), central_bias(hm, z, m)):
                out[f"{m}|{kk}"][s] = v
            for kk, N in (("Pc_r", Nrc), ("Pc_b", Nbc)):
                out[f"{m}|{kk}"][s] = 0.5*trapezoid(w[:, None]*(N/hm.ng*hm.M/H.RHO_M*se2)[:, None]*UE,
                                                    hm.lnM, axis=0)
            out[f"{m}|SdI_r"][s] = sat_terms(hm, Nrs, G)
            out[f"{m}|SdI_b"][s] = sat_terms(hm, Nbs, G)
            out[f"{m}|SdI_b06"][s] = sat_terms(hm, Nbs, G06)
            out[f"{m}|SII_rr"][s] = sat_ii(hm, Nrs, Nrs, G, G)
            out[f"{m}|SII_bb"][s] = sat_ii(hm, Nbs, Nbs, G, G)
            out[f"{m}|SII_bb06"][s] = sat_ii(hm, Nbs, Nbs, G06, G06)
            out[f"{m}|SII_rb06"][s] = sat_ii(hm, Nrs, Nbs, G, G06)
        # the split must add up to the one-population term exactly; the stored
        # halo_tables.npz (an earlier CAMB run) agrees with that to < 1e-3
        one = sat_terms(hm, hm.Ns, G)
        for m in MODELS:
            tot = out[f"{m}|SdI_r"][s] + out[f"{m}|SdI_b"][s]
            assert np.allclose(tot, one, rtol=1e-10, atol=0), "SdI split"
        nz = one[1:, hm.kin] != 0
        stored = float(np.max(np.abs(tab["SdI"][s][1:, hm.kin][nz]/one[1:, hm.kin][nz] - 1)))
        assert stored < 3e-3, stored
        f = {m: [out[f"{m}|{kk}"][s] for kk in ("f_rc", "f_bc", "f_rs", "f_bs")] for m in MODELS}
        print(f"z {z:4.2f}: red fraction (zt) {f['zt'][0] + f['zt'][2]:.3f} "
              f"[rc {f['zt'][0]:.3f} bc {f['zt'][1]:.3f} rs {f['zt'][2]:.3f} bs {f['zt'][3]:.3f}], "
              f"(zm) {f['zm'][0] + f['zm'][2]:.3f}; lgM rc/bc {out['zt|lgM_rc'][s]:.2f}/{out['zt|lgM_bc'][s]:.2f}; "
              f"b red/blue {out['zt|b_r'][s]:.2f}/{out['zt|b_b'][s]:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(HERE, "halo_redblue_tables.npz"), **out)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
