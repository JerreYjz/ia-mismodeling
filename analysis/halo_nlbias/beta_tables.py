#!/usr/bin/env python3
"""Nonlinear halo bias beta_NL(k, M1, M2, z) from Dark Emulator at the
FLAMINGO D3A cosmology, as in Mead & Verde 2021 (arXiv:2011.08858):

    beta_NL(k, M1, M2) = P_hh(k | M1, M2) / [B_12 P_lin(k)] - 1,

with P_hh the emulated halo-halo spectrum of two mass bins (Nishimichi et al.
2019, arXiv:1811.09504; halo exclusion included), P_lin the emulator's linear
spectrum and B_12 = <P_hh(k | M1, M2)/P_lin(k)> over 0.01 <= k <= 0.03 h/Mpc,
the pair's own large-scale amplitude, so that beta_NL -> 0 at low k for every
pair. (With the product b_E(M1) b_E(M2) of the auto-spectrum biases instead,
pairs of very different mass keep an offset of up to 6 percent at k = 0.01:
an inconsistency of the emulator's cross terms, stored as r_lowk.)

Mass bins: the halos between two consecutive cumulative-density thresholds
n(>M) = 10^ld, ld from -2.9 to -5.6 in NM_E steps; the spectrum of two bins
is the density-weighted combination of the four threshold spectra. Finite
bins avoid the +-1 percent mass differences of get_phh_mass, which are noisy
near the emulator's lowest mass. Bin mass: that of the geometric-mean density.

Coverage of the emulator, and what is done outside it:
  * mass: the direct emulation covers n(>M) from 10^-2.5 to 10^-5.75
    (h/Mpc)^3. Bins that start above n = 10^-2.9 are unstable at z >= 1:
    beta_NL of the lowest bin at k = 0.1 moves between -0.01 and -0.13 with
    the bin edges while every other bin is stable to 0.01 (the edge of the
    training range). The grid therefore uses 10^-2.9 to 10^-5.6, i.e. M_200m
    from about 10^12.8 to 10^14.4 M_sun/h at z = 0.3 and 10^12.6 to 10^14.0 at
    z = 1. Outside the grid beta_NL is held at the value of the nearest bin
    (the choice of Mead & Verde for low masses); see halo_nlbias.py.
  * redshift: 0 <= z <= 1.476 (21 snapshots). For a target z above 1.476 the
    table of z_E = 1.476 is assigned to the masses of the same peak height
    nu = delta_c/sigma(M, z) at the target z (lgM_E, the fiducial), or to the
    same masses (lgM_same, a bracket). Both are extrapolations.
  * cosmology: wCDM box (omega_b, omega_c, Omega_de, ln 10^10 A_s, n_s, w);
    FLAMINGO D3A (0.02254, 0.1187, 0.694, 3.044, 0.967, -1) is inside it.
    The emulator's neutrino mass is fixed at 0.06 eV, FLAMINGO's value.

Output beta_tables.npz: zs, k [nk], lgM_E [nz, NM_E], beta [nz, NM_E, NM_E, nk],
bE [nz, NM_E], r_lowk [nz, NM_E, NM_E], lgM_same [nz, NM_E] (NaN for
z <= 1.476). Runs in about 15 seconds.
    python beta_tables.py            # the nine core FLAMINGO snapshots
"""
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.normpath(os.path.join(HERE, '..', 'roman_dchi2', 'v6'))
sys.path.insert(0, V6)
import halo_ia as H  # noqa: E402

# the core FLAMINGO snapshots (analysis/roman_dchi2/v3/v3.npz, core)
ZS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0)
# the highest redshift of the emulator's snapshots
Z_EMU = 1.476
# k grid of the tables [h/Mpc]: log spaced, covering the FLAMINGO bins
K = np.logspace(-2.0, np.log10(20.0), 140)
# the k range that defines the large-scale bias b_E [h/Mpc]
K_BIAS = (0.01, 0.03)
# number of masses on the emulator grid, and its range in log10 n(>M)
NM_E = 9
LOGDENS = (-2.9, -5.6)


def cosmology():
    '''The FLAMINGO D3A parameters in the emulator's basis.'''
    h = H.h
    omnu = H.MNU/93.14/h**2
    return np.array([[H.OMB*h*h, (H.OMM - H.OMB - omnu)*h*h, 1 - H.OMM,
                      np.log(1e10*H.AS), H.NS, -1.0]])


def dens_to_mass(emu, z, logdens):
    '''Masses log10 M whose cumulative density n(>M) is 10^logdens at z.'''
    lgm = np.linspace(11.0, 16.0, 101)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ld = np.log10([emu.mass_to_dens(10**m, z) for m in lgm])
    ok = np.isfinite(ld)
    return np.interp(-np.asarray(logdens), -ld[ok], lgm[ok])


def beta_at(emu, z):
    '''beta_NL [NM_E, NM_E, nk], b_E [NM_E], the bin masses lgM [NM_E] and
    the large-scale cross-correlation coefficients r_lowk [NM_E, NM_E]
    at z (z <= Z_EMU), from mass bins between consecutive thresholds.

    Halos in bin i are those of threshold i (n(>M) = n_i) minus those of
    threshold i + 1, so their overdensity is (n_i d_i - n_{i+1} d_{i+1}) /
    (n_i - n_{i+1}) and P_hh of two bins is the same combination of the
    threshold spectra. The bin mass is that of the geometric-mean density.
    Bins of finite width avoid the +-1 percent mass differences of
    get_phh_mass, which are noisy near the emulator's lowest mass.'''
    plin = emu.get_pklin_from_z(K, z)
    ld = np.linspace(LOGDENS[0], LOGDENS[1], NM_E + 1)
    n = 10**ld
    ne = ld.size
    T = np.zeros((ne, ne, K.size))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for p in range(ne):
            for q in range(p, ne):
                T[p, q] = T[q, p] = emu.get_phh(K, ld[p], ld[q], z)
    w = np.zeros((NM_E, ne))
    for i in range(NM_E):
        w[i, i], w[i, i + 1] = n[i]/(n[i] - n[i + 1]), -n[i + 1]/(n[i] - n[i + 1])
    P = np.einsum('ip,jq,pqk->ijk', w, w, T)
    lowk = (K >= K_BIAS[0]) & (K <= K_BIAS[1])
    B12 = np.mean(P[..., lowk]/plin[lowk], axis=-1)
    bE = np.sqrt(np.diag(B12))
    beta = P/(B12[..., None]*plin) - 1.0
    lgM = dens_to_mass(emu, z, 0.5*(ld[:-1] + ld[1:]))
    return beta, bE, lgM, B12/np.outer(bE, bE)


def nu_of(lin, z, lgM):
    '''Peak height delta_c/sigma(M, z) of masses lgM (CAMB, CDM + baryons).'''
    kh, pk = lin.P(z)
    R = (3*10**np.asarray(lgM)/(4*np.pi*H.RHO_CB))**(1/3)
    return H.DELTA_C/np.sqrt(H.sigma2_R(kh, pk, R))


def main():
    from dark_emulator import darkemu
    t0 = time.time()
    emu = darkemu.base_class()
    emu.set_cosmology(cosmology())
    lin = H.Linear(list(ZS) + [Z_EMU])
    nz = len(ZS)
    beta = np.full((nz, NM_E, NM_E, K.size), np.nan)
    r_lowk = np.full((nz, NM_E, NM_E), np.nan)
    lgM_same = np.full((nz, NM_E), np.nan)
    bE = np.full((nz, NM_E), np.nan)
    lgM_E = np.full((nz, NM_E), np.nan)
    for s, z in enumerate(ZS):
        if z <= Z_EMU + 1e-9:
            beta[s], bE[s], lgM_E[s], r_lowk[s] = beta_at(emu, z)
        else:
            # the table of Z_EMU, assigned to the masses of the same peak
            # height at z; the bracket (lgM_same) assigns it to the same masses
            b_e, bE[s], lg_e, r_lowk[s] = beta_at(emu, Z_EMU)
            lgm_fine = np.linspace(9.0, 16.0, 701)
            lgM_E[s] = np.interp(nu_of(lin, Z_EMU, lg_e), nu_of(lin, z, lgm_fine), lgm_fine)
            beta[s] = b_e
            lgM_same[s] = lg_e
        ik = [int(np.argmin(np.abs(K - kk))) for kk in (0.1, 0.3, 0.5, 1.0, 2.0)]
        print(f'z {z:4.2f}: lgM_E {lgM_E[s, 0]:.2f} to {lgM_E[s, -1]:.2f}, b_E {bE[s, 0]:.2f} to {bE[s, -1]:.2f}; '
              f'beta(lowest, lowest) at k 0.1,0.3,0.5,1,2: {np.array2string(beta[s, 0, 0, ik], precision=3)}; '
              f'beta(lowest, highest): {np.array2string(beta[s, 0, -1, ik], precision=3)}; '
              f'max |beta| below k = 0.02: {np.abs(beta[s][..., K < 0.02]).max():.3f}  [{time.time() - t0:.0f} s]',
              flush=True)
    np.savez(os.path.join(HERE, 'beta_tables.npz'), zs=np.array(ZS), k=K, lgM_E=lgM_E, beta=beta, bE=bE,
             r_lowk=r_lowk, lgM_same=lgM_same, z_emu=Z_EMU, cosmo=cosmology())


if __name__ == '__main__':
    main()
