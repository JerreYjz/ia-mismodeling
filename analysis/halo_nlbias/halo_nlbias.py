#!/usr/bin/env python3
"""Halo-model 2-halo integrals with nonlinear halo bias (Mead & Verde 2021,
arXiv:2011.08858) for the intrinsic-alignment model of halo_nlbias.tex.

Every 2-halo spectrum between two fields X and Y is

    P_XY^2h(k) = P_lin(k) int dlnM1 dlnM2 n(M1) n(M2) b(M1) b(M2)
                 X(M1, k) Y(M2, k) [1 + beta_NL(k, M1, M2)]
               = P_lin(k) [I_X(k) I_Y(k) + I^beta_XY(k)],

with n = dn/dlnM (Tinker et al. 2010), b the Tinker bias and the per-halo
kernels (normalised so that int n b X dlnM is the field's large-scale bias):

    matter             X_m = (M/rho_cb) u(k|M)        (+ the mass below the grid)
    galaxies           X_g = [N_c + N_s u(k|M)]/n_g
    central shapes     X_c = N_c/(n_c b_c)            "bias-weighted": shape bias b_K(M) prop. to b(M)
                       X_c = N_c/(n_c b(M))           "tidal": b_K independent of M (variant T)

(the shape kernels are per unit -A1 C1bar, so int n b X_c = 1 and A1 is the
large-scale NLA amplitude per galaxy, as in HMc). The halo-model pieces,
the HOD per snapshot and the mass below the grid follow halo_ia.py and
halo_strict_tables.npz of analysis/roman_dchi2/v6 exactly.

beta_NL comes from beta_tables.npz (Dark Emulator). On the halo-model mass
grid it is interpolated linearly in log M and held at the nearest emulator
bin outside the emulator's mass range; in k it is interpolated in log k and
set to 0 below k = 0.01 h/Mpc. The mass below the grid (10^9 M_sun/h) is
put in the lowest bin with u = 1, as in halo_ia.strict_grid.

Output nlbias_tables.npz, all on the k grid of halo_central_tables.npz
(k_all, which contains the FLAMINGO bins), per snapshot [nz, nk]:
    Plin, Plin_v6          CAMB linear CDM + baryon spectrum, here and as stored in v6
    I1                     matter 2-halo integral (as I1m of v6)
    Ib_cm, Ib_cc           beta parts of central-shape x matter and x central-shape
    Ib_cm_T, Ib_cc_T       the same for the "tidal" weighting
    Ib_cm_lo, ...          the lower bracket: beta_NL = 0 below the emulator's masses
    Ib_cm_same, ...        the bracket for z > 1.476 (same masses, see beta_tables)
    bg_k, Ib_gc, Ib_gg     galaxy bias integral and beta parts (galaxy x shape, galaxy x galaxy)
    Pgg_1h                 galaxy 1-halo (Poisson satellites)
    Ib_mm, Pmm_1h          matter auto, for the check against P_NL
    Ib_gm, Pgm_1h          galaxy x matter, for the check against FLAMINGO's P_gm
Runs in about 1 minute.
    python halo_nlbias.py
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.normpath(os.path.join(HERE, '..', 'roman_dchi2', 'v6'))
sys.path.insert(0, V6)
import halo_ia as H  # noqa: E402

# the HOD of halo_strict_tables (sigma_lgM and the satellite slope)
SIG_LGM, ALPHA_SAT = 0.25, 1.0
# halo-model mass grid (as halo_central.py)
NM = 140
# below this k beta_NL is set to 0 [h/Mpc]
K_BETA_MIN = 0.01


def beta_on_grid(bt, s, lgM, k, which='lgM_E', low='hold'):
    '''beta_NL [nM, nM, nk] of snapshot s on the halo-model masses lgM and the
    wavenumbers k: linear in log M with the ends held, linear in log k. With
    low = 'zero', beta_NL is 0 whenever either mass is below the emulator's
    lowest bin (the lower bracket of the low-mass extrapolation).'''
    lgE = bt[which][s]
    bk = bt['beta'][s]                                    # [NE, NE, nkE]
    lk = np.log(bt['k'])
    # in k first (small), then in mass
    bkk = np.zeros(bk.shape[:2] + (k.size,))
    for i in range(bk.shape[0]):
        for j in range(bk.shape[1]):
            bkk[i, j] = np.interp(np.log(k), lk, bk[i, j])
    bkk[..., k < K_BETA_MIN] = 0.0
    x = np.clip(lgM, lgE[0], lgE[-1])
    j = np.clip(np.searchsorted(lgE, x) - 1, 0, lgE.size - 2)
    w = (x - lgE[j])/(lgE[j + 1] - lgE[j])
    W = np.zeros((lgM.size, lgE.size))
    W[np.arange(lgM.size), j] = 1 - w
    W[np.arange(lgM.size), j + 1] += w
    out = np.einsum('ap,bq,pqk->abk', W, W, bkk)
    if low == 'zero':
        below = lgM < lgE[0]
        out[below] = 0.0
        out[:, below] = 0.0
    return out


def tables(hm, beta, I0):
    '''The integrals of one snapshot from a Halo with its HOD set, beta_NL on
    its mass grid and the mass fraction I0 resolved by the grid.'''
    lnM, n, b, u, M = hm.lnM, hm.dndlnM, hm.b, hm.u, hm.M
    dl = np.gradient(lnM)                                  # trapezoid-like weights
    dl[0] *= 0.5
    dl[-1] *= 0.5
    nc = trapezoid(n*hm.Nc, lnM)
    bc = trapezoid(n*hm.Nc*b, lnM)/nc
    # kernels times n b dlnM, [nM, nk] (or [nM]); the matter kernel gets the
    # mass below the grid in its lowest bin with u = 1
    Km = (n*b*M/H.RHO_CB*dl)[:, None]*u
    Km[0] += 1 - I0
    Kc = n*b*hm.Nc/(nc*bc)*dl
    KcT = n*hm.Nc/nc*dl
    Kg = (n*b*dl)[:, None]*(hm.Nc[:, None] + hm.Ns[:, None]*u)/hm.ng
    out = {
        'I1': Km.sum(0), 'bg_k': Kg.sum(0), 'bc': bc, 'nc_over_ng': nc/hm.ng,
        'Ib_cm': np.einsum('a,abk,bk->k', Kc, beta, Km),
        'Ib_cc': np.einsum('a,abk,b->k', Kc, beta, Kc),
        'Ib_cm_T': np.einsum('a,abk,bk->k', KcT, beta, Km),
        'Ib_cc_T': np.einsum('a,abk,b->k', KcT, beta, KcT),
        'Ib_gc': np.einsum('ak,abk,b->k', Kg, beta, Kc),
        'Ib_gg': np.einsum('ak,abk,bk->k', Kg, beta, Kg),
        'Ib_mm': np.einsum('ak,abk,bk->k', Km, beta, Km),
        'Ib_gm': np.einsum('ak,abk,bk->k', Kg, beta, Km),
        'Pgm_1h': trapezoid((n*M/H.RHO_M)[:, None]*u*(hm.Nc[:, None] + hm.Ns[:, None]*u)/hm.ng, lnM, axis=0),
        'Pgg_1h': trapezoid((n*(2*hm.Nc*hm.Ns)/hm.ng**2)[:, None]*u + (n*hm.Ns**2/hm.ng**2)[:, None]*u**2, lnM, axis=0),
        'Pmm_1h': trapezoid((n*(M/H.RHO_M)**2)[:, None]*u**2, lnM, axis=0),
    }
    return out


def main():
    t0 = time.time()
    bt = dict(np.load(os.path.join(HERE, 'beta_tables.npz')))
    st = dict(np.load(os.path.join(V6, 'halo_strict_tables.npz')))
    ct = dict(np.load(os.path.join(V6, 'halo_central_tables.npz')))
    k = ct['k_all']
    zs_all = st['zs']
    lin = H.Linear(list(bt['zs']))
    res = {}
    for s, z in enumerate(bt['zs']):
        s6 = int(np.argmin(np.abs(zs_all - z)))
        assert abs(zs_all[s6] - z) < 1e-6
        hm = H.Halo(lin, float(z), k, nM=NM, nx=10)
        lgMmin, lgM1 = st['hod'][s6, :2]
        hm.set_hod(lgMmin, SIG_LGM, lgM1, alpha=ALPHA_SAT)
        assert abs(hm.ng/st['hod'][s6, 3] - 1) < 1e-3, (hm.ng, st['hod'][s6, 3])
        wb = hm.dndlnM*hm.b*hm.M/H.RHO_CB
        I0 = trapezoid(wb, hm.lnM)
        lgM = hm.lnM/np.log(10)
        out = tables(hm, beta_on_grid(bt, s, lgM, k), I0)
        zero = tables(hm, beta_on_grid(bt, s, lgM, k, low='zero'), I0)
        for q in ('Ib_cm', 'Ib_cc', 'Ib_cm_T', 'Ib_cc_T', 'Ib_gc', 'Ib_gg', 'Ib_mm', 'Ib_gm'):
            out[q + '_lo'] = zero[q]
        if np.isfinite(bt['lgM_same'][s, 0]):
            same = tables(hm, beta_on_grid(bt, s, lgM, k, 'lgM_same'), I0)
            for q in ('Ib_cm', 'Ib_cc', 'Ib_gc', 'Ib_gg'):
                out[q + '_same'] = same[q]
        out['Plin'] = np.exp(np.interp(np.log(k), np.log(lin.kh), np.log(lin.P(z)[1])))
        out['I1_v6'] = ct['I1m'][s6]
        # the stored v6 tables were made with another CAMB version, 0.9
        # percent below this one at k > 0.05; the fits use the stored P_lin
        # and I_1, so that they match the v6 and alpha-refit models
        assert np.allclose(out['Plin'], ct['Plin'][s6], rtol=0.02), 'P_lin differs from v6'
        out['Plin_v6'] = ct['Plin'][s6]
        for q, v in out.items():
            res.setdefault(q, []).append(v if np.ndim(v) else float(v))
        ik = [int(np.argmin(np.abs(k - kk))) for kk in (0.1, 0.3, 0.5, 0.7, 1.0)]
        print(f'z {z:4.2f}: b_c {out["bc"]:.2f}, n_c/n_g {out["nc_over_ng"]:.2f}, b_g {out["bg_k"][0]:.3f} '
              f'(v6 {st["hod"][s6, 4]:.3f}); at k 0.1,0.3,0.5,0.7,1: I1 {np.array2string(out["I1"][ik], precision=3)} '
              f'Ib_cm {np.array2string(out["Ib_cm"][ik], precision=3)} Ib_cm_T {np.array2string(out["Ib_cm_T"][ik], precision=3)} '
              f'Ib_cm_lo {np.array2string(out["Ib_cm_lo"][ik], precision=3)} Ib_cc {np.array2string(out["Ib_cc"][ik], precision=3)}  |I1 - I1_v6| max {np.abs(out["I1"] - out["I1_v6"]).max():.1e}'
              f'  [{time.time() - t0:.0f} s]', flush=True)
    nk = k.size
    arr = {q: np.array(vals) for q, vals in res.items() if len(vals) == len(bt['zs'])}
    for q in ('Ib_cm_same', 'Ib_cc_same', 'Ib_gc_same', 'Ib_gg_same'):
        full = np.full((len(bt['zs']), nk), np.nan)
        j = 0
        for s in range(len(bt['zs'])):
            if np.isfinite(bt['lgM_same'][s, 0]):
                full[s] = res[q][j]
                j += 1
        arr[q] = full
    np.savez(os.path.join(HERE, 'nlbias_tables.npz'), zs=bt['zs'], k=k, **arr)
    print('wrote nlbias_tables.npz')


if __name__ == '__main__':
    main()
