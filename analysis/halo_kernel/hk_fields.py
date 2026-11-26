#!/usr/bin/env python3
"""Fields and power spectra for the kernel-calibrated IA halo model.

Everything here reuses the FLAMINGO IA pipeline's own conventions
(flamingo/extraction): NGP assignment on an ng^3 mesh of the box folded nfold
times, the forward FFT normalised by 1/ng^3, density fields n/<n> - 1, shape
fields sum(e) ng^3/N (so a shape field is the ellipticity weighted by the
tracer's own number density, gamma = sum e / nbar), the E/B projection
E = e1 cos 2phi + e2 sin 2phi with phi the azimuth of k in the x-y plane
(line of sight z), Legendre multipoles in mu = k_z/k, and the 30 log bins
of 0.003 to 20 h/Mpc of ia_power.cfg.

A "tracer" is a set of points with optional ellipticities (e1, e2) along z.
power() returns the l = 0, 2, 4 multipoles of every requested pair
(density x density, density x E or B, E x E, B x B) in raw units, before any
conversion. convert() applies the conversions of
analysis/tatt_fit/fit_tatt_flamingo.load_snapshot to galaxy spectra:
shape legs x 1/(2R), R = 1 - e_rms^2; P_hE at mu = 0 = 3/2 x monopole;
P_EE, P_BB at mu = 0 = M0 - 5/2 M2 + 27/8 M4, divided by the NGP window.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'flamingo', 'extraction'))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'tatt_fit'))
from common import fft_r2c_inplace, new_grid, periodic_index  # noqa: E402

BOX = 681.0                     # L1_m8 box [Mpc/h]
KEDGES = np.logspace(np.log10(0.003), np.log10(20.0), 31)   # ia_power.cfg binning
# the folds and k ranges stitched into the FLAMINGO spectra (flamingo_truth.FOLDS),
# with ng = 512 here: fold 1 to 1.2 h/Mpc (0.5 k_Nyq), fold 4 above
FOLDS = ((1, 0.0, 1.2), (4, 1.2, 1e9))
NG = 512


def deposit(pos, values, ng, boxlen, norm, subtract_one):
    '''NGP grid of `values` at `pos` (box folded to boxlen), normalised, FFT'd.'''
    cell = periodic_index(pos*(ng/boxlen) + 1e-10, ng)
    rowlen = 2*(ng//2 + 1)
    flat = (cell[:, 0]*ng + cell[:, 1])*rowlen + cell[:, 2]
    grid = new_grid(ng)
    np.add.at(grid.reshape(-1), flat, values.astype(np.float32))
    real = grid[..., :ng]
    real *= np.float32(norm)
    if subtract_one:
        real -= np.float32(1.0)
    return fft_r2c_inplace(grid)


def tracer_fields(pos, e=None, ng=NG, nfold=1, density=True):
    '''FFT'd density grid and/or (e1, e2) shape grids of one tracer set.'''
    boxlen = BOX/nfold
    norm = float(ng)**3/len(pos)
    out = {}
    if density:
        out['d'] = deposit(pos, np.ones(len(pos)), ng, boxlen, norm, True)
    if e is not None:
        out['e1'] = deposit(pos, e[:, 0], ng, boxlen, norm, False)
        out['e2'] = deposit(pos, e[:, 1], ng, boxlen, norm, False)
    return out


def signed(n, ng):
    n = np.asarray(n)
    return np.where(n > ng//2, n - ng, n).astype(np.float64)


def power(fields, pairs, nfold=1, kedges=KEDGES):
    '''Multipoles l = 0, 2, 4 of every pair, binned in |k|.

    fields: name -> dict with 'd' and/or 'e1', 'e2' (FFT'd grids).
    pairs: list of (nameA, compA, nameB, compB), comp in d, E, B.
    Returns k (mean |k| per bin, mode weighted), nmodes, and
    P[pair] = array [3, nbin] of the bin means of P(k) L_l(mu), l = 0, 2, 4
    (raw units, no (2l + 1) factor, as ia_power.py).
    The half-space convention of ia_power.calc_power: k_z = 0 modes weight 1/2.'''
    any_field = next(iter(fields.values()))
    ng = next(iter(any_field.values())).shape[0]
    boxlen = BOX/nfold
    kfund = 2*np.pi/boxlen
    vol = BOX**3
    kmax = kedges[-1]
    nb = len(kedges) - 1
    jidx = np.arange(ng)
    jidx = jidx[np.abs(signed(jidx, ng))*kfund <= kmax]
    kidx = np.arange(ng//2)
    kidx = kidx[kidx*kfund <= kmax]
    jj = signed(jidx, ng)[:, None]
    kk = kidx.astype(np.float64)[None, :]
    weight = np.broadcast_to(np.where(kk == 0, 0.5, 1.0), (len(jidx), len(kidx))).ravel()
    acc = {p: np.zeros((3, nb)) for p in pairs}
    wsum = np.zeros(nb)
    ksum = np.zeros(nb)
    for i in range(ng):
        ii = float(signed(i, ng))
        if abs(ii)*kfund > kmax:
            continue
        kperp2 = np.broadcast_to(ii*ii + jj*jj, (len(jidx), len(kidx))).ravel()
        ktot2 = kperp2 + np.broadcast_to(kk*kk, (len(jidx), len(kidx))).ravel()
        valid = ktot2 > 0
        kp2, kt2, w = kperp2[valid], ktot2[valid], weight[valid]
        knorm = kfund*np.sqrt(kt2)
        b = np.digitize(knorm, kedges) - 1
        ok = (b >= 0) & (b < nb)
        jjf = np.broadcast_to(jj, (len(jidx), len(kidx))).ravel()[valid]
        mu2 = np.broadcast_to(kk, (len(jidx), len(kidx))).ravel()[valid]**2/kt2
        leg = (np.ones_like(mu2), 0.5*(3*mu2 - 1), 0.125*((35*mu2 - 30)*mu2 + 3))
        with np.errstate(invalid='ignore', divide='ignore'):
            fe = np.where(kp2 == 0, 1.0, (ii*ii - jjf*jjf)/kp2)
            fb = np.where(kp2 == 0, 0.0, 2*ii*jjf/kp2)
        plane = {}

        def get(name, comp):
            key = (name, comp)
            if key not in plane:
                f = fields[name]
                if comp == 'd':
                    plane[key] = np.asarray(f['d'][i][np.ix_(jidx, kidx)], dtype=np.complex128).ravel()[valid]
                else:
                    e1 = np.asarray(f['e1'][i][np.ix_(jidx, kidx)], dtype=np.complex128).ravel()[valid]
                    e2 = np.asarray(f['e2'][i][np.ix_(jidx, kidx)], dtype=np.complex128).ravel()[valid]
                    plane[(name, 'E')] = e1*fe + e2*fb
                    plane[(name, 'B')] = -e1*fb + e2*fe
            return plane[key]
        bo, wo = b[ok], w[ok]
        wsum += np.bincount(bo, weights=wo, minlength=nb)
        ksum += np.bincount(bo, weights=wo*knorm[ok], minlength=nb)
        lego = [lg[ok] for lg in leg]
        for p in pairs:
            a, c = get(p[0], p[1]), get(p[2], p[3])
            xo = (vol*(a.real*c.real + a.imag*c.imag))[ok]
            for l in range(3):
                acc[p][l] += np.bincount(bo, weights=wo*xo*lego[l], minlength=nb)
    with np.errstate(invalid='ignore', divide='ignore'):
        kmean = ksum/wsum
        # bin means of P L_l, without the (2l + 1): the "multipoles" ia_power.py
        # writes, which the mu = 0 coefficients (1, -5/2, 27/8) of convert assume
        P = {p: acc[p]/wsum for p in pairs}
    return kmean, wsum, P


def ngp_w2(k, cell):
    '''Angle-averaged NGP window <W^2>(k) (fit_tatt_flamingo.ngp_w2_avg).'''
    import fit_tatt_flamingo as F
    return F.ngp_w2_avg(k, cell)


def to_pipeline_mesh(k, P, nfold, noise=None):
    '''Rescale raw spectra of the NG = 512 mesh to the 1024 mesh of the FLAMINGO
    pipeline: both fields carry the NGP window, so every clustered spectrum
    scales by <W^2>_1024(k) / <W^2>_512(k). A white noise floor (shot noise of
    P_hh, shape noise of P_EE, P_BB; noise = {pair: level}) is not windowed: it
    is taken off the monopole first and put back after.'''
    noise = noise or {}
    with np.errstate(invalid='ignore'):
        kk = np.nan_to_num(k, nan=1.0)
        r = ngp_w2(kk, BOX/(nfold*1024))/ngp_w2(kk, BOX/(nfold*NG))
    out = {}
    for p, v in P.items():
        v = v.copy()
        n0 = noise.get(p, 0.0)
        v[0] -= n0
        v *= r[None, :]
        v[0] += n0
        out[p] = v
    return out


def stitch(per_fold):
    '''Stitch fold results {nfold: (k, nmodes, P)} over FOLDS into one k array.'''
    ks, Ps, ns = [], {}, []
    for nf, lo, hi in FOLDS:
        k, nm, P = per_fold[nf]
        m = np.isfinite(k) & (k >= lo) & (k < hi) & (nm > 0)
        ks.append(k[m])
        ns.append(nm[m])
        for p, v in P.items():
            Ps.setdefault(p, []).append(v[:, m])
    k = np.concatenate(ks)
    o = np.argsort(k)
    return k[o], np.concatenate(ns)[o], {p: np.concatenate(v, axis=1)[:, o] for p, v in Ps.items()}


def convert_galaxy(k, P, erms, nfold_of_k=None):
    '''Galaxy spectra in the cosmolike-fit convention (fit_tatt_flamingo.load_snapshot):
    P_hE(mu=0) = 3/2 s1 M0; P_EE(mu=0) = s1^2 (M0 - 5/2 M2 + 27/8 M4)/<W^2>.'''
    s1 = 1.0/(2.0*(1.0 - erms**2))
    c = np.array([1.0, -2.5, 27.0/8.0])
    cell = np.where(k < FOLDS[1][1], BOX/(FOLDS[0][0]*NG), BOX/(FOLDS[1][0]*NG))
    W2 = np.array([ngp_w2(np.array([kk]), cc)[0] for kk, cc in zip(k, cell)])
    out = {}
    for name, v in P.items():
        if name.endswith('hE'):
            out[name] = 1.5*s1*v[0]
        elif name.endswith('EE') or name.endswith('BB'):
            out[name] = s1**2*(c @ v)/W2
        else:
            out[name] = v[0]
    return out
