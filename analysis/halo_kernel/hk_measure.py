#!/usr/bin/env python3
"""Measure the kernels of the kernel-calibrated IA halo model in one FLAMINGO snapshot.

Inputs: the galaxy catalogue of the IA pipeline (galaxies_<sn>.hdf5: positions,
stellar shapes e1z/e2z, IsCentral) and the SOAP halo catalogue of
flamingo/extraction/soap_halos.py (halos_<sn>.hdf5: centre, M200m, R200m,
c200m, DM inertia tensor, and each galaxy's host row).

Outputs analysis/halo_kernel/kernels/kernels_<sn>.npz with, on the stitched
k grid (folds 1 and 4, 512 mesh rescaled to the pipeline's 1024 mesh):

  gravity layer (halos in mass bins i = 0..NB-1, M200m edges LGM_EDGES):
    P_hh[i, j]   halo density x halo density (shot noise off the autos)
    P_hE[i, j]   halo density (bin i) x halo DM-shape E (bin j), raw monopole
    P_EE[i, j], P_BB[i, j]   halo shape x halo shape, mu = 0 combination
                 (shape noise off the autos)
    n_h[i]       halo number density; e2_h[j] = <e1^2 + e2^2> of halo shapes
  galaxy layer (per host-mass bin):
    Nc[i], Ns[i]   mean number of sample centrals, satellites per halo
    kappa_c[i]     central regression on its halo: sum(e_g . e_h)/sum(e_h . e_h)
    kappa_s[i]     satellite regression on its host halo shape
    r_cs[i]        correlation coefficient of central and halo shapes
    us_k[i, k]     Fourier transform of the satellite radial profile (3D, r/R200m)
    erad_r[i, rb], rad_r_edges   mean radial ellipticity of satellites vs r_p/R200m
  galaxy spectra (the observable, and its 1-halo part):
    G_hE, G_EE, G_BB, G_hh      the full galaxy spectra (raw multipoles)
    G1h_hE, G1h_EE, G1h_BB      the same after shuffling every halo (with its
                                galaxies) to a random position, NSHUF times:
                                the pure 1-halo term (2-halo pairs destroyed)
    Gc_hE, Gs_hE                galaxy positions x central (satellite) shapes only
  bookkeeping: n_g, erms_g (e_rms per component of the sample), shape floors.

Run (cocoa python), about 10 minutes per snapshot:
    python hk_measure.py <sn> [<sn> ...]
"""
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hk_fields as HF  # noqa: E402

RUN = os.path.expanduser('~/ia_power_flamingo/run_L1_m8_particles/results')
GAL = os.path.join(RUN, 'Galaxies')
HALO = os.path.join(RUN, 'SOAP_halos')
OUT = os.path.join(HERE, 'kernels')
# host-halo mass bins, log10 M200m [Msun/h]; the lowest bin holds only the
# galaxy hosts below the halo-field threshold M_MIN = 10^11.3 of soap_halos.py
LGM_EDGES = np.array([9.0, 11.7, 12.1, 12.5, 12.9, 13.4, 16.0])
NB = len(LGM_EDGES) - 1
NSHUF = 2
RAD_EDGES = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
SEED = 12345


def ell_z(t):
    '''Projected (along z) ellipticity of 3D tensors (xx, yy, zz, xy, xz, yz),
    the definition of soap_galaxies.ellipticity_3d.'''
    with np.errstate(invalid='ignore', divide='ignore'):
        s = t[:, 0] + t[:, 1]
        return np.column_stack([(t[:, 0] - t[:, 1])/s, 2*t[:, 3]/s])


def wrap(dx, box):
    return (dx + 0.5*box) % box - 0.5*box


def load(sn):
    with h5py.File(os.path.join(GAL, f'galaxies_{sn}.hdf5'), 'r') as f:
        g = {'pos': f['Position'][()], 'e': np.column_stack([f['e1z'][()], f['e2z'][()]]),
             'cen': f['IsCentral'][()].astype(bool), 'z': float(f.attrs['Redshift'])}
    with h5py.File(os.path.join(HALO, f'halos_{sn}.hdf5'), 'r') as f:
        h = {k: f['halos'][k][()] for k in f['halos']}
        g['host'] = f['galaxies/host_row'][()]
    ok = np.all(np.isfinite(g['e']), axis=1)
    for k in ('pos', 'e', 'cen', 'host'):
        g[k] = g[k][ok]
    # the halo shape: the DM tensor within R200m (filled for every halo; the
    # bound-subhalo tensor is missing for most halos below 10^11.6 Msun/h)
    h['e'] = ell_z(h['so_tensor'])
    h['lgm'] = np.log10(np.maximum(h['M200m'], 1.0))
    h['bin'] = np.clip(np.digitize(h['lgm'], LGM_EDGES) - 1, 0, NB - 1)
    return g, h


def galaxy_kernels(g, h):
    '''HOD, central and satellite regressions, radial alignment, satellite profile.'''
    box = HF.BOX
    vol = box**3
    hb = h['bin'][g['host']]
    out = {k: np.zeros(NB) for k in ('Nc', 'Ns', 'kappa_c', 'kappa_s', 'r_cs', 'n_h', 'n_h_shape', 'e2_h')}
    out['erad_r'] = np.zeros((NB, len(RAD_EDGES) - 1))
    out['nsat_r'] = np.zeros((NB, len(RAD_EDGES) - 1))
    eh_host = h['e'][g['host']]
    dx = wrap(g['pos'] - h['centre'][g['host']], box)
    r3 = np.sqrt((dx**2).sum(1))/h['R200m'][g['host']]
    rp = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        c2 = (dx[:, 0]**2 - dx[:, 1]**2)/rp**2
        s2 = 2*dx[:, 0]*dx[:, 1]/rp**2
    erad = g['e'][:, 0]*c2 + g['e'][:, 1]*s2           # > 0: elongated along the projected radius
    rpn = rp/h['R200m'][g['host']]
    sat_r3, sat_bin = [], []
    for i in range(NB):
        inb = h['bin'] == i
        nh = int(inb.sum())
        out['n_h'][i] = nh/vol
        okh = inb & np.all(np.isfinite(h['e']), axis=1)
        out['n_h_shape'][i] = okh.sum()/vol
        out['e2_h'][i] = np.mean((h['e'][okh]**2).sum(1)) if okh.any() else np.nan
        c = g['cen'] & (hb == i)
        s = ~g['cen'] & (hb == i)
        out['Nc'][i] = c.sum()/max(nh, 1)
        out['Ns'][i] = s.sum()/max(nh, 1)
        for sel, key in ((c, 'kappa_c'), (s, 'kappa_s')):
            eh = eh_host[sel]
            ok = np.all(np.isfinite(eh), axis=1)
            num = np.sum(g['e'][sel][ok]*eh[ok])
            den = np.sum(eh[ok]**2)
            out[key][i] = num/den if den > 0 else np.nan
            if key == 'kappa_c' and ok.any():
                out['r_cs'][i] = num/np.sqrt(den*np.sum(g['e'][sel][ok]**2))
        rb = np.digitize(rpn[s], RAD_EDGES) - 1
        okr = (rb >= 0) & (rb < len(RAD_EDGES) - 1) & np.isfinite(erad[s])
        out['erad_r'][i] = np.bincount(rb[okr], weights=erad[s][okr], minlength=len(RAD_EDGES) - 1)[:len(RAD_EDGES) - 1] \
            / np.maximum(np.bincount(rb[okr], minlength=len(RAD_EDGES) - 1)[:len(RAD_EDGES) - 1], 1)
        out['nsat_r'][i] = np.bincount(rb[okr], minlength=len(RAD_EDGES) - 1)[:len(RAD_EDGES) - 1]
        sat_r3.append(r3[s]*h['R200m'][g['host'][s]])
        sat_bin.append(i)
    return out, sat_r3


def us_of_k(k, sat_r):
    '''Fourier transform of the measured 3D satellite distribution around the
    host centre, u_s(k) = <sin(k r)/(k r)> over satellites (exact for the sample).'''
    out = np.zeros((len(sat_r), len(k)))
    for i, r in enumerate(sat_r):
        if len(r) == 0:
            out[i] = 1.0
            continue
        r = r[np.isfinite(r)]
        x = np.outer(k, r)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[i] = np.where(x > 0, np.sin(x)/x, 1.0).mean(1)
    return out


def spectra(g, h, nf):
    '''All spectra of one fold (raw multipoles, 512 mesh rescaled to 1024).'''
    box = HF.BOX
    vol = box**3
    rng = np.random.default_rng(SEED + nf)
    res = {}
    # ---- gravity layer: halos in mass bins ----
    fl = {}
    noise = {}
    for i in range(NB):
        inb = h['bin'] == i
        okh = inb & np.all(np.isfinite(h['e']), axis=1)
        fl[f'd{i}'] = HF.tracer_fields(h['centre'][inb], None, nfold=nf)
        fl[f's{i}'] = HF.tracer_fields(h['centre'][okh], h['e'][okh], nfold=nf, density=False)
        noise[(f'd{i}', 'd', f'd{i}', 'd')] = vol/inb.sum()
        en = vol/okh.sum()*0.5*np.mean((h['e'][okh]**2).sum(1))
        noise[(f's{i}', 'E', f's{i}', 'E')] = en
        noise[(f's{i}', 'B', f's{i}', 'B')] = en
    pairs = []
    for i in range(NB):
        for j in range(NB):
            pairs.append((f'd{i}', 'd', f's{j}', 'E'))
            if j >= i:
                pairs += [(f'd{i}', 'd', f'd{j}', 'd'), (f's{i}', 'E', f's{j}', 'E'),
                          (f's{i}', 'B', f's{j}', 'B')]
    k, nm, P = HF.power(fl, pairs, nfold=nf)
    P = HF.to_pipeline_mesh(k, P, nf, noise)
    for p in noise:
        P[p][0] -= noise[p]
    res['halo'] = P
    del fl
    # ---- galaxies: full, central-shape-only and satellite-shape-only ----
    n = len(g['pos'])
    eg = g['e']
    shot_e = vol/n*0.5*np.mean((eg**2).sum(1))
    fl = {'g': HF.tracer_fields(g['pos'], eg, nfold=nf)}
    ec = np.where(g['cen'][:, None], eg, 0.0)
    fl['gc'] = HF.tracer_fields(g['pos'], ec, nfold=nf, density=False)     # normalised by N_g
    fl['gs'] = HF.tracer_fields(g['pos'], eg - ec, nfold=nf, density=False)
    gp = [('g', 'd', 'g', 'd'), ('g', 'd', 'g', 'E'), ('g', 'E', 'g', 'E'), ('g', 'B', 'g', 'B'),
          ('g', 'd', 'gc', 'E'), ('g', 'd', 'gs', 'E')]
    gnoise = {gp[0]: vol/n, gp[2]: shot_e, gp[3]: shot_e}
    _, _, PG = HF.power(fl, gp, nfold=nf)
    PG = HF.to_pipeline_mesh(k, PG, nf, gnoise)
    for p in gnoise:
        PG[p][0] -= gnoise[p]
    res['gal'] = PG
    # per host-mass bin: galaxy density x the shapes of the centrals (c<j>) and of
    # the satellites (t<j>) whose host is in bin j (normalised by N_g), for the
    # large-scale alignment kernels of hk_model.py
    hb = h['bin'][g['host']]
    for j in range(NB):
        for tag, sel in (('c', g['cen'] & (hb == j)), ('t', ~g['cen'] & (hb == j))):
            if not sel.any():
                continue
            ej = np.where(sel[:, None], eg, 0.0)
            fl[f'{tag}{j}'] = HF.tracer_fields(g['pos'], ej, nfold=nf, density=False)
            _, _, PJ = HF.power({'g': fl['g'], f'{tag}{j}': fl[f'{tag}{j}']}, [('g', 'd', f'{tag}{j}', 'E')], nfold=nf)
            res['gal'].update(HF.to_pipeline_mesh(k, PJ, nf))
            del fl[f'{tag}{j}']
    del fl
    # ---- 1-halo term: every halo moved, with its galaxies, to a random place ----
    acc = None
    rel = wrap(g['pos'] - h['centre'][g['host']], box)
    for _ in range(NSHUF):
        newc = rng.uniform(0, box, size=(len(h['centre']), 3))
        pos = (newc[g['host']] + rel) % box
        fl = {'g': HF.tracer_fields(pos, eg, nfold=nf)}
        _, _, PS = HF.power(fl, gp[:4], nfold=nf)
        PS = HF.to_pipeline_mesh(k, PS, nf, gnoise)
        for p in gnoise:
            PS[p][0] -= gnoise[p]
        acc = PS if acc is None else {p: acc[p] + PS[p] for p in PS}
        del fl
    res['gal1h'] = {p: v/NSHUF for p, v in acc.items()}
    res['shot_e'] = shot_e
    return k, nm, res


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in sys.argv[1:]:
        sn = int(s)
        t0 = time.time()
        g, h = load(sn)
        gk, sat_r = galaxy_kernels(g, h)
        print(f'sn {sn} (z = {g["z"]:.2f}): {len(g["pos"])} galaxies, {len(h["centre"])} halos; '
              f'Nc {np.round(gk["Nc"], 3)}, Ns {np.round(gk["Ns"], 3)}, kappa_c {np.round(gk["kappa_c"], 3)}, '
              f'kappa_s {np.round(gk["kappa_s"], 3)}, r_cs {np.round(gk["r_cs"], 3)}', flush=True)
        per = {}
        for nf, lo, hi in HF.FOLDS:
            k, nm, res = spectra(g, h, nf)
            flat = {}
            for grp in ('halo', 'gal', 'gal1h'):
                for p, v in res[grp].items():
                    flat[(grp,) + p] = v
            per[nf] = (k, nm, flat)
            print(f'  fold {nf} done [{time.time() - t0:.0f} s]', flush=True)
        k, nm, P = HF.stitch(per)
        out = dict(k=k, nmodes=nm, z=g['z'], lgm_edges=LGM_EDGES, rad_edges=RAD_EDGES,
                   n_g=len(g['pos'])/HF.BOX**3, erms_g=np.sqrt(0.5*np.mean((g['e']**2).sum(1))),
                   us_k=us_of_k(k, sat_r), **gk)
        # spectra stored as 'S|<group>|<A>|<compA>|<B>|<compB>', arrays [3 multipoles, nk]
        np.savez(os.path.join(OUT, f'kernels_{sn}.npz'), **out,
                 **{'S|' + '|'.join(key): v for key, v in P.items()})
        print(f'  wrote kernels_{sn}.npz [{time.time() - t0:.0f} s]', flush=True)


if __name__ == '__main__':
    main()
