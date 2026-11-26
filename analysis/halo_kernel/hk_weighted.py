#!/usr/bin/env python3
"""Mean-HOD-weighted halo fields for the kernel-calibrated IA halo model.

Why: hk_measure.py stores the halo spectra in six wide mass bins and weights
every halo of a bin equally. Inside a bin the sample galaxies sit in the more
massive (more biased) halos, so the binned prediction of the galaxy density,
sum_ij a_i a_j P_{h_i h_j}, misses 20 to 45 percent of the large-scale P_gg
(at z <= 1). A halo model avoids this by integrating over mass with the HOD
as weight; on a halo catalogue the equivalent is to weight each halo by the
mean occupation at its mass. That is what this script measures.

For each halo h (M200m, centre, DM shape e_h) the weights are the mean numbers
of sample centrals and satellites at its mass, <Nc>(M_h) and <Ns>(M_h), in
0.1 dex bins of log M200m (the halo catalogue is complete above
10^11.3 Msun/h; below, where only galaxy hosts are in the catalogue and
which holds 3 percent of the galaxies, a halo's actual counts are used).
Fields (all on the 512 mesh of hk_fields, folds 1 and 4, rescaled to the
pipeline's 1024 mesh):

  G      model galaxy density: centrals at halo centres plus satellites with
         the measured profile u_s,i(k) of the host-mass bin i of hk_measure,
         delta_G = sum_X (n_X/n_G) u_X delta_X, X = (centrals | satellites) x bin
  Ga     the same with each halo's actual counts (no HOD averaging): the
         difference to G is the assembly bias of the occupation
  H      all halos above 10^11.3 Msun/h, unweighted (the reference tracer
         of the deterministic II term)
  Ec<j>  shape field of the halos of bin j weighted by <Nc>/n_G:
         sum_h <Nc>_h e_h / (n_G V); the model central shape field per unit
         kernel is kappa_c,j Ec<j>
  Es<j>  the same with <Ns> and the satellite profile u_s,j(k)

Spectra stored (raw multipoles [3, nk], stitched; noise off the autos):
  G x G, Ga x Ga, H x H, G x H, Ga x H, G x Ga, and for every bin j
  G x Ec<j>, G x Es<j>, H x Ec<j>, H x Es<j>, Ga x Ec<j>, Ga x Es<j> (E part).

Writes kernels/weighted_<sn>.npz. About 4 minutes per snapshot.
    python hk_weighted.py <sn> [<sn> ...]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hk_fields as HF  # noqa: E402
import hk_measure as HM  # noqa: E402

M_COMPLETE = 11.3          # log10 M200m above which the halo catalogue is complete
DLGM = 0.1                 # width of the mean-HOD mass bins [dex]


def mean_hod(h, g):
    '''Per-halo actual and mean-HOD central and satellite counts.'''
    nh = len(h['centre'])
    nc = np.bincount(g['host'][g['cen']], minlength=nh).astype(float)
    ns = np.bincount(g['host'][~g['cen']], minlength=nh).astype(float)
    lgm = h['lgm']
    fine = np.floor((lgm - M_COMPLETE)/DLGM).astype(int)
    top = int(np.floor((15.3 - M_COMPLETE)/DLGM))
    fine = np.minimum(fine, top)                          # the rare clusters share one bin
    wc, ws = nc.copy(), ns.copy()
    hi = fine >= 0
    for b in np.unique(fine[hi]):
        m = hi & (fine == b)
        wc[m] = nc[m].mean()
        ws[m] = ns[m].mean()
    return nc, ns, wc, ws


def kgrid_abs(ng, nfold):
    '''|k| on the half-complex grid (ng, ng, ng//2 + 1), float32, built slab by slab.'''
    kf = 2*np.pi/(HF.BOX/nfold)
    s = np.fft.fftfreq(ng, 1.0/ng).astype(np.float32)*kf
    kz = np.arange(ng//2 + 1, dtype=np.float32)*kf
    return s, kz


def apply_filter(cg, nfold, k_tab, u_tab):
    '''Multiply the complex grid cg by u(|k|), tabulated on k_tab (held at the ends).'''
    ng = cg.shape[0]
    s, kz = kgrid_abs(ng, nfold)
    lk = np.log(k_tab)
    for i in range(ng):
        kk = np.sqrt(s[i]**2 + s[:, None]**2 + kz[None, :]**2)
        u = np.interp(np.log(np.maximum(kk, 1e-6)), lk, u_tab).astype(np.float32)
        cg[i] *= u


def combo_density(parts, nfold, us, kus):
    '''delta_G = sum_X f_X u_X delta_X over parts = [(pos, w, f, bin or None)].'''
    out = None
    for pos, w, f, ub in parts:
        if w.sum() <= 0:
            continue
        cg = HF.deposit(pos, w, HF.NG, HF.BOX/nfold, float(HF.NG)**3/w.sum(), True)
        if ub is not None:
            apply_filter(cg, nfold, kus, us[ub])
        cg *= np.float32(f)
        if out is None:
            out = cg.copy()
        else:
            out += cg
        del cg
    return out


def shape_field(pos, w, e, nG_V, nfold, u=None, kus=None):
    '''Shape grids sum_h w_h e_h ng^3 / (n_G V), optionally filtered by u(k).'''
    ng = HF.NG
    norm = float(ng)**3/nG_V
    out = {}
    for c, name in ((0, 'e1'), (1, 'e2')):
        cg = HF.deposit(pos, w*e[:, c], ng, HF.BOX/nfold, norm, False)
        if u is not None:
            apply_filter(cg, nfold, kus, u)
        out[name] = cg
    return out


def one_fold(g, h, ker, nf, t0):
    vol = HF.BOX**3
    nc, ns, wc, ws = h['nc'], h['ns'], h['wc'], h['ws']
    nb = HM.NB
    kus, us = ker['k'], ker['us_k']
    okk = np.isfinite(kus) & (kus > 0)
    kus, us = kus[okk], us[:, okk]
    nG = (wc.sum() + ws.sum())/vol
    nGa = (nc.sum() + ns.sum())/vol
    parts, parts_a = [], []
    for i in range(nb):
        inb = h['bin'] == i
        for w, wa, ub in ((wc, nc, None), (ws, ns, i)):
            parts.append((h['centre'][inb], w[inb], w[inb].sum()/vol/nG, ub))
            parts_a.append((h['centre'][inb], wa[inb], wa[inb].sum()/vol/nGa, ub))
    fields = {'G': {'d': combo_density(parts, nf, us, kus)},
              'Ga': {'d': combo_density(parts_a, nf, us, kus)}}
    hm = h['lgm'] >= M_COMPLETE
    fields['H'] = HF.tracer_fields(h['centre'][hm], None, nfold=nf)
    # shot noise of the weighted autos (satellites at u = 1; only used at low k)
    noise = {('G', 'd', 'G', 'd'): vol*np.sum((wc + ws)**2)/(wc.sum() + ws.sum())**2,
             ('Ga', 'd', 'Ga', 'd'): vol*np.sum((nc + ns)**2)/(nc.sum() + ns.sum())**2,
             ('H', 'd', 'H', 'd'): vol/hm.sum()}
    pairs = [('G', 'd', 'G', 'd'), ('Ga', 'd', 'Ga', 'd'), ('H', 'd', 'H', 'd'),
             ('G', 'd', 'H', 'd'), ('Ga', 'd', 'H', 'd'), ('G', 'd', 'Ga', 'd')]
    k, nm, P = HF.power(fields, pairs, nfold=nf)
    P = HF.to_pipeline_mesh(k, P, nf, noise)
    for p in noise:
        P[p][0] -= noise[p]
    print(f'    fold {nf}: density fields done [{time.time() - t0:.0f} s]', flush=True)
    okh = np.all(np.isfinite(h['e']), axis=1)
    e = np.where(okh[:, None], h['e'], 0.0)
    for tag, w in (('Ec', wc), ('Es', ws)):
        sh = {}
        for j in range(nb):
            inb = (h['bin'] == j) & okh
            if w[inb].sum() <= 0:
                continue
            sh[f'{tag}{j}'] = shape_field(h['centre'][inb], w[inb], e[inb], nG*vol, nf,
                                          u=us[j] if tag == 'Es' else None, kus=kus)
        prs = [(a, 'd', s, 'E') for s in sh for a in ('G', 'H', 'Ga')]
        _, _, PS = HF.power({**fields, **sh}, prs, nfold=nf)
        P.update(HF.to_pipeline_mesh(k, PS, nf))
        del sh
        print(f'    fold {nf}: {tag} shape fields done [{time.time() - t0:.0f} s]', flush=True)
    return k, nm, P, nG, nGa


def main():
    for s in sys.argv[1:]:
        sn = int(s)
        t0 = time.time()
        g, h = HM.load(sn)
        ker = dict(np.load(os.path.join(HM.OUT, f'kernels_{sn}.npz')))
        nc, ns, wc, ws = mean_hod(h, g)
        h.update(nc=nc, ns=ns, wc=wc, ws=ws)
        print(f'sn {sn} (z = {g["z"]:.2f}): {len(g["pos"])} galaxies; sum <Nc> {wc.sum():.0f} '
              f'(actual {nc.sum():.0f}), sum <Ns> {ws.sum():.0f} (actual {ns.sum():.0f})', flush=True)
        per = {}
        for nf, lo, hi in HF.FOLDS:
            k, nm, P, nG, nGa = one_fold(g, h, ker, nf, t0)
            per[nf] = (k, nm, P)
        k, nm, P = HF.stitch(per)
        nb = HM.NB
        vol = HF.BOX**3
        n_c = np.array([wc[h['bin'] == j].sum() for j in range(nb)])/vol
        n_s = np.array([ws[h['bin'] == j].sum() for j in range(nb)])/vol
        np.savez(os.path.join(HM.OUT, f'weighted_{sn}.npz'), k=k, nmodes=nm, z=g['z'], n_G=nG, n_Ga=nGa,
                 n_c=n_c, n_s=n_s, n_H=np.sum(h['lgm'] >= M_COMPLETE)/vol,
                 **{'S|' + '|'.join(key): v for key, v in P.items()})
        print(f'  wrote weighted_{sn}.npz [{time.time() - t0:.0f} s]', flush=True)


if __name__ == '__main__':
    main()
