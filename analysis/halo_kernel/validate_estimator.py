#!/usr/bin/env python3
"""Check that hk_fields reproduces the FLAMINGO IA spectra of the pipeline.

Rebuilds the galaxy density and shape fields of one snapshot from the
galaxy catalogue (ng = 512, folds 1 and 4) and compares the raw monopoles
P_hh, P_hE0 and the shape-noise-subtracted P_EE0, P_BB0, and P_EE2, with
flamingo/data/PS/*_sn<sn>_nfold<f>.dat (ng = 1024). Agreement to about 1
percent where the NGP windows of the two meshes agree (k < 0.5 k_Nyq of the
512 mesh) validates every convention used later.

    python validate_estimator.py [sn]        (default 58, z = 1)
"""
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hk_fields as HF  # noqa: E402

GAL = os.path.expanduser('~/ia_power_flamingo/run_L1_m8_particles/results/Galaxies')
PS = os.path.join(HF.REPO, 'flamingo', 'data', 'PS')


def main():
    sn = int(sys.argv[1]) if len(sys.argv) > 1 else 58
    with h5py.File(os.path.join(GAL, f'galaxies_{sn}.hdf5'), 'r') as f:
        pos = f['Position'][()]
        e = np.column_stack([f['e1z'][()], f['e2z'][()]])
    ok = np.all(np.isfinite(e), axis=1)
    pos, e = pos[ok], e[ok]
    n = len(pos)
    shot_e = HF.BOX**3/n*0.5*np.mean(e[:, 0]**2 + e[:, 1]**2)
    print(f'sn {sn}: {n} galaxies, Poisson shape floor {shot_e:.4g}')
    for nf, kmax_cmp in ((1, 1.2), (4, 4.0)):
        fl = {'g': HF.tracer_fields(pos, e, nfold=nf)}
        pairs = [('g', 'd', 'g', 'd'), ('g', 'd', 'g', 'E'), ('g', 'E', 'g', 'E'), ('g', 'B', 'g', 'B')]
        k, nm, P = HF.power(fl, pairs, nfold=nf)
        P = HF.to_pipeline_mesh(k, P, nf, noise={pairs[0]: HF.BOX**3/n, pairs[2]: shot_e, pairs[3]: shot_e})
        del fl
        for name, pair, l, sub in (('Phh', pairs[0], 0, 0.0), ('PhE0', pairs[1], 0, 0.0),
                                   ('PEE0', pairs[2], 0, shot_e), ('PBB0', pairs[3], 0, shot_e),
                                   ('PEE2', pairs[2], 1, 0.0)):
            ref = np.loadtxt(os.path.join(PS, f'{name}_sn{sn}_nfold{nf}.dat'))
            # the pipeline drops empty bins: match its rows to these bins by k
            b = np.digitize(ref[:, 0], HF.KEDGES) - 1
            mine = (P[pair][l] - sub)[b]
            kb = k[b]
            m = (nm[b] > 0) & (ref[:, 3] > 0) & (kb < kmax_cmp) & (kb > (0.02 if nf == 1 else 1.2))
            r = mine[m]/ref[m, 1]
            dev = (mine[m] - ref[m, 1])/np.maximum(ref[m, 2], 1e-30)
            print(f'  fold {nf} {name:5s}: ratio mine/pipeline median {np.median(r):.4f}, '
                  f'range {r.min():.3f} to {r.max():.3f}; |dev|/sigma max {np.abs(dev).max():.2f} '
                  f'over {m.sum()} bins; k {kb[m][0]:.3f} to {kb[m][-1]:.3f}', flush=True)


if __name__ == '__main__':
    main()
