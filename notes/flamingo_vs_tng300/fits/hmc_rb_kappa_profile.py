"""Profile of the joint chi2 against the blue misalignment factor kappa_b.

The halo-based colour split of hmc_rb_fit.py (every galaxy aligned like its
host halo, alignment proportional to b^beta, blue scaled by kappa_b) is
refitted with kappa_b fixed on a grid, every other parameter free, at
k_max = 0.52 h/Mpc, for alpha fixed (rbSb) and alpha X^eta_alpha (rbSbA),
with the fiducial red fractions. kappa_b = 1 is the one-population model
with A1 proportional to b_g^beta; kappa_b < 1 is the physical range (discs
less aligned than ellipticals); kappa_b = 0.04 is the Fortuna et al. 2021
ratio. Writes hmc_rb_kappa_profile.json.

    ~/cocoa/Cocoa/.local/bin/python hmc_rb_kappa_profile.py
"""
import json
import os
import time

import numpy as np

import hmc_rb_fit as R

KAPPAS = (0.0, 0.04, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
KMAX = 0.52
BASES = ('rbSb', 'rbSbA')


def main():
    t0 = time.time()
    fits = np.load(os.path.join(R.V2, 'fits.npz'))
    central = np.load(os.path.join(R.V6, 'halo_central_tables.npz'))
    sat = np.load(os.path.join(R.V6, 'halo_tables.npz'))
    rb = np.load(os.path.join(R.V6, 'halo_redblue_tables.npz'))
    core = np.where(np.load(os.path.join(R.V3, 'v3.npz'))['core'])[0]
    model = R.F.TATTModel(halofit='takahashi')
    snaps = [R.Snapshot(s, fits, central, sat, rb, model) for s in core]
    out = {'about': __doc__, 'kmax': KMAX, 'kappa': list(KAPPAS), 'profiles': {}}
    for base in BASES:
        names, lo, hi, rf, q = R.MODELS[base]
        i = names.index('kappa_b')
        rows, warm = [], None
        for kap in KAPPAS:
            name = f'{base}_kfix'
            R.MODELS[name] = (names[:i] + names[i + 1:], lo[:i] + lo[i + 1:], hi[:i] + hi[i + 1:], rf, q)
            R.KAPPA_FIXED[name] = kap
            r = R.fit_joint(snaps, name, KMAX, warm)
            warm = r['x']
            p = dict(zip(r['names'], r['x']))
            rows.append({'kappa_b': kap, 'chi2': r['chi2'], 'nu': r['nu'], 'at_bound': r['at_bound'],
                         'params': {n: p[n] for n in R.MODELS[name][0]}, 'a1_eff': r['a1_eff']})
            print(f'[{time.time() - t0:.0f}s] {base} kappa_b {kap:4.2f}: chi2 {r["chi2"]:.1f}/{r["nu"]}; '
                  + ', '.join(f'{n} {p[n]:+.3g}' for n in R.MODELS[name][0]) +
                  (f'  AT BOUND {r["at_bound"]}' if r['at_bound'] else ''), flush=True)
        out['profiles'][base] = rows
    json.dump(out, open(os.path.join(R.HERE, 'hmc_rb_kappa_profile.json'), 'w'), indent=1)
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
