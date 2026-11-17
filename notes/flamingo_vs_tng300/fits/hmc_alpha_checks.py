"""Physical-sensibility checks of the fits in hmc_alpha_fits.json.

A good chi2 is not the test. For every variant of hmc_alpha_fit.py this
prints, for one snapshot (default z = 1) against k_max and then for the core
snapshots at fixed k_max:

  1. the parameters with their errors, and which sit at a bound;
  2. alpha against its physical range [0, 1], a_c against its physical
     size (about 0.5), and the total white term N_s + W against zero;
  3. the drift of each amplitude with k_max (a stable physical parameter
     does not follow the fit range);
  4. the signs of the model: -P_dE, P_EE and P_BB positive at every bin up
     to K_SIGN;
  5. the model just outside the fit range: percent difference in P_dE at
     the next bins above k_max;
  6. the share of each term in P_dE at k_max.

Run with the cocoa Python, after hmc_alpha_fit.py:
    ~/cocoa/Cocoa/.local/bin/python hmc_alpha_checks.py [z]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hmc_alpha_fit as HA  # noqa: E402

# the signs are checked up to this k [h/Mpc]
K_SIGN = 4.5
# the k_max values of the per-snapshot table [h/Mpc]
K_TABLE = (0.52, 0.70, 0.94)
# number of bins above k_max used for the extrapolation check
N_OUT = 3


def main():
    zsel = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    with open(os.path.join(HERE, 'hmc_alpha_fits.json')) as fh:
        res = json.load(fh)
    zs, kgrid, core = np.array(res['zs']), np.array(res['kgrid']), np.array(res['core'])
    s0 = int(np.argmin(np.abs(zs - zsel)))
    fits = np.load(os.path.join(HA.V2, 'fits.npz'))
    central = np.load(os.path.join(HA.V6, 'halo_central_tables.npz'))
    sat = np.load(os.path.join(HA.V6, 'halo_tables.npz'))
    model = HA.F.TATTModel(halofit='takahashi')
    snaps = {s: HA.Snapshot(s, fits, central, sat, model) for s in np.where(core)[0]}
    snaps.setdefault(s0, HA.Snapshot(s0, fits, central, sat, model))
    names = res['names']
    np.set_printoptions(linewidth=200, precision=3, suppress=True)
    for variant, v in res['variants'].items():
        x = np.array([[[np.nan if q is None else q for q in row] for row in sn] for sn in v['x']])
        sig = np.array([[[np.nan if q is None else q for q in row] for row in sn] for sn in v['sig']])
        chi2, nu, p = np.array(v['chi2']), np.array(v['nu']), np.array(v['p'])
        bound = np.array(v['at_bound'])
        print(f'\n================ {variant} ================')
        print(f'z = {zs[s0]:g}: k_max, chi2/nu, p | ' + ', '.join(names)
              + ' | at a bound | -P_dE>0 P_EE>0 P_BB>=0 | % diff P_dE at the next bins | '
              'share of 2h / sat / central in P_dE at k_max')
        snap = snaps[s0]
        k = snap.k
        for j, kmax in enumerate(kgrid):
            if nu[s0, j] <= 0:
                continue
            parts = snap.parts(x[s0, j])
            de = parts['dE_2h'] + parts['dE_sat'] + parts['dE_c']
            sel = (k >= HA.KMIN) & (k <= K_SIGN)
            signs = (bool(np.all(-de[sel] > 0)), bool(np.all(parts['EE'][sel] > 0)),
                     bool(np.all(parts['BB'][sel] >= 0)))
            data_de = snap.y[0]/snap.bias
            above = np.where(k > kmax*(1 + 1e-9))[0][:N_OUT]
            pct = 100*(de[above] - data_de[above])/data_de[above]
            jk = int(np.argmin(np.abs(k - kmax)))
            share = np.array([parts['dE_2h'][jk], parts['dE_sat'][jk], parts['dE_c'][jk]])/de[jk]
            print(f'{kmax:5.2f} {chi2[s0, j]/nu[s0, j]:6.2f} {p[s0, j]:6.3f} | '
                  + ' '.join(f'{x[s0, j, q]:8.3f}' for q in range(7)) + ' | '
                  + (','.join(n for n, b in zip(names, bound[s0, j]) if b) or '-') + ' | '
                  + ' '.join('ok' if t else 'NEG' for t in signs) + ' | '
                  + ' '.join(f'{q:7.1f}' for q in pct) + ' | ' + ' '.join(f'{q:5.2f}' for q in share))
        print('errors at z = %g:' % zs[s0])
        for j, kmax in enumerate(kgrid):
            if nu[s0, j] > 0 and kmax > 0.3:
                print(f'{kmax:5.2f}        ' + ' '.join(f'{sig[s0, j, q]:8.3f}' for q in range(7))
                      + f'   corr(alpha, a_c / bTA / A2) = {v["corr_alpha_ac"][s0][j]} '
                      f'{v["corr_alpha_bTA"][s0][j]} {v["corr_alpha_A2"][s0][j]}')
        print(f'N_s + W at z = {zs[s0]:g}: ' + ' '.join(f'{snap.ns + x[s0, j, 5]:.3f}'
                                                        for j in range(len(kgrid))))
        for kk in K_TABLE:
            j = int(np.argmin(np.abs(kgrid - kk)))
            print(f'core snapshots at k_max = {kgrid[j]:.2f}: z, chi2/nu, p | ' + ', '.join(names)
                  + ' | at a bound')
            for s in np.where(core)[0]:
                print(f'  {zs[s]:4.2f} {chi2[s, j]/nu[s, j]:6.2f} {p[s, j]:6.3f} | '
                      + ' '.join(f'{x[s, j, q]:8.3f}' for q in range(7)) + ' | '
                      + (','.join(n for n, b in zip(names, bound[s, j]) if b) or '-'))
            print(f'  summed chi2/nu {chi2[core, j].sum()/nu[core, j].sum():.2f}')
        print('goodness-of-fit cuts (core): ' + ' '.join(
            'nan' if c is None else f'{c:.2f}' for c in np.array(v['cut'], dtype=object)[core]))


if __name__ == '__main__':
    main()
