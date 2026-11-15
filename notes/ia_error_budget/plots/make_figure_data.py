"""Write the data.json of every figure of the IA error-budget note.

The only file of the note that knows where the analysis outputs are, all in
analysis/roman_dchi2/v8/:
    clean_budget.json    the budget by linear response, checks, Roman fits
    nonlinear_tol.json   the tolerances, refitted where the linear one > 10%
    fid_fits.json        NLA and TATT fitted at the fiducial cosmology
    correlated_budget.json  the budget for random errors correlated in k
    correlated_bands.json   the same, confined to one decade of k: bands
Rerun this after those scripts, then rerun the plot scripts.

    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'analysis', 'roman_dchi2', 'v8'))
TRUTH_NAME = {'S1': 'stitched FLAMINGO truth (NLA below k_t(z), FLAMINGO above)',
              'prod': 'production FLAMINGO truth (FLAMINGO at every k >= 0.05 h/Mpc)'}
CUTS = ('none', 'K0/D 0.3')
COMMON = ('The IA error budget of Roman cosmic shear: the fractional error eps of the injected '
          'IA spectrum, P_X -> P_X [1 + eps f(k)] at every redshift, that moves Omega_m or S8 by '
          '0.3 of its marginalised error. Model agnostic: the analysis model is the truth itself '
          'with an IA amplitude and power-law evolution free, with Omega_m and A_s. Shapes f, all '
          'of one sign: bump, a Gaussian in ln k of width 0.3 at k; below, a smooth step equal to '
          '1 at all scales larger than k; above, 1 at all scales smaller than k. up / down: the '
          'tolerance for the spectrum too high / too low; linear response where the linear '
          'tolerance is below 10%, refitted above (nonlinear_tol.py); above about 20% with the cut '
          'the values are upper estimates; 1 = no limit within 100%. example1_cov. '
          '(analysis/roman_dchi2/v8/clean_budget.py, nonlinear_tol.py.)')


def load(name):
    with open(os.path.join(V8, name)) as f:
        return json.load(f)


def write(name, data):
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.relpath(os.path.join(folder, 'data.json'), HERE))


def main():
    src = load('clean_budget.json')
    nl = load('nonlinear_tol.json')
    ff = load('fid_fits.json')
    for tr in ('S1', 'prod'):
        t = nl['truths'][tr]
        write(f'bands_{tr.lower()}', {
            'about': COMMON + f' Truth: the {TRUTH_NAME[tr]}.',
            'k': src['nodes_h_per_Mpc'], 'truth': tr,
            'bands': {cut: {key: {'up': v['up'], 'down': v['down']} for key, v in t[cut].items()}
                      for cut in CUTS},
            'sigma': {cut: src['truths'][tr]['cuts'][cut]['sigma'] for cut in CUTS},
            'verify': {cut: src['truths'][tr]['cuts'][cut]['verify'] for cut in CUTS},
        })
        write(f'overlay_{tr.lower()}', {
            'about': (f'The error of NLA and TATT against the {TRUTH_NAME[tr]}, over the error '
                      'budget. bands[cut][X]: the up and down tolerance for one k range wrong (' +
                      COMMON + ') errors[cut][model][X]: P^model/P^truth - 1 at the core snapshots '
                      '(z_core) on k_dense, each model fitted at the fiducial cosmology (IA '
                      'parameters only) with the same cut (analysis/roman_dchi2/v8/fid_fits.py).'),
            'k': src['nodes_h_per_Mpc'], 'k_dense': ff['k_dense'], 'z_core': ff['z_core'],
            'bands': {cut: {X: t[cut][f'bump|{X}'] for X in ('GI', 'II')} for cut in CUTS},
            'errors': {cut: {m: ff['fits'][f'{tr}|{cut}|{m}']['error'] for m in ('NLA', 'TATT')}
                       for cut in CUTS},
            'fits': {f'{cut}|{m}': {k: ff['fits'][f'{tr}|{cut}|{m}'][k] for k in ('best', 'chi2', 'npts')}
                     for cut in CUTS for m in ('NLA', 'TATT')},
            'roman_fits_cosmology_free': {
                f'{cut}|{m}': {k: src['truths'][tr]['models'][f'{cut}|{m}'][k]
                               for k in ('dOm_over_sigma', 'dS8_over_sigma', 'chi2')}
                for cut in CUTS for m in ('NLA', 'TATT')},
        })

    cb = load('correlated_budget.json')
    write('corr_budget', {
        'about': ('The IA error budget for random errors correlated in k. At 31 nodes '
                  '(0.01 to 10 h/Mpc, 10 per decade) the fractional errors of P_X (X = GI or II, '
                  'the same at every z) are Gaussian with rms eps and correlation '
                  'exp(-(ln k_i - ln k_j)^2 / ell^2), splined in ln k. eps[q]: the rms at which a '
                  'fraction q of such errors keeps |dOmega_m| and |dS8| below 0.3 sigma (linear '
                  'response; checked by refits). Model agnostic, example1_cov. '
                  '(analysis/roman_dchi2/v8/correlated_budget.py.)'),
        'ell': cb['ells'],
        'eps': {tr: {cut: {X: {q: [r['eps'][q] for r in cb['truths'][tr][cut]['budget'][X]]
                               for q in ('0.68', '0.95')}
                           for X in ('GI', 'II')} for cut in CUTS} for tr in ('S1', 'prod')},
    })
    # example errors at the 95% budget, production truth, GI, no cut
    nodes = np.array(cb['nodes_h_per_Mpc'])
    kd = np.geomspace(nodes[0], nodes[-1], 300)
    rng = np.random.default_rng(7)
    ex = {}
    for ell in (0.3, 1.0, 3.0):
        C = np.exp(-(np.log(nodes)[:, None] - np.log(nodes)[None, :])**2/ell**2)
        w, V = np.linalg.eigh(C)
        L = V*np.sqrt(np.clip(w, 0, None))
        row = [r for r in cb['truths']['prod']['none']['budget']['GI'] if r['ell'] == ell][0]
        eps = row['eps']['0.95']
        A = np.array(cb['truths']['prod']['none']['A_GI'])
        draws = []
        for _ in range(4):
            dv = eps*(L @ rng.standard_normal(len(nodes)))
            draws.append({'f': CubicSpline(np.log(nodes), dv)(np.log(kd)).tolist(),
                          'shift_sigma': (A @ dv).tolist()})
        ex[f'{ell:g}'] = {'eps95': eps, 'draws': draws}
    write('corr_examples', {
        'about': ('Four random errors of the production FLAMINGO GI spectrum at the 95% budget '
                  '(no cut), for correlation lengths ell = 0.3, 1 and 3 in ln k: f(k), the '
                  'fractional error splined through the 31 nodes; shift_sigma: the linear '
                  '(dOmega_m, dS8) in units of sigma. eps95: the rms of the errors.'),
        'k': kd.tolist(), 'examples': ex,
    })
    cw = load('correlated_bands.json')
    write('corr_bands', {
        'about': ('The IA error budget as a band, for random errors correlated in k and confined '
                  'to one decade of k centred on k_c. Inside the window the fractional error of P_X '
                  '(the same at every z) is Gaussian with rms eps and correlation '
                  'exp(-(ln k_i - ln k_j)^2 / ell^2), splined in ln k through 31 nodes. '
                  'eps95[cut][X][ell]: the rms at which 95% of such errors keep |dOmega_m| and |dS8| '
                  'below 0.3 sigma; linear response below 5%, refitted above (200 errors, IA '
                  'amplitude and evolution exact); 1 = no limit within 100%. Production truth; '
                  'S1 holds the stitched truth. Model agnostic, example1_cov. '
                  '(analysis/roman_dchi2/v8/correlated_bands.py.)'),
        'k_c': cw['k_c'], 'ell': cw['ells'],
        'eps95': {cut: {X: {f'{e:g}': cw['truths']['prod'][cut][f'{X}|{e:g}']['eps95'] for e in cw['ells']}
                        for X in ('GI', 'II')} for cut in CUTS},
        'S1': {cut: {X: {f'{e:g}': cw['truths']['S1'][cut][f'{X}|{e:g}']['eps95'] for e in cw['ells']}
                     for X in ('GI', 'II')} for cut in CUTS},
    })


if __name__ == '__main__':
    main()
