"""Write the data.json of every figure of the IA error-budget note.

The only file of the note that knows where the analysis outputs are, all in
analysis/roman_dchi2/v8/:
    clean_budget.json    the budget by linear response, checks, Roman fits
    nonlinear_tol.json   the tolerances, refitted where the linear one > 10%
    fid_fits.json        NLA and TATT fitted at the fiducial cosmology
    correlated_budget.json  the budget for random errors correlated in k
    correlated_bands.json   the same, confined to one k range: bands
                            (corr_bands; corr_overlay with NLA and TATT)
and, for the v9 section (the correlated residual prior and Fisher method), in
analysis/roman_dchi2/v9/:
    ia_budget.json       every v9 budget (global, band, map, drift, marginal)
    validate_direct.json direct injections refitted against linear response
Rerun this after those scripts, then rerun the plot scripts.

    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py
"""
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'analysis', 'roman_dchi2', 'v8'))
V9 = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'analysis', 'roman_dchi2', 'v9'))
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


def load(name, folder=V8):
    with open(os.path.join(folder, name)) as f:
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
                  'to one k range around k_c (window exp(-ln^2(k/k_c)/(2*0.3^2)), the one-range bump). The fractional error of P_X '
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
        'one_sign_k': src['nodes_h_per_Mpc'],
        'one_sign': {cut: {X: nl['truths']['prod'][cut][f'bump|{X}'] for X in ('GI', 'II')} for cut in CUTS},
    })
    write('corr_overlay', {
        'about': ('The error of NLA and TATT against the production FLAMINGO truth, over the band '
                  'for random errors correlated in k (corr_bands). eps95[cut][X][ell]: the rms at '
                  'which 95% of random errors confined to one k range around k_c keep |dOmega_m| and '
                  '|dS8| below 0.3 sigma (correlated_bands.py), for ell = 0.3 and 1 in ln k. '
                  'errors[cut][model][X]: P^model/P^truth - 1 at the core snapshots (z_core) on '
                  'k_dense, each model fitted at the fiducial cosmology (IA parameters only) with '
                  'the same cut (fid_fits.py).'),
        'k_c': cw['k_c'], 'k_dense': ff['k_dense'], 'z_core': ff['z_core'],
        'eps95': {cut: {X: {e: cw['truths']['prod'][cut][f'{X}|{e}']['eps95'] for e in ('0.3', '1')}
                        for X in ('GI', 'II')} for cut in CUTS},
        'errors': {cut: {m: ff['fits'][f'prod|{cut}|{m}']['error'] for m in ('NLA', 'TATT')}
                   for cut in CUTS},
    })

    v9_figures()


V9_COMMON = ('The v9 IA error budget (analysis/roman_dchi2/v9), the correlated residual prior and '
             'Fisher method of the lssty1xdesi notes applied to the IA spectra: a random '
             'fractional error delta(X, z_s, k_i) of P_GI and P_II at 31 nodes in ln k (0.01 to 10 '
             'h/Mpc, splined) and the 13 FLAMINGO snapshots, Gaussian with rms sigma and '
             'correlation exp(-|ln k_i - ln k_j|/ell) exp(-|z_s - z_t|/L_z) (L_z = inf: the same '
             'error at every z), GI and II independent unless stated. The shifts of Omega_m and S8 '
             'follow from 1586 exact cosmolike responses (one per ln k tent and snapshot) and the '
             'Fisher matrix of the truth-as-model (Omega_m, A_s, A, eta free). sigma_max: the rms at '
             'which the rms shift of the limiting parameter reaches 0.3 of its error (eta = '
             'sqrt(1 + 0.3^2)); sigma_95: the rms at which 95% of errors keep both shifts below '
             '0.3 sigma (the v8 criterion). 1 = no limit within 100%. example1_cov.')


def v9_figures():
    """The data of the v9 figures."""
    b = load('ia_budget.json', V9)
    vd = load('validate_direct.json', V9)
    write('v9_budget', {
        'about': V9_COMMON + (' sigma_max[truth][cut][set][L_z]: against ell, exponential kernel '
                              'in ln k; sets: both = GI and II independent, both_rho1 = the same '
                              'error in GI and II; GI, II alone. (ia_budget.py, section global.)'),
        'ell': b['ells'], 'lz': b['lzs'],
        'sigma_max': {tr: {cut: {s: {lz: [r['sigma_max'] for r in b['truths'][tr][cut]['global']
                                           if r['set'] == s and r['kind'] == 'exp' and r['lz'] == lz]
                                     for lz in b['lzs']}
                                 for s in ('both', 'both_rho1', 'GI', 'II')}
                           for cut in ('none', 'K03')} for tr in ('prod', 'S1')},
        'sigma_95': {tr: {cut: {s: {lz: [r['sigma_95'] for r in b['truths'][tr][cut]['global']
                                          if r['set'] == s and r['kind'] == 'exp' and r['lz'] == lz]
                                    for lz in b['lzs']}
                                for s in ('both',)}
                          for cut in ('none', 'K03')} for tr in ('prod',)},
        'v8_ratio': {tr: {cut: [r['ratio'] for r in b['truths'][tr][cut]['v8_check']]
                          for cut in ('none', 'K03')} for tr in ('prod', 'S1')},
    })
    ff = load('fid_fits.json')
    write('v9_bands', {
        'about': V9_COMMON + (' The band: the error of one spectrum X confined to one k range '
                              'around k_c (a Gaussian window of width 0.3 in ln k, every z), '
                              'sigma_max[truth][cut][X][L_z] for ell = 1 (and ell = 0.3, 3 under '
                              'sigma_max_ell), sigma_95[cut][X][L_z] the same with the 95% '
                              'criterion. errors[cut][model][X]: P^model/P^truth - 1 at the '
                              'core snapshots (z_core) on k_dense, each model fitted at the '
                              'fiducial cosmology (IA parameters only) with the same cut '
                              '(v8/fid_fits.py). (ia_budget.py, section band.)'),
        'k_c': b['nodes'], 'k_dense': ff['k_dense'], 'z_core': ff['z_core'],
        'sigma_max': {tr: {cut: {X: {lz: b['truths'][tr][cut]['band'][f'{X}|1|{lz}']['sigma_max']
                                     for lz in ('0.3', 'inf')} for X in ('GI', 'II')}
                           for cut in ('none', 'K03')} for tr in ('prod', 'S1')},
        'sigma_max_ell': {cut: {X: {e: b['truths']['prod'][cut]['band'][f'{X}|{e}|0.3']['sigma_max']
                                    for e in ('0.3', '1', '3')} for X in ('GI', 'II')}
                          for cut in ('none', 'K03')},
        'sigma_95': {cut: {X: {lz: b['truths']['prod'][cut]['band'][f'{X}|1|{lz}']['sigma_95']
                               for lz in ('0.3', 'inf')} for X in ('GI', 'II')}
                     for cut in ('none', 'K03')},
        'errors': {cut9: {m: ff['fits'][f'prod|{cut8}|{m}']['error'] for m in ('NLA', 'TATT')}
                   for cut9, cut8 in (('none', 'none'), ('K03', 'K0/D 0.3'))},
    })
    write('v9_validation', {
        'about': ('Direct injection of random IA errors through cosmolike (spline on the table '
                  'grid), refitted with the full Gauss-Newton fit of v8 (CAMB at every step, A and '
                  'eta free), against the linear prediction of the v9 base; shifts in units of the '
                  'marginalised errors. Production truth, 5 errors per cut drawn at the cut\'s '
                  'sigma_max (exp kernel, ell = 1, L_z = 0.3, GI and II). '
                  '(analysis/roman_dchi2/v9/validate_direct.py.)'),
        'params': ['Omega_m', 'S8'],
        'cuts': {cut: {'sigma': rs[0]['sigma'], 'full': [r['full'] for r in rs],
                       'linear': [r['linear'] for r in rs]} for cut, rs in vd['cuts'].items()},
    })


if __name__ == '__main__':
    main()
