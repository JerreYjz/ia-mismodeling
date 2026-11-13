"""Write the data.json of every figure of the IA error-budget note.

The only file of the note that knows where the analysis outputs are, all in
analysis/roman_dchi2/v8/:
    clean_budget.json    the budget by linear response, checks, Roman fits
    nonlinear_tol.json   the tolerances, refitted where the linear one > 10%
    fid_fits.json        NLA and TATT fitted at the fiducial cosmology
Rerun this after those scripts, then rerun the plot scripts.

    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py
"""
import json
import os

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


if __name__ == '__main__':
    main()
