#!/usr/bin/env python3
"""Writes the data.json of every figure of notes/halo_ia_models from the
analysis outputs. Run from anywhere with any Python 3 with numpy:

    python make_figure_data.py [figure ...]      (default: all)

    gof         FLAMINGO goodness of fit against k_max per model
                (analysis/halo_nlbias/nlbias_fits.json)
    params      fitted parameters against k_max at z = 0.3, 1, 2 (same file)
    fob_k0scan  Roman figure of bias against the xi_+- cut, CP truth, Fisher
                (analysis/cut_validation/fisher_minimize.json)
The fourth figure, gi_kernel, is the folder of analysis/halo_nlbias/plots
copied as it is (its data.json comes from that folder's generator).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
FITS = os.path.join(REPO, 'analysis', 'halo_nlbias', 'nlbias_fits.json')
FOB = os.path.join(REPO, 'analysis', 'cut_validation', 'fisher_minimize.json')

# the models of the note, in the TATT and the NLA form: variant of
# nlbias_fit.py and model of fisher_minimize.py
MODELS = {
    'base': {'label': 'P_NL (no halo model)', 'TATT': ('tatt', 'TATT'), 'NLA': ('nla', 'NLA')},
    'hmc': {'label': 'HMc, I_1 P_lin', 'TATT': ('lin_tatt', 'TATTh'), 'NLA': ('lin_nla', 'NLAh')},
    'alpha': {'label': 'alpha free', 'TATT': ('alpha_tatt', 'TATTa'), 'NLA': ('alpha_nla', 'NLAa')},
    'alphaf': {'label': 'alpha of FLAMINGO', 'TATT': (None, 'TATTf'), 'NLA': (None, 'NLAf')},
    'nlb': {'label': 'beta_NL', 'TATT': ('nlb_tatt', 'TATTb'), 'NLA': ('nlb_nla', 'NLAb')},
}
FORMS = ('TATT', 'NLA')
K0_CS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)
K0_3X2 = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def clean(a):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), np.round(a, 6), None).tolist()


def num(v):
    return np.nan if v is None else v


def gof():
    d = json.load(open(FITS))
    kg = np.array(d['kgrid'])
    lines = {}
    for form in FORMS:
        for key, m in MODELS.items():
            var = m[form][0]
            if var is None:
                continue
            v = d['variants'][var]
            c2 = np.array([[num(x) for x in row] for row in v['chi2']], float)
            nu = np.array([[num(x) for x in row] for row in v['nu']], float)
            lines[f'{form}|{key}'] = {'variant': var, 'summed': clean(c2.sum(0)/nu.sum(0)),
                                      'per_snapshot': clean(c2/nu), 'cut': clean([num(c) for c in v['cut']])}
    return {'about': ('Producer: notes/halo_ia_models/plots/make_figure_data.py from '
                      'analysis/halo_nlbias/nlbias_fits.json (nlbias_fit.py). FLAMINGO P_hE, P_EE, P_BB '
                      'of the nine core snapshots, fitted on 0.05 <= k <= k_max with a free white term W, '
                      'a_c in [0, 1], no satellites. summed: sum chi2_min / sum nu over the snapshots; '
                      'per_snapshot: chi2_min/nu [snapshot][k_max]; cut: last k_max before two '
                      'consecutive p < 0.05.'),
            'kgrid': clean(kg), 'zs': d['zs'], 'models': {k: m['label'] for k, m in MODELS.items()},
            'lines': lines}


def params():
    d = json.load(open(FITS))
    kg = np.array(d['kgrid'])
    names = d['names']
    zsel = (0.3, 1.0, 2.0)
    out = {}
    for form in FORMS:
        for key, m in MODELS.items():
            var = m[form][0]
            if var is None:
                continue
            v = d['variants'][var]
            for z in zsel:
                s = d['zs'].index(z)
                x = np.array([[num(a) for a in row] for row in v['x'][s]], float)
                ab = np.array(v['at_bound'][s], bool)
                out[f'{form}|{key}|{z:g}'] = {n: clean(x[:, i]) for i, n in enumerate(names)}
                out[f'{form}|{key}|{z:g}']['at_bound'] = {n: ab[:, i].tolist() for i, n in enumerate(names)}
    return {'about': ('Producer: notes/halo_ia_models/plots/make_figure_data.py from '
                      'analysis/halo_nlbias/nlbias_fits.json. Best-fit parameters per k_max at '
                      'z = 0.3, 1, 2 (A1, A2, bTA, a_c, W, alpha, a_1h); at_bound marks a parameter '
                      'within 1 percent of its bound (A1 in [0, 20], A2 in [-5, 5], b_TA in [0, 2], '
                      'a_c in [0, 1], alpha in [-1, 2]). Fixed parameters are at their fixed value.'),
            'kgrid': clean(kg), 'zs': list(zsel), 'models': {k: m['label'] for k, m in MODELS.items()},
            'lines': out}


def fob():
    runs = json.load(open(FOB))['runs']
    lines, missing = {}, []
    for probe, k0s, sfx in (('cs', K0_CS, ''), ('3x2', K0_3X2, '|gt0.5')):
        for form in FORMS:
            for key, m in MODELS.items():
                mdl = m[form][1]
                rows = {'fob': [], 'mu_om': [], 'mu_s8': [], 'chi2_ref': [], 'at_bound_ref': []}
                for K in k0s:
                    r = runs.get(f'{probe}+cmb|{mdl}|{K:g}|CP{sfx}|fisher')
                    if r is None:
                        missing.append(f'{probe}|{mdl}|{K:g}')
                        for q in rows:
                            rows[q].append(None)
                        continue
                    rows['fob'].append(r['fob'])
                    rows['mu_om'].append(r['mu'][0])
                    rows['mu_s8'].append(r['mu'][1])
                    rows['chi2_ref'].append(r['chi2_ref'])
                    rows['at_bound_ref'].append(r['at_bound_ref'])
                lines[f'{probe}|{form}|{key}'] = dict(model=mdl, **rows)
    if missing:
        print('missing runs:', ', '.join(missing))
    return {'about': ('Producer: notes/halo_ia_models/plots/make_figure_data.py from '
                      'analysis/cut_validation/fisher_minimize.json. Roman with the CMB prior, every '
                      'parameter free, CP truth (stitched FLAMINGO with the close-pair II term), '
                      'Fisher-only shift of (Omega_m, S8) with the flat priors as Gaussians '
                      '(Kable et al. 2019). cs: xi_+- at k_UV = K0/D(z); 3x2: xi_+- at K0, gamma_t '
                      'of sources behind lenses at 0.5, w at 0.2. fob: 2D figure of bias; mu: shift '
                      'in units of the marginalized sigma; chi2_ref: the IA-only reference fit at the '
                      'fiducial cosmology.'),
            'K0': {'cs': list(K0_CS), '3x2': list(K0_3X2)}, 'threshold': 0.3,
            'models': {k: m['label'] for k, m in MODELS.items()}, 'lines': lines}


def main():
    todo = sys.argv[1:] or ['gof', 'params', 'fob_k0scan']
    make = {'gof': gof, 'params': params, 'fob_k0scan': fob}
    for name in todo:
        os.makedirs(os.path.join(HERE, name), exist_ok=True)
        with open(os.path.join(HERE, name, 'data.json'), 'w') as fh:
            json.dump(make[name](), fh, indent=1)
        print('wrote', name)


if __name__ == '__main__':
    main()
