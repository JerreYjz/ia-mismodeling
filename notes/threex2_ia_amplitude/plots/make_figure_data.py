#!/usr/bin/env python3
"""Writes the data.json of every figure of notes/threex2_ia_amplitude from the
analysis outputs (any Python 3 with numpy):

    python make_figure_data.py [figure ...]      (default: all)

    amplitude   the FLAMINGO truth's IA amplitude per snapshot against the power
                law and the freer A(z) forms, as fitted to the 3x2pt data vector with
                the truth's own shape; the sample's mean log M* per snapshot
                (analysis/cut_validation/hm_templates_cs_CP.npz, sample_mstar_hist.npz,
                fisher_minimize.json)
    fob         Roman figure of bias against the xi_+- cut for every A(z) strategy,
                cosmic shear and 3x2pt (fisher_minimize.json)
    cost        the marginalized sigma of Omega_m and S8 of each strategy over that of
                the power law with the same shape (fisher_minimize.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
CV = os.path.join(REPO, 'analysis', 'cut_validation')
X0 = 1.62
Z_NODES = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
ND_FREE = {0.0: 'zevo_n0', 0.2: 'zevo_n02', 0.5: 'zevo_n05', 2.0: 'zevo_n2'}
SN_FIXED = (8, 12)
K0 = {'cs': (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8), '3x2': (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)}
SUFFIX = {'cs': 'CP|fisher', '3x2': 'CP|gt0.5|fisher'}
# model key in fisher_minimize.json: (label, what it is)
MODELS = {
    'NLA': ('NLA', 'power law A1 x^eta1, 2 IA parameters'),
    'TATT': ('TATT', 'TATT, power laws, 5 IA parameters'),
    'NLAz': ('NLA, A1 per source bin', 'one A1 per source bin (cosmolike IA_redshift_evolution 2), 8 parameters'),
    'NLA+q': ('NLA, curved power law', 'A1 x^(eta1 + eta2 ln x), 3 parameters'),
    'NLA+m': ('NLA, mass-informed', 'A1 x^eta1 <(M*/10^10.5)^beta>_z, 3 parameters'),
    'NLA+mq': ('NLA, mass-informed + curvature', '4 parameters'),
    'NLA+lzhz': ('NLA, tents at z = 0, 0.2, 2, 3', '6 parameters'),
    'NLA+nd': ('NLA, nodes', 'g piecewise linear on z = 0, 0.2, 0.5, 2 (1 at z = 1, 4), 6 parameters'),
    'NLA+sn': ('NLA, free per snapshot', '13 parameters'),
    'TATT+q': ('TATT, curved power laws', '6 parameters'),
    'TATT+nd': ('TATT, nodes', '9 parameters'),
    'NLAs': ('HMk', 'NLA + HMk template, power law, 2 parameters'),
    'NLAs+q': ('HMk, curved power law', '3 parameters'),
    'NLAs+m': ('HMk, mass-informed', '3 parameters'),
    'NLAs+nd': ('HMk, nodes', '6 parameters'),
    'NLAt': ('truth shape (control)', 'NLA + the truth-shape template, power law, 2 parameters'),
    'NLAt+q': ('truth shape, curved', '3 parameters'),
    'NLAt+m': ('truth shape, mass-informed', '3 parameters'),
    'NLAt+nd': ('truth shape, nodes', '6 parameters'),
    'NLAt+lz': ('truth shape, tents at z = 0 and 0.2', '4 parameters'),
    'NLAt+sn': ('truth shape, free per snapshot (oracle)', '13 parameters'),
}


def clean(a, nd=5):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), np.round(a, nd), None).tolist()


_RUNS = None


def runs():
    '''fisher_minimize.json, read once under the lock its writers hold.'''
    global _RUNS
    if _RUNS is None:
        import fcntl
        with open(os.path.join(CV, 'fisher_minimize.json.lock'), 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            _RUNS = json.load(open(os.path.join(CV, 'fisher_minimize.json')))['runs']
            fcntl.flock(lk, fcntl.LOCK_UN)
    return _RUNS


def mstar():
    h = np.load(os.path.join(CV, 'sample_mstar_hist.npz'))
    return h['zs'], h['hist'].astype(float), 0.5*(h['edges'][1:] + h['edges'][:-1])


def g_of(kind, zs, p):
    '''The freer evolution g(z) of fisher_minimize.zevo_g at the snapshots.'''
    lx = np.log((1 + zs)/X0)
    if kind == '':
        return np.ones_like(zs)
    if kind == 'q':
        return np.exp(p['zevo_eta2']*lx**2)
    if kind == 'm':
        _, H, lm = mstar()
        w = (H*10**(p['zevo_beta']*(lm - 10.5))).sum(1)/H.sum(1)
        return w/w[6]
    if kind == 'nd':
        return 1 + np.interp(zs, Z_NODES, [p.get(ND_FREE[z], 0.0) if z in ND_FREE else 0.0 for z in Z_NODES])
    if kind == 'sn':
        c = np.zeros(13)
        c[[i for i in range(13) if i not in SN_FIXED]] = [p[f'zevo_s{i}'] for i in range(13) if i not in SN_FIXED]
        return 1 + c
    raise ValueError(kind)


def amplitude():
    R = runs()
    t = np.load(os.path.join(CV, 'hm_templates_cs_CP.npz'))
    zs, A = t['zs'], t['A_eff_hmt']
    zh, H, lm = mstar()
    assert np.allclose(zh, zs, atol=1e-3)
    fits = {}
    for mdl in ('NLAt', 'NLAt+q', 'NLAt+m', 'NLAt+nd'):
        r = R.get(f'3x2+cmb|{mdl}|0.3|CP|gt0.5|fisher')
        if r is None:
            print('missing', mdl)
            continue
        p = r['ref']
        kind = mdl.split('+')[1] if '+' in mdl else ''
        a = p['roman_A1_1']*((1 + zs)/X0)**p['roman_A1_2']*g_of(kind, zs, p)
        fits[mdl] = {'label': MODELS[mdl][0], 'A': clean(a, 4), 'chi2_ref': round(r['chi2_ref'], 3),
                     'slope': clean(np.diff(np.log(a))/np.diff(np.log(1 + zs)), 4),
                     'params': {k[6:] if k.startswith('roman_') else k: round(v, 4) for k, v in p.items()
                                if 'A1' in k or 'zevo' in k}}
    return {'about': ('Producer: notes/threex2_ia_amplitude/plots/make_figure_data.py. A_truth: the '
                      'effective NLA amplitude of the FLAMINGO CP truth per snapshot (A_eff_hmt of '
                      'analysis/cut_validation/hm_templates_cs_CP.npz: the amplitude at which the '
                      'truth-shape template reproduces the truth). fits: A(z) at the snapshots of the '
                      'reference IA fit (fixed cosmology) to the 3x2pt data vector at K0 = 0.3 with the '
                      'truth\'s own shape (fisher_minimize.json, models NLAt, NLAt+q, NLAt+m, NLAt+nd). '
                      'mean_logM: mean log10 M*/Msun of the IA sample per snapshot (sample_mstar_hist.npz); '
                      'logM_max: its upper edge (97.5th percentile of the target sample). slope_truth: '
                      'the local slope d ln A / d ln(1+z) of the truth between neighbouring snapshots, at '
                      'their mid redshifts slope_z.'),
            'zs': clean(zs), 'A_truth': clean(A, 4),
            'slope_z': clean(0.5*(zs[1:] + zs[:-1]), 3),
            'slope_truth': clean(np.diff(np.log(A))/np.diff(np.log(1 + zs)), 4),
            'mean_logM': clean((H*lm).sum(1)/H.sum(1), 4),
            'logM_max': clean([lm[np.nonzero(h)[0].max()] + 0.005 for h in H], 3),
            'fits': fits}


def table(probe, key):
    R = runs()
    out = {}
    for mdl in MODELS:
        rows = []
        for K in K0[probe]:
            r = R.get(f'{probe}+cmb|{mdl}|{K:g}|{SUFFIX[probe]}')
            rows.append(None if r is None else r[key])
        if any(x is not None for x in rows):
            out[mdl] = rows
    return out


def fob():
    lines = {}
    for probe in K0:
        for mdl, v in table(probe, 'fob').items():
            mu = table(probe, 'mu')[mdl]
            lines[f'{probe}|{mdl}'] = {'fob': clean(v, 4),
                                        'mu_om': clean([None if m is None else m[0] for m in mu], 3),
                                        'mu_s8': clean([None if m is None else m[1] for m in mu], 3)}
    return {'about': ('Producer: notes/threex2_ia_amplitude/plots/make_figure_data.py from '
                      'analysis/cut_validation/fisher_minimize.json. Roman, CMB prior, every parameter free, '
                      'CP truth, Fisher shift of (Omega_m, S8): cs = xi_+- at k_UV = K0/D(z); 3x2 = xi_+- '
                      'at K0, gamma_t of sources behind lenses at 0.5, w at 0.2. fob: 2D figure of bias; mu: '
                      'shift over the marginalized sigma.'),
            'K0': {p: list(v) for p, v in K0.items()}, 'threshold': 0.3,
            'models': {k: {'label': v[0], 'what': v[1]} for k, v in MODELS.items()}, 'lines': lines}


def cost():
    lines = {}
    for probe in K0:
        sig = table(probe, 'sigma')
        for mdl, v in sig.items():
            base = 'NLAs' if mdl.startswith('NLAs') else 'NLAt' if mdl.startswith('NLAt') else \
                'TATT' if mdl.startswith('TATT') else 'NLA'
            b = sig.get(base)
            if b is None or mdl == base:
                continue
            lines[f'{probe}|{mdl}'] = {
                'base': base,
                'om': clean([None if x is None or y is None else x[0]/y[0] for x, y in zip(v, b)], 4),
                's8': clean([None if x is None or y is None else x[1]/y[1] for x, y in zip(v, b)], 4)}
    return {'about': ('Producer: notes/threex2_ia_amplitude/plots/make_figure_data.py from '
                      'analysis/cut_validation/fisher_minimize.json. The marginalized Fisher sigma of '
                      'Omega_m (om) and S8 (s8) of each A(z) strategy over that of the power law with the '
                      'same IA shape (base), per K0: the price of the extra freedom.'),
            'K0': {p: list(v) for p, v in K0.items()}, 'lines': lines}


def main():
    todo = sys.argv[1:] or ['amplitude', 'fob', 'cost']
    make = {'amplitude': amplitude, 'fob': fob, 'cost': cost}
    for name in todo:
        os.makedirs(os.path.join(HERE, name), exist_ok=True)
        with open(os.path.join(HERE, name, 'data.json'), 'w') as fh:
            json.dump(make[name](), fh, indent=1)
        print('wrote', name)


if __name__ == '__main__':
    main()
