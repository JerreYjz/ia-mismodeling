#!/usr/bin/env python3
"""Every number of halo_ia_models.tex, written as LaTeX macros to numbers.tex.

Sources (each macro's comment in numbers.tex names its file and key):
    analysis/halo_nlbias/nlbias_fits.json          FLAMINGO fits (nlbias_fit.py)
    analysis/halo_nlbias/nlbias_roman_tables.npz    beta_NL integrals (nlbias_roman_tables.py)
    analysis/cut_validation/fisher_minimize.json    Roman figure of bias (fisher_minimize.py)
    notes/halo_ia_models/plots/fob_k0scan/data.json the same FoB numbers, as drawn
Run with any Python 3 with numpy:  python make_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
FITS = os.path.join(REPO, 'analysis', 'halo_nlbias', 'nlbias_fits.json')
TAB = os.path.join(REPO, 'analysis', 'halo_nlbias', 'nlbias_roman_tables.npz')
FOB = os.path.join(HERE, 'plots', 'fob_k0scan', 'data.json')
KEYS = ('base', 'hmc', 'alpha', 'alphaf', 'nlb')
# LaTeX macro names cannot hold digits or underscores
MNAME = {'base': 'Base', 'hmc': 'Hmc', 'alpha': 'Alpha', 'alphaf': 'Alphaf', 'nlb': 'Nlb'}
VAR = {('TATT', 'base'): 'tatt', ('TATT', 'hmc'): 'lin_tatt', ('TATT', 'alpha'): 'alpha_tatt',
       ('TATT', 'nlb'): 'nlb_tatt', ('NLA', 'base'): 'nla', ('NLA', 'hmc'): 'lin_nla',
       ('NLA', 'alpha'): 'alpha_nla', ('NLA', 'nlb'): 'nlb_nla'}
KSEL = {'A': 0.52, 'B': 0.70, 'C': 0.94}


def num(v):
    return np.nan if v is None else float(v)


def fmt(x, nd=2):
    return f'{x:.{nd}f}'


def main():
    out = []

    def mac(name, value, src):
        out.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {src}')

    d = json.load(open(FITS))
    kg = np.array(d['kgrid'])
    zs = d['zs']
    names = d['names']
    for (form, key), var in VAR.items():
        v = d['variants'][var]
        c2 = np.array([[num(x) for x in row] for row in v['chi2']])
        nu = np.array([[num(x) for x in row] for row in v['nu']])
        for tag, k in KSEL.items():
            j = int(np.argmin(np.abs(kg - k)))
            mac(f'gof{form.title()}{MNAME[key]}{tag}', fmt(c2[:, j].sum()/nu[:, j].sum()),
                f'nlbias_fits.json {var}: sum chi2 / sum nu at k_max {kg[j]:.3f}')
        cut = np.array([num(c) for c in v['cut']])
        mac(f'cut{form.title()}{MNAME[key]}Lo', fmt(np.nanmin(cut)), f'nlbias_fits.json {var}: min GOF cut over z')
        mac(f'cut{form.title()}{MNAME[key]}Hi', fmt(np.nanmax(cut)), f'nlbias_fits.json {var}: max GOF cut over z')
        mac(f'cut{form.title()}{MNAME[key]}Med', fmt(np.nanmedian(cut)), f'nlbias_fits.json {var}: median GOF cut')
        # physical checks over the nine snapshots at k_max = 0.52 and 0.94
        for tag in ('A', 'C'):
            j = int(np.argmin(np.abs(kg - KSEL[tag])))
            x = np.array([[num(a) for a in v['x'][s][j]] for s in range(len(zs))])
            ab = np.array([v['at_bound'][s][j] for s in range(len(zs))], bool)
            ia, ib, ic, ial = (names.index(n) for n in ('A1', 'bTA', 'a_c', 'alpha'))
            mac(f'acCap{form.title()}{MNAME[key]}{tag}', str(int(np.sum(ab[:, ic] & (x[:, ic] > 0.5)))),
                f'nlbias_fits.json {var}: snapshots (of 9) with a_c at its cap 1, k_max {kg[j]:.3f}')
            mac(f'aOne{form.title()}{MNAME[key]}{tag}Lo', fmt(x[:, ia].min()), f'{var}: min A1 over z, k_max {kg[j]:.3f}')
            mac(f'aOne{form.title()}{MNAME[key]}{tag}Hi', fmt(x[:, ia].max()), f'{var}: max A1 over z, k_max {kg[j]:.3f}')
            if form == 'TATT':
                mac(f'bta{MNAME[key]}{tag}Lo', fmt(x[:, ib].min()), f'{var}: min b_TA over z, k_max {kg[j]:.3f}')
                mac(f'bta{MNAME[key]}{tag}Hi', fmt(x[:, ib].max()), f'{var}: max b_TA over z, k_max {kg[j]:.3f}')
            if key == 'alpha':
                mac(f'alpha{form.title()}{tag}Lo', fmt(x[:, ial].min()), f'{var}: min alpha over z, k_max {kg[j]:.3f}')
                mac(f'alpha{form.title()}{tag}Hi', fmt(x[:, ial].max()), f'{var}: max alpha over z, k_max {kg[j]:.3f}')
        # drift of A1 between k_max = 0.29 and 0.94 (ratio, over z)
        j0, j1 = int(np.argmin(np.abs(kg - 0.29))), int(np.argmin(np.abs(kg - 0.94)))
        r = np.array([num(v['x'][s][j1][0])/num(v['x'][s][j0][0]) for s in range(len(zs))])
        mac(f'drift{form.title()}{MNAME[key]}Lo', fmt(r.min()), f'{var}: min A1(0.94)/A1(0.29) over z')
        mac(f'drift{form.title()}{MNAME[key]}Hi', fmt(r.max()), f'{var}: max A1(0.94)/A1(0.29) over z')

    # spread of A1 between the four models at k_max = 0.29, per snapshot
    j = int(np.argmin(np.abs(kg - 0.29)))
    for form in ('TATT', 'NLA'):
        A = np.array([[num(d['variants'][VAR[(form, k)]]['x'][s][j][0]) for s in range(len(zs))]
                      for k in ('base', 'hmc', 'alpha', 'nlb')])
        sp = 100*np.abs(A/A.mean(0) - 1).max(0)
        mac(f'lowkSpread{form.title()}Lo', f'{sp.min():.0f}', f'{form} form: min over z of max |A1/mean - 1| [%] at k_max {kg[j]:.3f}')
        mac(f'lowkSpread{form.title()}Hi', f'{sp.max():.0f}', f'{form} form: max over z of max |A1/mean - 1| [%] at k_max {kg[j]:.3f}')

    # the beta_NL integrals at z = 1 and the low-k dip
    t = np.load(TAB)
    s = int(np.argmin(np.abs(t['zs'] - 1.0)))
    for kk, tag in ((0.1, 'A'), (0.3, 'B'), (0.5, 'C'), (1.0, 'D')):
        i = int(np.argmin(np.abs(t['k'] - kk)))
        mac(f'ibcm{tag}', fmt(t['Ib_cm'][s, i]), f'nlbias_roman_tables.npz Ib_cm, z = 1, k = {t["k"][i]:.3f}')
        mac(f'ibcmLo{tag}', fmt(t['Ib_cm_lo'][s, i]), f'nlbias_roman_tables.npz Ib_cm_lo, z = 1, k = {t["k"][i]:.3f}')
    core = (t['zs'] >= 0.2) & (t['zs'] <= 2.0)
    m = (t['k'] > 0.05) & (t['k'] < 0.15)
    mac('dipLo', fmt(t['Ib_cm'][core][:, m].min()), 'Ib_cm, z = 0.2 to 2, 0.05 < k < 0.15: min')
    mac('dipHi', fmt(t['Ib_cm'][core][:, m].max()), 'Ib_cm, z = 0.2 to 2, 0.05 < k < 0.15: max')
    mac('lgMElo', fmt(t['lgM_E'][core][:, 0].min(), 1), 'nlbias_roman_tables.npz lgM_E: lowest emulator mass, z = 0.2 to 2')
    mac('lgMEhi', fmt(t['lgM_E'][core][:, 0].max(), 1), 'nlbias_roman_tables.npz lgM_E: lowest emulator mass, z = 0.2 to 2')

    # Roman figure of bias
    if os.path.isfile(FOB):
        f = json.load(open(FOB))
        for probe, kref in (('cs', 0.35), ('3x2', 0.2)):
            k0 = np.array(f['K0'][probe])
            jr = int(np.argmin(np.abs(k0 - kref)))
            for form in ('TATT', 'NLA'):
                for key in KEYS:
                    ln = f['lines'][f'{probe}|{form}|{key}']
                    fob = np.array([num(x) for x in ln['fob']])
                    p = 'Cs' if probe == 'cs' else 'Tx'
                    mac(f'fob{p}{form.title()}{MNAME[key]}', fmt(fob[jr]),
                        f'fob_k0scan/data.json {probe}|{form}|{key} at K0 = {k0[jr]:g}')
                    c2 = np.array([num(x) for x in ln['chi2_ref']])
                    mac(f'ctwo{p}{form.title()}{MNAME[key]}', fmt(c2[jr], 1),
                        f'fob_k0scan/data.json {probe}|{form}|{key}: chi2 of the IA reference fit at K0 = {k0[jr]:g}')
                    mac(f'ctwoTop{p}{form.title()}{MNAME[key]}', fmt(c2[-1], 1),
                        f'fob_k0scan/data.json {probe}|{form}|{key}: chi2 of the IA reference fit at K0 = {k0[-1]:g}')
                    mac(f'fobTop{p}{form.title()}{MNAME[key]}', fmt(fob[-1]),
                        f'fob_k0scan/data.json {probe}|{form}|{key} at K0 = {k0[-1]:g}')
                    mac(f'fobMax{p}{form.title()}{MNAME[key]}', fmt(np.nanmax(fob)),
                        f'fob_k0scan/data.json {probe}|{form}|{key}: max over K0')
                    ok = np.where(fob < f['threshold'])[0]
                    # the largest K0 below which every K0 has FoB < 0.3
                    jmax = -1
                    for j in range(len(k0)):
                        if fob[j] < f['threshold']:
                            jmax = j
                        else:
                            break
                    mac(f'kmax{p}{form.title()}{MNAME[key]}', f'{k0[jmax]:g}' if jmax >= 0 else 'none',
                        f'fob_k0scan/data.json {probe}|{form}|{key}: largest K0 with FoB < 0.3 at every smaller K0')
                    bnd = sorted({b for row in ln['at_bound_ref'] if row for b in row})
                    mac(f'bnd{p}{form.title()}{MNAME[key]}', ', '.join(b.replace('roman_', '').replace('_', r'\_') for b in bnd) or 'none',
                        f'fob_k0scan/data.json {probe}|{form}|{key}: reference parameters at a bound, any K0')
    with open(os.path.join(HERE, 'numbers.tex'), 'w') as fh:
        fh.write('% Written by make_numbers.py; do not edit by hand.\n' + '\n'.join(out) + '\n')
    print(f'wrote numbers.tex ({len(out)} macros)')


if __name__ == '__main__':
    main()
