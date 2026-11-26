#!/usr/bin/env python3
"""Every result number of threex2_ia_amplitude.tex, written as LaTeX macros to numbers.tex.

Sources (each macro's comment in numbers.tex names its file and key):
    plots/<figure>/data.json                           the numbers as drawn (make_figure_data.py)
    analysis/cut_validation/fisher_minimize.json       runs not drawn (2x2pt, minimizer checks)
    analysis/cut_validation/check_hm_CP.log            the pipeline consistency check
Run with any Python 3 with numpy:  python make_numbers.py
"""
import fcntl
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
CV = os.path.join(REPO, 'analysis', 'cut_validation')
PL = os.path.join(HERE, 'plots')
MAC = {'NLA': 'Nla', 'TATT': 'Tatt', 'NLAz': 'Nlaz', 'NLA+q': 'Q', 'NLA+m': 'M', 'NLA+mq': 'Mq',
       'NLA+lzhz': 'Lz', 'NLA+nd': 'Nd', 'NLA+sn': 'Sn', 'TATT+nd': 'TattNd', 'TATT+q': 'TattQ',
       'NLAs': 'Hmk', 'NLAs+q': 'HmkQ', 'NLAs+m': 'HmkM', 'NLAs+nd': 'HmkNd', 'NLAt': 'Ctrl',
       'NLAt+q': 'CtrlQ', 'NLAt+m': 'CtrlM', 'NLAt+nd': 'CtrlNd', 'NLAt+sn': 'Oracle',
       'NLAt+lz': 'CtrlLz'}
KW = {0.3: 'Three', 0.6: 'Six', 0.8: 'Eight'}
PROBE = {'cs': 'Cs', '3x2': 'Tx'}


def load(name):
    return json.load(open(os.path.join(PL, name, 'data.json')))


def arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def runs():
    with open(os.path.join(CV, 'fisher_minimize.json.lock'), 'a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        r = json.load(open(os.path.join(CV, 'fisher_minimize.json')))['runs']
        fcntl.flock(lk, fcntl.LOCK_UN)
    return r


def main():
    out = []

    def m(name, val, fmt, src):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            val, fmt = '--', '{}'
        out.append(f'\\newcommand{{\\{name}}}{{{fmt.format(val)}}}  % {src}')

    fob = load('fob')
    thr = fob['threshold']
    for probe, P in PROBE.items():
        k0 = np.array(fob['K0'][probe], float)
        for mdl, M in MAC.items():
            line = fob['lines'].get(f'{probe}|{mdl}')
            if line is None:
                continue
            y = arr(line['fob'])
            src = f'plots/fob/data.json {probe}|{mdl}'
            for K, w in KW.items():
                j = int(np.argmin(np.abs(k0 - K)))
                m(f'fob{P}{M}{w}', y[j], '{:.2f}', f'{src}: FoB at K0 = {K}')
            ok = np.isfinite(y)
            m(f'fob{P}{M}Max', np.max(y[ok]), '{:.2f}', f'{src}: max FoB over K0')
            m(f'fob{P}{M}MaxSix', np.max(y[ok & (k0 <= 0.61)]), '{:.2f}', f'{src}: max FoB over K0 <= 0.6')
            bad = np.where(y >= thr)[0]
            last = k0[-1] if bad.size == 0 else (k0[bad[0] - 1] if bad[0] > 0 else np.nan)
            m(f'Kzero{P}{M}', last, '{:g}', f'{src}: largest K0 with FoB < 0.3 at every K0 up to it')
            mu = arr(line['mu_om'])
            m(f'muOm{P}{M}Eight', mu[-1], '{:+.2f}', f'{src}: mu(Omega_m) at the largest K0')
    cost = load('cost')
    for probe, P in PROBE.items():
        k0 = np.array(cost['K0'][probe], float)
        j = int(np.argmin(np.abs(k0 - 0.6)))
        for mdl, M in MAC.items():
            line = cost['lines'].get(f'{probe}|{mdl}')
            if line is None:
                continue
            for q, Q in (('s8', 'S'), ('om', 'O')):
                v = arr(line[q])
                m(f'cost{P}{M}{Q}', 100*(v[j] - 1), '{:.0f}',
                  f'plots/cost/data.json {probe}|{mdl} {q}: sigma / sigma({line["base"]}) - 1 at K0 = 0.6 [percent]')
                m(f'cost{P}{M}{Q}Max', 100*(np.nanmax(v) - 1), '{:.0f}', f'the same, max over K0 [percent]')

    amp = load('amplitude')
    z = arr(amp['zs'])
    A = arr(amp['A_truth'])
    pl = arr(amp['fits']['NLAt']['A'])/A - 1
    m('plDevZero', 100*pl[0], '{:+.0f}', 'plots/amplitude/data.json: power-law fit (NLAt, 3x2pt, K0 = 0.3) / truth - 1 at z = 0 [percent]')
    m('plDevOne', 100*pl[1], '{:+.0f}', 'the same at z = 0.1')
    mid = (z >= 0.15) & (z <= 0.55)
    assert pl[mid].max() < 0 and pl[z >= 2.9].max() < 0     # the text says "below the truth"
    m('plDevMid', -100*pl[mid].min(), '{:.0f}', 'the same, largest shortfall at z = 0.2 to 0.5 [percent below]')
    m('plDevHigh', -100*pl[z >= 2.9].min(), '{:.0f}', 'the same, largest shortfall at z = 3 and 4 [percent below]')
    i02, i1, i4 = (int(np.argmin(np.abs(z - v))) for v in (0.2, 1.0, 4.0))
    m('slopeLow', np.log(A[i1]/A[i02])/np.log(2.0/1.2), '{:.2f}', 'plots/amplitude/data.json: d ln A_truth / d ln(1+z) from z = 0.2 to 1')
    m('slopeHigh', np.log(A[i4]/A[i1])/np.log(5.0/2.0), '{:.2f}', 'the same from z = 1 to 4')
    m('etaPl', amp['fits']['NLAt']['params']['A1_2'], '{:.2f}', 'plots/amplitude/data.json NLAt: eta1 of the power-law reference fit')
    for key, mac in (('NLAt+q', 'Q'), ('NLAt+m', 'M'), ('NLAt+nd', 'Nd')):
        f = amp['fits'].get(key)
        if f is None:
            continue
        dev = arr(f['A'])/A - 1
        core = z <= 2.05
        m(f'ampDev{mac}', 100*np.max(np.abs(dev[core])), '{:.0f}',
          f'plots/amplitude/data.json {key}: max |fit / truth - 1| at z <= 2 [percent]')
    m('ampDevPl', 100*np.max(np.abs(pl[z <= 2.05])), '{:.0f}', 'plots/amplitude/data.json NLAt: the same for the power law')
    f = amp['fits'].get('NLAt+q')
    if f:
        m('etaTwoCtrl', f['params']['zevo_eta2'], '{:.2f}', 'plots/amplitude/data.json NLAt+q: eta2 of the reference fit')
    if amp['fits'].get('NLAt+q'):
        e2 = amp['fits']['NLAt+q']['params']['zevo_eta2']
        g = np.exp(e2*np.log((1 + z[z <= 2.05])/1.62)**2)
        m('gQmax', 100*np.max(np.abs(g - 1)), '{:.0f}', 'plots/amplitude/data.json NLAt+q: max |g - 1| at z <= 2 [percent]')
    f = amp['fits'].get('NLAt+m')
    if f:
        m('betaCtrl', f['params']['zevo_beta'], '{:.2f}', 'plots/amplitude/data.json NLAt+m: beta of the reference fit')
    lmax, lmean = arr(amp['logM_max']), arr(amp['mean_logM'])
    for i, w in ((0, 'Zero'), (1, 'One'), (2, 'Two')):
        m(f'logMmax{w}', lmax[i], '{:.2f}', f'plots/amplitude/data.json logM_max at z = {z[i]:g}')
    m('logMmeanZero', lmean[0], '{:.2f}', 'plots/amplitude/data.json mean_logM at z = 0')
    m('logMmeanMid', np.mean(lmean[(z >= 0.15) & (z <= 1.05)]), '{:.2f}', 'the same, mean over z = 0.2 to 1')

    R = runs()
    for mdl, M in (('NLA', 'Nla'), ('NLAz', 'Nlaz')):
        v = [R[k]['fob'] for k in R if k.startswith(f'2x2+cmb|{mdl}|') and k.endswith('|CP|fisher')]
        m(f'fobTwo{M}Max', max(v), '{:.2f}', f'fisher_minimize.json 2x2+cmb|{mdl}|*|CP|fisher: max FoB over K0')
    c = R['cs+cmb|NLAt|0.8|CP|fisher']
    t = R['3x2+cmb|NLAt|0.8|CP|gt0.5|fisher']
    m('shiftOmCs', c['mu'][0]*c['sigma'][0], '{:+.4f}', 'fisher_minimize.json cs NLAt K0 0.8: Omega_m shift')
    m('shiftOmTx', t['mu'][0]*t['sigma'][0], '{:+.4f}', 'the same, 3x2pt')
    m('sigOmCs', c['sigma'][0], '{:.4f}', 'cs NLAt K0 0.8: sigma(Omega_m)')
    m('sigOmTx', t['sigma'][0], '{:.4f}', '3x2pt NLAt K0 0.8: sigma(Omega_m)')
    for key, mac in (('3x2+cmb|NLA+q|0.6|CP|gt0.5', 'QSix'), ('3x2+cmb|NLA+q|0.8|CP|gt0.5', 'QEight'),
                     ('3x2+cmb|NLAs+q|0.6|CP|gt0.5', 'HmkQSix'), ('3x2+cmb|NLAs+q|0.8|CP|gt0.5', 'HmkQEight'),
                     ('3x2+cmb|NLA|0.8|CP|gt0.5', 'NlaEight'), ('cs+cmb|NLAs+q|0.8|CP', 'CsHmkQEight')):
        r = R.get(key)
        m(f'minFob{mac}', None if r is None else r['fob'], '{:.2f}', f'fisher_minimize.json {key}: minimizer FoB')
    log = open(os.path.join(CV, 'check_hm_CP.log')).read()
    for probe, P in (('cs', 'Cs'), ('3x2', 'Tx')):
        v = re.search(rf'^{re.escape(probe)} A1 3.5 eta 0.7: .*chi2\(native - template\) ([0-9.]+)', log, re.M)
        m(f'checkChi{P}', float(v.group(1)), '{:.2f}', f'analysis/cut_validation/check_hm_CP.log {probe} A1 3.5 eta 0.7')
    with open(os.path.join(HERE, 'numbers.tex'), 'w') as fh:
        fh.write('% Written by make_numbers.py; do not edit by hand.\n' + '\n'.join(out) + '\n')
    print(f'wrote {len(out)} macros')


if __name__ == '__main__':
    main()
