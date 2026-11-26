#!/usr/bin/env python3
"""Every result number of halo_kernel_model.tex, written as LaTeX macros to numbers.tex.

Sources (each macro's comment in numbers.tex names its file and key):
    analysis/halo_kernel/model/hk_model.json     model, goodness of fit (hk_model.py)
    plots/<figure>/data.json                     the numbers as drawn (make_figure_data.py)
    analysis/cut_validation/hm_templates_cs_CP.npz   template amplitudes (hk_templates.py)
Run with any Python 3 with numpy:  python make_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
MODEL = os.path.join(REPO, 'analysis', 'halo_kernel', 'model', 'hk_model.json')
TPL = os.path.join(REPO, 'analysis', 'cut_validation', 'hm_templates_cs_CP.npz')
PL = os.path.join(HERE, 'plots')
CORE = ('74', '72', '70', '68', '66', '63', '58', '48', '38')


def load(name):
    return json.load(open(os.path.join(PL, name, 'data.json')))


def arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def at(k, v, k0):
    '''Value of v on the log-k grid k at k0 (log interpolation in k).'''
    k, v = arr(k), arr(v)
    return float(np.interp(np.log(k0), np.log(k), v))


def main():
    out = []

    def m(name, val, fmt, src):
        out.append(f'\\newcommand{{\\{name}}}{{{fmt.format(val)}}}  % {src}')

    gof = load('gof')
    kg = arr(gof['kgrid'])
    j = {k: int(np.argmin(np.abs(kg - k))) for k in (0.52, 0.94)}
    for key, mac in (('w', 'W'), ('wg', 'Wg'), ('v1', 'Vone'), ('tatt', 'Tatt'), ('lin', 'Lin'), ('emu', 'Emu'),
                     ('tab', 'Tab')):
        line = gof['lines'][key]
        s = arr(line['summed'])
        m(f'gof{mac}A', s[j[0.52]], '{:.2f}', f'plots/gof/data.json {key}: sum chi2 / sum nu at k_max 0.52')
        m(f'gof{mac}C', s[j[0.94]], '{:.2f}', f'plots/gof/data.json {key}: sum chi2 / sum nu at k_max 0.94')
        c = arr(line['cut'])
        c = c[np.isfinite(c)]
        if c.size:
            m(f'cut{mac}Lo', c.min(), '{:.2f}', f'plots/gof/data.json {key}: min GOF cut over the core z')
            m(f'cut{mac}Hi', c.max(), '{:.2f}', f'plots/gof/data.json {key}: max GOF cut')
            m(f'cut{mac}Med', np.median(c), '{:.2f}', f'plots/gof/data.json {key}: median GOF cut')

    sr = load('shape_response')['panels']['1']
    for key, mac in (('shape', 'Shape'), ('density', 'Dens'), ('matter', 'Matter'), ('emulator', 'Emu')):
        m(f'resp{mac}', at(sr['k'], sr[key], 0.94), '{:.1f}',
          f'plots/shape_response/data.json z = 1, {key} at k = 0.94')

    wt = load('weighting')['panels']['1']
    k = arr(wt['k'])
    low = (k >= 0.02) & (k <= 0.06)
    m('ggBinnedMiss', 100*(1 - np.mean(arr(wt['binned'])[low])), '{:.0f}',
      'plots/weighting/data.json z = 1: 1 - binned P_GG/P_gg, mean over k = 0.02 to 0.06 [percent]')
    m('ggMeanMiss', 100*(1 - np.mean(arr(wt['mean_hod'])[low])), '{:.0f}',
      'plots/weighting/data.json z = 1: 1 - mean-HOD P_GG/P_gg, k = 0.02 to 0.06 [percent]')
    ii = (k >= 0.1) & (k <= 0.5)
    both = [load('weighting')['panels'][z] for z in ('0.3', '1')]
    full = np.mean([np.mean(1/arr(p['ii_full'])[ii]) for p in both])
    det = np.mean([np.mean(np.abs(arr(p['ii_det'])[ii] - 1)) for p in both])
    m('iiFullExcess', 100*(full - 1), '{:.0f}',
      'plots/weighting/data.json: mean of prediction/galaxy II - 1, full halo-shape II, k = 0.1 to 0.5, z = 0.3 and 1 [percent]')
    m('iiDetDev', 100*det, '{:.0f}',
      'plots/weighting/data.json: mean |galaxy II/deterministic II - 1|, k = 0.1 to 0.5, z = 0.3 and 1 [percent]')

    res = json.load(open(MODEL))
    kgm = np.array([0.0899, 0.1204, 0.1615, 0.2167, 0.2907, 0.3900, 0.5229, 0.7013, 0.9406, 1.2618, 1.6919,
                    2.2686, 3.0424])
    j94, j126 = int(np.argmin(np.abs(kgm - 0.94))), int(np.argmin(np.abs(kgm - 1.26)))
    mid = [s for s in res if 0.15 < res[s]['z'] < 1.6]
    high = [s for s in res if res[s]['z'] > 1.9]
    j52 = int(np.argmin(np.abs(kgm - 0.52)))

    def grad(var, ss, j):
        return np.array([res[s][var]['alpha'][j] for s in ss])

    def shift(var, ss):
        return np.array([abs(res[s][var]['alpha'][j52] - res[s][var]['alpha'][j126]) for s in ss])

    def shift_sig(var, ss):
        return np.array([abs(res[s][var]['alpha'][j52] - res[s][var]['alpha'][j126]) / res[s][var]['alpha_err'][j52]
                         for s in ss])
    # the final model: saturated kernel gradient A (ws-ls-sat)
    a = np.concatenate([grad('ws-ls-sat', mid, j94), grad('ws-ls-sat', mid, j126)])
    m('ALo', a.min(), '{:.2f}', 'hk_model.json ws-ls-sat A at k_max 0.94 and 1.26: min over z = 0.2 to 1.5')
    m('AHi', a.max(), '{:.2f}', 'the same, max')
    m('AHighFit', np.mean(grad('ws-ls-sat', high, j126)), '{:.2f}', 'hk_model.json ws-ls-sat: mean A at k_max 1.26, z = 2 to 4')
    m('AShiftMid', shift('ws-ls-sat', mid).max(), '{:.2f}', 'hk_model.json ws-ls-sat: max |A(0.52) - A(1.26)|, z = 0.2 to 1.5')
    m('AShiftHigh', shift('ws-ls-sat', high).max(), '{:.2f}', 'the same, z = 2 to 4')
    m('AShiftSig', max(shift_sig('ws-ls-sat', mid).max(), shift_sig('ws-ls-sat', high).max()), '{:.0f}',
      'hk_model.json ws-ls-sat: max |A(0.52) - A(1.26)| / sigma(A(0.52)) over all snapshots')
    rl = np.array([res[s]['R_L'] for s in res])
    m('RLLo', rl.min(), '{:.1f}', 'hk_model.json R_L: min over snapshots (z = 4) [Mpc/h]')
    m('RLHi', rl.max(), '{:.1f}', 'hk_model.json R_L: max over snapshots (z = 0) [Mpc/h]')
    # the first version: k^2 gradient alpha on the deterministic II (w-ls-grad)
    m('alphaShiftHigh', shift('w-ls-grad', high).min(), '{:.2f}', 'hk_model.json w-ls-grad: min |alpha(0.52) - alpha(1.26)|, z = 2 to 4')
    m('alphaShiftHighSig', shift_sig('w-ls-grad', high).max(), '{:.0f}', 'the same in sigma, max over z = 2 to 4')
    m('alphaShiftHighSigK', shift_sig('ws-ls-grad', high).max(), '{:.0f}',
      'hk_model.json ws-ls-grad (residual II, k^2 gradient): max shift in sigma, z = 2 to 4')
    line = res['58']
    nc = np.array(line['Nc'])
    kc = np.array(line['kappa_w_c'])[nc >= 0.01]
    ks = np.array(line['kappa_w_s'])
    m('kcLo', kc.min(), '{:.2f}', 'hk_model.json z = 1 kappa_w_c (bins with Nc >= 0.01): min')
    m('kcHi', kc.max(), '{:.2f}', 'hk_model.json z = 1 kappa_w_c: max')
    m('ksLo', ks.min(), '{:.2f}', 'hk_model.json z = 1 kappa_w_s: min')
    m('ksHi', ks.max(), '{:.2f}', 'hk_model.json z = 1 kappa_w_s: max')
    rl = [res[s]['repro_maxdev_sigma'] for s in res]
    f = np.load(os.path.join(REPO, 'analysis', 'roman_dchi2', 'v2', 'fits.npz'))
    lowdev = []
    for s in res:
        L = res[s]
        si = int(np.argmin(np.abs(f['zs'] - L['z'])))
        kk = np.array(L['k'])
        mk = np.array(L['okk']) & (kk >= 0.05) & (kk <= 0.53)
        lowdev.append(np.max(np.abs(np.array(L['offset'])/f['err'][si])[:, mk]))
    m('reproLow', max(lowdev), '{:.1f}', 'hk_model.json offset/err: max over snapshots, k = 0.05 to 0.52')
    m('reproHigh', max(rl), '{:.1f}', 'hk_model.json repro_maxdev_sigma: max over snapshots (k <= 1.26)')

    t = np.load(TPL)
    A = t['A_eff_hms']
    m('AeffLo', A[0], '{:.1f}', 'hm_templates_cs_CP.npz A_eff_hms at z = 0')
    m('AeffHi', A[-1], '{:.1f}', 'hm_templates_cs_CP.npz A_eff_hms at z = 4')

    rs = load('residual')
    zr = arr(rs['zs'])
    for zz, mac in ((0.0, 'Zero'), (1.0, 'One'), (2.0, 'Two'), (4.0, 'Four')):
        i = int(np.argmin(np.abs(zr - zz)))
        m(f'krc{mac}', arr(rs['ratio_c'])[i], '{:.2f}', f'plots/residual/data.json ratio_c (kappa_r/kappa_d, centrals) at z = {zz:g}')
    i = int(np.argmin(np.abs(zr - 1.0)))
    m('krsOne', arr(rs['ratio_s'])[i], '{:.2f}', 'plots/residual/data.json ratio_s at z = 1')
    m('negIIMax', 100*rs['neg_ii_max'], '{:.0f}', 'plots/residual/data.json neg_ii_max [percent of model P_EE]')
    m('negBBMax', 100*rs['neg_bb_max'], '{:.0f}', 'plots/residual/data.json neg_bb_max [percent of model P_BB]')
    p2 = rs['panels']['2']
    kk = arr(p2['k'])
    band = (kk >= 0.1) & (kk <= 0.55)
    m('iiDetExcessTwo', 100*(np.mean(arr(p2['det'])[band]) - 1), '{:.0f}',
      'plots/residual/data.json z = 2: mean galaxy 2h II / deterministic II - 1, k = 0.1 to 0.55 [percent]')
    m('iiResExcessTwo', 100*(np.mean(arr(p2['det_res'])[band]) - 1), '{:.0f}',
      'the same over deterministic + residual II [percent]')
    m('iiFullExcessTwo', 100*(np.mean(arr(p2['full_ls'])[band]) - 1), '{:.0f}',
      'the same over kappa_d^2 x full halo-shape II [percent]')
    for sn, zz in (('38', 'Two'), ('18', 'Three'), ('14', 'Four')):
        for var, mac in (('w-ls-grad', 'Vone'), ('ws-ls-sat', 'Wg')):
            c = res[sn][var]['cut']
            m(f'cut{mac}Z{zz}', c, '{:.2f}', f'hk_model.json {var} GOF cut at z = {res[sn]["z"]:g}')
        m(f'gofWgZ{zz}', res[sn]['ws-ls-sat']['chi2'][j94]/res[sn]['ws-ls-sat']['nu'][j94], '{:.1f}',
          f'hk_model.json ws-ls-sat chi2/nu at k_max 0.94, z = {res[sn]["z"]:g}')
        m(f'gofVoneZ{zz}', res[sn]['w-ls-grad']['chi2'][j94]/res[sn]['w-ls-grad']['nu'][j94], '{:.1f}',
          f'hk_model.json w-ls-grad chi2/nu at k_max 0.94, z = {res[sn]["z"]:g}')

    fob = load('fob_k0scan')
    thr = fob['threshold']
    for probe, P in (('cs', 'Cs'), ('3x2', 'Tx')):
        k0 = np.array(fob['K0'][probe], float)
        for key, mac in (('NLA', 'Nla'), ('TATT', 'Tatt'), ('hms0', 'W'), ('hms', 'Wg'), ('hmsz', 'Kz'), ('hmk', 'Vone'),
                         ('ctrl', 'Ctrl')):
            y = arr(fob['lines'][f'{probe}|{key}']['fob'])
            m(f'fob{P}{mac}Eight', y[-1], '{:.2f}', f'plots/fob_k0scan/data.json {probe}|{key}: FoB at K0 = 0.8')
            m(f'fob{P}{mac}Lo', np.nanmin(y), '{:.2f}', f'plots/fob_k0scan/data.json {probe}|{key}: min FoB over K0')
            m(f'fob{P}{mac}Hi', np.nanmax(y), '{:.2f}', f'plots/fob_k0scan/data.json {probe}|{key}: max FoB over K0')
            # the largest K0 below which every FoB is under the threshold
            ok = np.where(y < thr)[0]
            last = k0[0] if y[0] >= thr else k0[np.argmax(y >= thr) - 1] if np.any(y >= thr) else k0[-1]
            m(f'Kzero{P}{mac}', last, '{:g}', f'plots/fob_k0scan/data.json {probe}|{key}: largest K0 with FoB < 0.3 at every smaller K0')
            if probe == '3x2' and key == 'ctrl':
                first = k0[np.argmax(y >= thr)] if np.any(y >= thr) else np.nan
                m('KzeroTxCtrl', first, '{:g}', 'plots/fob_k0scan/data.json 3x2|ctrl: first K0 with FoB >= 0.3')
    v1 = json.load(open(os.path.join(REPO, 'analysis', 'halo_kernel', 'model', 'fisher_hmk_v1.json')))['runs']
    m('fobVoneCsWgEight', v1['cs+cmb|NLAk|0.8|CP|fisher']['fob'], '{:.2f}',
      'analysis/halo_kernel/model/fisher_hmk_v1.json cs NLAk K0 0.8 (first template construction)')
    m('fobVoneCsWgTwo', v1['cs+cmb|NLAk|0.2|CP|fisher']['fob'], '{:.2f}', 'the same at K0 0.2')
    runs = json.load(open(os.path.join(REPO, 'analysis', 'cut_validation', 'fisher_minimize.json')))['runs']
    for mdl, mac in (('NLAkz', 'VoneKz'), ('NLAd1', 'DiagSat'), ('NLAd2', 'DiagRes')):
        m(f'fobCs{mac}Eight', runs[f'cs+cmb|{mdl}|0.8|CP|fisher']['fob'], '{:.2f}',
          f'analysis/cut_validation/fisher_minimize.json cs {mdl} K0 0.8 (diagnostic, truth shape above z = 1.5)')
    # KzeroTxCtrl is redefined above as the first crossing; drop the earlier definition
    seen, lines = set(), []
    for ln in reversed(out):
        name = ln.split('}')[0]
        if name not in seen:
            seen.add(name)
            lines.append(ln)
    with open(os.path.join(HERE, 'numbers.tex'), 'w') as fh:
        fh.write('% Written by make_numbers.py; do not edit by hand.\n')
        fh.write('\n'.join(reversed(lines)) + '\n')
    print(f'wrote {len(lines)} macros')


if __name__ == '__main__':
    main()
