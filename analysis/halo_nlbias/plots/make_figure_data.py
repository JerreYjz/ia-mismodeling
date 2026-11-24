#!/usr/bin/env python3
"""Write the data.json of every figure of halo_nlbias.tex from the analysis
outputs of this folder. The only script that knows where those files are.

    python make_figure_data.py [figure ...]      # default: all

Figures
  gi_kernel    the GI 2-halo spectrum per unit -A1 C1bar P_lin of HMc, of the
               alpha refit and of the nonlinear-bias model, with FLAMINGO's
               P_dE divided by the low-k NLA amplitude, at z = 0.3, 1, 2
  bias_checks  the halo model with and without beta_NL against FLAMINGO's own
               galaxy-matter (z = 1.5, 2) and galaxy-galaxy (z = 0.3, 1) spectra
  fit_gof      summed chi2/nu over the core snapshots against k_max
  fit_params   A1 over the low-k NLA amplitude, and a_1h, against k_max
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(NOTE, '..', '..'))
ANA = os.path.join(REPO, 'analysis')
sys.path.insert(0, os.path.join(ANA, 'tatt_fit'))

# the snapshots of the spectrum figures, and the k_max of the alpha used there
Z_SHOW = (0.3, 1.0, 2.0)
KMAX_ALPHA = 0.70
# k range drawn [h/Mpc]
K_DRAW = (0.06, 1.8)
# the k_max of the low-k NLA amplitude (v2 fits, NLA_x_raw) [h/Mpc]
KMAX_NLA = 0.22
# variants of the goodness-of-fit and parameter figures, with their labels
GOF_VARIANTS = {'lin_tatt': 'HMc ($I_1 P_{\\rm lin}$)', 'alpha_tatt': r'$\alpha$ refit',
                'nlb_tatt': r'$\beta_{\rm NL}$', 'nlb_tatt_sat': r'$\beta_{\rm NL}$ + satellites',
                'nlb_tt': r'$\beta_{\rm NL}$, $b_{\rm TA} = 0$'}


def load():
    fits = np.load(os.path.join(ANA, 'roman_dchi2', 'v2', 'fits.npz'))
    nlb = np.load(os.path.join(NOTE, 'nlbias_tables.npz'))
    with open(os.path.join(NOTE, 'nlbias_fits.json')) as fh:
        res = json.load(fh)
    return fits, nlb, res


def clean(a):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, None).tolist()


def column(res, variant, q, j):
    '''Parameter or statistic q of a variant at k_max index j, per snapshot.'''
    v = res['variants'][variant]
    if q == 'chi2' or q == 'nu':
        return np.array([np.nan if r is None or r[j] is None else r[j] for r in v[q]], float)
    i = res['names'].index(q)
    return np.array([np.nan if r[j] is None or r[j][i] is None else r[j][i] for r in v['x']], float)


def low_k_nla(fits, s):
    '''The NLA amplitude of v2 at k_max = KMAX_NLA.'''
    j = int(np.argmin(np.abs(fits['kgrid'] - KMAX_NLA)))
    return float(fits['NLA_x_raw'][s, j, 0]), float(fits['kgrid'][j])


def gi_kernel(fits, nlb, res):
    import fit_tatt_flamingo as F
    model = F.TATTModel(halofit='takahashi')
    kd = fits['k'][0]
    idn = np.searchsorted(nlb['k'], kd)
    kg = np.array(res['kgrid'])
    ja = int(np.argmin(np.abs(kg - KMAX_ALPHA)))
    m = (kd >= K_DRAW[0]) & (kd <= K_DRAW[1])
    panels = []
    for z in Z_SHOW:
        s = int(np.argmin(np.abs(fits['zs'] - z)))
        t = int(np.argmin(np.abs(nlb['zs'] - z)))
        r = res['zs'].index(float(nlb['zs'][t]))
        pl = nlb['Plin_v6'][t][idn]
        i1 = nlb['I1'][t][idn]
        pnl = fits['PNL'][s]
        alpha = column(res, 'alpha_tatt', 'alpha', ja)[r]
        a1, kmax_nla = low_k_nla(fits, s)
        c1 = a1*0.01389*0.306/model.D(float(fits['zs'][s]))
        pde = fits['y'][s][0]/fits['b'][s]
        ede = fits['err'][s][0]/fits['b'][s]
        panels.append({
            'z': float(fits['zs'][s]), 'alpha': float(alpha), 'A1_nla': a1,
            'HMc': clean(i1[m]),
            'nlb': clean((i1 + nlb['Ib_cm'][t][idn])[m]),
            'nlb_lo': clean((i1 + nlb['Ib_cm_lo'][t][idn])[m]),
            'alpha_refit': clean((i1 + alpha*(pnl/pl - i1))[m]),
            'NLA': clean((pnl/pl)[m]),
            'flamingo': clean((-pde/(c1*pl))[m]), 'flamingo_err': clean((ede/(c1*pl))[m])})
    return {'about': ('GI 2-halo spectra per unit -A1 C1bar P_lin(k) (dimensionless): HMc = I_1(k); '
                      'nlb = I_1 + I^beta_cm (Dark Emulator beta_NL held below its lowest mass); '
                      'nlb_lo = the same with beta_NL = 0 below that mass; alpha_refit = I_1 + alpha '
                      f'(P_NL/P_lin - I_1) with alpha of the alpha_tatt fit at k_max = {KMAX_ALPHA}; '
                      'NLA = P_NL/P_lin. flamingo: FLAMINGO P_dE = P_hE/b(k) over -A1 C1bar P_lin '
                      f'with A1 of the v2 NLA fit at k_max = {KMAX_NLA} (all terms of the data, '
                      'not only the A1 2-halo part). Written by plots/make_figure_data.py from '
                      'nlbias_tables.npz, nlbias_fits.json and analysis/roman_dchi2/v2/fits.npz.'),
            'k_h_per_Mpc': clean(kd[m]), 'kmax_alpha': KMAX_ALPHA, 'panels': panels}


def bias_checks(fits, nlb, res):
    gb = np.load(os.path.join(ANA, 'cut_validation', 'galaxy_bias', 'beta_tables.npz'))
    kd = fits['k'][0]
    idn = np.searchsorted(nlb['k'], kd)
    m = (kd >= K_DRAW[0]) & (kd <= 1.3)
    panels = []
    for kind, z in (('gm', 1.5), ('gm', 2.0), ('gg', 0.3), ('gg', 1.0)):
        s = int(np.argmin(np.abs(fits['zs'] - z)))
        t = int(np.argmin(np.abs(nlb['zs'] - z)))
        pl, pnl = nlb['Plin_v6'][t][idn], fits['PNL'][s]
        bg = nlb['bg_k'][t][idn]
        if kind == 'gm':
            assert str(gb['source'][s]) == 'P_gm'
            meas = bg[0]*np.interp(kd, gb['k'], gb['beta_gm'][s])*pnl

            def model(q):
                return pl*(bg*nlb['I1'][t][idn] + (nlb[q][t][idn] if q else 0.0)) + nlb['Pgm_1h'][t][idn]
            qs = (None, 'Ib_gm', 'Ib_gm_lo')
            label = 'P_gm'
        else:
            meas = fits['b'][s]**2*pnl

            def model(q):
                return pl*(bg**2 + (nlb[q][t][idn] if q else 0.0)) + nlb['Pgg_1h'][t][idn]
            qs = (None, 'Ib_gg', 'Ib_gg_lo')
            label = 'P_gg'
        panels.append({'kind': label, 'z': float(fits['zs'][s]),
                       'plain': clean((model(qs[0])/meas)[m]), 'nlb': clean((model(qs[1])/meas)[m]),
                       'nlb_lo': clean((model(qs[2])/meas)[m])})
    return {'about': ('Halo model over FLAMINGO for the galaxy-matter spectrum P_gm (z = 1.5, 2: '
                      'b_g beta_gm P_NL from analysis/cut_validation/galaxy_bias/beta_tables.npz, '
                      'measured with the dark-matter particles) and the galaxy spectrum P_gg (z = 0.3, 1: '
                      'b(k)^2 P_NL from v2/fits.npz, shot noise removed). plain: 2-halo on P_lin; nlb: '
                      'with beta_NL held below the emulator mass; nlb_lo: beta_NL = 0 below it. '
                      '1-halo terms from the v6 HOD (Poisson satellites). The FLAMINGO normalisation of '
                      'P_gm uses the halo-model b_g, equal to the measured b1 to 1 percent. '
                      'Written by plots/make_figure_data.py.'),
            'k_h_per_Mpc': clean(kd[m]), 'panels': panels}


def fit_gof(fits, nlb, res):
    kg = np.array(res['kgrid'])
    js = np.where((kg >= 0.28) & (kg <= 1.3))[0]
    lines = {}
    for v, lab in GOF_VARIANTS.items():
        out = []
        for j in js:
            c2, nu = column(res, v, 'chi2', j), column(res, v, 'nu', j)
            out.append(float(np.nansum(c2)/np.nansum(nu)))
        lines[v] = {'label': lab, 'chi2_nu': out}
    return {'about': ('Summed chi2_min / nu over the nine core FLAMINGO snapshots (z = 0.2 to 2) against '
                      'k_max, fits of nlbias_fit.py (a_c <= 1, W free). FLAMINGO errors are '
                      'underestimated by 3 to 9 times at high k: compare the lines, not the values with 1. '
                      'Written by plots/make_figure_data.py.'),
            'kmax_h_per_Mpc': kg[js].tolist(), 'lines': lines}


def fit_params(fits, nlb, res):
    kg = np.array(res['kgrid'])
    js = np.where((kg >= 0.28) & (kg <= 1.3))[0]
    panels = []
    for z in Z_SHOW:
        r = res['zs'].index(min(res['zs'], key=lambda q: abs(q - z)))
        s = int(np.argmin(np.abs(fits['zs'] - z)))
        a1n, _ = low_k_nla(fits, s)
        p = {'z': res['zs'][r], 'A1_nla': a1n, 'lines': {}}
        for v in ('alpha_tatt', 'nlb_tatt', 'nlb_tatt_sat'):
            p['lines'][v] = {'label': GOF_VARIANTS[v],
                             'A1_ratio': clean([column(res, v, 'A1', j)[r]/a1n for j in js]),
                             'a_1h': clean([column(res, v, 'a_1h', j)[r] for j in js]),
                             'a_c': clean([column(res, v, 'a_c', j)[r] for j in js])}
        panels.append(p)
    return {'about': ('A1 over the low-k NLA amplitude (v2 NLA fit at k_max = 0.22), and the satellite '
                      'amplitude a_1h, against k_max, at z = 0.3, 1, 2 (nlbias_fits.json); a1h_cap: the satellite '
                      'alignment cap gamma_max of F21, reached inside r_vir for a_1h >= 0.3. '
                      'Written by plots/make_figure_data.py.'),
            'kmax_h_per_Mpc': kg[js].tolist(), 'a1h_cap': 0.3, 'panels': panels}


FIGURES = {'gi_kernel': gi_kernel, 'bias_checks': bias_checks, 'fit_gof': fit_gof, 'fit_params': fit_params}


def main():
    fits, nlb, res = load()
    for name in (sys.argv[1:] or FIGURES):
        d = FIGURES[name](fits, nlb, res)
        os.makedirs(os.path.join(HERE, name), exist_ok=True)
        with open(os.path.join(HERE, name, 'data.json'), 'w') as fh:
            json.dump(d, fh, indent=1)
        print('wrote', name)


if __name__ == '__main__':
    main()
