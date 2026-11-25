#!/usr/bin/env python3
"""Every number quoted by halo_nlbias.tex, from the outputs of this folder,
with the physical-sensibility checks of the fits. Writes nlbias_numbers.json
and prints a readable log (nlbias_numbers_log.txt).

    python nlbias_numbers.py > nlbias_numbers_log.txt
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.normpath(os.path.join(HERE, '..'))
KS = (0.29, 0.39, 0.52, 0.70, 0.94, 1.26)
KMAXS = (0.52, 0.70, 0.94)


def main():
    fits = np.load(os.path.join(ANA, 'roman_dchi2', 'v2', 'fits.npz'))
    nlb = np.load(os.path.join(HERE, 'nlbias_tables.npz'))
    bt = np.load(os.path.join(HERE, 'beta_tables.npz'))
    gb = np.load(os.path.join(ANA, 'cut_validation', 'galaxy_bias', 'beta_tables.npz'))
    with open(os.path.join(HERE, 'nlbias_fits.json')) as fh:
        res = json.load(fh)
    kd = fits['k'][0]
    idn = np.searchsorted(nlb['k'], kd)
    ik = [int(np.argmin(np.abs(kd - k))) for k in KS]
    kg = np.array(res['kgrid'])
    jk = {k: int(np.argmin(np.abs(kg - k))) for k in (0.39, 0.52, 0.70, 0.94, 1.26)}
    names = res['names']
    out = {'k': [float(kd[i]) for i in ik]}

    def x_of(v, s, j, q):
        r = res['variants'][v]['x'][s][j]
        return np.nan if r is None or r[names.index(q)] is None else r[names.index(q)]

    # 1. beta_NL itself: the lowest emulator bin with itself, at k = 0.3, 0.5, 1
    kb = bt['k']
    out['beta'] = {}
    for s, z in enumerate(bt['zs']):
        out['beta'][f'{z:g}'] = {'lgM_lowest': float(bt['lgM_E'][s, 0]), 'lgM_highest': float(bt['lgM_E'][s, -1]),
                                 'beta_low_low': [float(np.interp(k, kb, bt['beta'][s, 0, 0])) for k in (0.3, 0.5, 1.0)],
                                 'max_abs_beta_k_below_0.03': float(np.abs(bt['beta'][s][..., kb < 0.03]).max())}
    # 2. the gap: alpha correction against I^beta_cm, per snapshot
    ja = jk[0.70]
    out['gap'] = {}
    print('Share of the alpha correction alpha (P_NL/P_lin - I_1) given by I^beta_cm, alpha at k_max = 0.70')
    print('   z  alpha  ' + '  '.join(f'k={k:.2f}' for k in KS))
    for s, z in enumerate(nlb['zs']):
        s2 = int(np.argmin(np.abs(fits['zs'] - z)))
        pl, i1 = nlb['Plin_v6'][s][idn], nlb['I1'][s][idn]
        alpha = x_of('alpha_tatt', s, ja, 'alpha')
        corr = alpha*(fits['PNL'][s2]/pl - i1)
        share = nlb['Ib_cm'][s][idn]/corr
        share_lo = nlb['Ib_cm_lo'][s][idn]/corr
        j39 = KS.index(0.39)
        out['gap'][f'{z:g}'] = {'alpha': alpha, 'share': [float(share[i]) for i in ik],
                                'alpha_from_beta_k0.39': float(share[ik[j39]]*alpha),
                                'share_lo': [float(share_lo[i]) for i in ik],
                                'Ib_cm': [float(nlb['Ib_cm'][s][idn][i]) for i in ik],
                                'alpha_corr': [float(corr[i]) for i in ik],
                                'Ib_cm_k0.1': float(np.interp(0.1, nlb['k'], nlb['Ib_cm'][s]))}
        print(f'{z:5.2f}  {alpha:5.2f}  ' + '  '.join(f'{share[i]:6.2f}' for i in ik)
              + f'   (I^beta_cm at k = 0.1: {np.interp(0.1, nlb["k"], nlb["Ib_cm"][s]):+.3f})')
    # 3. checks with FLAMINGO's own clustering
    print('\nHalo model / FLAMINGO, k = ' + ', '.join(f'{k:.2f}' for k in KS))
    out['checks'] = {}
    for kind, z in (('gm', 1.5), ('gm', 2.0), ('gg', 0.3), ('gg', 1.0), ('mm', 0.3), ('mm', 1.0)):
        s = int(np.argmin(np.abs(nlb['zs'] - z)))
        s2 = int(np.argmin(np.abs(fits['zs'] - z)))
        pl, pnl, bg = nlb['Plin_v6'][s][idn], fits['PNL'][s2], nlb['bg_k'][s][idn]
        i1 = nlb['I1'][s][idn]
        if kind == 'gm':
            meas = bg[0]*np.interp(kd, gb['k'], gb['beta_gm'][s2])*pnl
            mods = {q: pl*(bg*i1 + (nlb[t][s][idn] if t else 0)) + nlb['Pgm_1h'][s][idn]
                    for q, t in (('plain', None), ('nlb', 'Ib_gm'), ('nlb_lo', 'Ib_gm_lo'))}
        elif kind == 'gg':
            meas = fits['b'][s2]**2*pnl
            mods = {q: pl*(bg**2 + (nlb[t][s][idn] if t else 0)) + nlb['Pgg_1h'][s][idn]
                    for q, t in (('plain', None), ('nlb', 'Ib_gg'), ('nlb_lo', 'Ib_gg_lo'))}
        else:
            meas = pnl
            mods = {q: pl*(i1**2 + (nlb[t][s][idn] if t else 0)) + nlb['Pmm_1h'][s][idn]
                    for q, t in (('plain', None), ('nlb', 'Ib_mm'), ('nlb_lo', 'Ib_mm_lo'))}
        out['checks'][f'{kind}_{z:g}'] = {q: [float(v[i]/meas[i]) for i in ik] for q, v in mods.items()}
        for q, v in mods.items():
            print(f'  P_{kind} z = {z:3.1f} {q:7s}: ' + ' '.join(f'{v[i]/meas[i]:6.3f}' for i in ik))
    # 4. fits: summed chi2/nu, and the parameters with their checks
    print('\nSummed chi2/nu over the core snapshots at k_max = ' + ', '.join(map(str, KMAXS)))
    out['gof'] = {}
    for v in res['variants']:
        c2 = np.array([[np.nan if q is None else q for q in r] for r in res['variants'][v]['chi2']], float)
        nu = np.array([[np.nan if q is None else q for q in r] for r in res['variants'][v]['nu']], float)
        out['gof'][v] = [float(np.nansum(c2[:, jk[k]])/np.nansum(nu[:, jk[k]])) for k in KMAXS]
        print(f'  {v:13s} ' + ' '.join(f'{x:7.2f}' for x in out['gof'][v]))
    print('\nz > 1.476: chi2/nu at k_max = 0.52, 0.70 with beta_NL assigned by peak height (nlb_tt) and to the '
          'same masses (nlbsame_tt)')
    out['same_bracket'] = {}
    for s, z in enumerate(nlb['zs']):
        if z < 1.4:
            continue
        row = {}
        for v in ('nlb_tt', 'nlbsame_tt', 'nlb_tatt'):
            vv = res['variants'][v]
            row[v] = [vv['chi2'][s][jk[k]]/vv['nu'][s][jk[k]] for k in (0.52, 0.70)]
        out['same_bracket'][f'{z:g}'] = row
        print(f'  z {z:3.1f}: ' + '  '.join(f'{v} {r[0]:6.2f} {r[1]:6.2f}' for v, r in row.items()))
    print('\nParameters (A1/A1_NLA, A2, b_TA, a_c, a_1h) per snapshot at k_max = 0.39, 0.52, 0.70, 0.94')
    jn = int(np.argmin(np.abs(fits['kgrid'] - 0.22)))
    out['params'] = {}
    for v in ('alpha_tatt', 'nlb_tatt', 'nlb_tt', 'nlb_tatt_sat'):
        if v not in res['variants']:
            continue
        out['params'][v] = {}
        print(f'  {v}')
        for s, z in enumerate(nlb['zs']):
            s2 = int(np.argmin(np.abs(fits['zs'] - z)))
            a1n = float(fits['NLA_x_raw'][s2, jn, 0])
            row = {q: [x_of(v, s, jk[k], q) for k in (0.39, 0.52, 0.70, 0.94)] for q in names}
            row['A1_over_NLA'] = [a/a1n for a in row['A1']]
            row['A1_NLA'] = a1n
            out['params'][v][f'{z:g}'] = row
            fmt = lambda a: ' '.join('  nan' if not np.isfinite(q) else f'{q:5.2f}' for q in a)  # noqa: E731
            print(f'    z {z:4.2f} A1/A1_NLA {fmt(row["A1_over_NLA"])} | A2 {fmt(row["A2"])} | bTA {fmt(row["bTA"])}'
                  f' | a_c {fmt(row["a_c"])} | a_1h {fmt(row["a_1h"])} | alpha {fmt(row["alpha"])}')
    # 5. signs: every best-fit model spectrum, all FLAMINGO k bins, every k_max
    import nlbias_fit as NF
    fits_, central, nlb_, model, kgrid, sat = NF.load()
    print('\nSign checks of the best fits (all k bins of the data, every k_max):')
    out['signs'] = {}
    for v in ('nlb_tatt', 'nlb_tt', 'nlb_tatt_sat', 'alpha_tatt'):
        mode, suffix = NF.VARIANTS[v][:2]
        bad = {'GI_2h_wrong_sign': 0, 'PdE_total_wrong_sign': 0, 'EE_negative': 0, 'BB_negative': 0, 'EE_2h_negative': 0}
        nfit = 0
        for s, z in enumerate(nlb['zs']):
            snap = NF.Snapshot(float(z), fits_, central, nlb_, model, sat)
            for j in range(len(kgrid)):
                r = res['variants'][v]['x'][s][j]
                if r is None:
                    continue
                x = np.array([np.nan if q is None else q for q in r], float)
                p = snap.parts(x, mode, suffix)
                nfit += 1
                bad['GI_2h_wrong_sign'] += int(np.any(p['dE_2h'] > 0))
                bad['PdE_total_wrong_sign'] += int(np.any(p['dE_2h'] + p['dE_c'] + p['dE_sat'] > 0))
                bad['EE_negative'] += int(np.any(p['EE'] <= 0))
                bad['BB_negative'] += int(np.any(p['BB'] <= 0))
                bad['EE_2h_negative'] += int(np.any(p['EE_2h'] < 0))
        out['signs'][v] = dict(bad, n_fits=nfit)
        print(f'  {v:13s} fits with a wrong sign anywhere: ' + ', '.join(f'{q} {n}' for q, n in bad.items()) + f' (of {nfit})')
    mins = {q: float(np.min(1 + nlb[q][:, nlb['k'] <= 2.0])) for q in ('Ib_cc',)}
    mins['I1_plus_Ib_cm'] = float(np.min((nlb['I1'] + nlb['Ib_cm'])[:, nlb['k'] <= 2.0]))
    out['kernel_min_k_below_2'] = mins
    print('  minimum over k <= 2 and all snapshots: 1 + Ib_cc', f"{mins['Ib_cc']:.3f}", ', I1 + Ib_cm', f"{mins['I1_plus_Ib_cm']:.3f}")
    with open(os.path.join(HERE, 'nlbias_numbers.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
