"""Write the data.json of every figure of the scale-cut summary note.

This is the only file of the note that knows where the analysis outputs
are. Each figure folder plots/<name>/ then redraws from its data.json alone
(plot_<name>.py). Rerun this script after the analysis is rerun, then rerun
every plot script and refresh the numbers block of scale_cuts_summary.tex.

Inputs, all under analysis/roman_dchi2/ of this repository:
    v2/fits.npz                    FLAMINGO spectra in the cosmolike convention
    v3/cuts.csv, v3/knl.csv        per-snapshot simulation cuts; D(z)
    v4/derived.json                volume scaling, FoB-precision trade-off,
                                   accuracy of the anchored fits
    v4/roman_strategies.json       the accuracy Roman requires
    v3/mask_scan_ex1_{D1,flat}.json   Roman FoB scans, example1 covariance
    v6/hmc_scan_ex1_{D1,flat}.json    the same with the halo model (HMc)

Run with the cocoa Python from any folder:
    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'analysis', 'roman_dchi2'))
# the redshifts of the core snapshots used for every simulation statement
CORE_ZMIN, CORE_ZMAX = 0.2, 2.0
# the cuts of the Roman scans [h/Mpc]
K0_GRID = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def path(*parts):
    return os.path.join(ANA, *parts)


def load_json(*parts):
    with open(path(*parts)) as f:
        return json.load(f)


def clean(x):
    '''A float, or None for NaN (a gap in the figure).'''
    x = float(x)
    return None if not np.isfinite(x) else x


def write(name, data):
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.relpath(os.path.join(folder, 'data.json'), HERE))


def growth_curve():
    '''D(z) of the D3A cosmology on a fine grid, interpolated linearly from
    the 13 snapshot values of v3/knl.csv (D is smooth on that spacing).'''
    rows = list(csv.DictReader(open(path('v3', 'knl.csv'))))
    z = np.array([float(r['z']) for r in rows])
    D = np.array([float(r['D']) for r in rows])
    zf = np.linspace(CORE_ZMIN, CORE_ZMAX, 91)
    return zf, np.interp(zf, z, D)


def sim_cuts():
    rows = [r for r in csv.DictReader(open(path('v3', 'cuts.csv'))) if r['core'] == '1']
    zf, Df = growth_curve()
    write('sim_cuts', {
        'about': ('Goodness-of-fit scale cuts of NLA and TATT in FLAMINGO per core snapshot: '
                  'the last k_max [h/Mpc] before two consecutive p < 0.05, N_s fixed to the '
                  'B-mode plateau, fits over 0.05 <= k <= k_max to P_hE, P_EE, P_BB with the '
                  'measured errors (analysis/roman_dchi2/v3/v3_flamingo.py -> v3/cuts.csv). '
                  'null: the cut lies below the first grid node, k = 0.09. '
                  'curve_*: the production mask family k_UV = K0/D(z), D from v3/knl.csv.'),
        'z': [float(r['z']) for r in rows],
        'NLA': [clean(r['NLA_GoF']) for r in rows],
        'TATT': [clean(r['TATT_GoF']) for r in rows],
        'kgrid_first': 0.08988,
        'curve_z': zf.tolist(),
        'curve_K0': 0.3,
        'curve_kuv': (0.3/Df).tolist(),
    })


def cut_volume():
    vol = load_json('v4', 'derived.json')['volume']
    write('cut_volume', {
        'about': ('The goodness-of-fit cut, combined over the nine core snapshots, against the '
                  'simulation volume factor f (errors sigma/sqrt(f); FLAMINGO L1000 = 1), from '
                  'the non-centrality estimate <chi2>(f) = nu + f (chi2_meas - nu) '
                  '(analysis/roman_dchi2/v4/plot_strategies.py -> v4/derived.json, '
                  'volume.<model>.comb_lambda). null: below the first grid node. '
                  'roman_band: the ratio of Roman to FLAMINGO Delta chi2 for the actual NLA and '
                  'TATT residuals (fits with k_max <= 0.22), i.e. the volume factor at which '
                  'FLAMINGO sees the residuals as Roman does.'),
        'f': vol['NLA']['f'],
        'NLA': [clean(v) for v in vol['NLA']['comb_lambda']],
        'TATT': [clean(v) for v in vol['TATT']['comb_lambda']],
        'roman_band': [0.02, 0.04],
    })


def white_excess():
    f = np.load(path('v2', 'fits.npz'))
    zs, k = f['zs'], f['k'][0]
    y, err, b, pnl, ns = f['y'], f['err'], f['b'], f['PNL'], f['Ns_true']
    # X = (P_EE - N_s) - P_dE^2 / P_NL with P_dE = P_hE / b(k): zero for any
    # model in which GI and II come from one tidal field
    X = (y[:, 1] - ns[:, None]) - (y[:, 0]/b)**2/pnl
    dX = np.sqrt(err[:, 1]**2 + (2*(y[:, 0]/b)*err[:, 0]/b/pnl)**2)
    sel = (k >= 0.05) & (k <= 0.6)
    snaps = []
    for zt in (0.3, 1.0, 2.0):
        s = int(np.argmin(np.abs(zs - zt)))
        snaps.append({'z': float(zs[s]), 'X_over_Ns': (X[s, sel]/ns[s]).tolist(),
                      'err': (dX[s, sel]/ns[s]).tolist()})
    write('white_excess', {
        'about': ('The white II excess X(k) = (P_EE - N_s) - P_dE^2/P_NL in units of the B-mode '
                  'shape-noise level N_s, for three snapshots, with the propagated FLAMINGO '
                  'errors (computed here from analysis/roman_dchi2/v2/fits.npz: y = [P_hE, '
                  'P_EE + N_s, P_BB + N_s], P_dE = P_hE/b(k); as cocoa/plot_white.py). '
                  'k in h/Mpc.'),
        'k': k[sel].tolist(),
        'snapshots': snaps,
    })


def required_accuracy():
    rs = load_json('v4', 'roman_strategies.json')
    ach = load_json('v4', 'derived.json')['achieved']
    k = np.load(path('v2', 'fits.npz'))['k'][0]
    keep = (k >= 0.1) & (k <= 1.3)
    out = {'about': ('The accuracy Roman requires of the IA spectra against what NLA and TATT '
                     'achieve. required: the coherent fractional error epsilon at every '
                     "k' >= k (a step) that gives Delta chi2_Roman = 1 on the mask "
                     'K0 = 0.3/D(z), Gaussian covariance, propagated with the cell Jacobian '
                     '(analysis/roman_dchi2/v4/roman_strategies.py -> roman_strategies.json, '
                     'tolerance["n1:0.3"], eps_dchi1; K0 = 0.2 and 0.4 kept for the caption). '
                     'achieved: |M/T - 1| of the fits anchored at k <= 0.16, mean over the '
                     'core snapshots (v4/derived.json, achieved). k in h/Mpc.'),
           'panels': {}}
    for which, idx in (('GI', 0), ('II', 1)):
        panel = {}
        for key in ('n1:0.2', 'n1:0.3', 'n1:0.4'):
            rows = [x for x in rs['tolerance'][key]['TATT']
                    if x['shape'] == 'step' and x['which'] == which]
            panel['required_' + key.split(':')[1]] = {
                'k': [x['kx'] for x in rows], 'eps': [clean(x['eps_dchi1']) for x in rows]}
        for mdl in ('NLA', 'TATT'):
            panel[mdl] = {'k': k[keep].tolist(),
                          'mean': np.array(ach[mdl]['mean'])[idx][keep].tolist(),
                          'max': np.array(ach[mdl]['max'])[idx][keep].tolist()}
        out['panels'][which] = panel
    write('required_accuracy', out)


def bias_floor():
    trade = load_json('v4', 'derived.json')['tradeoff']
    out = {'about': ('The figure of bias of every mask against its precision sigma(S_8) '
                     '(Fisher, case A), Gaussian covariance, for the three redshift shapes of '
                     'the mask k_UV = K0 D(z)^-p: p = 1 (D1), 2/3 (D23), 0 (flat). '
                     '(analysis/roman_dchi2/v4/plot_strategies.py -> v4/derived.json, '
                     'tradeoff; FoB from the full fits of v3/mask_scan.json.) abs_dS8: '
                     'the constant absolute S_8 shifts drawn as references, FoB = dS8/sigma.'),
           'abs_dS8': [0.001, 0.002, 0.003], 'threshold': 0.3, 'models': {}}
    for mdl in ('NLA', 'TATT'):
        out['models'][mdl] = {fam: {'K0': trade[mdl][fam]['K0'],
                                    'sigma_S8': trade[mdl][fam]['sS8'],
                                    'FoB': trade[mdl][fam]['FoB'],
                                    'abs_dS8': trade[mdl][fam]['dS8abs']}
                              for fam in ('D1', 'D23', 'flat')}
    write('bias_floor', out)


def scan_rows(fname):
    rows = load_json(*fname)
    return {round(r['K0'], 4): r for r in rows}


def fob_k0():
    out = {'about': ('Roman cosmic-shear figure of bias against the cut K0 with the production '
                     'covariance example1_cov, for the 1/D(z) family (k_UV = K0/D(z)) and a '
                     'constant cut (k_UV = K0), the FLAMINGO GI and II spectra injected, every '
                     'parameter free. case_A: Omega_m, A_s and IA free; FoB_sig its '
                     'uncertainty from the FLAMINGO errors; case_B: also n_s, H0, Omega_b with '
                     'Gaussian priors; white_A: case A with one free white-II amplitude. '
                     'npts: points kept of 1080; sigma_S8: case A, NLA. '
                     '(analysis/roman_dchi2/v3/mask_scan.py with COV=example1 -> '
                     'v3/mask_scan_ex1_{D1,flat}.json.)'),
           'K0': K0_GRID, 'threshold': 0.3, 'families': {}}
    for fam, fname in (('D1', ('v3', 'mask_scan_ex1_D1.json')),
                       ('flat', ('v3', 'mask_scan_ex1_flat.json'))):
        rows = scan_rows(fname)
        fam_out = {'npts': [rows[k]['npts'] for k in K0_GRID]}
        for mdl in ('NLA', 'TATT'):
            fam_out[mdl] = {
                'case_A': [clean(rows[k][mdl + '_A']['FoB']) for k in K0_GRID],
                'FoB_sig': [clean(rows[k][mdl + '_A']['FoB_sig']) for k in K0_GRID],
                'case_B': [clean(rows[k][mdl + '_B']['FoB']) for k in K0_GRID],
                'white_A': [clean(rows[k][mdl + 'w_A']['FoB']) for k in K0_GRID],
                'dS8_A': [clean(rows[k][mdl + '_A']['dS8']) for k in K0_GRID],
            }
        out['families'][fam] = fam_out
    write('fob_k0', out)


def hmc_fob():
    out = {'about': ('Roman cosmic-shear figure of bias against K0 with example1_cov, case B '
                     '(Omega_m, A_s, n_s, H0, Omega_b and IA free), for NLA, TATT and the halo '
                     'model with central one-halo alignment (HMc), the 1/D(z) family and a '
                     'constant cut; FoB_sig: uncertainty from the FLAMINGO errors; case A kept '
                     'for the text. The HMc templates are evaluated at the fiducial cosmology. '
                     '(analysis/roman_dchi2/v6/hmc_scan.py -> v6/hmc_scan_ex1_{D1,flat}.json.)'),
           'threshold': 0.3, 'families': {}}
    for fam, fname in (('D1', ('v6', 'hmc_scan_ex1_D1.json')),
                       ('flat', ('v6', 'hmc_scan_ex1_flat.json'))):
        rows = load_json(*fname)
        fam_out = {'K0': [r['K0'] for r in rows], 'npts': [r['npts'] for r in rows]}
        for mdl in ('NLA', 'TATT', 'HMc'):
            fam_out[mdl] = {
                'case_B': [clean(r[mdl + '_B']['FoB']) for r in rows],
                'FoB_sig_B': [clean(r[mdl + '_B']['FoB_sig']) for r in rows],
                'case_A': [clean(r[mdl + '_A']['FoB']) for r in rows],
            }
        out['families'][fam] = fam_out
    write('hmc_fob', out)


def main():
    sim_cuts()
    cut_volume()
    white_excess()
    required_accuracy()
    bias_floor()
    fob_k0()
    hmc_fob()


if __name__ == '__main__':
    main()
