#!/usr/bin/env python3
"""Writes the data.json of every figure of notes/halo_kernel_model from the
analysis outputs. Run with the cocoa Python (it imports analysis/halo_kernel):

    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py [figure ...]   (default: all)

    ingredients     HOD, halo and shape biases, alignment kernels per mass bin, z = 1
                    (analysis/halo_kernel/kernels/kernels_58.npz, model/hk_model.json)
    shape_response  galaxy-weighted nonlinear response of halo shapes and halo
                    densities, z = 0.3 and 1 (kernels_<sn>.npz through hk_model.build)
    weighting       model galaxy density over the measured one (binned, mean HOD,
                    actual occupations) and galaxy II over the halo-shape II
    spectra         FLAMINGO spectra and the final model at z = 0.3 and 1
    gof             goodness of fit against k_max per model (model/hk_model.json;
                    TATT from notes/halo_ia_models/plots/gof/data.json)
    residual        object-level over large-scale kernels against z, and the galaxy
                    2-halo II over the deterministic and deterministic + residual II
    checks          saturated kernel gradient A and alignment kernels against z
    fob_k0scan      Roman figure of bias against the xi_+- cut, CP truth, Fisher
                    (analysis/cut_validation/fisher_minimize.json)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
HK = os.path.join(REPO, 'analysis', 'halo_kernel')
sys.path.insert(0, HK)
MODEL = os.path.join(HK, 'model', 'hk_model.json')
FITS = os.path.join(REPO, 'analysis', 'roman_dchi2', 'v2', 'fits.npz')
FOB = os.path.join(REPO, 'analysis', 'cut_validation', 'fisher_minimize.json')
OLD_GOF = os.path.join(REPO, 'notes', 'halo_ia_models', 'plots', 'gof', 'data.json')

# core snapshots of the goodness of fit (as the halo_ia_models note): z = 0.2 to 2
CORE = (74, 72, 70, 68, 66, 63, 58, 48, 38)
# the models of the goodness-of-fit figure: key of hk_model.json and its label
GOF_MODELS = {
    'lin': ('lin-ls', 'linear gravity layer (beta = 0)'),
    'emu': ('emu-ls', 'density beta_NL (Dark Emulator)'),
    'tab': ('tab-ls', 'binned halo spectra, full II'),
    'v1': ('w-ls-grad', 'first HMk: deterministic II, k^2 kernel gradient (1 parameter)'),
    'w': ('ws-ls', 'HMk: mean-HOD halo fields, deterministic + residual II (0 parameters)'),
    'wg': ('ws-ls-sat', 'HMk + saturating kernel gradient A (1 parameter)'),
}
K_SPEC = 1.26          # k_max of the gradient fit drawn in the spectra figure
K0_CS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)
K0_3X2 = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)
FOB_MODELS = {'NLA': 'NLA', 'TATT': 'TATT', 'hms0': 'NLAs0', 'hms': 'NLAs', 'hmsz': 'NLAsz', 'hmk': 'NLAk',
              'ctrl': 'NLAt'}


def clean(a, nd=6):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), np.round(a, nd), None).tolist()


def sig(a, n=5):
    '''Round to n significant digits (spectra span many decades).'''
    a = np.asarray(a, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.array([float(f'{x:.{n}g}') if np.isfinite(x) else np.nan for x in a.ravel()]).reshape(a.shape)
    return clean(out, 12)


def cut_rule(ok, kgrid):
    for j in range(len(kgrid) - 1):
        if not ok[j] and not ok[j + 1]:
            return float(kgrid[j - 1]) if j > 0 else float('nan')
    return float(kgrid[-1] if ok[-1] else kgrid[-2])


def build_one(sn):
    '''hk_model.build at one snapshot (the arrays the json does not keep).'''
    import hk_model as M
    d = M.load(sn)
    z = float(d['z'])
    kh, pk = M.HIA.Linear([z]).P(z)
    plin = np.exp(np.interp(np.log(d['k']), np.log(kh), np.log(pk)))
    mod, meas, aux = M.build(d, plin)
    return M, d, z, plin, mod, meas, aux


def ingredients():
    res = json.load(open(MODEL))
    line = res['58']
    edges = np.array(line['lgm_edges'])
    lo = np.maximum(edges[:-1], 11.3)
    hi = np.minimum(edges[1:], 14.5)
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py from '
                      'analysis/halo_kernel/model/hk_model.json (hk_model.py, kernels_58.npz). FLAMINGO '
                      'L1_m8 at z = 1, host-halo bins in log10 M200m [Msun/h] (lowest bin: galaxy hosts '
                      'from 10^9, all halos above 10^11.3; edges clipped to 11.3 and 14.5 for drawing). '
                      'Nc, Ns: mean sample centrals, satellites per halo. b: large-scale halo bias; bK: '
                      'large-scale halo-shape bias (P_hE = b bK P_lin at k = 0.02 to 0.08). kappa_obj: '
                      'per-object regression of galaxy on halo ellipticity; kappa_w: large-scale kernel '
                      'on the mean-HOD halo fields (k = 0.02 to 0.10); null where Nc < 0.01.'),
            'z': line['z'], 'lgm_lo': clean(lo), 'lgm_hi': clean(hi), 'lgm_mid': clean(0.5*(lo + hi)),
            'Nc': clean(line['Nc']), 'Ns': clean(line['Ns']), 'b': clean(line['b']), 'bK': clean(line['bK']),
            'kappa_c_obj': clean(line['kappa_c']), 'kappa_s_obj': clean(line['kappa_s']),
            'kappa_c_w': clean(np.where(np.array(line['Nc']) >= 0.01, line['kappa_w_c'], np.nan)),
            'kappa_s_w': clean(line['kappa_w_s'])}


def shape_response():
    out = {}
    for sn in (72, 58):
        M, d, z, plin, mod, meas, aux = build_one(sn)
        k = d['k']
        m = (k >= 0.05) & (k <= 3.1)
        f = np.load(FITS)
        s = int(np.argmin(np.abs(f['zs'] - z)))
        pnl = np.exp(np.interp(np.log(k), np.log(f['k'][s]), np.log(f['PNL'][s])))
        a = aux['a']
        hh2 = np.einsum('ik,jk,ijk->k', a, a, d['Phh'][:, :, 0])
        hhl = np.einsum('ik,jk->k', a*aux['b'][:, None], a*aux['b'][:, None])*aux['W2']*plin
        out[f'{z:g}'] = {
            'k': clean(k[m]),
            'shape': clean((mod['tab-ls']['2h']['hE']/mod['lin-ls']['2h']['hE'])[m]),
            'shape_II': clean((mod['tab-ls']['2h']['EE']/mod['lin-ls']['2h']['EE'])[m]),
            'density': clean((hh2/hhl)[m]),
            'emulator': clean((mod['emu-ls']['2h']['hE']/mod['lin-ls']['2h']['hE'])[m]),
            'matter': clean((pnl/plin)[m])}
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py through '
                      'analysis/halo_kernel/hk_model.build (kernels_72.npz, kernels_58.npz). Galaxy-weighted '
                      'nonlinear responses, each a 2-halo sum over the six host-mass bins divided by the '
                      'same sum with linear biases on P_lin: shape = sum a_i c_j P_{h_i E_j} / sum a_i c_j '
                      'b_i bK_j P_lin (1 + beta_K); shape_II = the same for P_{E_i E_j}; density = sum a_i '
                      'a_j P_{h_i h_j} / sum a_i a_j b_i b_j P_lin; emulator = 1 + beta_NL of Dark Emulator '
                      'on the same weights; matter = P_NL/P_lin of FLAMINGO.'),
            'panels': out}


def weighting():
    out = {}
    for sn in (72, 58):
        M, d, z, plin, mod, meas, aux = build_one(sn)
        k = d['k']
        m = (k >= 0.02) & (k <= 1.0)
        a = aux['a']
        gg = d['G']['hh'][0]
        binned = np.einsum('ik,jk,ijk->k', a, a, d['Phh'][:, :, 0])/gg
        gal_ii = meas['EE'] - mod['det-ls']['1h']['EE']
        out[f'{z:g}'] = {'k': clean(k[m]), 'binned': clean(binned[m]),
                         'mean_hod': clean(aux['bias_ratio']['G'][m]),
                         'actual': clean(aux['bias_ratio']['Ga'][m]),
                         'ii_full': clean(np.where(k >= 0.04, gal_ii/mod['tab-ls']['2h']['EE'], np.nan)[m]),
                         'ii_det': clean(np.where(k >= 0.04, gal_ii/mod['w-ls']['2h']['EE'], np.nan)[m])}
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py through '
                      'analysis/halo_kernel/hk_model.build (kernels_<sn>.npz, weighted_<sn>.npz). Top: the '
                      'model galaxy density power over the measured P_gg (monopole, shot noise off): binned = '
                      'sum a_i a_j P_{h_i h_j} over six mass bins with equal halo weights; mean_hod = halo '
                      'field weighted by <Nc>(M), <Ns>(M) in 0.1 dex bins; actual = weighted by each halo\'s '
                      'own counts. Bottom: the 2-halo galaxy II (measured P_EE minus the measured 1-halo '
                      'term) over two predictions: ii_full = kappa^2 times the full halo-shape II (binned '
                      'kernels), ii_det = the deterministic halo-shape II on the mean-HOD fields (null below '
                      'k = 0.04, a handful of modes). k in h/Mpc.'),
            'panels': out}


def spectra():
    res = json.load(open(MODEL))
    f = np.load(FITS)
    out = {}
    for sn in (58, 38):
        line = res[str(sn)]
        s = int(np.argmin(np.abs(f['zs'] - line['z'])))
        k = np.array(line['k'])
        m = np.array(line['okk']) & (k >= 0.05) & (k <= 3.1)
        jk = int(np.argmin(np.abs(np.array(KG) - K_SPEC)))
        parts = line['ws-ls']['parts']
        out[f'{line["z"]:g}'] = {
            'k': clean(k[m]), 'y': sig(f['y'][s][:, m]), 'err': sig(f['err'][s][:, m]),
            'final': sig(np.array(line['ws-ls-sat']['pred_kmax'][jk])[:, m]),
            'A': line['ws-ls-sat']['alpha'][jk],
            'zero': sig(np.array(line['ws-ls']['pred'])[:, m]),
            'two_halo': sig(np.array([parts[p]['2h'] for p in ('hE', 'EE', 'BB')])[:, m]),
            'one_halo': sig(np.array([parts[p]['1h'] for p in ('hE', 'EE', 'BB')])[:, m])}
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py from '
                      'analysis/halo_kernel/model/hk_model.json and analysis/roman_dchi2/v2/fits.npz. '
                      'FLAMINGO P_hE, P_EE, P_BB at mu = 0 in the cosmolike-fit convention [(Mpc/h)^3], '
                      'rows (hE, EE, BB), z = 1 and 2; final: HMk with the saturating kernel gradient A '
                      f'fitted to k <= {K_SPEC}; zero: HMk with no fitted parameter (deterministic + residual '
                      'II); two_halo, one_halo: the parts of '
                      'zero (2-halo with alpha = 0, measured 1-halo term). Predictions include the '
                      'estimator offset (fits.npz minus the catalogue measurement).'),
            'kmax_fit': K_SPEC, 'panels': out}


KG = None


def gof():
    res = json.load(open(MODEL))
    old = json.load(open(OLD_GOF))
    kg = np.array(KG)
    lines = {}
    for key, (name, label) in GOF_MODELS.items():
        C, NU, CUT = [], [], []
        for sn in CORE:
            v = res[str(sn)][name]
            C.append(v['chi2'])
            NU.append(v['nu'])
            CUT.append(v['cut'])
        C, NU = np.array(C), np.array(NU)
        lines[key] = {'model': name, 'summed': clean(C.sum(0)/NU.sum(0)),
                      'per_snapshot': [clean(c/n) for c, n in zip(C, NU)], 'cut': clean(CUT)}
    t = old['lines']['TATT|base']
    lines['tatt'] = {'model': 'TATT + W (halo_ia_models)', 'summed': t['summed'],
                     'per_snapshot': t['per_snapshot'], 'cut': t['cut']}
    assert np.allclose(np.array(old['kgrid']), kg, rtol=2e-3), 'k grids differ'
    zs = [res[str(sn)]['z'] for sn in CORE]
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py from '
                      'analysis/halo_kernel/model/hk_model.json (hk_model.py). FLAMINGO P_hE, P_EE, P_BB of '
                      'the nine core snapshots on 0.05 <= k <= k_max; HMk variants have no parameter fitted '
                      'to the spectra except wg (the saturating kernel gradient A) and v1 (the k^2 '
                      'gradient alpha of the first version); tatt: TATT with A1, A2, b_TA '
                      'and a white term W fitted (notes/halo_ia_models). summed: sum chi2 / sum nu; '
                      'per_snapshot: chi2/nu [snapshot][k_max]; cut: last k_max before two consecutive '
                      'p < 0.05.'),
            'kgrid': clean(kg), 'zs': clean(zs),
            'models': {k: v[1] for k, v in GOF_MODELS.items()} | {'tatt': 'TATT + W'}, 'lines': lines}


def residual():
    '''kappa_r / kappa_d against z, and the galaxy 2-halo II over the two II models.'''
    res = json.load(open(MODEL))
    sns = sorted(res, key=lambda s: res[s]['z'])
    rat = {'c': [], 's': []}
    for s in sns:
        L = res[s]
        nh, Nc, Ns = (np.array(L[q]) for q in ('n_h', 'Nc', 'Ns'))
        for t, n in (('c', nh*Nc), ('s', nh*Ns)):
            kd, kr = np.array(L[f'kappa_w_{t}']), np.array(L[f'kappa_r_{t}'])
            use = (kd != 0) & (kr != 0)
            rat[t].append(float(np.sum((n*kr)[use])/np.sum((n*kd)[use])))
    # sign check: the largest negative value of the residual II and of the 2-halo BB at
    # 0.1 <= k <= 1.3, in units of the model's total P_EE and P_BB, over all snapshots
    neg_ii, neg_bb = 0.0, 0.0
    for s in sns:
        M, d, z, plin, mod, meas, aux = build_one(int(s))
        k = d['k']
        band = (k >= 0.1) & (k <= 1.3)
        r = mod['ws-ls']['2h']['EE'] - mod['w-ls']['2h']['EE']
        bb = mod['ws-ls']['2h']['BB']
        neg_ii = max(neg_ii, float(np.max(np.clip(-r, 0, None)[band]/mod['ws-ls']['tot']['EE'][band])))
        neg_bb = max(neg_bb, float(np.max(np.clip(-bb, 0, None)[band]/mod['ws-ls']['tot']['BB'][band])))
    panels = {}
    for sn in (58, 38):
        M, d, z, plin, mod, meas, aux = build_one(sn)
        k = d['k']
        m = (k >= 0.04) & (k <= 1.0)
        gal = meas['EE'] - mod['ws-ls']['1h']['EE']
        panels[f'{z:g}'] = {'k': clean(k[m]), 'det': clean((gal/mod['w-ls']['2h']['EE'])[m]),
                            'det_res': clean((gal/mod['ws-ls']['2h']['EE'])[m]),
                            'full_ls': clean((gal/mod['wf-ls']['2h']['EE'])[m])}
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py from '
                      'analysis/halo_kernel/model/hk_model.json and hk_model.build (kernels_<sn>.npz, '
                      'weighted_<sn>.npz, stoch_<sn>.npz). Top: kappa_r/kappa_d, the object-level regression '
                      'kernel over the large-scale kernel, weighted by the number of centrals (c) or '
                      'satellites (s) per host-mass bin, per snapshot z. Bottom: the 2-halo galaxy II '
                      '(measured P_EE minus the measured 1-halo term) over three 2-halo predictions on the '
                      'mean-HOD fields: det = deterministic II only, det_res = deterministic + residual II '
                      '(the residual halo-shape II with kappa_r), full_ls = the full halo-shape II with the '
                      'large-scale kernels; z = 1 and 2, k in h/Mpc. neg_ii_max, neg_bb_max: the largest '
                      'negative value of the residual II and of the 2-halo BB at 0.1 <= k <= 1.3 over all '
                      'snapshots, in units of the model total P_EE, P_BB.'),
            'zs': clean([res[s]['z'] for s in sns]), 'ratio_c': clean(rat['c']), 'ratio_s': clean(rat['s']),
            'neg_ii_max': round(neg_ii, 4), 'neg_bb_max': round(neg_bb, 4),
            'panels': panels}


def checks():
    res = json.load(open(MODEL))
    sns = sorted(res, key=lambda s: res[s]['z'])
    zs = [res[s]['z'] for s in sns]
    kg = np.array(KG)
    al, ae = {}, {}
    for km in (0.52, 0.94, 1.26):
        j = int(np.argmin(np.abs(kg - km)))
        al[f'{km:g}'] = clean([res[s]['ws-ls-sat']['alpha'][j] for s in sns])
        ae[f'{km:g}'] = clean([res[s]['ws-ls-sat']['alpha_err'][j] for s in sns])
    kc = np.array([res[s]['kappa_w_c'] for s in sns])
    ks = np.array([res[s]['kappa_w_s'] for s in sns])
    nc = np.array([res[s]['Nc'] for s in sns])
    kc = np.where((nc >= 0.01) & (kc != 0), kc, np.nan)        # 0: bin left out (too few galaxies)
    ks = np.where(ks != 0, ks, np.nan)
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py from '
                      'analysis/halo_kernel/model/hk_model.json. A: saturated kernel gradient (kappa -> '
                      'kappa [1 + A k^2 R_L^2/(1 + k^2 R_L^2)], R_L the galaxy-weighted Lagrangian radius of '
                      'the hosts [Mpc/h]) fitted to k <= k_max, per snapshot; kappa_c, kappa_s: '
                      'large-scale kernels on the mean-HOD halo fields [z][mass bin] (null where a bin holds '
                      'fewer than 1000 galaxies of the kind or Nc < 0.01). A_err: 1 sigma from the chi2 '
                      'curvature.'),
            'zs': clean(zs), 'A': al, 'A_err': ae, 'A_range': [-0.5, 3.0], 'R_L': clean([res[s]['R_L'] for s in sns]),
            'kappa_c': [clean(r) for r in kc], 'kappa_s': [clean(r) for r in ks],
            'lgm_edges': res['58']['lgm_edges']}


def fob():
    runs = json.load(open(FOB))['runs']
    lines, missing = {}, []
    for probe, k0s, sfx in (('cs', K0_CS, ''), ('3x2', K0_3X2, '|gt0.5')):
        for key, mdl in FOB_MODELS.items():
            rows = {'fob': [], 'mu_om': [], 'mu_s8': [], 'chi2_ref': []}
            for K in k0s:
                r = runs.get(f'{probe}+cmb|{mdl}|{K:g}|CP{sfx}|fisher')
                if r is None:
                    missing.append(f'{probe}|{mdl}|{K:g}')
                    for q in rows:
                        rows[q].append(None)
                    continue
                rows['fob'].append(round(r['fob'], 4))
                rows['mu_om'].append(round(r['mu'][0], 4))
                rows['mu_s8'].append(round(r['mu'][1], 4))
                rows['chi2_ref'].append(round(r['chi2_ref'], 3))
            lines[f'{probe}|{key}'] = dict(model=mdl, **rows)
    if missing:
        print('missing runs:', ', '.join(missing))
    return {'about': ('Producer: notes/halo_kernel_model/plots/make_figure_data.py from '
                      'analysis/cut_validation/fisher_minimize.json. Roman with the CMB prior, every '
                      'parameter free, CP truth, Fisher-only shift of (Omega_m, S8). cs: xi_+- at '
                      'k_UV = K0/D(z); 3x2: xi_+- at K0, gamma_t of sources behind lenses at 0.5, w at 0.2. '
                      'Models: NLA, TATT; hms0 = NLA + HMk template (deterministic + residual II, no fitted '
                      'parameter); hms = the same with the saturating kernel gradient fitted to k <= 1.26; hmsz = '
                      'hms at z <= 1.5 and the truth\'s own shape above (diagnostic); hmk = the first HMk '
                      '(deterministic II, k^2 gradient); ctrl = NLA + the template of the truth\'s own '
                      'shape (matched control). Templates use the model on 0.2 <= k <= 1.26 and hold its ratio '
                      'to the truth outside. fob: 2D figure of bias; mu: shift in units '
                      'of the marginalized sigma; chi2_ref: IA-only reference fit at the fiducial cosmology.'),
            'K0': {'cs': list(K0_CS), '3x2': list(K0_3X2)}, 'threshold': 0.3,
            'models': FOB_MODELS, 'lines': lines}


def main():
    global KG
    import hk_model as M
    KG = list(M.KGRID)
    todo = sys.argv[1:] or ['ingredients', 'shape_response', 'weighting', 'residual', 'spectra', 'gof', 'checks',
                            'fob_k0scan']
    make = {'ingredients': ingredients, 'shape_response': shape_response, 'weighting': weighting,
            'residual': residual, 'spectra': spectra, 'gof': gof, 'checks': checks, 'fob_k0scan': fob}
    for name in todo:
        os.makedirs(os.path.join(HERE, name), exist_ok=True)
        with open(os.path.join(HERE, name, 'data.json'), 'w') as fh:
            json.dump(make[name](), fh, indent=1)
        print('wrote', name)


if __name__ == '__main__':
    main()
