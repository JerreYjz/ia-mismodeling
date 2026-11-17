"""Write data.json of the five figures of this folder.

    ia_ps_z1    FLAMINGO vs TNG300 IA power spectra at z = 1
    ia_fit_z1   the best NLA and TATT fits to the FLAMINGO spectra at z = 1
    ia_hmc_z1   the best halo model (HMc + W) fit to the same spectra
    ia_hmca_z1  the halo model refit with a share alpha of the nonlinear power
                in the 2-halo term and the central amplitude capped at 1
    ia_hmcas_z1 the same refit with the satellite term free, over a wider
                fit range

This is the only file that knows where the measured spectra and the fits
are. Each figure folder then redraws from its data.json alone.

ia_ps_z1
--------

Both simulations are put in the same convention, the one of sim_IA.c
(the cosmolike injection) and of ia_halofit_v2/stitch.py:

  * shear units: P_dE / (2R), P_EE / (2R)^2 with R = 1 - e_rms^2 and
    e_rms the mean of the e1z, e2z rms of the hinfo file;
  * transverse modes (mu = 0): P_dE = 3/2 x monopole,
    P_EE = M0 - 5/2 M2 + 27/8 M4, with the NGP mesh window removed
    from EE;
  * errors: the sigma column of the measured files (Gaussian, from the
    number of modes of each bin), propagated linearly.

TNG300-1 (box 205 Mpc/h, mesh 1024, nfold 1) has the matter field, so
P_dE is measured directly. FLAMINGO L1_m8 (box 681 Mpc/h, mesh 1024,
nfold 1, 4, 16 stitched) has no matter field in this pipeline, so
P_dE = P_hE / b(k), b(k) = sqrt[(P_hh - V/N_gal) / P_mm(k)] with the
matter P_mm(k) measured by the FLAMINGO team. The error of b(k) is not
propagated.

Inputs:
    TNG300   ~/Downloads/PS_2re_unred_3 (P{dE,EE}{0,2,4}_sn50_nfold1.dat)
             ~/Downloads/hinfo/hinfo_fixng_ms_sub_red_sn50_nfold1.dat
    FLAMINGO flamingo/data/{PS,hinfo,flamingo_pmm} of this repository,
             snapshot 58
The TNG300 spectra are a collaborator's measurements and are not in this
repository; ia_ps_z1/data.json holds the converted z = 1 numbers.

ia_fit_z1
---------
The fits are those of analysis/roman_dchi2/v2/fit_models.py (stored in
fits.npz): per snapshot and k_max, NLA (a1, Ns) and TATT (a1, a2, b_TA, Ns)
fitted to the measured P_hE, P_EE, P_BB over 0.05 <= k <= k_max with the
FLAMINGO error bars, P_hE = b(k) P_dE, b(k) = sqrt(P_hh / P_NL) with the
CAMB halofit (takahashi) P_NL, Ns a white shape noise common to P_EE and
P_BB, cosmolike conventions (c1rhocrit = 0.01389). "Best fit" of a model
is its fit at the largest k_max that is still an acceptable fit, the
goodness-of-fit cut of kmax_gof.csv (the last k_max before two consecutive
p < 0.05). The models are re-evaluated here at the stored best-fit
parameters with fit_tatt_flamingo.TATTModel (CAMB and FAST-PT), and the
stored chi2 is reproduced as a check.

Inputs:
    analysis/roman_dchi2/v2/{fits.npz,kmax_gof.csv}
    analysis/tatt_fit/fit_tatt_flamingo.py

ia_hmc_z1
---------
The halo model with a central 1-halo term and a free white E = B term
(variant HMc_W of analysis/roman_dchi2/v6/halo_central_fit.py, stored in
f21v_HMc_W.npz), fitted to the same P_hE, P_EE, P_BB over
0.05 <= k <= k_max, N_s fixed to the B plateau:

    P_dE = f_2h P_dE^2h[A1 terms on I_1 P_lin] - a_1h f_1h S_dI - a_c P_c
    P_hE = b(k) P_dE
    P_EE = f_2h P_EE^2h[A1 terms on P_lin] + a_1h^2 f_1h S_II + N_s + W
    P_BB = f_2h P_BB^2h + N_s + W,      f_2h = exp[-(k/6)^2]

"Best fit" is again the fit at the goodness-of-fit cut (HMc_W_cut). The
model is rebuilt here from the tables (halo_central_tables.npz: I_1, P_lin,
P_c) and fit_tatt_flamingo.TATTModel, with no import of the v6 scripts
(they write log files next to themselves). This needs a_1h = 0 at the best
fit, which is checked, as is the stored chi2.

Inputs:
    analysis/roman_dchi2/v6/{f21v_HMc_W.npz, halo_central_tables.npz},
    analysis/roman_dchi2/v2/fits.npz

ia_hmca_z1, ia_hmcas_z1
-----------------------
The refit of ../fits/hmc_alpha_fit.py (results in ../fits/hmc_alpha_fits.json):
HMc + W with the A1 terms on I_1 P_lin + alpha (P_NL - I_1 P_lin) in GI and
P_lin + alpha (P_NL - P_lin) in II, alpha free, and the central 1-halo
amplitude a_c bounded by 1. HMCA_FIGURES gives the variant and the fit
range of each figure:
  * ia_hmca_z1: no satellite term (variant alpha_cap_nosat), at the fit
    range of ia_hmc_z1. With the satellites free their amplitude is
    unconstrained below k of about 1 and its noise value drives the model
    above the fit range (see ../fits/hmc_alpha_checks.py).
  * ia_hmcas_z1: satellites free (variant alpha_cap), at the goodness-of-fit
    cut of that variant at z = 1, where their amplitude is constrained.
The model is rebuilt here on a fine k grid and must reproduce the chi2
stored by the fit.

Inputs:
    ../fits/hmc_alpha_fits.json
    analysis/roman_dchi2/v6/{halo_central_tables.npz, halo_tables.npz},
    analysis/roman_dchi2/v2/fits.npz

Run with the cocoa Python from any folder (all figures, or the named ones):
    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py [figure ...]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
HOME = os.path.expanduser('~')

TNG_PS = os.path.join(HOME, 'Downloads', 'PS_2re_unred_3')
TNG_HINFO = os.path.join(HOME, 'Downloads', 'hinfo')
FLA = os.path.join(REPO, 'flamingo', 'data')
ANA = os.path.join(REPO, 'analysis')
FITS = os.path.join(ANA, 'roman_dchi2', 'v2')
HMC = os.path.join(ANA, 'roman_dchi2', 'v6')

# redshift of the comparison and the snapshot of each simulation there
Z = 1.0
TNG_SN, FLA_SN = 50, 58
# box [Mpc/h], mesh, folds
TNG_BOX, TNG_NG = 205.0, 1024
FLA_BOX, FLA_NG, FLA_FOLDS = 681.0, 1024, (1, 4, 16)
# FLAMINGO h, to convert the matter P(k) file from Mpc to Mpc/h
FLA_H = 0.681
# index of z = 1.00 in flamingo_pmm/output_list_ps.txt
FLA_PMM_INDEX = 102
# stellar-mass selections [log10 Msun], from masses.py (TNG300) and
# SUMMARY_2026-10-26.md (FLAMINGO)
TNG_LOGM, FLA_LOGM = (9.35, 10.65), (10.0, 11.07)


def ngp_w2_avg(k, cell, n=48):
    '''Angle-averaged NGP window <prod_i sinc^2(k_i cell/2)>, as sim_IA.c.'''
    mu = (np.arange(n) + 0.5)/n
    ph = (np.arange(n) + 0.5)/n*(0.5*np.pi)
    mu_grid, ph_grid = np.meshgrid(mu, ph, indexing='ij')
    st = np.sqrt(1.0 - mu_grid**2)
    dirs = np.stack([st*np.cos(ph_grid), st*np.sin(ph_grid), mu_grid], -1).reshape(-1, 3)
    out = []
    for kk in np.atleast_1d(k):
        x = 0.5*kk*cell*dirs
        out.append(np.mean(np.prod((np.sin(x)/x)**2, axis=1)))
    return np.array(out)


def hinfo(path):
    '''(N_gal used, e_rms, shear factor 1/(2R)) from a hinfo file.'''
    vals = open(path).read().split('\n')[1].split()
    erms = 0.5*(float(vals[3]) + float(vals[4]))
    return float(vals[2]), erms, 1.0/(2.0*(1.0 - erms**2))


def ee_mu0(p0, p2, p4):
    '''P_EE(mu = 0) and its error from the l = 0, 2, 4 multipole tables.'''
    val = p0[:, 1] - 2.5*p2[:, 1] + 27.0/8.0*p4[:, 1]
    err = np.sqrt(p0[:, 2]**2 + (2.5*p2[:, 2])**2 + (27.0/8.0*p4[:, 2])**2)
    return val, err


def tng():
    '''TNG300 at sn 50: k, P_dE, sigma, P_EE, sigma (shear units, mu = 0).'''
    load = lambda name: np.loadtxt(os.path.join(TNG_PS, f'{name}_sn{TNG_SN}_nfold1.dat'))
    ngal, erms, fsh = hinfo(os.path.join(TNG_HINFO,
                                         f'hinfo_fixng_ms_sub_red_sn{TNG_SN}_nfold1.dat'))
    de = load('PdE0')
    k = de[:, 0]
    ee, see = ee_mu0(load('PEE0'), load('PEE2'), load('PEE4'))
    w2 = ngp_w2_avg(k, TNG_BOX/TNG_NG)
    return {'k': k, 'PdE': 1.5*de[:, 1]*fsh, 'sig_dE': 1.5*de[:, 2]*fsh,
            'PEE': ee/w2*fsh**2, 'sig_EE': see/w2*fsh**2,
            'ngal': ngal, 'e_rms': erms, 'shear_factor': fsh}


def flamingo():
    '''FLAMINGO at sn 58: the same, with the folds stitched and P_dE = P_hE/b.'''
    load = lambda name, f: np.loadtxt(os.path.join(FLA, 'PS', f'{name}_sn{FLA_SN}_nfold{f}.dat'))
    ngal, erms, fsh = hinfo(os.path.join(FLA, 'hinfo',
                                         f'hinfo_fixng_ms_sub_red_sn{FLA_SN}_nfold1.dat'))
    pmm = np.loadtxt(os.path.join(FLA, 'flamingo_pmm', f'power_matter_{FLA_PMM_INDEX:04d}.txt'))
    if abs(pmm[0, 0] - Z) > 1e-3:
        raise ValueError(f'matter P(k) file is at z = {pmm[0, 0]}, not {Z}')
    kp, pp = pmm[:, 1]/FLA_H, pmm[:, 2]*FLA_H**3
    kny = lambda f: np.pi*FLA_NG*f/FLA_BOX
    rows = []
    for f in FLA_FOLDS:
        # fold f is used for 0.25 k_Nyq(f/4) < k <= 0.25 k_Nyq(f)
        lo, hi = (0.25*kny(f/4) if f > 1 else 0.0), 0.25*kny(f)
        hh, he = load('Phh', f), load('PhE0', f)
        ee, see = ee_mu0(load('PEE0', f), load('PEE2', f), load('PEE4', f))
        k = hh[:, 0]
        m = (k > lo) & (k <= hi)
        w2 = ngp_w2_avg(k[m], FLA_BOX/(f*FLA_NG))
        rows.append(np.column_stack([k[m], hh[m, 1], he[m, 1], he[m, 2], ee[m]/w2, see[m]/w2]))
    c = np.concatenate(rows)
    k = c[:, 0]
    bias = np.sqrt(np.clip(c[:, 1] - FLA_BOX**3/ngal, 1e-30, None)
                   / np.exp(np.interp(np.log(k), np.log(kp), np.log(pp))))
    return {'k': k, 'PdE': 1.5*c[:, 2]/bias*fsh, 'sig_dE': 1.5*c[:, 3]/bias*fsh,
            'PEE': c[:, 4]*fsh**2, 'sig_EE': c[:, 5]*fsh**2, 'bias': bias,
            'ngal': ngal, 'e_rms': erms, 'shear_factor': fsh}


def ratio(fla, tng_, key, sig):
    '''FLAMINGO / TNG300 at the TNG300 bins inside the FLAMINGO k range.

    FLAMINGO (dense, small errors) is interpolated linearly in ln k; the
    two relative errors are added in quadrature.'''
    m = (tng_['k'] >= fla['k'].min()) & (tng_['k'] <= fla['k'].max())
    lk = np.log(tng_['k'][m])
    f = np.interp(lk, np.log(fla['k']), fla[key])
    sf = np.interp(lk, np.log(fla['k']), fla[sig])
    t, st = tng_[key][m], tng_[sig][m]
    r = f/t
    return tng_['k'][m], r, np.abs(r)*np.sqrt((sf/f)**2 + (st/t)**2)


def lst(a):
    return [float(v) for v in a]


# the probes of the fit, in the order of fits.npz, and its lower k [h/Mpc]
FIT_PROBES = ('hE', 'EE', 'BB')
FIT_KMIN = 0.05
# the free parameters of each model, in the order of fits.npz
FIT_FREE = {'NLA': ('a1', 'Ns'), 'TATT': ('a1', 'a2', 'bTA', 'Ns')}
# the smooth model curves: k range [h/Mpc] and number of points
CURVE_K = (0.02, 6.0, 160)


def fit_figure():
    '''data.json of ia_fit_z1: FLAMINGO at z = 1 and its best NLA and TATT fits.'''
    sys.path.insert(0, os.path.join(ANA, 'tatt_fit'))
    import fit_tatt_flamingo as F
    d = np.load(os.path.join(FITS, 'fits.npz'))
    i = list(d['sns']).index(FLA_SN)
    z = float(d['zs'][i])
    if abs(z - Z) > 1e-6:
        raise ValueError(f'snapshot {FLA_SN} is at z = {z}, not {Z}')
    with open(os.path.join(FITS, 'kmax_gof.csv')) as fh:
        gof = [row for row in csv.DictReader(fh) if abs(float(row['z']) - z) < 1e-6][0]
    k, bias, pnl = d['k'][i], d['b'][i], d['PNL'][i]
    y = dict(zip(FIT_PROBES, d['y'][i]))
    err = dict(zip(FIT_PROBES, d['err'][i]))
    model = F.TATTModel(halofit='takahashi')
    kern = model.kernels(z, k)
    kc = np.logspace(np.log10(CURVE_K[0]), np.log10(CURVE_K[1]), CURVE_K[2])
    out = {
        'about': ('FLAMINGO L1_m8 snapshot 58 (z = 1.00) and the best NLA and TATT fits of '
                  'analysis/roman_dchi2/v2/fits.npz, shear units at mu = 0, k in h/Mpc, P in '
                  '(Mpc/h)^3. Data: P_dE = P_hE / b(k) (signed, negative), b(k) = '
                  'sqrt(P_hh / P_NL) with the CAMB halofit P_NL, and the measured P_EE (it '
                  'holds a white shape-noise term). Models: fitted to P_hE, P_EE, P_BB over '
                  '0.05 <= k <= kmax, kmax the goodness-of-fit cut of kmax_gof.csv; PEE of a '
                  'model includes its fitted white noise Ns. pct: 100 (model - data) / data '
                  'at the data bins. a1, a2, bTA in the cosmolike convention (c1rhocrit = '
                  '0.01389). Written by plots/make_figure_data.py.'),
        'z': z, 'kmin_fit': FIT_KMIN,
        'data': {'k_h_per_Mpc': lst(k), 'PdE': lst(y['hE']/bias), 'sig_dE': lst(err['hE']/bias),
                 'PEE': lst(y['EE']), 'sig_EE': lst(err['EE']), 'bias': lst(bias)},
        'models': {},
    }
    for name, free in FIT_FREE.items():
        kmax = float(gof['kmax_' + name])
        j = int(np.argmin(np.abs(d['kgrid'] - kmax)))
        th = dict(a1=0.0, a2=0.0, bTA=0.0, Ns=0.0)
        th.update(zip(free, d[name + '_x_raw'][i, j]))
        pr = model.predict(z, k, th['a1'], th['a2'], th['bTA'], 1.0, K=kern, PNL=pnl,
                           Ns=th['Ns'], bk=bias)
        m = (k >= FIT_KMIN) & (k <= d['kgrid'][j]*(1 + 1e-9))
        chi2 = float(sum(np.sum((((y[p] - pr[p])/err[p])[m])**2) for p in FIT_PROBES))
        stored = float(d[name + '_chi2_raw'][i, j])
        if abs(chi2 - stored) > 1e-3*max(1.0, stored):
            raise ValueError(f'{name}: chi2 {chi2} does not reproduce the stored {stored}')
        cv = model.predict(z, kc, th['a1'], th['a2'], th['bTA'], 1.0, Ns=th['Ns'])
        out['models'][name] = {
            'kmax_fit': float(d['kgrid'][j]),
            'params': {q: float(th[q]) for q in free},
            'param_errors': dict(zip(free, lst(d[name + '_e_raw'][i, j]))),
            'chi2': chi2, 'dof': int(d[name + '_dof'][i, j]),
            'p_value': float(d[name + '_p_raw'][i, j]),
            'curve_k_h_per_Mpc': lst(kc), 'curve_PdE': lst(cv['dE']), 'curve_PEE': lst(cv['EE']),
            'PdE': lst(pr['dE']), 'PEE': lst(pr['EE']),
            'pct_dE': lst(100.0*(pr['hE'] - y['hE'])/y['hE']),
            'pct_EE': lst(100.0*(pr['EE'] - y['EE'])/y['EE']),
        }
        print(name, out['models'][name]['kmax_fit'], out['models'][name]['params'],
              'chi2/dof', round(chi2, 2), out['models'][name]['dof'])
    path = os.path.join(HERE, 'ia_fit_z1', 'data.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)


# the halo model variant drawn, and the scale of its 2-halo window [h/Mpc]
HMC_VARIANT = 'HMc_W'
HMC_K2H = 6.0


def hmc_spectra(model, z, k, kern, plin, i1, pc, x, white):
    '''The halo model spectra at wavenumbers k for the parameters
    x = (A1, A2, bTA, a_c): the 2-halo and central 1-halo parts of P_dE,
    the 2-halo part of P_EE and P_BB, and the totals with the white term.'''
    a1, a2, bta, ac = x
    f2h = np.exp(-(k/HMC_K2H)**2)
    gi = model.predict(z, k, a1, a2, bta, 1.0, K=kern, PNL=i1*plin, Ns=0.0)
    ii = model.predict(z, k, a1, a2, bta, 1.0, K=kern, PNL=plin, Ns=0.0)
    de_2h, de_c = gi['dE']*f2h, -ac*pc
    ee_2h, bb_2h = ii['EE']*f2h, ii['BB']*f2h
    return {'dE_2h': de_2h, 'dE_c': de_c, 'dE': de_2h + de_c,
            'EE_2h': ee_2h, 'EE': ee_2h + white, 'BB': bb_2h + white}


def hmc_figure():
    '''data.json of ia_hmc_z1: FLAMINGO at z = 1 and its best halo model fit.'''
    sys.path.insert(0, os.path.join(ANA, 'tatt_fit'))
    import fit_tatt_flamingo as F
    d = np.load(os.path.join(FITS, 'fits.npz'))
    h = np.load(os.path.join(HMC, f'f21v_{HMC_VARIANT}.npz'))
    ct = np.load(os.path.join(HMC, 'halo_central_tables.npz'))
    i = list(d['sns']).index(FLA_SN)
    z = float(d['zs'][i])
    if abs(z - Z) > 1e-6 or abs(float(h['zs'][i]) - z) > 1e-6 or abs(float(ct['zs'][i]) - z) > 1e-6:
        raise ValueError(f'snapshot {FLA_SN} is not at z = {Z} in every input')
    k, bias, ns = d['k'][i], d['b'][i], float(d['Ns_true'][i])
    y = dict(zip(FIT_PROBES, d['y'][i]))
    err = dict(zip(FIT_PROBES, d['err'][i]))
    kmax = float(h[HMC_VARIANT + '_cut'][i])
    j = int(np.argmin(np.abs(h['kgrid'] - kmax)))
    a1, a2, bta, a1h, ac, w = (float(v) for v in h[HMC_VARIANT + '_x'][i, j, :6])
    if abs(a1h) > 1e-8:
        raise ValueError(f'a_1h = {a1h}: the satellite term is not rebuilt here')
    model = F.TATTModel(halofit='takahashi')
    # at the measured bins: the fitted quantities and the chi2 check
    idx = np.searchsorted(ct['k_all'], k)
    if not np.allclose(ct['k_all'][idx], k):
        raise ValueError('the measured bins are not on the grid of the halo tables')
    at = hmc_spectra(model, z, k, model.kernels(z, k), ct['Plin'][i][idx], ct['I1m'][i][idx],
                     ct['Pc_m'][i][idx], (a1, a2, bta, ac), ns + w)
    pr = {'hE': bias*at['dE'], 'EE': at['EE'], 'BB': at['BB']}
    m = (k >= FIT_KMIN) & (k <= h['kgrid'][j]*(1 + 1e-9))
    chi2 = float(sum(np.sum((((y[p] - pr[p])/err[p])[m])**2) for p in FIT_PROBES))
    stored = float(h[HMC_VARIANT + '_chi2'][i, j])
    if abs(chi2 - stored) > 1e-3*max(1.0, stored):
        raise ValueError(f'chi2 {chi2} does not reproduce the stored {stored}')
    # the smooth curves, on the grid of the halo tables
    sel = (ct['k_all'] >= CURVE_K[0]) & (ct['k_all'] <= CURVE_K[1])
    kc = ct['k_all'][sel]
    cv = hmc_spectra(model, z, kc, model.kernels(z, kc), ct['Plin'][i][sel], ct['I1m'][i][sel],
                     ct['Pc_m'][i][sel], (a1, a2, bta, ac), ns + w)
    out = {
        'about': ('FLAMINGO L1_m8 snapshot 58 (z = 1.00) and the best halo model fit (HMc with a '
                  'central 1-halo term and a free white E = B term W, variant HMc_W of '
                  'analysis/roman_dchi2/v6), shear units at mu = 0, k in h/Mpc, P in (Mpc/h)^3. '
                  'Data: P_dE = P_hE / b(k) (signed, negative), b(k) = sqrt(P_hh / P_NL), and '
                  'the measured P_EE. Model: fitted to P_hE, P_EE, P_BB over 0.05 <= k <= kmax, '
                  'kmax the goodness-of-fit cut; a_1h = 0 at this fit. curve_PdE = '
                  'curve_PdE_2h + curve_PdE_c (2-halo and central 1-halo); curve_PEE = '
                  'curve_PEE_2h + white, white = N_s + W. pct: 100 (model - data) / data at '
                  'the data bins. A1, A2, bTA in the cosmolike convention (c1rhocrit = '
                  '0.01389). Written by plots/make_figure_data.py.'),
        'z': z, 'kmin_fit': FIT_KMIN,
        'data': {'k_h_per_Mpc': lst(k), 'PdE': lst(y['hE']/bias), 'sig_dE': lst(err['hE']/bias),
                 'PEE': lst(y['EE']), 'sig_EE': lst(err['EE']), 'bias': lst(bias)},
        'model': {
            'kmax_fit': float(h['kgrid'][j]),
            'params': {'A1': a1, 'A2': a2, 'bTA': bta, 'a_1h': a1h, 'a_c': ac, 'W': w},
            'N_s': ns, 'white': ns + w,
            'chi2': chi2, 'dof': int(h[HMC_VARIANT + '_nu'][i, j]),
            'p_value': float(h[HMC_VARIANT + '_p'][i, j]),
            'curve_k_h_per_Mpc': lst(kc), 'curve_PdE': lst(cv['dE']),
            'curve_PdE_2h': lst(cv['dE_2h']), 'curve_PdE_c': lst(cv['dE_c']),
            'curve_PEE': lst(cv['EE']), 'curve_PEE_2h': lst(cv['EE_2h']),
            'PdE': lst(at['dE']), 'PEE': lst(at['EE']),
            'pct_dE': lst(100.0*(pr['hE'] - y['hE'])/y['hE']),
            'pct_EE': lst(100.0*(pr['EE'] - y['EE'])/y['EE']),
        },
    }
    print(HMC_VARIANT, out['model']['kmax_fit'], out['model']['params'],
          'chi2/dof', round(chi2, 2), out['model']['dof'])
    path = os.path.join(HERE, 'ia_hmc_z1', 'data.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)


# the refit figures: the variant of hmc_alpha_fit.py and the fit range
# [h/Mpc] of each; and the scale of the 1-halo window [h/Mpc]
HMCA_FIGURES = {'ia_hmca_z1': ('alpha_cap_nosat', 0.70), 'ia_hmcas_z1': ('alpha_cap', 1.69)}
HMC_K1H = 4.0
HMCA_NAMES = ('A1', 'A2', 'bTA', 'a_1h', 'a_c', 'W', 'alpha')


def hmca_spectra(model, z, k, kern, tab, x, ns):
    '''The refitted halo model at wavenumbers k for the parameters x (in the
    order of HMCA_NAMES), with tab the tables at k: Plin, Pnl, I1, Pc, and
    the satellite tables SdI, SII [n_a, nk] on the grid log_a of ln a_1h.'''
    a1, a2, bta, a1h, ac, w, alpha = x
    f2h, f1h = np.exp(-(k/HMC_K2H)**2), -np.expm1(-(k/HMC_K1H)**2)
    p_gi = tab['I1']*tab['Plin'] + alpha*(tab['Pnl'] - tab['I1']*tab['Plin'])
    p_ii = tab['Plin'] + alpha*(tab['Pnl'] - tab['Plin'])
    gi = model.predict(z, k, a1, a2, bta, 1.0, K=kern, PNL=p_gi, Ns=0.0)
    ii = model.predict(z, k, a1, a2, bta, 1.0, K=kern, PNL=p_ii, Ns=0.0)
    la = tab['log_a']
    xa = np.log(max(a1h, 1e-6))
    j = int(np.clip(np.searchsorted(la, xa), 1, la.size - 1))
    wa = np.clip((xa - la[j - 1])/(la[j] - la[j - 1]), 0, 1)
    sdi = tab['SdI'][j - 1]*(1 - wa) + tab['SdI'][j]*wa
    sii = tab['SII'][j - 1]*(1 - wa) + tab['SII'][j]*wa
    out = {'dE_2h': gi['dE']*f2h, 'dE_sat': -a1h*f1h*sdi, 'dE_c': -ac*tab['Pc'],
           'EE_2h': ii['EE']*f2h, 'EE_sat': a1h**2*f1h*sii, 'BB_2h': ii['BB']*f2h}
    out['dE'] = out['dE_2h'] + out['dE_sat'] + out['dE_c']
    out['EE'] = out['EE_2h'] + out['EE_sat'] + ns + w
    out['BB'] = out['BB_2h'] + ns + w
    return out


def hmca_figure(name):
    '''data.json of a refit figure of HMCA_FIGURES: FLAMINGO at z = 1 and the
    halo model refit.'''
    variant, kmax = HMCA_FIGURES[name]
    sys.path.insert(0, os.path.join(ANA, 'tatt_fit'))
    import fit_tatt_flamingo as F
    d = np.load(os.path.join(FITS, 'fits.npz'))
    ct = np.load(os.path.join(HMC, 'halo_central_tables.npz'))
    st = np.load(os.path.join(HMC, 'halo_tables.npz'))
    with open(os.path.join(HERE, '..', 'fits', 'hmc_alpha_fits.json')) as fh:
        res = json.load(fh)
    i = list(d['sns']).index(FLA_SN)
    z = float(d['zs'][i])
    if abs(z - Z) > 1e-6 or abs(res['zs'][i] - z) > 1e-6 or list(res['names']) != list(HMCA_NAMES):
        raise ValueError('the fit results do not match the snapshot or the parameter order')
    if not np.allclose(ct['k_all'], st['k_all']):
        raise ValueError('the central and satellite tables are on different k grids')
    kgrid = np.array(res['kgrid'])
    j = int(np.argmin(np.abs(kgrid - kmax)))
    k, bias, ns = d['k'][i], d['b'][i], float(d['Ns_true'][i])
    y = dict(zip(FIT_PROBES, d['y'][i]))
    err = dict(zip(FIT_PROBES, d['err'][i]))
    model = F.TATTModel(halofit='takahashi')
    log_a = np.log(np.maximum(st['a_grid'], 1e-6))

    def tables(sel, pnl):
        return {'Plin': ct['Plin'][i][sel], 'I1': ct['I1m'][i][sel], 'Pc': ct['Pc_m'][i][sel],
                'Pnl': pnl, 'SdI': st['SdI'][i][:, sel], 'SII': st['SII'][i][:, sel],
                'log_a': log_a}
    idx = np.searchsorted(ct['k_all'], k)
    if not np.allclose(ct['k_all'][idx], k):
        raise ValueError('the measured bins are not on the grid of the halo tables')
    tab_k, kern_k = tables(idx, d['PNL'][i]), model.kernels(z, k)
    m = (k >= FIT_KMIN) & (k <= kgrid[j]*(1 + 1e-9))

    def at_bins(variant):
        '''The stored fit of a variant at the measured bins, chi2 checked.'''
        v = res['variants'][variant]
        x = np.array(v['x'][i][j], dtype=float)
        at = hmca_spectra(model, z, k, kern_k, tab_k, x, ns)
        pr = {'hE': bias*at['dE'], 'EE': at['EE'], 'BB': at['BB']}
        chi2 = float(sum(np.sum((((y[p] - pr[p])/err[p])[m])**2) for p in FIT_PROBES))
        if abs(chi2 - v['chi2'][i][j]) > 1e-3*max(1.0, v['chi2'][i][j]):
            raise ValueError(f'{variant}: chi2 {chi2} does not reproduce {v["chi2"][i][j]}')
        return v, x, at, pr, chi2
    v, x, at, pr, chi2 = at_bins(variant)
    sel = np.where((ct['k_all'] >= CURVE_K[0]) & (ct['k_all'] <= CURVE_K[1]))[0]
    kc = ct['k_all'][sel]
    cv = hmca_spectra(model, z, kc, model.kernels(z, kc), tables(sel, model.Pnl(z, kc)), x, ns)
    w = float(x[5])
    out = {
        'about': ('FLAMINGO L1_m8 snapshot 58 (z = 1.00) and the halo model refit of '
                  f'fits/hmc_alpha_fit.py, variant {variant}: HMc + W with the A1 terms on '
                  'I_1 P_lin + alpha (P_NL - I_1 P_lin) in GI and P_lin + alpha (P_NL - P_lin) '
                  'in II, alpha free in [-1, 2], central 1-halo amplitude a_c in [0, 1], '
                  'satellite amplitude a_1h free in [0, 3] or fixed to 0 (see params and '
                  'param_errors). Shear units at mu = 0, k in h/Mpc, P in (Mpc/h)^3. Data: '
                  'P_dE = P_hE / b(k) (signed, negative) and the measured P_EE. Fitted to '
                  'P_hE, P_EE, P_BB over 0.05 <= k <= kmax_fit. curve_PdE = 2h + sat + c '
                  '(2-halo, radial satellites, central 1-halo); curve_PEE = 2h + sat + white, '
                  'white = N_s + W; the central 1-halo term has no P_EE part (one central '
                  'per halo, its self pair is shape noise). param_errors: null means fixed '
                  'or at a bound. pct: 100 (model - data) / data at the data bins. Written '
                  'by plots/make_figure_data.py.'),
        'z': z, 'kmin_fit': FIT_KMIN,
        'data': {'k_h_per_Mpc': lst(k), 'PdE': lst(y['hE']/bias), 'sig_dE': lst(err['hE']/bias),
                 'PEE': lst(y['EE']), 'sig_EE': lst(err['EE']), 'bias': lst(bias)},
        'model': {
            'kmax_fit': float(kgrid[j]),
            'params': dict(zip(HMCA_NAMES, lst(x))),
            'param_errors': dict(zip(HMCA_NAMES, v['sig'][i][j])),
            'at_bound': [n for n, b in zip(HMCA_NAMES, v['at_bound'][i][j]) if b],
            'ac_cap': res['ac_cap'], 'N_s': ns, 'white': ns + w,
            'chi2': chi2, 'dof': int(v['nu'][i][j]), 'p_value': float(v['p'][i][j]),
            'curve_k_h_per_Mpc': lst(kc), 'curve_PdE': lst(cv['dE']),
            'curve_PdE_2h': lst(cv['dE_2h']), 'curve_PdE_sat': lst(cv['dE_sat']),
            'curve_PdE_c': lst(cv['dE_c']), 'curve_PEE': lst(cv['EE']),
            'curve_PEE_2h': lst(cv['EE_2h']), 'curve_PEE_sat': lst(cv['EE_sat']),
            'PdE': lst(at['dE']), 'PEE': lst(at['EE']),
            'pct_dE': lst(100.0*(pr['hE'] - y['hE'])/y['hE']),
            'pct_EE': lst(100.0*(pr['EE'] - y['EE'])/y['EE']),
        },
    }
    # the fine-grid curves against the values at the measured bins
    inside = (k > kc[0]) & (k < kc[-1])
    for key in ('dE', 'EE'):
        c = np.interp(np.log(k[inside]), np.log(kc), cv[key])
        dev = float(np.max(np.abs(c/at[key][inside] - 1)))
        if dev > 5e-3:
            raise ValueError(f'the fine-grid P_{key} deviates from the bins by {dev:.3g}')
    print(name, variant, out['model']['kmax_fit'], out['model']['params'], 'at a bound:',
          out['model']['at_bound'], 'chi2/dof', round(chi2, 2), out['model']['dof'])
    path = os.path.join(HERE, name, 'data.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)


def sims_figure():
    '''data.json of ia_ps_z1: FLAMINGO and TNG300 at z = 1.'''
    t, f = tng(), flamingo()
    out = {
        'about': ('IA power spectra of TNG300-1 (snapshot 50) and FLAMINGO L1_m8 (snapshot 58) '
                  'at z = 1.00, both in shear units at mu = 0 (transverse modes), the '
                  'convention of sim_IA.c. k in h/Mpc, P in (Mpc/h)^3, errors are the Gaussian '
                  'mode-count sigma of the measured files. P_dE is signed (negative: radial '
                  'alignment). TNG300 P_dE uses the matter field; FLAMINGO P_dE = P_hE / b(k), '
                  'b(k) from P_hh (shot noise removed) and the measured matter P(k), the error '
                  'of b(k) not propagated. ratio: FLAMINGO / TNG300 at the TNG300 bins, FLAMINGO '
                  'interpolated in ln k. Written by plots/make_figure_data.py.'),
        'z': Z,
    }
    for name, d, box, logm in (('tng300', t, TNG_BOX, TNG_LOGM), ('flamingo', f, FLA_BOX, FLA_LOGM)):
        out[name] = {'box_Mpc_per_h': box, 'ngal': int(d['ngal']), 'e_rms': d['e_rms'],
                     'shear_factor': d['shear_factor'], 'logMstar_range_Msun': list(logm),
                     'k_h_per_Mpc': lst(d['k']), 'PdE': lst(d['PdE']), 'sig_dE': lst(d['sig_dE']),
                     'PEE': lst(d['PEE']), 'sig_EE': lst(d['sig_EE'])}
    out['flamingo']['bias'] = lst(f['bias'])
    kr, r_de, s_de = ratio(f, t, 'PdE', 'sig_dE')
    _, r_ee, s_ee = ratio(f, t, 'PEE', 'sig_EE')
    out['ratio'] = {'k_h_per_Mpc': lst(kr), 'dE': lst(r_de), 'sig_dE': lst(s_de),
                    'EE': lst(r_ee), 'sig_EE': lst(s_ee)}
    path = os.path.join(HERE, 'ia_ps_z1', 'data.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)


def main():
    figures = {'ia_ps_z1': sims_figure, 'ia_fit_z1': fit_figure, 'ia_hmc_z1': hmc_figure}
    for name in HMCA_FIGURES:
        figures[name] = lambda name=name: hmca_figure(name)
    for name in (sys.argv[1:] or figures):
        figures[name]()


if __name__ == '__main__':
    main()
