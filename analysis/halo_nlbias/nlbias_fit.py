#!/usr/bin/env python3
"""Fits of the halo model with nonlinear halo bias (halo_nlbias.tex) to the
FLAMINGO IA spectra, against the alpha refit it replaces.

Data and conventions are those of the alpha refit
(notes/flamingo_vs_tng300/fits/hmc_alpha_fit.py, variant alpha_cap_nosat):
v2/fits.npz (P_hE, P_EE, P_BB at mu = 0, FLAMINGO errors, 0.05 <= k <= k_max),
P_hE = b(k) P_dE with the measured b(k), N_s fixed, a free white term W,
the windows f_2h = exp[-(k/6)^2], the central 1-halo term -a_c P_c with
a_c in [0, 1], no satellites. Only the spectra under the tidal-alignment (A1)
terms change:

    mode   GI (per -A1 C1bar)                    II (per A1^2 C1bar^2)
    lin    I_1 P_lin                              P_lin                    (HMc)
    alpha  I_1 P_lin + alpha (P_NL - I_1 P_lin)   P_lin + alpha (P_NL - P_lin)
    nlb    P_lin [I_1 + I^beta_cm]                P_lin [1 + I^beta_cc]    (this note)

with I^beta from nlbias_tables.npz (halo_nlbias.py): shape bias proportional
to b(M) (fiducial), "T" for a mass-independent shape bias, "lo" for beta_NL
set to 0 below the emulator's masses, "same" for the z > 1.476 bracket. The
A2 (tidal torquing) and b_TA (density weighting) terms keep their one-loop
TATT kernels.

Variants: name -> (mode, table suffix, A2 free, b_TA free, alpha free/fixed).
Writes nlbias_fits.json; with names as arguments only those are refitted.
    python nlbias_fit.py [variant ...]         # all: about 5 minutes
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
ANA = os.path.join(REPO, 'analysis')
sys.path.insert(0, os.path.join(ANA, 'tatt_fit'))
sys.path.insert(0, os.path.join(REPO, 'notes', 'flamingo_vs_tng300', 'fits'))
import fit_tatt_flamingo as F  # noqa: E402
import hmc_alpha_fit as HA  # noqa: E402  (constants, cut_rule, nan_list)

KMIN, K2H = HA.KMIN, HA.K2H
AC_CAP = HA.AC_CAP
ALPHA_LO, ALPHA_HI = HA.ALPHA_LO, HA.ALPHA_HI
NAMES = ('A1', 'A2', 'bTA', 'a_c', 'W', 'alpha', 'a_1h')
# name: (mode, table suffix, A2 free, b_TA free, alpha free (None: fixed at
# the value given), satellite amplitude a_1h free)
VARIANTS = {
    'lin_tatt': ('lin', '', True, True, 0.0, False),
    'alpha_tatt': ('alpha', '', True, True, None, False),
    'nlb_tatt': ('nlb', '', True, True, 0.0, False),
    'nlb_tt': ('nlb', '', True, False, 0.0, False),
    'nlb_nla': ('nlb', '', False, False, 0.0, False),
    'nlbT_tt': ('nlb', '_T', True, False, 0.0, False),
    'nlblo_tt': ('nlb', '_lo', True, False, 0.0, False),
    'nlbsame_tt': ('nlb', '_same', True, False, 0.0, False),
    'alpha_nla': ('alpha', '', False, False, None, False),
    'nla': ('alpha', '', False, False, 1.0, False),
    'nlb_tatt_sat': ('nlb', '', True, True, 0.0, True),
    'nlb_tt_sat': ('nlb', '', True, False, 0.0, True),
    'lin_tatt_sat': ('lin', '', True, True, 0.0, True),
}
# upper bound of the satellite amplitude (as hmc_alpha_fit)
A1H_HI = 3.0
AT_BOUND = HA.AT_BOUND


class Snapshot:
    '''Data, tables and model of one core snapshot.'''

    def __init__(self, z, fits, central, nlb, model, sat):
        s = int(np.argmin(np.abs(fits['zs'] - z)))
        self.z = float(fits['zs'][s])
        self.k = fits['k'][s]
        self.y, self.err = fits['y'][s], fits['err'][s]
        self.bias, self.pnl, self.ns = fits['b'][s], fits['PNL'][s], float(fits['Ns_true'][s])
        idx = np.searchsorted(central['k_all'], self.k)
        assert np.allclose(central['k_all'][idx], self.k)
        self.plin, self.i1, self.pc = (central[n][s][idx] for n in ('Plin', 'I1m', 'Pc_m'))
        t = int(np.argmin(np.abs(nlb['zs'] - z)))
        assert abs(nlb['zs'][t] - self.z) < 1e-6
        idn = np.searchsorted(nlb['k'], self.k)
        assert np.allclose(nlb['k'][idn], self.k)
        assert np.allclose(nlb['I1'][t][idn], self.i1, rtol=1e-6)
        self.ib = {q: nlb[q][t][idn] for q in nlb.files if q.startswith('Ib_')}
        self.model = model
        self.kern = model.kernels(self.z, self.k)
        self.f2h = np.exp(-(self.k/K2H)**2)
        self.f1h = -np.expm1(-(self.k/HA.K1H)**2)
        idx6 = np.searchsorted(sat['k_all'], self.k)
        assert np.allclose(sat['k_all'][idx6], self.k)
        self.sdi, self.sii = sat['SdI'][s][:, idx6], sat['SII'][s][:, idx6]
        self.log_a = np.log(np.maximum(sat['a_grid'], 1e-6))

    def sat_at(self, table, a1h):
        '''A satellite table interpolated in ln a_1h (as hmc_alpha_fit).'''
        x = np.log(max(a1h, 1e-6))
        j = int(np.clip(np.searchsorted(self.log_a, x), 1, self.log_a.size - 1))
        w = np.clip((x - self.log_a[j - 1])/(self.log_a[j] - self.log_a[j - 1]), 0, 1)
        return table[j - 1]*(1 - w) + table[j]*w

    def spectra(self, mode, suffix, alpha):
        '''The spectra under the A1 terms, GI and II.'''
        if mode == 'lin':
            return self.i1*self.plin, self.plin
        if mode == 'alpha':
            return (self.i1*self.plin + alpha*(self.pnl - self.i1*self.plin),
                    self.plin + alpha*(self.pnl - self.plin))
        cm, cc = self.ib['Ib_cm' + suffix], self.ib['Ib_cc' + suffix]
        return self.plin*(self.i1 + cm), self.plin*(1 + cc)

    def parts(self, x, mode, suffix):
        '''The model terms at x = (A1, A2, b_TA, a_c, W, alpha, a_1h); the
        satellites are the radial F21 term of v6 (halo_tables.npz).'''
        a1, a2, bta, ac, w, alpha, a1h = x
        p_gi, p_ii = self.spectra(mode, suffix, alpha)
        gi = self.model.predict(self.z, self.k, a1, a2, bta, 1.0, K=self.kern, PNL=p_gi, Ns=0.0)
        ii = self.model.predict(self.z, self.k, a1, a2, bta, 1.0, K=self.kern, PNL=p_ii, Ns=0.0)
        sat_ee = a1h**2*self.f1h*self.sat_at(self.sii, a1h) if a1h > 0 else 0.0*self.k
        return {'dE_2h': gi['dE']*self.f2h, 'dE_c': -ac*self.pc,
                'dE_sat': -a1h*self.f1h*self.sat_at(self.sdi, a1h) if a1h > 0 else 0.0*self.k,
                'EE_2h': ii['EE']*self.f2h, 'BB_2h': ii['BB']*self.f2h, 'EE_sat': sat_ee,
                'EE': ii['EE']*self.f2h + sat_ee + self.ns + w, 'BB': ii['BB']*self.f2h + self.ns + w}

    def predict(self, x, mode, suffix):
        p = self.parts(x, mode, suffix)
        return np.array([self.bias*(p['dE_2h'] + p['dE_c'] + p['dE_sat']), p['EE'], p['BB']])

    def fit(self, variant, kmax, warm=None):
        mode, suffix, a2_free, bta_free, alpha_fix, sat_free = VARIANTS[variant]
        alpha_free = alpha_fix is None
        a_fix = 0.0 if alpha_free else alpha_fix
        lo = np.array([0.0, -5.0 if a2_free else 0.0, 0.0, 0.0, -self.ns, ALPHA_LO if alpha_free else a_fix, 0.0])
        hi = np.array([20.0, 5.0 if a2_free else 0.0, 2.0 if bta_free else 0.0, AC_CAP, 5*self.ns,
                       ALPHA_HI if alpha_free else a_fix, A1H_HI if sat_free else 0.0])
        free = np.array([True, a2_free, bta_free, True, True, alpha_free, sat_free])
        m = (self.k >= KMIN) & (self.k <= kmax*(1 + 1e-9))

        def full(xf):
            x = lo.copy()
            x[free] = xf
            return x

        def resid(xf):
            return ((self.y - self.predict(full(xf), mode, suffix))/self.err)[:, m].ravel()
        starts = []
        for a1, a2, bta, ac in ([4.0, -1.0, 0.1, 0.0], [4.0, -1.0, 0.3, 0.5], [3.0, -1.5, 0.3, 1.0],
                                [4.0, 0.0, 0.5, 0.5], [4.0, 1.0, 0.1, 0.0]):
            for w in (0.0, 0.5*self.ns):
                for alpha in ((0.0, 0.5, 1.0) if alpha_free else (a_fix,)):
                    for a1h in ((0.0, 1e-3, 0.02, 0.1) if sat_free else (0.0,)):
                        starts.append([a1, a2, bta, ac, w, alpha, a1h])
        if warm is not None:
            starts.append(list(warm))
        best = None
        for x0 in starts:
            x0 = np.clip(np.array(x0, float), lo, hi)[free]
            r = least_squares(resid, x0, bounds=(lo[free], hi[free]), x_scale='jac', max_nfev=400)
            if best is None or r.cost < best.cost:
                best = r
        x = full(best.x)
        width = hi - lo
        at_bound = free & ((x - lo < AT_BOUND*np.maximum(width, 1e-30)) | (hi - x < AT_BOUND*np.maximum(width, 1e-30)))
        act = np.where(free & ~at_bound)[0]
        jac = best.jac[:, [list(np.where(free)[0]).index(q) for q in act]]
        cov = np.linalg.pinv(jac.T @ jac)
        sig = np.full(7, np.nan)
        sig[act] = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = np.full((7, 7), np.nan)
        corr[np.ix_(act, act)] = cov/np.outer(sig[act], sig[act])
        return {'x': x, 'chi2': float(2*best.cost), 'nk': int(m.sum()), 'sig': sig, 'corr': corr,
                'at_bound': at_bound, 'npar': int(free.sum())}


def load():
    '''The inputs: FLAMINGO data, v6 central tables, this folder's tables, the
    TATT model, the k_max grid and the core mask.'''
    fits = np.load(os.path.join(ANA, 'roman_dchi2', 'v2', 'fits.npz'))
    central = np.load(os.path.join(ANA, 'roman_dchi2', 'v6', 'halo_central_tables.npz'))
    nlb = np.load(os.path.join(HERE, 'nlbias_tables.npz'))
    sat = np.load(os.path.join(ANA, 'roman_dchi2', 'v6', 'halo_tables.npz'))
    v3 = np.load(os.path.join(ANA, 'roman_dchi2', 'v3', 'v3.npz'))
    model = F.TATTModel(halofit='takahashi')
    return fits, central, nlb, model, v3['kgrid'], sat


def main():
    t0 = time.time()
    fits, central, nlb, model, kgrid, sat = load()
    zs = nlb['zs']
    snaps = [Snapshot(float(z), fits, central, nlb, model, sat) for z in zs]
    path = os.path.join(HERE, 'nlbias_fits.json')
    out = {'about': ('Fits of the halo model with nonlinear halo bias (nlbias_fit.py) and of the '
                     'alpha refit it replaces, to FLAMINGO P_hE, P_EE, P_BB of the core snapshots. '
                     'Arrays are [snapshot][k_max]; parameters in the order of names; None: fixed '
                     'or at a bound.'),
           'names': list(NAMES), 'zs': zs.tolist(), 'kgrid': kgrid.tolist(), 'kmin': KMIN,
           'ac_cap': AC_CAP, 'variants': {}}
    todo = sys.argv[1:] or list(VARIANTS)
    if sys.argv[1:] and os.path.isfile(path):
        with open(path) as fh:
            out['variants'] = json.load(fh)['variants']
    for variant in todo:
        rows_all = {q: [] for q in ('x', 'sig', 'chi2', 'nu', 'p', 'at_bound', 'corr_A1_bTA', 'corr_A1_A2', 'corr_A1_alpha')}
        cuts, used = [], []
        for snap in snaps:
            if VARIANTS[variant][1] == '_same' and not np.all(np.isfinite(snap.ib['Ib_cm_same'])):
                used.append(False)
                for q in rows_all:
                    rows_all[q].append([None]*len(kgrid))
                cuts.append(None)
                continue
            used.append(True)
            rows = {q: [] for q in rows_all}
            warm = None
            for kmax in kgrid:
                r = snap.fit(variant, kmax, warm)
                warm = r['x']
                nu = 3*r['nk'] - r['npar']
                rows['x'].append(HA.nan_list(r['x']))
                rows['sig'].append(HA.nan_list(r['sig']))
                rows['chi2'].append(r['chi2'])
                rows['nu'].append(nu)
                rows['p'].append(float(chi2_dist.sf(r['chi2'], nu)) if nu > 0 else 1.0)
                rows['at_bound'].append(r['at_bound'].tolist())
                rows['corr_A1_bTA'].append(HA.nan_list(r['corr'][0, 2]))
                rows['corr_A1_A2'].append(HA.nan_list(r['corr'][0, 1]))
                rows['corr_A1_alpha'].append(HA.nan_list(r['corr'][0, 5]))
            for q in rows_all:
                rows_all[q].append(rows[q])
            cuts.append(HA.cut_rule(np.array(rows['p']) >= 0.05, kgrid))
        out['variants'][variant] = dict(rows_all, cut=cuts, used=used, spec=list(map(str, VARIANTS[variant])))
        js = [int(np.argmin(np.abs(kgrid - kk))) for kk in (0.39, 0.52, 0.70, 0.94, 1.26)]
        ok = np.array(used)
        c2 = np.array([[np.nan if v is None else v for v in row] for row in rows_all['chi2']], float)
        nu = np.array([[np.nan if v is None else v for v in row] for row in rows_all['nu']], float)
        print(f'{variant:11s} summed chi2/nu at k_max 0.39, 0.52, 0.70, 0.94, 1.26: '
              + ' '.join(f'{np.nansum(c2[ok, j])/np.nansum(nu[ok, j]):6.2f}' for j in js)
              + f'   [{time.time() - t0:.0f} s]', flush=True)
        with open(path, 'w') as fh:
            json.dump(out, fh)
    print('wrote', path)


if __name__ == '__main__':
    main()
