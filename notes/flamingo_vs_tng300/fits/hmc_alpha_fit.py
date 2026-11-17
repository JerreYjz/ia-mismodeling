"""Refit of the halo model (HMc + W) to FLAMINGO with a free share of the
nonlinear power in the 2-halo term and a physical cap on the central 1-halo
amplitude.

The question: in HMc the tidal-alignment (A1) terms sit on I_1 P_lin (GI) and
P_lin (II), and the fit then needs a central 1-halo amplitude a_c of 20 to
300, far above its physical size (about 0.5 for a galaxy that copies the
shape of its halo). Is that amplitude compensating for nonlinear power
missing from the 2-halo term?

The test: put the A1 terms on

    P_GI = I_1 P_lin + alpha (P_NL - I_1 P_lin)
    P_II = P_lin + alpha (P_NL - P_lin)

alpha = 0 is HMc, alpha = 1 is the nonlinear power of NLA and TATT. Everything
else is the HMc_W variant of analysis/roman_dchi2/v6/halo_central_fit.py:
same data (v2/fits.npz: P_hE, P_EE, P_BB with the FLAMINGO errors,
0.05 <= k <= k_max), same b(k), N_s, windows, satellite tables and bounds.

Variants (parameters A1, A2, b_TA, a_1h, a_c, W, alpha):
    base        alpha = 0, a_c in [0, 300]        (= stored HMc_W, a check)
    cap         alpha = 0, a_c in [0, AC_CAP]
    alpha_cap   alpha in [ALPHA_LO, ALPHA_HI], a_c in [0, AC_CAP]
    alpha_free  alpha in [ALPHA_LO, ALPHA_HI], a_c in [0, 300]
    alpha_cap_nosat   alpha_cap with the satellite amplitude a_1h fixed to 0.
                Added after the checks (hmc_alpha_checks.py) showed that a_1h is
                unconstrained for k_max below about 1 (errors of hundreds) and
                that its noise value then drives the model above the fit range.
alpha is allowed outside its physical range [0, 1] on purpose, so that a
fit that wants an unphysical value shows it instead of hiding at a bound.

The model is rebuilt from the tables (no import of the v6 scripts, which
write log files next to themselves); the base variant must reproduce the
stored HMc_W chi2, and the script prints how well it does.

Writes hmc_alpha_fits.json next to this file; with variant names as
arguments only those are (re)fitted and the others are kept from the file.
Run with the cocoa Python:
    ~/cocoa/Cocoa/.local/bin/python hmc_alpha_fit.py [variant ...]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'analysis'))
V2 = os.path.join(ANA, 'roman_dchi2', 'v2')
V3 = os.path.join(ANA, 'roman_dchi2', 'v3')
V6 = os.path.join(ANA, 'roman_dchi2', 'v6')
sys.path.insert(0, os.path.join(ANA, 'tatt_fit'))
import fit_tatt_flamingo as F  # noqa: E402

# lowest k of every fit [h/Mpc]
KMIN = 0.05
# scales of the 1-halo and 2-halo windows [h/Mpc]
K1H, K2H = 4.0, 6.0
# the physical cap of the central 1-halo amplitude, and its stored bound
AC_CAP, AC_FREE = 1.0, 300.0
# the allowed range of alpha (the physical range is 0 to 1)
ALPHA_LO, ALPHA_HI = -1.0, 2.0
# parameter names, in the order of the parameter vector
NAMES = ('A1', 'A2', 'bTA', 'a_1h', 'a_c', 'W', 'alpha')
# variants: (upper bound of a_c, alpha free?, a_1h free?)
VARIANTS = {'base': (AC_FREE, False, True), 'cap': (AC_CAP, False, True),
            'alpha_cap': (AC_CAP, True, True), 'alpha_free': (AC_FREE, True, True),
            'alpha_cap_nosat': (AC_CAP, True, False)}
# a parameter closer to a bound than this fraction of the bound width is
# reported as sitting at the bound
AT_BOUND = 1e-3


class Snapshot:
    '''The data and the tables of one snapshot, and the model on them.'''

    def __init__(self, s, fits, central, sat, model):
        self.z = float(fits['zs'][s])
        self.k = fits['k'][s]
        self.y, self.err = fits['y'][s], fits['err'][s]
        self.bias, self.pnl, self.ns = fits['b'][s], fits['PNL'][s], float(fits['Ns_true'][s])
        idx = np.searchsorted(central['k_all'], self.k)
        assert np.allclose(central['k_all'][idx], self.k)
        self.plin, self.i1, self.pc = (central[n][s][idx] for n in ('Plin', 'I1m', 'Pc_m'))
        idx6 = np.searchsorted(sat['k_all'], self.k)
        assert np.allclose(sat['k_all'][idx6], self.k)
        self.sdi, self.sii = sat['SdI'][s][:, idx6], sat['SII'][s][:, idx6]
        self.log_a = np.log(np.maximum(sat['a_grid'], 1e-6))
        self.model = model
        self.kern = model.kernels(self.z, self.k)
        self.f1h = -np.expm1(-(self.k/K1H)**2)
        self.f2h = np.exp(-(self.k/K2H)**2)

    def sat_at(self, table, a1h):
        '''A satellite table interpolated in ln a_1h (as f21_variants.interp_a).'''
        x = np.log(max(a1h, 1e-6))
        j = int(np.clip(np.searchsorted(self.log_a, x), 1, self.log_a.size - 1))
        w = np.clip((x - self.log_a[j - 1])/(self.log_a[j] - self.log_a[j - 1]), 0, 1)
        return table[j - 1]*(1 - w) + table[j]*w

    def parts(self, x):
        '''The terms of the model at x: the 2-halo, satellite and central
        parts of P_dE, and P_EE, P_BB with the white term.'''
        a1, a2, bta, a1h, ac, w, alpha = x
        p_gi = self.i1*self.plin + alpha*(self.pnl - self.i1*self.plin)
        p_ii = self.plin + alpha*(self.pnl - self.plin)
        gi = self.model.predict(self.z, self.k, a1, a2, bta, 1.0, K=self.kern, PNL=p_gi, Ns=0.0)
        ii = self.model.predict(self.z, self.k, a1, a2, bta, 1.0, K=self.kern, PNL=p_ii, Ns=0.0)
        return {'dE_2h': gi['dE']*self.f2h,
                'dE_sat': -a1h*self.f1h*self.sat_at(self.sdi, a1h),
                'dE_c': -ac*self.pc,
                'EE': ii['EE']*self.f2h + a1h**2*self.f1h*self.sat_at(self.sii, a1h) + self.ns + w,
                'BB': ii['BB']*self.f2h + self.ns + w}

    def predict(self, x):
        '''[P_hE, P_EE, P_BB] at x.'''
        p = self.parts(x)
        return np.array([self.bias*(p['dE_2h'] + p['dE_sat'] + p['dE_c']), p['EE'], p['BB']])

    def fit(self, variant, kmax, warm=None):
        '''Bounded least squares over 0.05 <= k <= kmax from several starts.
        Returns the parameters, chi2, the number of k bins, the parameter
        errors (NaN for fixed parameters and for those at a bound) and the
        correlation matrix of the free parameters.'''
        ac_hi, alpha_free, sat_free = VARIANTS[variant]
        lo = np.array([0.0, -5.0, 0.0, 0.0, 0.0, -self.ns, ALPHA_LO if alpha_free else 0.0])
        hi = np.array([20.0, 5.0, 2.0, 3.0 if sat_free else 0.0, ac_hi, 5*self.ns,
                       ALPHA_HI if alpha_free else 0.0])
        free = np.array([True, True, True, sat_free, True, True, alpha_free])
        m = (self.k >= KMIN) & (self.k <= kmax*(1 + 1e-9))

        def full(xf):
            x = np.zeros(7)
            x[free] = xf
            return x

        def resid(xf):
            return ((self.y - self.predict(full(xf)))/self.err)[:, m].ravel()
        starts = []
        for a1, a2, bta, a1h, ac in ([4.0, -1.0, 0.1, 1e-3, 0.0], [4.0, -1.0, 0.3, 0.02, 10.0],
                                     [3.0, -1.5, 0.3, 1e-3, 40.0], [4.0, 0.0, 0.5, 0.05, 0.5]):
            for w in (0.0, 0.5*self.ns):
                for alpha in ((0.0, 0.5, 1.0) if alpha_free else (0.0,)):
                    starts.append([a1, a2, bta, a1h, ac, w, alpha])
        if warm is not None:
            starts.append(list(warm))
        best = None
        for x0 in starts:
            x0 = np.clip(np.array(x0), lo, hi)[free]
            r = least_squares(resid, x0, bounds=(lo[free], hi[free]), x_scale='jac', max_nfev=400)
            if best is None or r.cost < best.cost:
                best = r
        x = full(best.x)
        width = hi - lo
        at_bound = free & ((x - lo < AT_BOUND*np.maximum(width, 1e-30))
                           | (hi - x < AT_BOUND*np.maximum(width, 1e-30)))
        # errors from the Jacobian over the parameters that are free and
        # away from their bounds
        act = np.where(free & ~at_bound)[0]
        jac = best.jac[:, [list(np.where(free)[0]).index(q) for q in act]]
        cov = np.linalg.pinv(jac.T @ jac)
        sig = np.full(7, np.nan)
        sig[act] = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = np.full((7, 7), np.nan)
        corr[np.ix_(act, act)] = cov/np.outer(sig[act], sig[act])
        return {'x': x, 'chi2': float(2*best.cost), 'nk': int(m.sum()), 'sig': sig, 'corr': corr,
                'at_bound': at_bound, 'npar': int(free.sum())}


def cut_rule(ok, kgrid):
    '''Last k_max before two consecutive failures (as v4/simfit.cut_rule).'''
    for j in range(len(kgrid) - 1):
        if not ok[j] and not ok[j + 1]:
            return float(kgrid[j - 1]) if j > 0 else float('nan')
    return float(kgrid[-1] if ok[-1] else kgrid[-2])


def nan_list(a):
    '''A float array as a nested list with None for NaN (valid JSON).'''
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, None).tolist() if a.ndim else float(a)


def main():
    t0 = time.time()
    fits = np.load(os.path.join(V2, 'fits.npz'))
    central = np.load(os.path.join(V6, 'halo_central_tables.npz'))
    sat = np.load(os.path.join(V6, 'halo_tables.npz'))
    stored = np.load(os.path.join(V6, 'f21v_HMc_W.npz'))
    v3 = np.load(os.path.join(V3, 'v3.npz'))
    kgrid, core = v3['kgrid'], v3['core']
    assert np.allclose(kgrid, stored['kgrid'])
    model = F.TATTModel(halofit='takahashi')
    zs = fits['zs']
    out = {'about': ('Fits of analysis/roman_dchi2 HMc_W variants to FLAMINGO with a free share '
                     'alpha of the nonlinear power in the A1 terms and a capped central 1-halo '
                     'amplitude; see hmc_alpha_fit.py. Arrays are [snapshot][k_max] and '
                     'parameters are in the order of names. None: fixed, or at a bound.'),
           'names': list(NAMES), 'zs': zs.tolist(), 'sns': fits['sns'].tolist(),
           'kgrid': kgrid.tolist(), 'core': core.tolist(), 'kmin': KMIN,
           'ac_cap': AC_CAP, 'ac_free': AC_FREE, 'alpha_range': [ALPHA_LO, ALPHA_HI],
           'variants': {}}
    snaps = [Snapshot(s, fits, central, sat, model) for s in range(len(zs))]
    path = os.path.join(HERE, 'hmc_alpha_fits.json')
    todo = sys.argv[1:] or list(VARIANTS)
    if sys.argv[1:] and os.path.isfile(path):
        with open(path) as fh:
            out['variants'] = json.load(fh)['variants']
    for variant in todo:
        res = {q: [] for q in ('x', 'sig', 'chi2', 'nu', 'p', 'at_bound', 'corr_alpha_ac',
                               'corr_alpha_bTA', 'corr_alpha_A2')}
        cuts = []
        for snap in snaps:
            rows = {q: [] for q in res}
            warm = None
            for kmax in kgrid:
                r = snap.fit(variant, kmax, warm)
                warm = r['x']
                nu = 3*r['nk'] - r['npar']
                rows['x'].append(r['x'])
                rows['sig'].append(r['sig'])
                rows['chi2'].append(r['chi2'])
                rows['nu'].append(nu)
                rows['p'].append(float(chi2_dist.sf(r['chi2'], nu)) if nu > 0 else 1.0)
                rows['at_bound'].append(r['at_bound'].tolist())
                rows['corr_alpha_ac'].append(r['corr'][6, 4])
                rows['corr_alpha_bTA'].append(r['corr'][6, 2])
                rows['corr_alpha_A2'].append(r['corr'][6, 1])
            for q in res:
                res[q].append(rows[q])
            cuts.append(cut_rule(np.array(rows['p']) >= 0.05, kgrid))
        chi2 = np.array(res['chi2'])
        nu = np.array(res['nu'], dtype=float)
        if variant == 'base':
            diff = chi2 - stored['HMc_W_chi2']
            print(f'base against the stored HMc_W chi2: max |difference| {np.abs(diff).max():.3f}, '
                  f'largest improvement {diff.min():.3f}, largest loss {diff.max():.3f}')
        out['variants'][variant] = {
            'x': nan_list(res['x']), 'sig': nan_list(res['sig']), 'chi2': chi2.tolist(),
            'nu': nu.tolist(), 'p': res['p'], 'at_bound': res['at_bound'], 'cut': nan_list(cuts),
            'corr_alpha_ac': nan_list(res['corr_alpha_ac']),
            'corr_alpha_bTA': nan_list(res['corr_alpha_bTA']),
            'corr_alpha_A2': nan_list(res['corr_alpha_A2'])}
        js = [int(np.argmin(np.abs(kgrid - kk))) for kk in (0.39, 0.52, 0.70, 0.94, 3.04)]
        print(f'{variant:10s} summed chi2/nu over the core snapshots at k_max 0.39, 0.52, 0.70, '
              f'0.94, 3.04: ' + ' '.join(f'{chi2[core, j].sum()/nu[core, j].sum():7.2f}' for j in js)
              + f'   cuts {np.round(np.array(cuts)[core], 2)}  [{time.time() - t0:.0f} s]',
              flush=True)
    with open(path, 'w') as fh:
        json.dump(out, fh)
    print('wrote', path)


if __name__ == '__main__':
    main()
