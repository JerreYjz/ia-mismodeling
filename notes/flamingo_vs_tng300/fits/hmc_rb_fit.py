"""Halo model with and without a red/blue split, fitted jointly to the
FLAMINGO spectra of the core snapshots (0.2 <= z <= 2).

Why joint. Fitted snapshot by snapshot, a red/blue split is only a
relabelling: the effective amplitudes absorb any fraction. The split says
something only when each colour has one amplitude at every redshift
(Fortuna et al. 2021: "we do not include any intrinsic redshift evolution in
our IA model"), so every model here is fitted to the nine core snapshots at
once, with one set of physical parameters and a white term W_s per snapshot.

Data, b(k), N_s, windows, the I_1 P_lin + alpha (P_NL - I_1 P_lin) 2-halo
power (GI) and P_lin + alpha (P_NL - P_lin) (II), and the bounds follow
hmc_alpha_fit.py (variant alpha_cap: a_c <= 1). Every model has the TATT
torquing term A2 X^eta2 (except rbF21), b_TA, the satellite amplitude a_1h,
the central 1-halo a_c and alpha. X = (1 + z)/1.62; f_rc, f_bc, f_rs, f_bs:
red/blue central/satellite fractions; b_r, b_b: mean bias of red and blue
galaxies on the HOD (analysis/roman_dchi2/v6/halo_redblue.py; red
fractions "zt": Zu & Mandelbaum 2016 at z <= 0.1, Tinker et al. 2013 at
z = 0.36 to 0.88, held above; "zm": Zu & Mandelbaum at every z).

No colour split:
  1p      A1 X^eta1
  1pB     A1 b_g/1.5 (b_g: the HOD bias matched to FLAMINGO)
  1pBb    A1 (b_g/1.5)^beta
  1pA, 1pBbA   1p, 1pBb with alpha X^eta_alpha
  1pF, 1pFA    the floor: A1 free at each of the nine snapshots (alpha
          fixed; evolving): the best any law A1(z), colour-based or not, can do
Colour split:
  rbF21   Fortuna et al. 2021 (Eqs. 3-4): red and blue centrals NLA with
          constant amplitudes A_r, A_b, GI amplitude f_rc A_r + f_bc A_b, II
          (f_rc A_r)^2 + (f_bc A_b)^2 without the red-blue cross term
  rbS     halo-based: every galaxy, central or satellite, aligned like its
          host halo, whose alignment scales with its bias (Xia et al. 2017;
          Kurita et al. 2021), blue galaxies scaled by a misalignment factor
          kappa_b (Tenneti et al. 2016: discs are more misaligned):
              a1 = A_h [(f_rc + f_rs)(b_r/1.5)^beta + kappa_b (f_bc + f_bs)(b_b/1.5)^beta],
          beta = 1, kappa_b in [0, 2] (physical: 0 to 1); kappa_b = 1 is 1pB
          (1pBb) exactly, so the split can only match or improve it; the
          TATT terms carry the red-blue cross term
  rbS_zm  rbS with the "zm" fractions
  rbSb, rbSbP   beta free; kappa_b in [0, 2], in [0, 1]
  rbSbA, rbSbAP rbSb, rbSbP with alpha X^eta_alpha
Satellites (both families with colour): red a_1h, blue 0.6 a_1h (F21
Table 3); central 1-halo a_c on red centrals.

Writes hmc_rb_fits.json: per model and k_max the parameters, their errors,
chi2, nu, the parameters at a bound, the effective per-snapshot amplitudes
a1(z), a2(z), and chi2 per snapshot.

    ~/cocoa/Cocoa/.local/bin/python hmc_rb_fit.py [model ...]
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

KMIN = 0.05
K1H, K2H = 4.0, 6.0
AC_CAP = 1.0
X0 = 1.62
KMAXES = (0.39, 0.52, 0.70, 0.94)
AT_BOUND = 1e-3
Q_BLUE = 0.6
C1RHOCRIT = 0.01389

NCORE = 9
_COMMON = (('bTA', 0, 2), ('a_1h', 0, 3), ('a_c', 0, AC_CAP), ('alpha', -1, 2))
_A2 = (('A2', -10, 10), ('eta2', -5, 5))
_EA = (('eta_alpha', -5, 5),)
_A1S = tuple((f'a1_{i}', 0, 40) for i in range(NCORE))
_HALO = (('A_h', 0, 60), ('kappa_b', 0, 2))
_HALOP = (('A_h', 0, 60), ('kappa_b', 0, 1))
_BETA = (('beta', -2, 5),)


def _model(spec, rf=None, q=None):
    '''(parameter names, lower bounds, upper bounds, red-fraction model, blue satellite factor).'''
    return (tuple(n for n, _, _ in spec), tuple(l for _, l, _ in spec), tuple(h for _, _, h in spec), rf, q)


MODELS = {
    # no colour split
    '1p': _model((('A1', 0, 40), ('eta1', -5, 5)) + _A2 + _COMMON),
    '1pB': _model((('A1', 0, 40),) + _A2 + _COMMON),
    '1pBb': _model((('A1', 0, 40),) + _BETA + _A2 + _COMMON),
    '1pA': _model((('A1', 0, 40), ('eta1', -5, 5)) + _A2 + _COMMON + _EA),
    '1pBbA': _model((('A1', 0, 40),) + _BETA + _A2 + _COMMON + _EA),
    '1pF': _model(_A2 + _COMMON + _A1S),
    '1pFA': _model(_A2 + _COMMON + _EA + _A1S),
    # colour split
    'rbF21': _model((('A_r', 0, 60), ('A_b', -40, 40)) + _COMMON, 'zt', Q_BLUE),
    'rbS': _model(_HALO + _A2 + _COMMON, 'zt', Q_BLUE),
    'rbS_zm': _model(_HALO + _A2 + _COMMON, 'zm', Q_BLUE),
    'rbSb': _model(_HALO + _BETA + _A2 + _COMMON, 'zt', Q_BLUE),
    'rbSbP': _model(_HALOP + _BETA + _A2 + _COMMON, 'zt', Q_BLUE),
    'rbSbA': _model(_HALO + _BETA + _A2 + _COMMON + _EA, 'zt', Q_BLUE),
    'rbSbAP': _model(_HALOP + _BETA + _A2 + _COMMON + _EA, 'zt', Q_BLUE),
}
# kappa_b fixed by model name (hmc_rb_kappa_profile.py adds entries)
KAPPA_FIXED = {}
B_PIV = 1.5


def interp_a(log_a, table, a):
    '''A table [n_a, nk] interpolated in ln a (as hmc_alpha_fit.sat_at).'''
    x = np.log(max(a, 1e-6))
    j = int(np.clip(np.searchsorted(log_a, x), 1, log_a.size - 1))
    w = np.clip((x - log_a[j - 1])/(log_a[j] - log_a[j - 1]), 0, 1)
    return table[j - 1]*(1 - w) + table[j]*w


class Snapshot:
    '''Data and tables of one snapshot; the model pieces per population.'''

    def __init__(self, s, fits, central, sat, rb, model):
        self.z = float(fits['zs'][s])
        self.k = fits['k'][s]
        self.y, self.err = fits['y'][s], fits['err'][s]
        self.bias, pnl, self.ns = fits['b'][s], fits['PNL'][s], float(fits['Ns_true'][s])
        idx = np.searchsorted(central['k_all'], self.k)
        assert np.allclose(central['k_all'][idx], self.k)
        plin, i1 = central['Plin'][s][idx], central['I1m'][s][idx]
        self.pc_all = central['Pc_m'][s][idx]
        self.gi0, self.gi1 = i1*plin, pnl - i1*plin           # p_gi = gi0 + alpha gi1
        self.ii0, self.ii1 = plin, pnl - plin
        idx6 = np.searchsorted(sat['k_all'], self.k)
        assert np.allclose(sat['k_all'][idx6], self.k)
        self.log_a = np.log(np.maximum(sat['a_grid'], 1e-6))
        self.sdi_all, self.sii_all = sat['SdI'][s][:, idx6], sat['SII'][s][:, idx6]
        idr = np.searchsorted(rb['k_all'], self.k)
        assert np.allclose(rb['k_all'][idr], self.k) and np.allclose(rb['a_grid'], sat['a_grid'])
        self.rb = {}
        for m in ('zt', 'zm'):
            d = {kk: float(rb[f'{m}|{kk}'][s]) for kk in ('f_rc', 'f_bc', 'f_rs', 'f_bs')}
            d['pc_r'] = rb[f'{m}|Pc_r'][s][idr]
            d['b_r'], d['b_b'] = float(rb[f'{m}|b_r'][s]), float(rb[f'{m}|b_b'][s])
            d['b_rc'], d['b_bc'] = float(rb[f'{m}|b_rc'][s]), float(rb[f'{m}|b_bc'][s])
            for kk in ('SdI_r', 'SdI_b06', 'SII_rr', 'SII_bb06', 'SII_rb06'):
                d[kk] = rb[f'{m}|{kk}'][s][:, idr]
            self.rb[m] = d
        D = model.D(self.z)
        self.c1 = model.cosmo['Om']*C1RHOCRIT/D
        self.c2 = model.cosmo['Om']*C1RHOCRIT/D**2
        self.K = model.kernels(self.z, self.k)
        self.f1h = -np.expm1(-(self.k/K1H)**2)
        self.f2h = np.exp(-(self.k/K2H)**2)
        self.X = (1 + self.z)/X0
        self.bg = float(sat['hod'][s, 4])

    def two_halo(self, a1, a2, bta, alpha, a1sq=None):
        '''TATT 2-halo pieces with amplitudes a1, a2 (a1sq: the amplitude
        squared of the A1^2 terms, for the F21 split without cross term).'''
        K, c1, c2 = self.K, self.c1, self.c2
        a1sq = a1*a1 if a1sq is None else a1sq
        pgi = self.gi0 + alpha*self.gi1
        pii = self.ii0 + alpha*self.ii1
        dE = -(c1*a1*pgi + c1*a1*bta*K['dE'] - 5*c2*a2*K['mix'])
        EE = (c1**2*a1sq*pii + c1**2*a1sq*(bta**2*K['ta'] + 2*bta*K['dE'])
              - 10*c1*a1*c2*a2*(K['mix'] + bta*K['mixEE']) + 25*c2**2*a2**2*K['tt'])
        BB = c1**2*a1sq*bta**2*K['taBB'] - 10*c1*a1*bta*c2*a2*K['mixBB'] + 25*c2**2*a2**2*K['ttB']
        return dE*self.f2h, EE*self.f2h, BB*self.f2h

    def predict(self, name, p, w):
        '''[P_hE, P_EE, P_BB] for model name with global parameters p (dict) and white term w.'''
        rf, q = MODELS[name][3], MODELS[name][4]
        alpha = p['alpha']*self.X**p.get('eta_alpha', 0.0)
        bta, a1h, ac = p['bTA'], p['a_1h'], p['a_c']
        if rf is None:
            a1, a2 = self.effective(name, p)
            dE, EE, BB = self.two_halo(a1, a2, bta, alpha)
            dE_sat = -a1h*self.f1h*interp_a(self.log_a, self.sdi_all, a1h)
            EE_sat = a1h**2*self.f1h*interp_a(self.log_a, self.sii_all, a1h)
            dE_c = -ac*self.pc_all
        else:
            d = self.rb[rf]
            if name == 'rbF21':
                ar, ab_ = d['f_rc']*p['A_r'], d['f_bc']*p['A_b']
                dE, EE, BB = self.two_halo(ar + ab_, 0.0, bta, alpha, a1sq=ar**2 + ab_**2)
            else:
                dE, EE, BB = self.two_halo(*self.effective(name, p), bta, alpha)
            ab = q*a1h
            dE_sat = -self.f1h*(a1h*interp_a(self.log_a, d['SdI_r'], a1h)
                                + ab*interp_a(self.log_a, d['SdI_b06'], a1h))
            EE_sat = self.f1h*(a1h**2*interp_a(self.log_a, d['SII_rr'], a1h)
                               + ab**2*interp_a(self.log_a, d['SII_bb06'], a1h)
                               + 2*a1h*ab*interp_a(self.log_a, d['SII_rb06'], a1h))
            if q not in (0.0, Q_BLUE):
                raise ValueError('blue satellite tables exist for q = 0 and 0.6 only')
            dE_c = -ac*d['pc_r']
        return np.array([self.bias*(dE + dE_sat + dE_c), EE + EE_sat + self.ns + w, BB + self.ns + w])

    def effective(self, name, p):
        '''The effective one-population amplitudes a1(z), a2(z) of a model.'''
        rf = MODELS[name][3]
        if rf is None:
            if name.startswith('1pF'):
                a1 = p[f'a1_{self.idx}']
            elif name.startswith('1pB'):
                a1 = p['A1']*(self.bg/B_PIV)**p.get('beta', 1.0)
            else:
                a1 = p['A1']*self.X**p.get('eta1', 0.0)
            return a1, p['A2']*self.X**p.get('eta2', 0.0)
        d = self.rb[rf]
        if name == 'rbF21':
            return d['f_rc']*p['A_r'] + d['f_bc']*p['A_b'], 0.0
        be = p.get('beta', 1.0)
        kap = KAPPA_FIXED.get(name, p.get('kappa_b'))
        a1 = p['A_h']*((d['f_rc'] + d['f_rs'])*(d['b_r']/B_PIV)**be
                       + kap*(d['f_bc'] + d['f_bs'])*(d['b_b']/B_PIV)**be)
        return a1, p['A2']*self.X**p['eta2']


def fit_joint(snaps, name, kmax, warm=None):
    names, lo, hi = MODELS[name][0], np.array(MODELS[name][1], float), np.array(MODELS[name][2], float)
    ng = len(names)
    for i, s in enumerate(snaps):
        s.idx = i
    lo_w = np.array([-s.ns for s in snaps])
    hi_w = np.array([5*s.ns for s in snaps])
    LO, HI = np.r_[lo, lo_w], np.r_[hi, hi_w]
    masks = [(s.k >= KMIN) & (s.k <= kmax*(1 + 1e-9)) for s in snaps]

    def resid(x):
        p = dict(zip(names, x[:ng]))
        return np.concatenate([((s.y - s.predict(name, p, x[ng + i]))/s.err)[:, m].ravel()
                               for i, (s, m) in enumerate(zip(snaps, masks))])
    starts = []
    base = {'A1': 3.5, 'A2': -1.0, 'eta1': 0.0, 'eta2': 0.0, 'bTA': 0.3, 'a_1h': 1e-3, 'a_c': 0.5,
            'alpha': 0.5, 'A_r': 12.0, 'A_b': 1.0, 'A_h': 10.0, 'kappa_b': 0.3,
            'beta': 1.0, 'eta_alpha': 0.0, **{f'a1_{i}': 3.5 + 0.2*i for i in range(NCORE)}}
    for over in ({}, {'A_r': 6.0, 'A1': 2.5, 'alpha': 0.2}, {'A_r': 20.0, 'A2': -3.0, 'bTA': 0.6},
                 {'A_b': 5.0, 'A2': 1.0, 'a_1h': 0.05, 'alpha': 0.8}, {'eta1': 1.0},
                 {'kappa_b': 1.0, 'A_h': 6.0}, {'kappa_b': 0.05, 'A_h': 14.0},
                 # the 1pB solution, which rbS contains at kappa_b = 1
                 {'kappa_b': 1.0, 'A_h': 3.54, 'A1': 3.54, 'A2': 0.67, 'eta2': 1.73, 'bTA': 0.0, 'alpha': 0.5,
                  'a_1h': 0.0, 'a_c': 0.0},
                 # the 1p solution, which 1pA and 1pF contain
                 {'A1': 3.54, 'eta1': 0.54, 'A2': 0.19, 'eta2': 4.1, 'bTA': 0.01, 'alpha': 0.64,
                  'a_1h': 0.0, 'a_c': 0.0, **{f'a1_{i}': 3.54*((1 + z)/X0)**0.54 for i, z in
                                              enumerate((0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0))}}):
        b = dict(base, **over)
        starts.append(np.r_[[b[n] for n in names], 0.3*np.array([s.ns for s in snaps])])
    if warm is not None:
        starts.append(np.asarray(warm))
    best = None
    for x0 in starts:
        x0 = np.clip(x0, LO + 1e-9*(HI - LO), HI - 1e-9*(HI - LO))
        r = least_squares(resid, x0, bounds=(LO, HI), x_scale='jac', max_nfev=2000)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    width = HI - LO
    at_bound = (x - LO < AT_BOUND*width) | (HI - x < AT_BOUND*width)
    act = np.where(~at_bound)[0]
    J = best.jac[:, act]
    cov = np.linalg.pinv(J.T @ J)
    sig = np.full(x.size, np.nan)
    sig[act] = np.sqrt(np.clip(np.diag(cov), 0, None))
    nk = int(sum(3*m.sum() for m in masks))
    nu = nk - x.size
    chi2 = float(2*best.cost)
    p = dict(zip(names, x[:ng]))
    eff = [s.effective(name, p) for s in snaps]
    return {'x': x.tolist(), 'names': list(names) + [f'W_{s.z:g}' for s in snaps], 'sig': sig.tolist(),
            'chi2': chi2, 'nu': nu, 'p': float(chi2_dist.sf(chi2, nu)),
            'at_bound': [n for n, b in zip(list(names) + [f'W_{s.z:g}' for s in snaps], at_bound) if b],
            'a1_eff': [float(e[0]) for e in eff], 'a2_eff': [float(e[1]) for e in eff],
            'chi2_snap': [float(np.sum(((s.y - s.predict(name, p, x[ng + i]))/s.err)[:, m]**2))
                          for i, (s, m) in enumerate(zip(snaps, masks))]}


def main():
    t0 = time.time()
    fits = np.load(os.path.join(V2, 'fits.npz'))
    central = np.load(os.path.join(V6, 'halo_central_tables.npz'))
    sat = np.load(os.path.join(V6, 'halo_tables.npz'))
    rb = np.load(os.path.join(V6, 'halo_redblue_tables.npz'))
    v3 = np.load(os.path.join(V3, 'v3.npz'))
    core = np.where(v3['core'])[0]
    model = F.TATTModel(halofit='takahashi')
    snaps = [Snapshot(s, fits, central, sat, rb, model) for s in core]
    path = os.path.join(HERE, 'hmc_rb_fits.json')
    out = {'about': __doc__, 'zs': [s.z for s in snaps], 'kmaxes': list(KMAXES), 'models': {},
           'fractions': {m: {kk: [s.rb[m][kk] for s in snaps] for kk in ('f_rc', 'f_bc', 'f_rs', 'f_bs')}
                         for m in ('zt', 'zm')}}
    todo = sys.argv[1:] or list(MODELS)
    if sys.argv[1:] and os.path.isfile(path):
        out['models'] = json.load(open(path))['models']
    for name in todo:
        res, warm = [], None
        for kmax in KMAXES:
            r = fit_joint(snaps, name, kmax, warm)
            warm = r['x']
            res.append(r)
            ng = len(MODELS[name][0])
            print(f'[{time.time() - t0:.0f}s] {name:9s} k_max {kmax:.2f}: chi2/nu {r["chi2"]:.1f}/{r["nu"]} = '
                  f'{r["chi2"]/r["nu"]:.2f}; ' + ', '.join(f'{n} {v:+.3g}' for n, v in zip(r['names'][:ng], r['x'][:ng]))
                  + (f'  AT BOUND {[b for b in r["at_bound"]]}' if r['at_bound'] else ''), flush=True)
        out['models'][name] = res
        json.dump(out, open(path, 'w'), indent=1)
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
