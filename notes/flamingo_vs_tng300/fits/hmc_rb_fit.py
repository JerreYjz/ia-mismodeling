"""Halo model with red and blue galaxies, fitted jointly to the FLAMINGO
spectra of the core snapshots (0.2 <= z <= 2).

Why joint. Fitted snapshot by snapshot, a red/blue split is only a
relabelling: the effective amplitudes f_rc A_r + f_bc A_b absorb any
fraction. The split says something only when the amplitudes of each colour
are the same at every redshift (Fortuna et al. 2021: "we do not include any
intrinsic redshift evolution in our IA model") and the redshift dependence
of the signal comes from the red and blue fractions of the sample. So every
model here is fitted to the nine core snapshots at once, with one set of
physical amplitudes and a white term W_s per snapshot.

Data, b(k), N_s, windows, the I_1 P_lin + alpha (P_NL - I_1 P_lin) 2-halo
power (GI) and P_lin + alpha (P_NL - P_lin) (II), and the bounds follow
hmc_alpha_fit.py (variant alpha_cap: a_c <= 1).

Models (X = (1 + z)/1.62; f_rc, f_bc: red and blue central fractions of the
sample from analysis/roman_dchi2/v6/halo_redblue.py):
  1p      one population, power-law evolution: A1 X^eta1, A2 X^eta2, b_TA,
          satellites a_1h, central 1-halo a_c (all centrals), alpha
  1p0     1p with eta1 = eta2 = 0 (no evolution)
  rbF21   red centrals NLA A_r, blue centrals NLA A_b, as Fortuna et al.
          2021 Eqs. 3-4: GI amplitude f_rc A_r + f_bc A_b, II amplitude
          (f_rc A_r)^2 + (f_bc A_b)^2 without the red-blue cross term;
          red satellites a_1h, blue satellites 0.6 a_1h (F21 Table 3);
          central 1-halo a_c on red centrals only
  rbTT    red centrals NLA A_r, blue centrals tidal torquing A2_b (the TATT
          quadratic term with amplitude f_bc A2_b); TATT then carries the
          red-blue cross terms it implies; satellites and a_c as rbF21
  rbTT_q0 rbTT with unaligned blue satellites
  rbTT_zm rbTT with the red fractions of Zu & Mandelbaum 2016 at every z
  rbTT_eta rbTT with A_r X^eta_r: does the red fraction remove the need
          for an evolution law?
Added after the first fits, which needed the amplitude to rise with z as
the galaxy bias does (A1/b_g of the per-snapshot fits is 2.4 to 2.6):
  1pB     one population with A1 b_g(z)/B_PIV in place of A1 X^eta1 (b_g:
          the HOD bias matched to FLAMINGO, halo_sim.py); A2 X^eta2 kept
  rbB     red and blue centrals both linear (NLA), each with its amplitude
          times its own bias, A_r b_r/B_PIV and A_b b_b/B_PIV (with the
          red-blue cross term: two linear populations share the tidal
          field); blue tidal torquing f_bc A2_b; satellites and a_c as rbTT.
          A_b = A_r means alignment that does not care about colour.
Halo-based colour split (added after scrutiny of the models above, which
give each colour an amplitude that ignores its halo): a central takes the
alignment of its halo, which scales with the halo bias (Xia et al. 2017:
the mass and formation-time dependence of halo alignment is explained by
halo bias; Kurita et al. 2021), times a misalignment factor that depends on
colour (Tenneti et al. 2016: discs are more misaligned with their halos
than ellipticals), 1 for red and kappa_b for blue:
    a1(z) = A_h [f_rc b_rc + kappa_b f_bc b_bc],
b_rc, b_bc the mean bias of red and blue centrals (halo_redblue.py), the
satellites not aligned on large scales (F21). Both colours align linearly
with the same tidal field, so P_II keeps the red-blue cross term.
  rbH     kappa_b free in [0, 2] (physical: 0 to 1), blue tidal torquing
          f_bc A2_b
  rbH_A2  rbH with torquing on every central, A2 X^eta2, not by colour
  rbH_k1  kappa_b = 1 (colour-blind)
  rbH_kF  kappa_b = 0.04 (the F21 ratio A_blue/A_red = 0.21/5.3)
  rbH_zm  rbH with the Zu & Mandelbaum fractions at every z
The rbH fits need the satellites to carry large-scale alignment (with the
centrals alone the amplitude rises too steeply with z), so:
  rbS     every galaxy, central or satellite, aligned like its host halo:
          a1 = A_h [(f_rc + f_rs) b_r + kappa_b (f_bc + f_bs) b_b]/1.5,
          torquing A2 X^eta2 on every galaxy. kappa_b = 1 is 1pB exactly
          (the bracket is then b_g), so the colour split can only improve it.
  rbS_bt  rbS with the torquing on blue galaxies only, f_bc A2_b
  rbS_zm  rbS with the Zu & Mandelbaum fractions at every z
Diagnostics of what limits the joint fits (added after rbS fell short of 1p):
  1pF     one population with A1 free at every snapshot (nine values), the
          rest shared: the floor any law A1(z), colour-based or not, can reach
  1pA     1p with alpha X^eta_alpha (the per-snapshot alpha rises with z)
  1pFA    1pF with alpha X^eta_alpha
  1pBb    one population, A1 (b_g/1.5)^beta: alignment growing as a power of
          the bias (misalignment falls with halo mass, Tenneti et al. 2016)
  1pBbA   1pBb with alpha X^eta_alpha
  rbSb    rbS with b^beta for each colour (beta = 1 is rbS)
  rbSbA   rbSb with alpha X^eta_alpha
  rbSbP, rbSbAP   rbSb and rbSbA with kappa_b in its physical range [0, 1]
          (the free fits reach the bound kappa_b = 2: blue galaxies twice as
          aligned as red, which uses the rising blue fraction as an evolution law)
The red fractions are fixed inputs (FLAMINGO has no colour split): model
"zt" (Zu & Mandelbaum 2016 at z <= 0.1, Tinker et al. 2013 at z = 0.36 to
0.88, held above) unless stated.

Writes hmc_rb_fits.json: per model and k_max the parameters, their errors,
chi2, nu, the parameters at a bound, and the effective per-snapshot
amplitudes a1(z), a2(z).

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

# model: (global parameter names, bounds lo, bounds hi, red-fraction model, blue satellites q)
MODELS = {
    '1p': (('A1', 'A2', 'eta1', 'eta2', 'bTA', 'a_1h', 'a_c', 'alpha'),
           (0, -10, -5, -5, 0, 0, 0, -1), (40, 10, 5, 5, 2, 3, AC_CAP, 2), None, None),
    '1p0': (('A1', 'A2', 'bTA', 'a_1h', 'a_c', 'alpha'),
            (0, -10, 0, 0, 0, -1), (40, 10, 2, 3, AC_CAP, 2), None, None),
    'rbF21': (('A_r', 'A_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
              (0, -40, 0, 0, 0, -1), (60, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbTT': (('A_r', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
             (0, -40, 0, 0, 0, -1), (60, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbTT_q0': (('A_r', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
                (0, -40, 0, 0, 0, -1), (60, 40, 2, 3, AC_CAP, 2), 'zt', 0.0),
    'rbTT_zm': (('A_r', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
                (0, -40, 0, 0, 0, -1), (60, 40, 2, 3, AC_CAP, 2), 'zm', Q_BLUE),
    'rbTT_eta': (('A_r', 'A2_b', 'eta_r', 'bTA', 'a_1h', 'a_c', 'alpha'),
                 (0, -40, -5, 0, 0, 0, -1), (60, 40, 5, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    '1pB': (('A1', 'A2', 'eta2', 'bTA', 'a_1h', 'a_c', 'alpha'),
            (0, -10, -5, 0, 0, 0, -1), (40, 10, 5, 2, 3, AC_CAP, 2), None, None),
    'rbB': (('A_r', 'A_b', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
            (0, -40, -40, 0, 0, 0, -1), (60, 60, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbH': (('A_h', 'kappa_b', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
            (0, 0, -40, 0, 0, 0, -1), (60, 2, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbH_A2': (('A_h', 'kappa_b', 'A2', 'eta2', 'bTA', 'a_1h', 'a_c', 'alpha'),
               (0, 0, -10, -5, 0, 0, 0, -1), (60, 2, 10, 5, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbH_k1': (('A_h', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
               (0, -40, 0, 0, 0, -1), (60, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbH_kF': (('A_h', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
               (0, -40, 0, 0, 0, -1), (60, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbH_zm': (('A_h', 'kappa_b', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
               (0, 0, -40, 0, 0, 0, -1), (60, 2, 40, 2, 3, AC_CAP, 2), 'zm', Q_BLUE),
    'rbS': (('A_h', 'kappa_b', 'A2', 'eta2', 'bTA', 'a_1h', 'a_c', 'alpha'),
            (0, 0, -10, -5, 0, 0, 0, -1), (60, 2, 10, 5, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbS_bt': (('A_h', 'kappa_b', 'A2_b', 'bTA', 'a_1h', 'a_c', 'alpha'),
               (0, 0, -40, 0, 0, 0, -1), (60, 2, 40, 2, 3, AC_CAP, 2), 'zt', Q_BLUE),
    'rbS_zm': (('A_h', 'kappa_b', 'A2', 'eta2', 'bTA', 'a_1h', 'a_c', 'alpha'),
               (0, 0, -10, -5, 0, 0, 0, -1), (60, 2, 10, 5, 2, 3, AC_CAP, 2), 'zm', Q_BLUE),
}
NCORE = 9
_COMMON = (('bTA', 0, 2), ('a_1h', 0, 3), ('a_c', 0, AC_CAP), ('alpha', -1, 2))
_A2 = (('A2', -10, 10), ('eta2', -5, 5))
_EA = (('eta_alpha', -5, 5),)
_A1S = tuple((f'a1_{i}', 0, 40) for i in range(NCORE))


def _model(spec, rf=None, q=None):
    return (tuple(n for n, _, _ in spec), tuple(l for _, l, _ in spec), tuple(h for _, _, h in spec), rf, q)


MODELS.update({
    '1pF': _model(_A2 + _COMMON + _A1S),
    '1pA': _model((('A1', 0, 40), ('eta1', -5, 5)) + _A2 + _COMMON + _EA),
    '1pFA': _model(_A2 + _COMMON + _EA + _A1S),
    '1pBb': _model((('A1', 0, 40), ('beta', -2, 5)) + _A2 + _COMMON),
    '1pBbA': _model((('A1', 0, 40), ('beta', -2, 5)) + _A2 + _COMMON + _EA),
    'rbSb': _model((('A_h', 0, 60), ('kappa_b', 0, 2), ('beta', -2, 5)) + _A2 + _COMMON, 'zt', Q_BLUE),
    'rbSbA': _model((('A_h', 0, 60), ('kappa_b', 0, 2), ('beta', -2, 5)) + _A2 + _COMMON + _EA, 'zt', Q_BLUE),
    'rbSbP': _model((('A_h', 0, 60), ('kappa_b', 0, 1), ('beta', -2, 5)) + _A2 + _COMMON, 'zt', Q_BLUE),
    'rbSbAP': _model((('A_h', 0, 60), ('kappa_b', 0, 1), ('beta', -2, 5)) + _A2 + _COMMON + _EA, 'zt', Q_BLUE),
})
KAPPA_FIXED = {'rbH_k1': 1.0, 'rbH_kF': 0.21/5.3}
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
            ar = p.get('A_r', 0.0)*self.X**p.get('eta_r', 0.0)
            if name.startswith('rbF21'):
                a1 = d['f_rc']*ar + d['f_bc']*p['A_b']
                dE, EE, BB = self.two_halo(a1, 0.0, bta, alpha,
                                           a1sq=(d['f_rc']*ar)**2 + (d['f_bc']*p['A_b'])**2)
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
        ar = p.get('A_r', 0.0)*self.X**p.get('eta_r', 0.0)
        if name.startswith('rbS'):
            be = p.get('beta', 1.0)
            kap = KAPPA_FIXED.get(name, p.get('kappa_b'))
            a1 = p['A_h']*((d['f_rc'] + d['f_rs'])*(d['b_r']/B_PIV)**be
                           + kap*(d['f_bc'] + d['f_bs'])*(d['b_b']/B_PIV)**be)
            a2 = d['f_bc']*p['A2_b'] if name == 'rbS_bt' else p['A2']*self.X**p['eta2']
            return a1, a2
        if name.startswith('rbH'):
            kap = KAPPA_FIXED.get(name, p.get('kappa_b'))
            a1 = p['A_h']*(d['f_rc']*d['b_rc'] + kap*d['f_bc']*d['b_bc'])
            a2 = p['A2']*self.X**p['eta2'] if name == 'rbH_A2' else d['f_bc']*p['A2_b']
            return a1, a2
        if name == 'rbB':
            return ((d['f_rc']*p['A_r']*d['b_r'] + d['f_bc']*p['A_b']*d['b_b'])/B_PIV,
                    d['f_bc']*p['A2_b'])
        if name.startswith('rbF21'):
            return d['f_rc']*ar + d['f_bc']*p['A_b'], 0.0
        return d['f_rc']*ar, d['f_bc']*p['A2_b']


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
            'alpha': 0.5, 'A_r': 12.0, 'A_b': 1.0, 'A2_b': -2.0, 'eta_r': 0.0, 'A_h': 10.0, 'kappa_b': 0.3,
            'beta': 1.0, 'eta_alpha': 0.0, **{f'a1_{i}': 3.5 + 0.2*i for i in range(NCORE)}}
    for over in ({}, {'A_r': 6.0, 'A1': 2.5, 'alpha': 0.2}, {'A_r': 20.0, 'A2_b': -6.0, 'A2': -3.0, 'bTA': 0.6},
                 {'A_b': 5.0, 'A2_b': 2.0, 'A2': 1.0, 'a_1h': 0.05, 'alpha': 0.8}, {'eta1': 1.0, 'eta_r': -1.0},
                 {'kappa_b': 1.0, 'A_h': 6.0}, {'kappa_b': 0.05, 'A_h': 14.0, 'A2_b': 1.0},
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
