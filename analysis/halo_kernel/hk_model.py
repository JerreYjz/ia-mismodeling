#!/usr/bin/env python3
"""The kernel-calibrated IA halo model (HMk): assembly and FLAMINGO comparison.

Galaxy shape field, halo by halo (all ellipticities projected along z):
    central:    e_c = kappa_c(M) e_h + noise           (misalignment kernel)
    satellite:  e_s = kappa_s(M) e_h(host) + radial + noise
with e_h the projected DM shape of the (host) halo. Galaxy positions: centrals
at the halo centre, satellites with the measured profile u_s(k|M).

2-halo terms (halo mass bins i, j; n_i halo density, n_g galaxy density):
    a_i(k) = n_i [Nc_i + Ns_i u_s,i(k)] / n_g                  position weights
    c_j(k) = n_j [Nc_j kappa_c,j + Ns_j kappa_s,j u_s,j(k)] / n_g   shape weights
    P_gE^2h(k) = sum_ij a_i c_j P_{h_i E_j}(k)
    P_EE^2h(k) = sum_ij c_i c_j P_{E_i E_j}(k),  same for BB
where P_{h_i E_j} is the gravity layer: halo positions (bin i) x halo DM shapes
(bin j), and P_{E_i E_j} halo shapes x halo shapes. Three gravity layers:
    tab  the measured halo spectra (tests the decomposition itself)
    lin  b_i b_K,j P_lin                   (beta = 0: the HMc assumption)
    emu  b_i b_K,j P_lin [1 + beta_NL]     (Dark Emulator density beta)
with b_i, b_K,j the measured large-scale halo and shape biases (k = 0.02 to
0.08). 1-halo term: the measured 1-halo spectra of the galaxy sample (all
halos shuffled), the calibrated 1-halo kernel. Spectra are in the
cosmolike-fit convention of v2/fits.npz (mu = 0, shear units).

    python hk_model.py            # writes model/hk_model.json and prints the GOF
    python hk_model.py 38 58      # those snapshots only, to model/hk_model_test.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hk_fields as HF  # noqa: E402

sys.path.insert(0, os.path.join(HF.REPO, 'analysis', 'roman_dchi2', 'v6'))
import halo_ia as HIA  # noqa: E402
import fit_tatt_flamingo as F  # noqa: E402

KERN = os.path.join(HERE, 'kernels')
FITS = os.path.join(HF.REPO, 'analysis', 'roman_dchi2', 'v2', 'fits.npz')
KLOW = (0.02, 0.08)            # large-scale range for b_i and b_K,j [h/Mpc]
KMIN = 0.05                    # lowest k of the FLAMINGO fits
LAYERS = ('tab', 'det', 'lin', 'emu')
# galaxy-layer kernels: per-object regression (obj) or large-scale ratio (ls)
KAPPAS = ('ls', 'obj')
KLS = (0.02, 0.10)            # k range of the large-scale alignment ratio [h/Mpc]
# the measured 1-halo term is noisy at low k (shape noise of the shuffled catalogue):
# below K1H it is replaced by its physical limit, white for II (the mean over
# K1H to 2 K1H) and proportional to k^2 for GI (matched at K1H to 2 K1H)
K1H = 0.2
NC_MIN = 0.01                 # bins with fewer centrals per halo get kappa_c = 0
# a kernel calibrated on fewer galaxies than this is noise (kappa_c = -4 from one
# central at z = 3); those galaxies, under 0.1 percent of the sample, get kappa = 0
N_MIN = 1000
# the residual (stochastic) 2-halo II and BB are a difference of noisy spectra at
# k < K_RES (a few hundred modes); like the 1-halo term they take their k -> 0
# limit there, a constant equal to their mean over K_RES to 2 K_RES
K_RES = 0.1


def load(sn):
    d = dict(np.load(os.path.join(KERN, f'kernels_{sn}.npz')))
    nb = len(d['lgm_edges']) - 1
    S = {k[2:]: v for k, v in d.items() if k.startswith('S|')}

    def mat(fmt, sym):
        m = np.zeros((nb, nb, 3, len(d['k'])))
        for i in range(nb):
            for j in range(nb):
                key = fmt(i, j) if (not sym or j >= i) else fmt(j, i)
                m[i, j] = S[key]
        return m
    d['PhE'] = mat(lambda i, j: f'halo|d{i}|d|s{j}|E', False)
    d['Phh'] = mat(lambda i, j: f'halo|d{i}|d|d{j}|d', True)
    d['PEE'] = mat(lambda i, j: f'halo|s{i}|E|s{j}|E', True)
    d['PBB'] = mat(lambda i, j: f'halo|s{i}|B|s{j}|B', True)
    d['G'] = {p: S[f'gal|g|d|g|{p}'] if p != 'hh' else S['gal|g|d|g|d'] for p in ('E', 'hh')}
    d['G'] = {'hE': S['gal|g|d|g|E'], 'EE': S['gal|g|E|g|E'], 'BB': S['gal|g|B|g|B'], 'hh': S['gal|g|d|g|d'],
              'hEc': S['gal|g|d|gc|E'], 'hEs': S['gal|g|d|gs|E']}
    d['G1h'] = {'hE': S['gal1h|g|d|g|E'], 'EE': S['gal1h|g|E|g|E'], 'BB': S['gal1h|g|B|g|B']}
    # galaxy density x shapes of centrals (satellites) with host in bin j
    zero = np.zeros((3, len(d['k'])))
    d['Gc'] = np.array([S.get(f'gal|g|d|c{j}|E', zero) for j in range(nb)])
    d['Gt'] = np.array([S.get(f'gal|g|d|t{j}|E', zero) for j in range(nb)])
    d['nb'] = nb
    d['sn'] = sn
    fw = os.path.join(KERN, f'weighted_{sn}.npz')
    if os.path.isfile(fw):
        w = dict(np.load(fw))
        assert np.allclose(w['k'], d['k'], rtol=1e-6, equal_nan=True), 'k grids differ'
        d['W'] = {key[2:]: v for key, v in w.items() if key.startswith('S|')}
    fs = os.path.join(KERN, f'stoch_{sn}.npz')
    if os.path.isfile(fs):
        t = dict(np.load(fs))
        assert np.allclose(t['k'], d['k'], rtol=1e-6, equal_nan=True), 'k grids differ'
        d['St'] = {key[2:]: v for key, v in t.items() if key.startswith('S|')}
        d['St_kappa'] = {key: t[key] for key in ('kappa_d_c', 'kappa_d_s', 'kappa_r_c', 'kappa_r_s')}
    return d


def mu0(v, kind):
    '''mu = 0 value from the multipole means: 3/2 M0 for hE, M0 - 5/2 M2 + 27/8 M4 for EE, BB.'''
    if kind == 'hE':
        return 1.5*v[..., 0, :]
    return v[..., 0, :] - 2.5*v[..., 1, :] + 27.0/8.0*v[..., 2, :]


def window(k):
    '''<W^2> of the 1024 pipeline mesh at each stitched k (fold 1 below 1.2, fold 4 above).'''
    cell = np.where(k < HF.FOLDS[1][1], HF.BOX/(HF.FOLDS[0][0]*1024), HF.BOX/(HF.FOLDS[1][0]*1024))
    return np.array([HF.ngp_w2(np.array([kk]), cc)[0] for kk, cc in zip(k, cell)])


def emu_beta(z, lgm, k):
    '''Dark Emulator beta_NL at the bin masses lgm (held outside its range), on k.'''
    t = np.load(os.path.join(KERN, 'beta_emu.npz'))
    s = int(np.argmin(np.abs(t['zs'] - z)))
    lgE, bt = t['lgM_E'][s], t['beta'][s]
    x = np.clip(lgm, lgE[0], lgE[-1])
    jj = np.clip(np.searchsorted(lgE, x) - 1, 0, len(lgE) - 2)
    w = (x - lgE[jj])/(lgE[jj + 1] - lgE[jj])
    W = np.zeros((len(lgm), len(lgE)))
    W[np.arange(len(lgm)), jj] = 1 - w
    W[np.arange(len(lgm)), jj + 1] += w
    bm = np.einsum('ap,bq,pqk->abk', W, W, bt)
    out = np.zeros(bm.shape[:2] + (len(k),))
    for a in range(bm.shape[0]):
        for b in range(bm.shape[1]):
            out[a, b] = np.interp(np.log(k), np.log(t['k']), bm[a, b])
    out[..., k < 0.01] = 0.0
    return out


def regularize_1h(k, v, kind):
    '''The 1-halo term below K1H from its physical limit (see K1H).'''
    v = v.copy()
    band = (k >= K1H) & (k <= 2*K1H)
    low = k < K1H
    if kind == 'hE':
        amp = np.mean(v[band]/k[band]**2)
        v[low] = amp*k[low]**2
    else:
        v[low] = np.mean(v[band])
    return v


def white_below(k, v, k0):
    '''v below k0 replaced by its mean over k0 to 2 k0 (the white k -> 0 limit of a stochastic term).'''
    v = v.copy()
    v[k < k0] = np.mean(v[(k >= k0) & (k <= 2*k0)])
    return v


def build(d, plin):
    '''Model spectra (mu = 0, raw shape units, 1024-mesh windowed) per gravity layer.'''
    k = d['k']
    nb = d['nb']
    W2 = window(k)
    low = (k >= KLOW[0]) & (k <= KLOW[1])
    wl = d['nmodes'][low]
    # large-scale halo and shape biases (monopole normalisation)
    hh0 = np.array([d['Phh'][i, i, 0] for i in range(nb)])
    b = np.sqrt(np.array([np.sum(wl*hh0[i][low]/(W2*plin)[low])/wl.sum() for i in range(nb)]))
    b = np.nan_to_num(b)               # an empty bin (the top bin at z = 4) has no halo bias
    he0 = d['PhE'][:, :, 0]
    num = np.array([np.sum(d['n_h'][:, None]*np.array([np.sum(wl*he0[i, j][low])/wl.sum() for i in range(nb)])[:, None])
                    for j in range(nb)])
    den = np.sum(d['n_h']*b)*np.sum(wl*(W2*plin)[low])/wl.sum()
    bK = num/den
    a = d['n_h'][:, None]*(d['Nc'][:, None] + d['Ns'][:, None]*d['us_k'])/d['n_g']
    # large-scale alignment kernels: the measured cross of galaxy density with the
    # shapes of centrals (satellites) in host bin j, over the same cross predicted
    # for kappa = 1 from the halo shapes of bin j, averaged over KLS
    hEm = mu0(d['PhE'], 'hE')
    ls = (k >= KLS[0]) & (k <= KLS[1])
    wls = d['nmodes'][ls]
    kap = {'obj': (np.nan_to_num(d['kappa_c']), np.nan_to_num(d['kappa_s']))}
    kc, ks = np.zeros(nb), np.zeros(nb)
    for j in range(nb):
        unit = np.einsum('ik,ik->k', a, hEm[:, j])*d['n_h'][j]/d['n_g']
        for arr, G, w in ((kc, d['Gc'], d['Nc'][j]), (ks, d['Gt'], d['Ns'][j]*d['us_k'][j])):
            den = np.sum(wls*(unit*w)[ls])
            arr[j] = np.sum(wls*mu0(G[j], 'hE')[ls])/den if abs(den) > 0 else 0.0
        if d['Nc'][j] < NC_MIN:
            kc[j] = 0.0
    kap['ls'] = (kc, ks)
    cs = {kk: d['n_h'][:, None]*(d['Nc'][:, None]*v[0][:, None] + d['Ns'][:, None]*v[1][:, None]*d['us_k'])/d['n_g']
          for kk, v in kap.items()}
    # effective bin masses: the HOD-weighted mean log M is not stored; use the bin centres
    edges = d['lgm_edges'].copy()
    edges[0] = max(edges[0], 11.3)
    edges[-1] = min(edges[-1], 14.5)
    lgm = 0.5*(edges[:-1] + edges[1:])
    beta = emu_beta(float(d['z']), lgm, k)
    # deterministic halo-shape field: the part of each bin's halo shapes that is
    # correlated with the halo density (all halos above 10^11.3, weights n_i),
    # P_{E_i E_j}^det = P_{H E_i} P_{H E_j} / P_HH; its B mode vanishes
    w = d['n_h']/d['n_h'].sum()
    PHE = np.einsum('i,ijk->jk', w, hEm)
    PHH = np.einsum('i,j,ijk->k', w, w, d['Phh'][:, :, 0])
    EEdet = PHE[:, None, :]*PHE[None, :, :]/PHH
    out = {}
    for layer, kname in [(L, K) for K in KAPPAS for L in LAYERS]:
        c = cs[kname]
        if layer == 'tab':
            hE = mu0(d['PhE'], 'hE')
            EE = mu0(d['PEE'], 'EE')
            BB = mu0(d['PBB'], 'EE')
        elif layer == 'det':
            hE = mu0(d['PhE'], 'hE')
            EE = EEdet
            BB = 0.0*EE
        else:
            bb = beta if layer == 'emu' else 0.0
            hE = 1.5*W2*plin*b[:, None, None]*bK[None, :, None]*(1 + bb)
            EE = 2.25*W2*plin*bK[:, None, None]*bK[None, :, None]*(1 + bb)
            BB = 0.0*EE
        two = {'hE': np.einsum('ik,jk,ijk->k', a, c, hE),
               'EE': np.einsum('ik,jk,ijk->k', c, c, EE),
               'BB': np.einsum('ik,jk,ijk->k', c, c, BB)}
        one = {p: regularize_1h(k, mu0(d['G1h'][p], 'hE' if p == 'hE' else 'EE'), p) for p in ('hE', 'EE', 'BB')}
        out[f'{layer}-{kname}'] = {'2h': two, '1h': one, 'tot': {p: two[p] + one[p] for p in two}}
    meas = {p: mu0(d['G'][p], 'hE' if p.startswith('hE') else 'EE') for p in ('hE', 'EE', 'BB', 'hEc', 'hEs')}
    aux = dict(b=b, bK=bK, a=a, c=cs, W2=W2, lgm=lgm, kappa_ls_c=kc, kappa_ls_s=ks)
    if 'W' in d:
        wout, waux = build_weighted(d, k, ls, wls)
        out.update(wout)
        aux.update(waux)
    return out, meas, aux


def build_weighted(d, k, ls, wls):
    '''The final model on the mean-HOD-weighted halo fields (hk_weighted.py).

    Galaxy density model G (or Ga, actual occupations); per bin j the shape
    fields Ec<j>, Es<j> per unit kernel. Large-scale kernels
        kappa_c,j = <Gc_j> / <P_{G,Ec_j}>,  kappa_s,j = <Gt_j> / <P_{G,Es_j}>  over KLS,
    2-halo GI  = sum_j kappa_c,j P_{G,Ec_j} + kappa_s,j P_{G,Es_j},
    2-halo II  = (sum_j kappa_c,j P_{H,Ec_j} + kappa_s,j P_{H,Es_j})^2 / P_HH  (deterministic),
    2-halo BB  = 0; 1-halo terms as the binned model.'''
    W = d['W']
    nb = d['nb']
    out, aux = {}, {}
    one = {p: regularize_1h(k, mu0(d['G1h'][p], 'hE' if p == 'hE' else 'EE'), p) for p in ('hE', 'EE', 'BB')}
    zero = np.zeros((3, len(k)))
    for ref, name in (('G', 'w'), ('Ga', 'wa')):
        kc, ks = np.zeros(nb), np.zeros(nb)
        gi, hi = np.zeros(len(k)), np.zeros(len(k))
        for j in range(nb):
            for arr, tag, G in ((kc, 'Ec', d['Gc']), (ks, 'Es', d['Gt'])):
                x = W.get(f'{ref}|d|{tag}{j}|E', zero)
                den = np.sum(wls*mu0(x, 'hE')[ls])
                arr[j] = np.sum(wls*mu0(G[j], 'hE')[ls])/den if abs(den) > 0 else 0.0
            nobj = d['n_h'][j]*HF.BOX**3*np.array([d['Nc'][j], d['Ns'][j]])
            if d['Nc'][j] < NC_MIN or nobj[0] < N_MIN:
                kc[j] = 0.0
            if nobj[1] < N_MIN:
                ks[j] = 0.0
            for kap, tag in ((kc[j], 'Ec'), (ks[j], 'Es')):
                gi += kap*mu0(W.get(f'{ref}|d|{tag}{j}|E', zero), 'hE')
                hi += kap*mu0(W.get(f'H|d|{tag}{j}|E', zero), 'hE')
        ee = hi**2/W['H|d|H|d'][0]
        two = {'hE': gi, 'EE': ee, 'BB': 0.0*ee}
        out[f'{name}-ls'] = {'2h': two, '1h': one, 'tot': {p: two[p] + one[p] for p in two}}
        aux[f'kappa_{name}_c'] = kc
        aux[f'kappa_{name}_s'] = ks
        if name == 'w' and 'St' in d:
            # deterministic + residual (hk_stoch.py): the residual halo-halo shape
            # correlation inherited with the object-level kernels kappa_r
            St = d['St']
            assert np.allclose(d['St_kappa']['kappa_d_c'], kc) and np.allclose(d['St_kappa']['kappa_d_s'], ks), \
                'stoch_<sn>.npz was measured with other kappa_d: rerun hk_stoch.py'
            PHH = St['H|d|H|d'][0]
            Td, Tr = mu0(St['H|d|Ed|E'], 'hE'), mu0(St['H|d|Er|E'], 'hE')
            res = white_below(k, mu0(St['Er|E|Er|E'], 'EE') - Tr**2/PHH, K_RES)
            two = {'hE': gi, 'EE': Td**2/PHH + res, 'BB': white_below(k, mu0(St['Er|B|Er|B'], 'EE'), K_RES)}
            out['ws-ls'] = {'2h': two, '1h': one, 'tot': {p: two[p] + one[p] for p in two}}
            # the same with the large-scale kernels on the residual (galaxy = kappa_d x halo shape)
            full = mu0(St['Ed|E|Ed|E'], 'EE')
            two = {'hE': gi, 'EE': full, 'BB': mu0(St['Ed|B|Ed|B'], 'EE')}
            out['wf-ls'] = {'2h': two, '1h': one, 'tot': {p: two[p] + one[p] for p in two}}
            aux['det_check'] = Td**2/PHH/ee
            aux['kappa_r_c'] = d['St_kappa']['kappa_r_c']
            aux['kappa_r_s'] = d['St_kappa']['kappa_r_s']
    aux['bias_ratio'] = {r: W[f'{r}|d|{r}|d'][0]/d['G']['hh'][0] for r in ('G', 'Ga')}
    return out, aux


def to_fit_units(v, kind, erms, W2):
    '''Shape legs to shear (x 1/(2R) each, R = 1 - e_rms^2) and the window of P_EE, P_BB.'''
    s1 = 1.0/(2.0*(1.0 - erms**2))
    return v*s1 if kind == 'hE' else v*s1**2/W2


# k_max grid of the goodness-of-fit scans (the fits.npz k values, as in the
# halo_ia_models note) and the p-value of a failed fit
KGRID = (0.0899, 0.1204, 0.1615, 0.2167, 0.2907, 0.3900, 0.5229, 0.7013, 0.9406, 1.2618, 1.6919, 2.2686, 3.0424)
P_FAIL = 0.05
# kernel gradient: c_j(k) -> c_j(k) (1 + alpha k^2), alpha in [h^-1 Mpc]^2
ALPHA_RANGE = (-0.5, 3.0)
ZERO_PARAM = ('lin-ls', 'emu-ls', 'tab-ls', 'det-ls', 'tab-obj', 'w-ls', 'wa-ls', 'ws-ls', 'wf-ls')
GRAD = ('det-ls', 'tab-ls', 'w-ls', 'wa-ls', 'ws-ls')


# saturating kernel gradient (variant <name>-sat): the galaxy kernel grows above its
# large-scale value inside the Lagrangian patch of its host halo and saturates,
#     kappa(k) = kappa_ls [1 + A x(k)],   x = k^2 R_L^2 / (1 + k^2 R_L^2),
# with R_L the galaxy-weighted mean Lagrangian radius of the hosts,
# R_L = (3 M200m / (4 pi rho_m))^(1/3); A (dimensionless) is the only fitted number
SAT = ('w-ls', 'ws-ls')
A_RANGE = (-0.5, 3.0)
RHO_M = 2.775e11*HIA.OMM          # comoving mean matter density [h^2 Msun / Mpc^3]


def lagrangian_radius(sn):
    '''<R_L> over the sample galaxies of their host halos [Mpc/h], cached in kernels/.'''
    fn = os.path.join(KERN, 'lagrangian_radius.json')
    cache = json.load(open(fn)) if os.path.isfile(fn) else {}
    if str(sn) not in cache:
        import hk_measure as HM
        g, h = HM.load(sn)
        cache[str(sn)] = float(np.mean((3*h['M200m'][g['host']]/(4*np.pi*RHO_M))**(1.0/3.0)))
        json.dump(cache, open(fn, 'w'), indent=1)
    return cache[str(sn)]


def cut_rule(ok, kgrid):
    '''Last k_max before two consecutive failures (as v4/simfit.cut_rule).'''
    for j in range(len(kgrid) - 1):
        if not ok[j] and not ok[j + 1]:
            return float(kgrid[j - 1]) if j > 0 else float('nan')
    return float(kgrid[-1] if ok[-1] else kgrid[-2])


def predictor(parts, idx, erms, W2, k, x=None):
    '''pred(alpha) in fit units on the fits k grid from the 2-halo and 1-halo parts;
    the kernels scale as 1 + alpha x(k), x = k^2 by default.'''
    two = {p: to_fit_units(parts['2h'][p][idx], p, erms, W2[idx]) for p in ('hE', 'EE', 'BB')}
    one = {p: to_fit_units(parts['1h'][p][idx], p, erms, W2[idx]) for p in ('hE', 'EE', 'BB')}
    x = k**2 if x is None else x

    def pred(alpha=0.0):
        g = 1.0 + alpha*x
        return np.array([two['hE']*g + one['hE'], two['EE']*g**2 + one['EE'], two['BB']*g**2 + one['BB']])
    return pred


def main():
    from scipy.optimize import minimize_scalar
    fits = np.load(FITS)
    zs_fit = fits['zs']
    files = sorted(f for f in os.listdir(KERN) if f.startswith('kernels_'))
    # optional snapshot subset: a test run, written to hk_model_test.json
    only = [int(a) for a in sys.argv[1:]]
    if only:
        files = [f for f in files if int(f.split('_')[1].split('.')[0]) in only]
    res = {}
    for fn in files:
        sn = int(fn.split('_')[1].split('.')[0])
        d = load(sn)
        z = float(d['z'])
        lin = HIA.Linear([z])
        kh, pk = lin.P(z)
        plin = np.exp(np.interp(np.log(d['k']), np.log(kh), np.log(pk)))
        mod, meas, aux = build(d, plin)
        erms = F.SNAPSHOTS[sn][1]
        s = int(np.argmin(np.abs(zs_fit - z)))
        kf = fits['k'][s]
        b = np.digitize(kf, HF.KEDGES) - 1
        kb = np.digitize(d['k'], HF.KEDGES) - 1
        idx = np.array([np.flatnonzero(kb == bb)[0] if np.any(kb == bb) else -1 for bb in b])
        okk = idx >= 0
        y, err = fits['y'][s], fits['err'][s]
        # estimator offset: the same galaxies measured here minus fits.npz (512-mesh
        # aliasing near k = 1, see validate_estimator.py); added to every prediction,
        # so that each chi2 is the model against the identical estimator
        rep = np.array([to_fit_units(meas[p][idx], p, erms, aux['W2'][idx]) for p in ('hE', 'EE', 'BB')])
        off = np.where(okk, y - rep, 0.0)
        line = {'z': z, 'sn': sn, 'b': aux['b'].tolist(), 'bK': aux['bK'].tolist(),
                'kappa_c': d['kappa_c'].tolist(), 'kappa_s': d['kappa_s'].tolist(),
                'Nc': d['Nc'].tolist(), 'Ns': d['Ns'].tolist(), 'n_h': d['n_h'].tolist(),
                'n_g': float(d['n_g']), 'lgm_edges': d['lgm_edges'].tolist(),
                'kappa_ls_c': aux['kappa_ls_c'].tolist(), 'kappa_ls_s': aux['kappa_ls_s'].tolist(),
                'k': kf.tolist(), 'okk': okk.tolist(), 'offset': off.tolist()}
        print(f'z {z:.2f} (sn {sn}): b {np.round(aux["b"], 2)}, b_K {np.round(aux["bK"], 4)}', flush=True)
        print(f'   kappa_c obj {np.round(d["kappa_c"], 3)}  ls {np.round(aux["kappa_ls_c"], 3)}; '
              f'kappa_s obj {np.round(d["kappa_s"], 3)}  ls {np.round(aux["kappa_ls_s"], 3)}', flush=True)

        def chi2_at(pred, kmax):
            m = okk & (kf >= KMIN) & (kf <= kmax*1.001)
            r = ((y - pred - off)/err)[:, m]
            return float(np.sum(r**2)), int(r.size)
        if 'det_check' in aux:
            dc = aux['det_check'][(d['k'] > 0.03) & (d['k'] < 1.3)]
            print(f'   ws-ls: det II from stoch file / w-ls det II: {dc.min():.4f} to {dc.max():.4f}', flush=True)
        for key in ('kappa_w_c', 'kappa_w_s', 'kappa_wa_c', 'kappa_wa_s', 'kappa_r_c', 'kappa_r_s'):
            if key in aux:
                line[key] = aux[key].tolist()
        if 'bias_ratio' in aux:
            line['bias_ratio'] = {r: v.tolist() for r, v in aux['bias_ratio'].items()}
            line['kk'] = d['k'].tolist()
            print(f'   kappa_c w {np.round(aux["kappa_w_c"], 3)}; kappa_s w {np.round(aux["kappa_w_s"], 3)}', flush=True)
        for name in ZERO_PARAM:
            if name not in mod:
                continue
            P = predictor(mod[name], idx, erms, aux['W2'], kf)
            pred = P(0.0)
            chis = [chi2_at(pred, km) for km in KGRID]
            pv = [float(chi2_dist.sf(c, n)) for c, n in chis]
            line[name] = {'pred': (pred + off).tolist(), 'chi2': [c for c, _ in chis], 'nu': [n for _, n in chis],
                          'p': pv, 'cut': cut_rule(np.array(pv) >= P_FAIL, KGRID)}
            print(f'   {name:8s} chi2/nu: ' + ' '.join(f'{c/n:7.2f}' for c, n in chis)
                  + f'  cut {line[name]["cut"]}', flush=True)
        RL = lagrangian_radius(sn)
        line['R_L'] = RL
        for name, key, xk, rng in [(n, n + '-grad', None, ALPHA_RANGE) for n in GRAD] + \
                [(n, n + '-sat', (kf*RL)**2/(1 + (kf*RL)**2), A_RANGE) for n in SAT]:
            if name not in mod:
                continue
            P = predictor(mod[name], idx, erms, aux['W2'], kf, xk)
            al, sig, chis, preds = [], [], [], []
            for km in KGRID:
                r = minimize_scalar(lambda a: chi2_at(P(a), km)[0], bounds=rng, method='bounded',
                                    options=dict(xatol=1e-4))
                c, n = chi2_at(P(r.x), km)
                # 1 sigma of alpha from the chi2 curvature (Delta chi2 = 1)
                h = 0.02
                curv = (chi2_at(P(r.x + h), km)[0] + chi2_at(P(r.x - h), km)[0] - 2*c)/h**2
                sig.append(float(np.sqrt(2.0/curv)) if curv > 0 else float('nan'))
                al.append(float(r.x))
                chis.append((c, n - 1))
                preds.append((P(r.x) + off).tolist())
            pv = [float(chi2_dist.sf(c, max(n, 1))) for c, n in chis]
            line[key] = {'alpha': al, 'alpha_err': sig, 'pred_kmax': preds, 'chi2': [c for c, _ in chis], 'nu': [n for _, n in chis],
                         'p': pv, 'cut': cut_rule(np.array(pv) >= P_FAIL, KGRID)}
            print(f'   {key:12s} chi2/nu: ' + ' '.join(f'{c/n:7.2f}' for c, n in chis)
                  + f'  cut {line[key]["cut"]}; alpha ' + ' '.join(f'{a:5.2f}' for a in al), flush=True)
        # 2-halo and 1-halo parts of the final model (alpha = 0), fit units
        for name in ('det-ls', 'w-ls', 'ws-ls'):
            if name in mod:
                line[name]['parts'] = {
                    p: {'2h': to_fit_units(mod[name]['2h'][p][idx], p, erms, aux['W2'][idx]).tolist(),
                        '1h': to_fit_units(mod[name]['1h'][p][idx], p, erms, aux['W2'][idx]).tolist()}
                    for p in ('hE', 'EE', 'BB')}
        m = okk & (kf >= KMIN) & (kf <= 1.26)
        line['repro_maxdev_sigma'] = float(np.max(np.abs(off/err)[:, m]))
        res[str(sn)] = line
        print(f'   catalogue reproduction of fits.npz: max |dev|/sigma {line["repro_maxdev_sigma"]:.2f}', flush=True)
    os.makedirs(os.path.join(HERE, 'model'), exist_ok=True)
    with open(os.path.join(HERE, 'model', 'hk_model_test.json' if only else 'hk_model.json'), 'w') as fh:
        json.dump(res, fh)


if __name__ == '__main__':
    main()
