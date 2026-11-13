#!/usr/bin/env python3
"""Fit the TATT intrinsic-alignment model to FLAMINGO IA power spectra.

Fits, per snapshot, the cosmolike TATT amplitudes (a1, a2, b_TA) and a
linear galaxy bias b1 to the measured galaxy-galaxy (P_hh), galaxy-shape
(P_hE), and shape-shape (P_EE, P_BB) spectra, then fits the cosmolike
redshift-evolution form to the per-snapshot results:

    a1(z) = A1 ((1+z)/(1+z0))^eta1      roman_A1_1, roman_A1_2
    a2(z) = A2 ((1+z)/(1+z0))^eta2      roman_A2_1, roman_A2_2
    b_TA  = const                       roman_BTA_1
    1 + z0 = 1.62 (cosmolike nuisance.oneplusz0_ia)

Conventions are those of cosmolike (IA.c, cosmo2D.c TATT cores), so the
output can be used directly as a cosmolike fiducial:

    C1 = a1 Omega_m c1rhocrit / D(z),   C2 = a2 Omega_m c1rhocrit / D(z)^2
    c1rhocrit = 0.01389,  D(z=0) = 1
    P_dE = -[C1 P_NL + C1 bTA (dE1 + dE2) - 5 C2 (mixA + mixB)]
    P_EE = C1^2 P_NL + C1^2 [bTA^2 ta + 2 bTA (dE1 + dE2)]
           - 10 C1 C2 (mixA + mixB) - 10 C1 bTA C2 mixEE + 25 C2^2 tt
    P_BB = C1^2 bTA^2 taBB - 10 C1 bTA C2 mixBB + 25 C2^2 ttBB
    P_hE = b P_dE,   P_hh = b1^2 P_NL
    P_EE, P_BB data also carry a free white shape-noise residual Ns
with the FAST-PT one-loop kernels from P_lin(z=0) times D(z)^4. The
physical P_dE sign follows cosmolike: C1 > 0 for a1 > 0 and P_dE < 0.

Data conventions are those of sim_IA.c (the injection code), applied to
the raw Pds files:
    shape legs: distortion -> shear, x 1/(2R) per leg, R = 1 - e_rms^2
    P_hE: mu = 0 from the monopole, x 3/2 (linear-alignment factor)
    P_EE, P_BB: mu = 0 from multipoles, M0 - (5/2) M2 + (27/8) M4,
                divided by the angle-averaged NGP window <W^2>(k)
    P_hh: monopole minus the Poisson shot noise 1/nbar
Galaxy bias (--bias):
    scale  b(k) = sqrt(P_hh/P_mm) from the data (deterministic bias, as the
           FLAMINGO pipeline used to build P_dE = P_hE/b(k) for sim_IA.c), so
           the fit targets the same P_dE that Part 2 injects. Default.
    const  one linear b1, fitted jointly with P_hh = b1^2 P_NL; this is the
           form cosmolike's gs TATT core uses. Breaks down at high z
           (b > 2) where the bias is scale dependent.

Validated against the 2026-10-25/26/27 FLAMINGO reports: the NLA limit
reproduces their a_NLA within ~2% (they use FLAMINGO's own P_mm; this
script uses CAMB halofit unless --pmm-dir is given).

The evolution fit defaults to 0.5 <= z <= 1.5, where the FLAMINGO sample's
mass edges are fixed (log M* 10.0-11.07/11.10): at z <= 0.4 the upper edge
moves (a1 jumps between z = 0.1 and 0.2) and at z >= 2 it rises again, and
including either range flips the poorly constrained (a2, bTA) pair to a
different branch with a similar chi2.

Usage:
    python fit_tatt_flamingo.py --ps-dir PS --out fit_out --plots
    python fit_tatt_flamingo.py --ps-dir PS --pmm-dir Pmm --bias scale
    python fit_tatt_flamingo.py --ps-dir PS --zmin-global 0 --zmax-global 3
    python fit_tatt_flamingo.py --ps-dir PS --snapshots 58 48 --plots
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# FLAMINGO fiducial cosmology (Schaye et al. 2023, "D3A": DES Y3 3x2pt + all
# external). Check against the simulation's parameter file before relying
# on per-percent amplitudes.
COSMO = dict(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, sigma8=0.807, mnu=0.06)

BOX_HMPC = 681.0       # L1_m8: 1 Gpc = 681 Mpc/h
NGRID = 1024           # mesh of the Pds run
C1RHOCRIT = 0.01389    # cosmolike nuisance.c1rhocrit_ia
C1RHOCRIT_REPORT = 0.0134  # value used for a_NLA in the FLAMINGO reports
ONEPLUSZ0 = 1.62       # cosmolike nuisance.oneplusz0_ia

# Snapshot table from the 2026-10-27 report:
# sn: (z, e_rms, N_galaxies, bias_report, aNLA_report [c1rhocrit = 0.0134])
SNAPSHOTS = {
    78: (0.00, 0.250, 2031304, 1.16, 1.86),
    76: (0.10, 0.254, 3686178, 1.20, 2.17),
    74: (0.20, 0.262, 5532432, 1.24, 2.88),
    72: (0.30, 0.268, 5771587, 1.27, 3.17),
    70: (0.40, 0.274, 5854842, 1.31, 3.35),
    68: (0.50, 0.280, 5933593, 1.35, 3.50),
    66: (0.60, 0.285, 5934007, 1.39, 3.56),
    63: (0.75, 0.293, 5883782, 1.46, 3.78),
    58: (1.00, 0.305, 5657613, 1.60, 4.14),
    48: (1.50, 0.323, 4808380, 1.94, 4.94),
    38: (2.00, 0.335, 3661123, 2.36, 6.04),
    18: (3.00, 0.358, 1613396, 3.56, 8.32),
    14: (4.00, 0.371, 533335, 5.26, 10.02),
}

PARAM_NAMES = ["a1", "a2", "bTA", "b1", "Ns"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ngp_w2_avg(k, D):
    """Angle-averaged NGP window <W^2>(k), mesh spacing D (as sim_IA.c)."""
    nmu = nphi = 48
    mu = (np.arange(nmu) + 0.5)/nmu
    ph = (np.arange(nphi) + 0.5)/nphi*0.5*np.pi
    MU, PH = np.meshgrid(mu, ph, indexing="ij")
    st = np.sqrt(1.0 - MU**2)
    out = np.empty_like(np.atleast_1d(k), dtype=float)
    for i, kk in enumerate(np.atleast_1d(k)):
        x = 0.5*kk*D*np.array([st*np.cos(PH), st*np.sin(PH), MU])
        w = np.prod(np.sinc(x/np.pi)**2, axis=0)
        out[i] = w.mean()
    return out


def _load(ps_dir, tag, sn, nfold):
    for name in (f"{tag}_sn{sn}_nfold{nfold}.dat",
                 f"{tag}_sn{sn:03d}_nfold{nfold}.dat"):
        path = os.path.join(ps_dir, name)
        if os.path.exists(path):
            d = np.loadtxt(path)
            return d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    raise FileNotFoundError(f"{tag} sn{sn} nfold{nfold} not in {ps_dir}")


def load_snapshot(ps_dir, sn, nfold=1, shot_noise=True):
    """Converted spectra of one snapshot and fold (h/Mpc, (Mpc/h)^3)."""
    z, erms, ngal = SNAPSHOTS[sn][:3]
    R = 1.0 - erms**2
    s1 = 1.0/(2.0*R)                 # one shape leg
    out = {"z": z, "sn": sn}

    k, P, sig, nmod = _load(ps_dir, "Phh", sn, nfold)
    nbar = ngal/BOX_HMPC**3
    out["k"] = k
    out["nmodes"] = nmod
    out["hh"] = (P - (1.0/nbar if shot_noise else 0.0), sig)

    _, P, sig, _ = _load(ps_dir, "PhE0", sn, nfold)
    out["hE"] = (P*s1*1.5, sig*s1*1.5)

    W2 = ngp_w2_avg(k, BOX_HMPC/(nfold*NGRID))
    for tag in ("EE", "BB"):
        M = [_load(ps_dir, f"P{tag}{l}", sn, nfold) for l in (0, 2, 4)]
        c = (1.0, -2.5, 27.0/8.0)
        P = sum(ci*m[1] for ci, m in zip(c, M))
        sig = np.sqrt(sum((ci*m[2])**2 for ci, m in zip(c, M)))
        out[tag] = (P*s1**2/W2, sig*s1**2/W2)

    # null spectra (should vanish): diagnostics only
    for tag, leg in (("hB0", s1), ("EB0", s1**2)):
        _, P, sig, _ = _load(ps_dir, f"P{tag}", sn, nfold)
        out[tag[:2]] = (P*leg, sig*leg)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TATTModel:
    """CAMB P_lin/P_NL plus FAST-PT TATT kernels at the FLAMINGO cosmology."""

    def __init__(self, cosmo=COSMO, pmm_dir=None, halofit="mead2020"):
        import camb
        from fastpt import FASTPT
        self.cosmo = cosmo
        h, Om, Ob, mnu = cosmo["h"], cosmo["Om"], cosmo["Ob"], cosmo["mnu"]
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100*h, ombh2=Ob*h*h,
                           omch2=(Om - Ob)*h*h - mnu/93.14,
                           mnu=mnu, num_massive_neutrinos=1)
        pars.InitPower.set_params(ns=cosmo["ns"], As=2.1e-9)
        zgrid = np.linspace(0.0, 5.0, 51)
        pars.set_matter_power(redshifts=list(zgrid[::-1]), kmax=60.0)
        res = camb.get_results(pars)
        As = 2.1e-9*(cosmo["sigma8"]/res.get_sigma8_0())**2
        pars.InitPower.set_params(ns=cosmo["ns"], As=As)
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params(halofit_version=halofit)
        res = camb.get_results(pars)
        self.Plin = res.get_matter_power_interpolator(
            nonlinear=False, hubble_units=True, k_hunit=True)
        self.Pnl_camb = res.get_matter_power_interpolator(
            nonlinear=True, hubble_units=True, k_hunit=True)
        p0 = self.Plin.P(0.0, 1e-3)
        self._D = lambda z: np.sqrt(self.Plin.P(z, 1e-3)/p0)
        self.pmm_dir = pmm_dir

        # FAST-PT kernels from P_lin(z=0)
        kf = np.logspace(-4, np.log10(30.0), 1200)
        fpt = FASTPT(kf, to_do=["IA"], low_extrap=-5, high_extrap=3,
                     n_pad=int(0.5*len(kf)))
        P0 = self.Plin.P(0.0, kf)
        tt, ttB = fpt.IA_tt(P0, C_window=0.75)
        dE1, dE2, ta, taBB = fpt.IA_ta(P0, C_window=0.75)
        mA, mB, mEE, mBB = fpt.IA_mix(P0, C_window=0.75)
        lk = np.log(kf)
        self._K = {name: interp1d(lk, arr, kind="cubic")
                   for name, arr in dict(tt=tt, ttB=ttB, dE=dE1 + dE2,
                                         ta=ta, taBB=taBB, mix=mA + mB,
                                         mixEE=mEE, mixBB=mBB).items()}

    def D(self, z):
        return float(self._D(z))

    def Pnl(self, z, k, sn=None):
        """Nonlinear matter power; FLAMINGO's own P_mm if --pmm-dir given."""
        if self.pmm_dir is not None and sn is not None:
            path = os.path.join(self.pmm_dir, f"Pmm_sn{sn}.dat")
            if os.path.exists(path):
                d = np.loadtxt(path)
                return np.exp(np.interp(np.log(k), np.log(d[:, 0]),
                                        np.log(np.abs(d[:, 1]))))
        return self.Pnl_camb.P(z, k)

    def kernels(self, z, k):
        g4 = self.D(z)**4
        return {n: g4*f(np.log(k)) for n, f in self._K.items()}

    def predict(self, z, k, a1, a2, bTA, b1, sn=None, K=None, PNL=None,
                Ns=0.0, bk=None):
        D = self.D(z)
        Om = self.cosmo["Om"]
        C1 = a1*Om*C1RHOCRIT/D
        C2 = a2*Om*C1RHOCRIT/D**2
        K = self.kernels(z, k) if K is None else K
        P = self.Pnl(z, k, sn) if PNL is None else PNL
        PdE = -(C1*P + C1*bTA*K["dE"] - 5.0*C2*K["mix"])
        PEE = (C1**2*P + C1**2*(bTA**2*K["ta"] + 2.0*bTA*K["dE"])
               - 10.0*C1*C2*K["mix"] - 10.0*C1*bTA*C2*K["mixEE"]
               + 25.0*C2**2*K["tt"])
        PBB = (C1**2*bTA**2*K["taBB"] - 10.0*C1*bTA*C2*K["mixBB"]
               + 25.0*C2**2*K["ttB"])
        b = b1 if bk is None else bk
        return {"hh": b1**2*P, "hE": b*PdE, "EE": PEE + Ns, "BB": PBB + Ns,
                "dE": PdE}


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def k_masks(d, z, D, args):
    """Fit ranges: IA probes up to kmax0/D(z) (capped), P_hh up to khh0/D."""
    k = d["k"]
    kmax_ia = min(args.kmax0/D, args.kmax_cap)
    kmax_hh = min(args.khh0/D, args.khh_cap)
    m_ia = (k >= args.kmin) & (k <= kmax_ia)
    m_hh = (k >= args.kmin) & (k <= kmax_hh)
    return {"hh": m_hh, "hE": m_ia, "EE": m_ia, "BB": m_ia}, kmax_ia, kmax_hh


def scale_bias(d, PNL):
    """Deterministic b(k) = sqrt(P_hh/P_mm), lightly smoothed in ln k."""
    r = np.clip(d["hh"][0], 1e-30, None)/PNL
    b = np.sqrt(r)
    bs = b.copy()
    for i in range(1, len(b) - 1):  # 3-point running mean, keeps the edges
        bs[i] = (b[i-1] + b[i] + b[i+1])/3.0
    return bs


def default_free(args, nla=False):
    f = ["a1"] if nla else ["a1", "a2", "bTA"]
    if args.bias == "const":
        f.append("b1")
    if args.shape_noise and ("EE" in args.probes or "BB" in args.probes):
        f.append("Ns")
    return tuple(f)


def fit_snapshot(model, d, args, free=None, probes=None, kband=None):
    z, sn, k = d["z"], d["sn"], d["k"]
    D = model.D(z)
    masks, kmax_ia, kmax_hh = k_masks(d, z, D, args)
    if kband is not None:  # fixed k band for all probes (validation fits)
        m = (k >= kband[0]) & (k <= kband[1])
        masks = {p: m for p in masks}
        kmax_ia = kband[1]
    probes = list(args.probes if probes is None else probes)
    free = default_free(args) if free is None else tuple(free)
    K = model.kernels(z, k)
    PNL = model.Pnl(z, k, sn)
    bk = scale_bias(d, PNL) if args.bias == "scale" else None
    if bk is not None and "hh" in probes:
        probes.remove("hh")  # P_hh defines b(k); nothing left to fit

    # fixed per-probe error: scaled mode scatter + fractional floor
    err = {}
    for p in probes:
        P, s = d[p]
        err[p] = np.sqrt((args.err_scale*s)**2 + (args.sys_frac*np.abs(P))**2)

    b_rep, a_rep = SNAPSHOTS[sn][3], SNAPSHOTS[sn][4]
    base = dict(a1=a_rep*C1RHOCRIT_REPORT/C1RHOCRIT, a2=0.0, bTA=0.0,
                b1=b_rep, Ns=0.0)
    idx = [PARAM_NAMES.index(n) for n in free]

    def full(x):
        th = dict(base)
        for n, v in zip(free, x):
            th[n] = v
        return th

    def resid(x):
        th = full(x)
        m = model.predict(z, k, th["a1"], th["a2"], th["bTA"], th["b1"],
                          K=K, PNL=PNL, Ns=th["Ns"], bk=bk)
        return np.concatenate([((d[p][0] - m[p])/err[p])[masks[p]]
                               for p in probes])

    lo = np.array([-30.0, -100.0, -20.0, 0.3, -50.0])[idx]
    hi = np.array([60.0, 100.0, 20.0, 15.0, 50.0])[idx]
    best = None
    starts = [[]]
    for n in free:  # small grid of starts in the quadratic parameters
        vals = {"a2": [-2.0, 0.0, 2.0], "bTA": [-1.0, 0.0, 1.0]}.get(n, [base[n]])
        starts = [s + [v] for s in starts for v in vals]
    for x0 in starts:
        r = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6),
                          bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(free), len(free)), np.nan)
    r = resid(best.x)
    chi2 = float(r @ r)
    ndata = len(r)
    if args.rescale_errors and ndata > len(free):
        cov = cov*chi2/(ndata - len(free))
    th = full(best.x)
    m = model.predict(z, k, th["a1"], th["a2"], th["bTA"], th["b1"],
                      K=K, PNL=PNL, Ns=th["Ns"], bk=bk)
    if bk is not None:
        m["hh"] = bk**2*PNL
    chi2_p = {p: float(np.sum((((d[p][0] - m[p])/err[p])[masks[p]])**2))
              for p in probes}
    n_p = {p: int(masks[p].sum()) for p in probes}
    errs = {n: float(np.sqrt(cov[i, i])) for i, n in enumerate(free)}
    return dict(sn=sn, z=z, D=D, theta=th, err=errs, cov=cov.tolist(),
                free=list(free), chi2=chi2, ndata=ndata, chi2_probe=chi2_p,
                n_probe=n_p, kmax_ia=kmax_ia, kmax_hh=kmax_hh, model=m,
                masks=masks, errors=err)


def null_test(d, args, kmax):
    """chi2 of the hB and EB spectra against zero (should be ~ N)."""
    m = (d["k"] >= args.kmin) & (d["k"] <= kmax)
    out = {}
    for p in ("hB", "EB"):
        P, s = d[p]
        out[p] = (float(np.sum((P[m]/(args.err_scale*s[m]))**2)), int(m.sum()))
    return out


def fit_redshift_evolution(model, data, results, args):
    """Joint fit of the cosmolike evolution form to all spectra at once.

    Parameters: A1, eta1, A2, eta2, bTA (global) plus one shape-noise
    residual Ns per snapshot (and one b1 per snapshot with --bias const).
    Fitting the spectra directly, rather than the per-snapshot (a1, a2, bTA),
    avoids the near-singular a2-bTA covariances of the short low-z k ranges.
    """
    rows = [(d, r) for d, r in zip(data, results)
            if args.zmin_global - 1e-9 <= d["z"] <= args.zmax_global + 1e-9]
    if len(rows) < 3:
        return None
    nz = len(rows)
    use_ns = args.shape_noise and ("EE" in args.probes or "BB" in args.probes)
    use_b1 = args.bias == "const"
    prep = []
    for d, r in rows:
        z, sn, k = d["z"], d["sn"], d["k"]
        masks, _, _ = k_masks(d, z, model.D(z), args)
        K = model.kernels(z, k)
        PNL = model.Pnl(z, k, sn)
        bk = scale_bias(d, PNL) if args.bias == "scale" else None
        probes = [p for p in args.probes if not (bk is not None and p == "hh")]
        err = {p: np.sqrt((args.err_scale*d[p][1])**2
                          + (args.sys_frac*np.abs(d[p][0]))**2) for p in probes}
        prep.append((d, r, masks, K, PNL, bk, probes, err))

    def unpack(x):
        g = x[:5]
        i = 5
        ns = x[i:i + nz] if use_ns else np.zeros(nz)
        i += nz if use_ns else 0
        b1 = x[i:i + nz] if use_b1 else np.ones(nz)
        return g, ns, b1

    def resid(x):
        g, ns, b1 = unpack(x)
        out = []
        for j, (d, r, masks, K, PNL, bk, probes, err) in enumerate(prep):
            xx = (1.0 + d["z"])/ONEPLUSZ0
            m = model.predict(d["z"], d["k"], g[0]*xx**g[1], g[2]*xx**g[3],
                              g[4], b1[j], K=K, PNL=PNL, Ns=ns[j], bk=bk)
            out += [((d[p][0] - m[p])/err[p])[masks[p]] for p in probes]
        return np.concatenate(out)

    # start from the best-measured snapshots (median of the per-z fits)
    a1s = np.array([r["theta"]["a1"] for _, r in rows])
    zs = np.array([d["z"] for d, _ in rows])
    eta0 = np.polyfit(np.log((1 + zs)/ONEPLUSZ0), np.log(np.abs(a1s)), 1)[0]
    x0 = [np.median(a1s), eta0, np.median([r["theta"]["a2"] for _, r in rows]),
          0.0, np.median([r["theta"]["bTA"] for _, r in rows])]
    x0 += [r["theta"]["Ns"] for _, r in rows] if use_ns else []
    x0 += [r["theta"]["b1"] for _, r in rows] if use_b1 else []
    best = None
    for a2s in (-2.0, -1.0, 0.0, 1.0):
        for btas in (-0.5, 0.0, 0.5):
            xs = np.array(x0, dtype=float)
            xs[2], xs[4] = a2s, btas
            rr = least_squares(resid, xs, x_scale="jac")
            if best is None or rr.cost < best.cost:
                best = rr
    J = best.jac
    cov = np.linalg.pinv(J.T @ J)
    chi2 = float(best.fun @ best.fun)
    ndata = len(best.fun)
    names = ["A1", "eta1", "A2", "eta2", "bTA"]
    err = np.sqrt(np.diag(cov))[:5]
    if args.rescale_errors:
        err = err*np.sqrt(chi2/(ndata - len(best.x)))
    g, ns, b1 = unpack(best.x)
    c5 = cov[:5, :5]/np.outer(np.sqrt(np.diag(cov))[:5], np.sqrt(np.diag(cov))[:5])
    return dict(params=dict(zip(names, best.x[:5].tolist())),
                err=dict(zip(names, err.tolist())),
                corr=c5.tolist(), chi2=chi2, ndata=ndata,
                dof=ndata - len(best.x), zmax=args.zmax_global,
                zmin=args.zmin_global,
                snapshots=[d["sn"] for d, _ in rows],
                Ns=dict(zip([d["sn"] for d, _ in rows], ns.tolist())),
                b1=dict(zip([d["sn"] for d, _ in rows], b1.tolist())))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def plot_snapshot(d, res, nla, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = d["k"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, p, lab in zip(ax, ("hh", "hE", "EE", "BB"),
                         (r"$P_{hh}$", r"$-P_{hE}$", r"$P_{EE}$", r"$P_{BB}$")):
        sgn = -1.0 if p == "hE" else 1.0
        P, s = d[p]
        a.errorbar(k, sgn*P, res["errors"].get(p, s), fmt="o", ms=3,
                   color="k", label="FLAMINGO")
        a.plot(k, sgn*res["model"][p], "C0-", label="TATT")
        a.plot(k, sgn*nla["model"][p], "C1--", label="NLA")
        m = res["masks"].get(p)
        if m is not None and m.any():
            a.axvspan(k[m].min(), k[m].max(), color="C0", alpha=0.08)
        a.set_xscale("log")
        a.set_yscale("symlog", linthresh=1e-3*np.max(np.abs(P)))
        a.set_xlim(0.008, 3)
        a.set_title(lab)
        a.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
    ax[0].legend(fontsize=8)
    th = res["theta"]
    fig.suptitle(f"sn{d['sn']}  z={d['z']:.2f}   a1={th['a1']:.2f}  "
                 f"a2={th['a2']:.2f}  bTA={th['bTA']:.2f}  b1={th['b1']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_evolution(results, evo, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.array([r["z"] for r in results])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, n in zip(ax, ("a1", "a2", "bTA")):
        v = [r["theta"][n] for r in results]
        e = [r["err"].get(n, np.nan) for r in results]
        a.errorbar(z, v, e, fmt="o", color="k")
        if evo is not None:
            zz = np.linspace(0, z.max(), 100)
            x = (1 + zz)/ONEPLUSZ0
            p = evo["params"]
            curve = {"a1": p["A1"]*x**p["eta1"], "a2": p["A2"]*x**p["eta2"],
                     "bTA": np.full_like(zz, p["bTA"])}[n]
            a.plot(zz, curve, "C0-")
            a.axvline(evo["zmax"], color="0.6", ls=":")
        a.set_xlabel("z")
        a.set_title(n)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ps-dir", required=True, help="folder of the raw Pds files")
    ap.add_argument("--out", default="tatt_fit_out")
    ap.add_argument("--snapshots", type=int, nargs="*", default=None)
    ap.add_argument("--nfold", type=int, default=1,
                    help="fold to fit (1 covers k <~ 2 h/Mpc)")
    ap.add_argument("--probes", nargs="+", default=["hh", "hE", "EE", "BB"],
                    choices=["hh", "hE", "EE", "BB"])
    ap.add_argument("--kmin", type=float, default=0.03)
    ap.add_argument("--kmax0", type=float, default=0.3,
                    help="IA kmax at z = 0; kmax(z) = kmax0/D(z)")
    ap.add_argument("--kmax-cap", type=float, default=0.6)
    ap.add_argument("--khh0", type=float, default=0.15,
                    help="P_hh kmax at z = 0 (linear-bias regime)")
    ap.add_argument("--khh-cap", type=float, default=0.3)
    ap.add_argument("--err-scale", type=float, default=1.0,
                    help="multiply the mode-scatter errors (reports: 3-9x "
                         "too small at high k)")
    ap.add_argument("--sys-frac", type=float, default=0.02,
                    help="fractional error floor added in quadrature")
    ap.add_argument("--bias", choices=["scale", "const"], default="scale",
                    help="scale: b(k) = sqrt(P_hh/P_mm) from data (default); "
                         "const: one linear b1 fitted with P_hh")
    ap.add_argument("--no-shape-noise", dest="shape_noise",
                    action="store_false",
                    help="do not fit the white shape-noise residual in EE/BB")
    ap.add_argument("--rescale-errors", action="store_true",
                    help="scale parameter covariances by chi2/dof")
    ap.add_argument("--no-shot-noise", action="store_true",
                    help="do not subtract 1/nbar from P_hh")
    ap.add_argument("--pmm-dir", default=None,
                    help="folder with FLAMINGO Pmm_sn<sn>.dat (k, P) to use "
                         "instead of CAMB halofit")
    ap.add_argument("--zmax-global", type=float, default=1.5,
                    help="snapshots used in the redshift-evolution fit")
    ap.add_argument("--zmin-global", type=float, default=0.5,
                    help="lowest snapshot z in the evolution fit (the sample's "
                         "mass edge moves at z <= 0.4; 0.5 keeps it fixed)")
    ap.add_argument("--halofit", default="takahashi",
                    help="CAMB halofit_version for P_NL; match the yaml "
                         "(cocoa's Roman yamls use takahashi)")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sns = args.snapshots or sorted(SNAPSHOTS, key=lambda s: SNAPSHOTS[s][0])
    print("Building CAMB + FAST-PT model ...", flush=True)
    model = TATTModel(pmm_dir=args.pmm_dir, halofit=args.halofit)

    results = []
    data = []
    hdr = (f"{'sn':>3} {'z':>5} {'kmax':>5} {'a1':>7} {'a2':>7} {'bTA':>6} "
           f"{'Ns':>6} {'chi2/N':>11} {'dchi2':>7} | {'aNLA':>5} {'rep':>5}")
    print(hdr)
    for sn in sns:
        d = load_snapshot(args.ps_dir, sn, args.nfold,
                          shot_noise=not args.no_shot_noise)
        tatt = fit_snapshot(model, d, args)
        nla = fit_snapshot(model, d, args, free=default_free(args, nla=True))
        # the reports' recipe: NLA, P_hE only, k = 0.05-0.3 h/Mpc
        chk = fit_snapshot(model, d, args, probes=["hE"], kband=(0.05, 0.3),
                           free=("a1",))
        nulls = null_test(d, args, tatt["kmax_ia"])
        th, e = tatt["theta"], tatt["err"]
        anla_rep_units = chk["theta"]["a1"]*C1RHOCRIT/C1RHOCRIT_REPORT
        print(f"{sn:>3} {d['z']:5.2f} {tatt['kmax_ia']:5.2f} "
              f"{th['a1']:7.3f} {th['a2']:7.3f} {th['bTA']:6.2f} "
              f"{th['Ns']:6.3f} {tatt['chi2']:6.1f}/{tatt['ndata']:<4d} "
              f"{nla['chi2'] - tatt['chi2']:7.1f} | "
              f"{anla_rep_units:5.2f} {SNAPSHOTS[sn][4]:5.2f}", flush=True)
        rec = {kk: v for kk, v in tatt.items()
               if kk not in ("model", "masks", "errors")}
        rec["nla"] = {kk: v for kk, v in nla.items()
                      if kk not in ("model", "masks", "errors")}
        rec["aNLA_check_report_units"] = anla_rep_units
        rec["nulls"] = nulls
        results.append(rec)
        data.append(d)
        if args.plots:
            plot_snapshot(d, tatt, nla,
                          os.path.join(args.out, f"fit_sn{sn}.png"))

    print("\nJoint redshift-evolution fit to all spectra ...", flush=True)
    evo = fit_redshift_evolution(model, data, results, args)
    with open(os.path.join(args.out, "tatt_fit.json"), "w") as f:
        json.dump(dict(args=vars(args), cosmo=COSMO, snapshots=results,
                       evolution=evo), f, indent=1)
    with open(os.path.join(args.out, "tatt_fit.csv"), "w") as f:
        f.write("sn,z,a1,a1_err,a2,a2_err,bTA,bTA_err,b1,Ns,chi2,ndata,"
                "a1_nla,chi2_nla,aNLA_check_c1_0.0134\n")
        for r in results:
            t, e = r["theta"], r["err"]
            f.write(f"{r['sn']},{r['z']},{t['a1']:.5f},{e['a1']:.5f},"
                    f"{t['a2']:.5f},{e['a2']:.5f},{t['bTA']:.5f},"
                    f"{e['bTA']:.5f},{t['b1']:.5f},{t['Ns']:.5f},"
                    f"{r['chi2']:.3f},{r['ndata']},"
                    f"{r['nla']['theta']['a1']:.5f},{r['nla']['chi2']:.3f},"
                    f"{r['aNLA_check_report_units']:.4f}\n")

    print("\naNLA: the reports' recipe (NLA, P_hE only, k = 0.05-0.3) in "
          f"their units (c1rhocrit = {C1RHOCRIT_REPORT}); rep = report value.")
    print("dchi2 = chi2(NLA) - chi2(TATT) on the same data, 2 extra "
          "parameters. Ns: white shape-noise residual in P_EE, P_BB.")
    if evo is not None:
        p, e = evo["params"], evo["err"]
        print(f"Redshift evolution ({evo['zmin']} <= z <= {evo['zmax']}, 1+z0 = {ONEPLUSZ0}, "
              f"chi2 = {evo['chi2']:.1f}/{evo['dof']}, "
              f"{len(evo['snapshots'])} snapshots):")
        for n in p:
            print(f"  {n:5s} = {p[n]:8.4f} +- {e[n]:.4f}")
        print("\ncosmolike yaml (IA_redshift_evolution = 3, IA_model = 1):")
        print(f"  roman_A1_1: {p['A1']:.4f}\n  roman_A1_2: {p['eta1']:.4f}\n"
              f"  roman_A2_1: {p['A2']:.4f}\n  roman_A2_2: {p['eta2']:.4f}\n"
              f"  roman_BTA_1: {p['bTA']:.4f}")
        if args.plots:
            plot_evolution(results, evo, os.path.join(args.out, "evolution.png"))
    print(f"\nWrote {args.out}/tatt_fit.json and tatt_fit.csv")


if __name__ == "__main__":
    sys.exit(main())
