#!/usr/bin/env python3
"""Cut validation by minimizer plus full Fisher, Roman cosmic shear or 3x2pt.

For one IA model, one cut K0 (k_UV = K0/D(z), masks_n1) and one FLAMINGO
truth, with every parameter of the chains free (5 cosmological, 8 Delta z
with N(0, 0.002), 8 m with N(0, 0.005), the IA parameters):
    data     d = GG(D3A) + xi_IA^FLAMINGO(truth), noise free, the GG of the
             chain's own theory (emulmps + cosmolike, use_emulator 2)
    ref      theta_ref: D3A, Delta z = m = 0, the IA parameters fitted to d
             at fixed cosmology (the reference the bias is measured from)
    Fisher   F = J^T J + P at theta_ref, J the derivatives of the whitened
             data vector (central differences, absolute steps), P the
             Gaussian priors; flat priors left out. sigma(Omega_m, S8) is
             marginalized over all the other parameters.
    min      the maximum a posteriori from theta_ref and 2 starts drawn from
             F^-1 (scipy least_squares, trust region with the flat prior
             bounds, the same J); its shift of (Omega_m, S8) from D3A has no
             projection effect. mu = shift/sigma_F, FoB = sqrt(dp C^-1 dp).
    linear   the Fisher prediction of the same shift, F^-1 J^T r(theta_ref).
Models: NLA (A1, eta1 power law), NLAz (NLA with one A1 per source bin,
IA_redshift_evolution 2), TATT (A1, eta1, A2, eta2, bTA).
Probes:
    cs     xi_+- at K0, the example1 covariance (cov_example1_xi.npy)
    3x2    xi_+- at K0, gamma_t of sources behind lenses at K0 = 0.3, w at
           K0 = 0.2 (masks of v9/ia_base_3x2_common.npz), its Gaussian 3x2pt
           covariance, the 8 linear biases free in [0.4, 4] (truth B1_FID of
           cocoa/driver3x2.py), lens Delta z = source Delta z, Limber for
           gamma_t and w (as the base the IA vector comes from), the IA part
           of the data xi_ia of v9/ia_base_3x2_<truth>.npz (xi_+- and gamma_t)
    +cmb   Gaussian priors at D3A of the width of Planck 2018 on n_s (0.0042),
           Omega_b h^2 (0.00015) and H0 (0.54); none on A_s or Omega_m.
Run from ~/cocoa/Cocoa after start_cocoa.sh:
    python <this> [--probe cs|3x2] [--cmb] <model> <K0> <truth> [...]
Each run appends to fisher_minimize.json, key <probe>[+cmb]|<model>|<K0>|<truth>
(the first cosmic-shear runs, without prior, are under <model>|<K0>|<truth>).
"""
import fcntl
import json
import os
import sys
import time

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "roman_dchi2")
LIKS = {"cs": "roman_real.cosmic_shear", "3x2": "roman_real.combo_3x2pt"}
NXI, NDV = 1080, 2115
B1_FID = [1.24, 1.36, 1.55, 1.71, 1.90, 2.15, 2.52, 2.44]
BB = [f"roman_B1_{i}" for i in range(1, 9)]
OBH2_FID = 0.0486*0.681**2
# CMB prior rows: (name, function of the parameter dict, centre, sigma)
CMB = [("ns", lambda p: p["ns"], 0.967, 0.0042),
       ("H0", lambda p: p["H0"], 68.1, 0.54),
       ("ombh2", lambda p: p["omegab"]*(p["H0"]/100)**2, OBH2_FID, 0.00015)]
OUT = os.path.join(HERE, "fisher_minimize.json")

FID = dict(As_1e9=2.099, ns=0.967, H0=68.1, omegab=0.0486, omegam=0.306)
COSMO = list(FID)
BOUNDS = dict(As_1e9=(0.5, 5), ns=(0.87, 1.07), H0=(55, 91), omegab=(0.03, 0.07), omegam=(0.1, 0.9))
DZ = [f"roman_DZ_S{i}" for i in range(1, 9)]
MM = [f"roman_M{i}" for i in range(1, 9)]
PRIOR = {**{p: 0.002 for p in DZ}, **{p: 0.005 for p in MM}}
# absolute derivative steps, about a tenth of each parameter's error
STEP = dict(As_1e9=0.02, ns=0.004, H0=0.4, omegab=0.001, omegam=0.002,
            **{p: 0.0004 for p in DZ}, **{p: 0.001 for p in MM}, **{p: 0.02 for p in BB})
BOUNDS.update({p: (0.4, 4.0) for p in BB})
IA = {
    "NLA":  dict(model=0, evo=3, free={"roman_A1_1": (-5, 5, 0.03), "roman_A1_2": (-5, 5, 0.02)}),
    "NLAz": dict(model=0, evo=2, free={f"roman_A1_{i}": (-5, 10, 0.03) for i in range(1, 9)}),
    "TATT": dict(model=1, evo=3, free={"roman_A1_1": (-5, 5, 0.03), "roman_A1_2": (-5, 5, 0.02),
                                       "roman_A2_1": (-5, 5, 0.05), "roman_A2_2": (-5, 5, 0.1),
                                       "roman_BTA_1": (0, 2, 0.03)}),
}
IA_START = {"NLA": [3.5, 0.7], "NLAz": [3.0]*8, "TATT": [3.4, 0.7, -1.0, -2.0, 0.5]}
ALL_IA = ["roman_A1_%d" % i for i in range(1, 9)] + ["roman_A2_1", "roman_A2_2", "roman_BTA_1"]


class Setup:
    """The cobaya model of one IA model, the data, mask and whitening."""

    def __init__(self, mdl, K0, truth, probe="cs", cmb=False):
        info = yaml_load_file(os.path.join(HERE, "NLA_on_flam_k0p2.yaml"))
        for k in ("sampler", "output"):
            info.pop(k, None)
        LIK = LIKS[probe]
        self.probe, self.LIK = probe, LIK
        L = info["likelihood"].pop(LIKS["cs"])
        info["likelihood"][LIK] = L
        # cutval_3x2_ones: all points, the Gaussian 3x2pt covariance (the shipped
        # example1_cov is not positive definite and cosmolike refuses it)
        L["data_file"] = "ones.dataset" if probe == "cs" else "cutval_3x2_ones.dataset"
        if probe == "3x2":
            L.update(lmax=100000, ggl_exclude=[[6, 0], [7, 0], [7, 1]], adopt_limber_gs=1, adopt_limber_gg=1)
        L["IA_model"], L["IA_redshift_evolution"] = IA[mdl]["model"], IA[mdl]["evo"]
        pr = info["params"]
        for p in ALL_IA:
            pr[p] = {"value": 0.0}
        for p, (lo, hi, _) in IA[mdl]["free"].items():
            pr[p] = {"prior": {"min": lo, "max": hi}, "ref": 1.0, "proposal": 0.1}
        if probe == "3x2":
            for p in BB:
                pr[p] = {"prior": {"min": 0.4, "max": 4.0}, "ref": 1.5, "proposal": 0.03}
        self.model = get_model(info)
        self.lik = self.model.likelihood[LIK]
        orig = self.lik.get_datavector
        self.last = None
        self.neval = 0

        def keep(**params):                      # the logp call keeps its vector
            self.last = np.array(orig(**params))
            return self.last
        self.lik.get_datavector = keep
        self.names = COSMO + DZ + MM + (BB if probe == "3x2" else []) + list(IA[mdl]["free"])
        self.lo = np.array([BOUNDS[p][0] if p in BOUNDS else IA[mdl]["free"][p][0] if p in IA[mdl]["free"]
                            else -np.inf for p in self.names])
        self.hi = np.array([BOUNDS[p][1] if p in BOUNDS else IA[mdl]["free"][p][1] if p in IA[mdl]["free"]
                            else np.inf for p in self.names])
        self.h = np.array([STEP.get(p, IA[mdl]["free"].get(p, (0, 0, 0))[2]) for p in self.names])
        if probe == "cs":
            m = np.load(os.path.join(RD, "cocoa", "run", "cosmolike_xi_ia.npz"))
            iK = int(np.argmin(np.abs(m["K_n1"] - K0)))
            assert abs(m["K_n1"][iK] - K0) < 1e-6
            self.mask = m["masks_n1"][iK].astype(bool)
            cov = np.load(os.path.join(RD, "cocoa", "analysis", "cov_example1_xi.npy"))
            self.n = NXI
        else:
            c = np.load(os.path.join(RD, "v9", "ia_base_3x2_common.npz"))
            self.mask = np.concatenate([c[f"xi|{K0:g}"], c["gt|0.3"] & c["behind"], c["w|0.2"]]).astype(bool)
            assert self.mask.size == NDV
            cov = c["cov"]
            self.n = NDV
        self.Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(self.mask, self.mask)]))
        self.ip = np.array([self.names.index(p) for p in PRIOR])
        self.sp = np.array([PRIOR[p] for p in PRIOR])
        self.cmb = CMB if cmb else []
        gg, _ = self.vector(self.theta(FID, {}, IA_START[mdl], zero_ia=True))
        base = "ia_base" if probe == "cs" else "ia_base_3x2"
        xi_ia = np.load(os.path.join(RD, "v9", f"{base}_{truth}.npz"))["xi_ia"]
        assert xi_ia.size == self.n
        self.d = gg + xi_ia

    def theta(self, cosmo, nuis, ia, zero_ia=False):
        t = [cosmo[p] for p in COSMO] + [nuis.get(p, 0.0) for p in DZ + MM]
        if self.probe == "3x2":
            t += [nuis.get(p, b) for p, b in zip(BB, B1_FID)]
        return np.array(t + [0.0 if zero_ia else v for v in ia])

    def vector(self, th):
        p = dict(zip(self.names, th))
        r = self.model.logposterior(p, as_dict=True)
        self.neval += 1
        if "S8" not in r["derived"]:
            raise RuntimeError(f"no S8 (logpost {r.get('logpost')}, logpriors {r.get('logpriors')}) at "
                               + ", ".join(f"{k} {v:.4g}" for k, v in p.items()))
        return self.last[:self.n].copy(), r["derived"]["S8"]

    def resid(self, th):
        v, _ = self.vector(th)
        return np.concatenate([self.Lw @ (self.d - v)[self.mask], -th[self.ip]/self.sp, self.cmb_rows(th)])

    def cmb_rows(self, th):
        p = dict(zip(self.names, th))
        return np.array([-(f(p) - c)/sg for _, f, c, sg in self.cmb])

    def jac(self, th, idx=None):
        """d resid/d theta by central differences (columns idx only), and dS8/d theta."""
        idx = range(len(th)) if idx is None else idx
        nd, npr = self.mask.sum(), len(self.ip)
        J = np.zeros((nd + npr + len(self.cmb), len(th)))
        g = np.zeros(len(th))
        for i in idx:
            tp, tm = th.copy(), th.copy()
            # the stencil stays inside the flat prior (one-sided at a bound)
            tp[i] = min(th[i] + self.h[i], self.hi[i])
            tm[i] = max(tp[i] - 2*self.h[i], self.lo[i])
            tp[i] = tm[i] + 2*self.h[i]
            vp, sp_ = self.vector(tp)
            vm, sm_ = self.vector(tm)
            J[:nd, i] = -self.Lw @ ((vp - vm)[self.mask])/(tp[i] - tm[i])
            g[i] = (sp_ - sm_)/(tp[i] - tm[i])
            if self.cmb:
                J[nd + npr:, i] = (self.cmb_rows(tp) - self.cmb_rows(tm))/(tp[i] - tm[i])
        J[nd + np.arange(npr), self.ip] = -1.0/self.sp
        return J, g


def solve(S, th0, free):
    """least_squares over the parameters free (indices), the others held."""
    def f(x):
        th = th0.copy()
        th[free] = x
        return S.resid(th)

    def jf(x):
        th = th0.copy()
        th[free] = x
        return S.jac(th, free)[0][:, free]
    r = least_squares(f, th0[free], jac=jf, bounds=(S.lo[free], S.hi[free]),
                      x_scale=S.h[free]*10, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=60)
    th = th0.copy()
    th[free] = r.x
    return th, float(r.fun @ r.fun), r


def run(mdl, K0, truth, probe="cs", cmb=False):
    t0 = time.time()
    S = Setup(mdl, K0, truth, probe, cmb)
    n = len(S.names)
    iia = np.arange(n - len(IA[mdl]["free"]), n)
    ic = [S.names.index("omegam")]
    # reference: IA fitted at the fiducial cosmology
    ref, chi2_ref, _ = solve(S, S.theta(FID, {}, IA_START[mdl]), iia)
    # the flat priors as Gaussians only where no Gaussian prior already acts
    J, g = S.jac(ref)
    F = J.T @ J
    Finv = np.linalg.inv(F)
    # the flat priors as Gaussians of the same variance (width^2/12): for the
    # starts of the minimizer, and quoted as a second sigma
    wflat = np.where(np.isfinite(S.lo), (S.hi - S.lo)**2/12, np.inf)
    if cmb:   # n_s and H0 carry the CMB prior; Omega_b is tied through Omega_b h^2
        for q in ("ns", "H0", "omegab"):
            wflat[S.names.index(q)] = np.inf
    Fflat_inv = np.linalg.inv(F + np.diag(np.where(np.isfinite(wflat), 1/wflat, 0.0)))
    G = np.zeros((2, n))
    G[0, ic[0]] = 1.0
    G[1] = g
    C2 = G @ Finv @ G.T
    sig = np.sqrt(np.diag(C2))
    r_ref = S.resid(ref)
    lin = -Finv @ J.T @ r_ref                  # Gauss-Newton step from ref
    dlin = G @ lin
    # the same with the flat priors as Gaussians: the linear (Gaussian)
    # estimate of the shift of the posterior mean, the chains' measure
    dlinP = G @ (-Fflat_inv @ J.T @ r_ref)
    C2P = G @ Fflat_inv @ G.T
    sigP = np.sqrt(np.diag(C2P))
    # minimizer: ref and 2 starts drawn from F^-1, inside the bounds
    rng = np.random.default_rng(1)
    fin = np.isfinite(S.lo)
    pad = np.where(fin, 1e-3*(S.hi - S.lo), 0.0)
    starts = [ref]
    for _ in range(2):
        x = ref + rng.multivariate_normal(np.zeros(n), 0.25*Fflat_inv)
        starts.append(np.where(fin, np.clip(x, S.lo + pad, S.hi - pad), x))
    best, runs = None, []
    for s in starts:
        th, c2, r = solve(S, s, np.arange(n))
        runs.append(dict(chi2=c2, nfev=int(r.nfev), status=int(r.status)))
        if best is None or c2 < best[1]:
            best = (th, c2)
    th, c2 = best
    _, s8 = S.vector(th)
    _, s8_ref = S.vector(S.theta(FID, {}, [0.0]*len(iia)))
    dp = np.array([th[ic[0]] - FID["omegam"], s8 - s8_ref])
    mu, fob = dp/sig, float(np.sqrt(dp @ np.linalg.solve(C2, dp)))
    mu_lin = dlin/sig
    names = S.names
    at_bound = [p for p, v, lo, hi in zip(names, th, S.lo, S.hi)
                if np.isfinite(lo) and ((v - lo) < 1e-3*(hi - lo) or (hi - v) < 1e-3*(hi - lo))]
    row = dict(model=mdl, K0=K0, truth=truth, probe=probe, cmb=cmb, npts=int(S.mask.sum()), names=names,
               ref=dict(zip(names, ref.tolist())), chi2_ref=chi2_ref,
               best=dict(zip(names, th.tolist())), chi2_min=c2, starts=runs,
               S8_truth=s8_ref, S8_min=s8, shift=dp.tolist(), sigma=sig.tolist(),
               corr=float(C2[0, 1]/np.prod(sig)), mu=mu.tolist(), fob=fob,
               mu_linear=mu_lin.tolist(),
               flatprior=dict(sigma=sigP.tolist(), mu_map=(dp/sigP).tolist(),
                              fob_map=float(np.sqrt(dp @ np.linalg.solve(C2P, dp))),
                              mu_mean_linear=(dlinP/sigP).tolist(),
                              fob_mean_linear=float(np.sqrt(dlinP @ np.linalg.solve(C2P, dlinP)))),
               sigma_all=dict(zip(names, np.sqrt(np.diag(Finv)).tolist())),
               sigma_flatprior=np.sqrt(np.diag(G @ Fflat_inv @ G.T)).tolist(),
               at_bound=at_bound, neval=S.neval, seconds=time.time() - t0)
    key = f"{probe}{'+cmb' if cmb else ''}|{mdl}|{K0:g}|{truth}"
    with open(OUT + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.load(open(OUT)) if os.path.exists(OUT) else dict(about=__doc__, runs={})
        out["about"] = __doc__
        out["runs"][key] = row
        json.dump(out, open(OUT, "w"), indent=1)
    iav = ", ".join(f"{p.replace('roman_', '')} {th[names.index(p)]:+.2f}" for p in IA[mdl]["free"])
    print(f"{key}: {S.mask.sum()} pts, chi2 ref {chi2_ref:.3f} -> min {c2:.3f} "
          f"(starts {[round(r['chi2'], 3) for r in runs]}); sigma ({sig[0]:.4f}, {sig[1]:.4f}); "
          f"mu ({mu[0]:+.3f}, {mu[1]:+.3f}) [linear ({mu_lin[0]:+.3f}, {mu_lin[1]:+.3f})], FoB {fob:.3f}; "
          f"flat-prior sigma ({sigP[0]:.4f}, {sigP[1]:.4f}): MAP mu ({dp[0]/sigP[0]:+.3f}, {dp[1]/sigP[1]:+.3f}) "
          f"FoB {float(np.sqrt(dp @ np.linalg.solve(C2P, dp))):.3f}, mean(lin) mu ({dlinP[0]/sigP[0]:+.3f}, "
          f"{dlinP[1]/sigP[1]:+.3f}); "
          f"H0 {th[names.index('H0')]:.1f} ns {th[names.index('ns')]:.3f}; {iav}"
          + (f"  AT BOUND {at_bound}" if at_bound else "") + f"  [{time.time() - t0:.0f}s, {S.neval} evals]",
          flush=True)


def main():
    a = sys.argv[1:]
    probe = "cs"
    if "--probe" in a:
        probe = a.pop(a.index("--probe") + 1)
        a.remove("--probe")
    cmb = "--cmb" in a
    if cmb:
        a.remove("--cmb")
    for i in range(0, len(a), 3):
        run(a[i], float(a[i + 1]), a[i + 2], probe, cmb)


if __name__ == "__main__":
    main()
