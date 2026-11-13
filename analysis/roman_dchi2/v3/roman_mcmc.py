#!/usr/bin/env python3
"""MCMC check of the MAP shift at one mask (K0 = 0.3/D), case A.

Posterior over (Omega_m, A_s, IA) with flat chain priors; IA exact
(cosmolike at every step), cosmology through its linear response evaluated
at the MAP IA. emcee, walkers started around the MAP. Two runs per model:
data = GG + FLAMINGO (mismodelling) and data = TATT fiducial (matched
control for the projection/prior-volume shift).
    python roman_mcmc.py <data> <model: NLA|TATT> <truth: FLAMINGO|TATT> [nsteps]
"""
import json
import os
import sys
import time

import numpy as np
import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
import bias_analysis as B      # noqa: E402
import bias_v2 as V            # noqa: E402

K0 = 0.3
FREE = ["omegam", "As_1e9"]


def main():
    data, mdl, truth = sys.argv[1], sys.argv[2], sys.argv[3]
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    os.chdir(COCOA)
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    m = M1[K1.index(K0)]
    C = cov[np.ix_(m, m)]
    Ci = np.linalg.inv(C)
    Lw = np.linalg.inv(np.linalg.cholesky(C))
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d = xi_gg + R["xi_flam"] if truth == "FLAMINGO" else cl.model("TATT", B.FID, B.TATT_FID)
    names = FREE + B.IAP[mdl]
    nf = len(FREE)
    lo = np.array([V.BOUNDS[n][0] for n in names])
    hi = np.array([V.BOUNDS[n][1] for n in names])
    x_fid = np.array([B.FID[n] for n in FREE])
    # MAP (bias_v2 machinery)
    r = V.fit(cl, mdl, d, Ci, Lw, m, FREE, print, nla_sol=(3.5, 0.68) if truth == "FLAMINGO" else (3.8, 0.57))
    x_map = np.array([r["best"][n] for n in names])
    ia_map = x_map[nf:]
    Jc = np.zeros((m.sum(), nf))
    for i, n in enumerate(FREE):
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        Jc[:, i] = (cl.model(mdl, pp, ia_map) - cl.model(mdl, pm, ia_map))[m]/(2*B.STEP[n])
    # S8 response (linear)
    s8_fid = cl.s8(B.FID)
    gS8 = np.array([(cl.s8({**B.FID, n: B.FID[n] + B.STEP[n]}) - cl.s8({**B.FID, n: B.FID[n] - B.STEP[n]}))/(2*B.STEP[n]) for n in FREE])

    def lnpost(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        mod = cl.model(mdl, B.FID, x[nf:])[m] + Jc @ (x[:nf] - x_fid)
        v = Lw @ (d[m] - mod)
        return -0.5*float(v @ v)
    ndim = len(names)
    nwalk = max(16, 2*ndim + 2)
    rng = np.random.default_rng(1)
    sig0 = np.array([r["flat"]["sOm"], 0.03] + [0.05]*(ndim - nf))
    p0 = np.clip(x_map + sig0*rng.normal(size=(nwalk, ndim))*0.5, lo + 1e-3, hi - 1e-3)
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnpost)
    tag = f"{mdl}_{truth}"
    chainfile = os.path.join(HERE, f"chain_{tag}.npy")
    for i, _ in enumerate(sampler.sample(p0, iterations=nsteps, progress=False)):
        if (i + 1) % 50 == 0:
            np.save(chainfile, sampler.get_chain())
            print(f"{tag}: step {i + 1}/{nsteps}, {time.time() - t0:.0f}s, acc {np.mean(sampler.acceptance_fraction):.2f}", flush=True)
    chain = sampler.get_chain()
    np.save(chainfile, chain)
    burn = nsteps//3
    flat = chain[burn:].reshape(-1, ndim)
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = np.full(ndim, np.nan)
    mean = flat.mean(0)
    std = flat.std(0)
    S8 = s8_fid + (flat[:, :nf] - x_fid) @ gS8
    S8_map = s8_fid + (x_map[:nf] - x_fid) @ gS8
    out = dict(model=mdl, truth=truth, mask=K0, names=names, nwalk=nwalk, nsteps=nsteps, burn=burn,
               tau=tau.tolist(), map=x_map.tolist(), mean=mean.tolist(), std=std.tolist(),
               S8_fid=float(s8_fid), S8_map=float(S8_map), S8_mean=float(S8.mean()), S8_std=float(S8.std()),
               Om_fid=B.FID["omegam"], map_FoB=r["flat"]["FoB"], map_chi2=r["chi2_bestfit"],
               dOm_mean_sig=float((mean[0] - B.FID["omegam"])/std[0]), dS8_mean_sig=float((S8.mean() - s8_fid)/S8.std()),
               dOm_map_sig=float((x_map[0] - B.FID["omegam"])/std[0]), dS8_map_sig=float((S8_map - s8_fid)/S8.std()))
    # 2D FoB of the posterior mean in the posterior covariance
    th = np.column_stack([flat[:, 0], S8])
    Sig = np.cov(th.T)
    dmean = th.mean(0) - np.array([B.FID["omegam"], s8_fid])
    out["FoB_mean"] = float(np.sqrt(dmean @ np.linalg.solve(Sig, dmean)))
    dmap = np.array([x_map[0], S8_map]) - np.array([B.FID["omegam"], s8_fid])
    out["FoB_map_postcov"] = float(np.sqrt(dmap @ np.linalg.solve(Sig, dmap)))
    json.dump(out, open(os.path.join(HERE, f"mcmc_{tag}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
