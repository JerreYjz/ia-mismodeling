#!/usr/bin/env python3
"""GGL and 3x2pt: figure of bias with the FLAMINGO IA injected into gamma_t
as well as xi_+-.

Truth: d = [GG + galaxy terms](fiducial cosmology, b_i = B1_FID) + FLAMINGO IA
(sim_IA tables: GI + II in xi_+-, GI = b_i P_dE in gamma_t; w has no IA).
Model: native cosmolike NLA / TATT 3x2pt, power-law IA evolution, with
Omega_m, A_s, the eight linear biases b_i and the IA parameters free
(roman_test.fit; IA bounded to the chain priors, b_i in [0.8, 3.0]).
Masks: IR/UV (alpha 0.05, GG + NLA baseline, monotonic rows) with separate
k_UV(z) = K0/D(z) per probe: cosmic shear K0_cs, GGL K0_ggl, clustering
K0_gc = 0.2 (the plan's value). Runs:
  3x2pt  : K0_cs in {0.2, 0.3} x K0_ggl in {0.1, 0.15, 0.2, 0.3, 0.4}
  GGL+w  : the same K0_ggl with every xi_+- row masked (lens probes alone)
  *_behind: the same with only the source bins behind the lens bin (s > l)
           kept in gamma_t (the conservative GGL configuration; the same-bin
           pairs are GI-dominated with a shared lens/source n(z))
Covariance: Gaussian 3x2pt (cov3x2.py). FoB in (Omega_m, S8) with b_i and IA
marginalised. No FLAMINGO-noise band here (the xi J_cells Jacobian does not
cover gamma_t).
    python ggl3x2.py <cocoa data dir> [part: all|3x2|ggl]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
sys.path.insert(0, HERE)
import bias_analysis as B                                             # noqa: E402
import bias_v2 as V                                                   # noqa: E402
from cosmolike_xi_ia import LN_FLOOR, mask_rows                       # noqa: E402
from driver3x2 import Cosmolike3x2, B1_FID                            # noqa: E402
import roman_test as RT                                               # noqa: E402

BN = [f"b{i + 1}" for i in range(8)]
K0_CS, K0_GGL, K0_GC = (0.2, 0.3), (0.1, 0.15, 0.2, 0.3, 0.4), 0.2
NXI, NGT = 1080, 915
SUFFIX = os.environ.get("GGL_SUFFIX", "")


class CL3(Cosmolike3x2):
    """3x2pt driver with the cached-CAMB cosmology interface of bias_analysis.CL."""
    cache = B.CL.cache
    order = B.CL.order
    at = B.CL.at
    s8 = B.CL.s8

    def model(self, mdl, p, ia):
        self.at(p)
        b1 = [p[n] for n in BN]
        if mdl == "NLA":
            return self.dv_native(0, A1=[ia[0], ia[1]], A2=[0.0, 0.0], BTA=[0.0], B1=b1)
        return self.dv_native(1, A1=[ia[0], ia[1]], A2=[ia[2], ia[3]], BTA=[ia[4]], B1=b1)


def main():
    data = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    os.chdir(COCOA)
    for i, n in enumerate(BN):                      # bias parameters as "cosmology-like" free parameters
        B.FID[n], B.STEP[n], V.BOUNDS[n] = B1_FID[i], 0.02, (0.8, 3.0)
    cov = np.load("analysis/cov3x2_gauss_d3a.npy")
    fits = np.load("../v2/fits.npz")
    zs, sns = fits["zs"], fits["sns"]
    cl = CL3(os.path.join(COCOA, "lib"), data, threads=1)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    dv_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    cl.use_sim_tables("run/tables_stitch/prod", sns, [float(z) for z in zs], 1)
    dv_sim = cl.dv_sim()
    xi_fl = dv_sim - dv_gg
    R = np.load("run/cosmolike_xi_ia.npz")
    assert np.max(np.abs(xi_fl[:NXI] - R["xi_flam"])) < 1e-12*np.max(np.abs(R["xi_flam"])) + 1e-30
    d_fl = dv_gg + xi_fl
    np.savez(os.path.join(HERE, "dv3x2_flam.npz"), dv_gg=dv_gg, xi_fl=xi_fl, pairs=np.array(cl.gs_pairs()))
    logf = open(os.path.join(HERE, f"ggl3x2_log_{part}.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    # ---------------- masks ----------------
    cl.ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    cl.sim = False
    k = 10.0**cl.lk/h
    z2 = cl.z2
    lnPL = np.log(cl.PKL.P(z2, 10.0**cl.lk)) + np.log(h**3)
    lnPNL = np.log(cl.PKNL.P(z2, 10.0**cl.lk)) + np.log(h**3)
    lnPLcb = np.log(cl.PKLcb.P(z2, 10.0**cl.lk)) + np.log(h**3)
    zg = cl.z1[cl.z1 <= z2[-1]]
    G = np.sqrt(cl.PKL.P(zg, 0.05)/cl.PKL.P(0, 0.05))*(1 + zg)
    G = np.asarray(G/(np.sqrt(cl.PKL.P(z2[-1], 0.05)/cl.PKL.P(0, 0.05))*(1 + z2[-1]))).ravel()
    Dz2 = np.sqrt(np.asarray(cl.PKL.P(z2, 1e-3))/cl.PKL.P(0.0, 1e-3)).ravel()

    def dv_with(lnL, lnN):
        cl.at(B.FID)
        cl.ci.set_cosmology(omegam=cl.cosmo["omegam"], omegab=cl.cosmo["omegab"], omegan2=cl.omegan2,
                            H0=cl.cosmo["H0"], log10k_2D=cl.lk - np.log10(h), z_2D=z2,
                            lnP_linear=lnL.flatten(order="F"), lnP_linear_cb=lnPLcb.flatten(order="F"),
                            lnP_nonlinear=lnN.flatten(order="F"), G=G, z_G=zg, z_1D=cl.z1,
                            chi=cl.res.comoving_radial_distance(cl.z1)*h)
        cl._nuisance([0.6, -1.5], [0.0], [0.0])
        return np.asarray(cl.ci.compute_data_vector_masked(), dtype=float)
    full = dv_with(lnPL, lnPNL)

    def ir_vector(K0):
        kuv = K0/Dz2
        keep = k[None, :] <= kuv[:, None]
        return dv_with(np.where(keep, lnPL, lnPL + LN_FLOOR), np.where(keep, lnPNL, lnPNL + LN_FLOOR))

    def frac(ir):
        uv = full - ir
        den = np.abs(uv) + np.abs(ir)
        return np.where(den > 0, np.abs(uv)/np.where(den > 0, den, 1), 0)
    ir_cache = {}

    pairs = cl.gs_pairs()
    behind = part.endswith("_behind")          # keep only source bins behind the lens bin (s > l)

    def mask3x2(K0cs, K0ggl, K0gc, with_xi=True):
        f = np.zeros_like(full)
        for K0, sl in ((K0cs, slice(0, NXI)), (K0ggl, slice(NXI, NXI + NGT)), (K0gc, slice(NXI + NGT, None))):
            if K0 not in ir_cache:
                ir_cache[K0] = frac(ir_vector(K0))
            f[sl] = ir_cache[K0][sl]
        m = mask_rows(f)
        if not with_xi:
            m[:NXI] = False
        if behind:
            for p, (l, s_) in enumerate(pairs):
                if s_ <= l:
                    m[NXI + 15*p:NXI + 15*(p + 1)] = False
        return m

    runs = []
    if part in ("all", "3x2", "3x2_behind"):
        for K0cs in K0_CS:
            for K0g in K0_GGL:
                runs.append((f"3x2pt cs {K0cs:g} ggl {K0g:g} gc {K0_GC:g}", dict(probe="3x2pt", K0_cs=K0cs, K0_ggl=K0g, K0_gc=K0_GC), mask3x2(K0cs, K0g, K0_GC)))
    if part in ("all", "ggl", "ggl_behind"):
        for K0g in K0_GGL:
            runs.append((f"GGL+w ggl {K0g:g} gc {K0_GC:g}", dict(probe="GGL+w", K0_cs=None, K0_ggl=K0g, K0_gc=K0_GC), mask3x2(0.2, K0g, K0_GC, with_xi=False)))
    # denser TATT start grid: both solution branches seen in the first pass
    _starts = V.starts

    def starts(mdl, nla_sol):
        if mdl != "TATT":
            return _starts(mdl, nla_sol)
        a1, e1 = nla_sol
        extra = [[a1, e1, 0.6, -4.5, 0.3], [a1, e1, 0.06, 4.9, 0.0], [a1, e1, -1.0, -0.7, 0.2],
                 [a1, e1, 1.0, 0.0, 0.5], [a1, e1, 0.0, 2.0, 1.0], [a1, e1, -0.5, 4.0, 0.0], [a1, e1, 2.0, -2.0, 0.1]]
        return list(_starts(mdl, nla_sol)) + [np.array(x) for x in extra]
    V.starts = starts
    free = ["omegam", "As_1e9"] + BN
    out = []
    for lab, meta, m in runs:
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        npts = dict(xi=int(m[:NXI].sum()), gt=int(m[NXI:NXI + NGT].sum()), w=int(m[NXI + NGT:].sum()))
        log(f"[{lab}: points xi {npts['xi']} gt {npts['gt']} w {npts['w']}  ({time.time() - t0:.0f}s)]")
        rn = RT.fit(cl, "NLA", d_fl, Ci, Lw, m, free, {}, log)
        rt = RT.fit(cl, "TATT", d_fl, Ci, Lw, m, free, {}, log, nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]))
        row = dict(label=lab, npts=npts, **meta)
        for mdl, r in (("NLA", rn), ("TATT", rt)):
            row[mdl] = dict(FoB=r["FoB"], chi2=r["chi2_bestfit"], dOm=r["dOm"]/r["sOm"], dS8=r["dS8"]/r["sS8"],
                            sOm=r["sOm"], sS8=r["sS8"], at_bound=r["at_bound"], ia=[r["best"][n] for n in B.IAP[mdl]],
                            bias=[r["best"][n] for n in BN])
        out.append(row)
        json.dump(out, open(os.path.join(HERE, f"ggl3x2_{part}{SUFFIX}.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
