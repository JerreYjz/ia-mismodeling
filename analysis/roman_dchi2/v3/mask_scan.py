#!/usr/bin/env python3
"""Roman figure of bias against the mask for three one-parameter mask families
    D1   : k_UV(z) = K0 / D(z)          (n = -1, the production family)
    D23  : k_UV(z) = K0 / D(z)^(2/3)    (n = 0, the fit to the TATT cuts, Sec. 10)
    flat : k_UV(z) = K0                 (constant comoving k, the NLA cuts)
K0 = 0.15 ... 0.5 h/Mpc. IR/UV masks as make_uv_scale_cuts (GG + NLA
baseline, alpha 0.05, monotonic rows). Data = GG + FLAMINGO (production
truth). Fits (roman_test.fit): NLA and TATT with every parameter free, case
A (Omega_m, A_s) and case B (+ n_s, H0, Omega_b with the chain priors);
NLA + white template and TATT + white template (free amplitude a_w) in case
A. FoB uncertainty from the simulation's error bars (J_cells).
    python mask_scan.py <cocoa data dir>
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
from cosmolike_xi_ia import KMIN, LN_FLOOR, mask_rows                 # noqa: E402
import roman_test as RT                                               # noqa: E402

K0S = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
FAMILIES = {"D1": 1.0, "D23": 2.0/3.0, "flat": 0.0}
if os.environ.get("FAMS"):
    FAMILIES = {k: v for k, v in FAMILIES.items() if k in os.environ["FAMS"].split(",")}
TAG = os.environ.get("OUTTAG", "")


def main():
    data = sys.argv[1]
    t0 = time.time()
    os.chdir(COCOA)
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load(os.environ.get("COV", "analysis/cov_gauss_d3a.npy"))
    fits = np.load("../v2/fits.npz")
    zs, sns, kd = fits["zs"], fits["sns"], fits["k"][0]
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]

    # ---------------- mask machinery (as v3_roman / cosmolike_xi_ia) ----------------
    cl.ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
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
        cl.ci.set_cosmology(omegam=cl.cosmo["omegam"], omegab=cl.cosmo["omegab"], omegan2=cl.omegan2,
                            H0=cl.cosmo["H0"], log10k_2D=cl.lk - np.log10(h), z_2D=z2,
                            lnP_linear=lnL.flatten(order="F"), lnP_linear_cb=lnPLcb.flatten(order="F"),
                            lnP_nonlinear=lnN.flatten(order="F"), G=G, z_G=zg, z_1D=cl.z1,
                            chi=cl.res.comoving_radial_distance(cl.z1)*h)
        cl._nuisance([0.6, -1.5], [0.0], [0.0])
        return cl.xi_vector()
    full = dv_with(lnPL, lnPNL)

    def mask_for(kuv_z2):
        keep = k[None, :] <= kuv_z2[:, None]
        ir = dv_with(np.where(keep, lnPL, lnPL + LN_FLOOR), np.where(keep, lnPNL, lnPNL + LN_FLOOR))
        uv = full - ir
        den = np.abs(uv) + np.abs(ir)
        frac = np.where(den > 0, np.abs(uv)/np.where(den > 0, den, 1), 0)
        return mask_rows(frac)

    # noise propagation
    J_cells = np.load("analysis/J_cells.npy")
    bk, Pf, err = fits["b"], fits["PNL"], fits["err"]
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    sig = []
    for s in range(len(zs)):
        for x in range(2):
            conv = PNLd[s]/(bk[s]*Pf[s]) if x == 0 else PNLd[s]/Pf[s]
            sig.extend((err[s, x]*conv)[nodes])
    Jcells_S = J_cells*np.array(sig)[None, :]

    # white-template models
    B.IAP["NLAw"] = ["A1", "eta1", "aw"]
    B.IAP["TATTw"] = ["A1", "eta1", "A2", "eta2", "bTA", "aw"]
    V.BOUNDS["aw"] = (-5, 5)
    V.IA_SIG["aw"] = 10/np.sqrt(12)
    _orig = B.CL.model

    def model_w(self, mdl, p, ia):
        if mdl == "NLAw":
            return _orig(self, "NLA", p, ia[:2]) + ia[2]*TPL
        if mdl == "TATTw":
            return _orig(self, "TATT", p, ia[:5]) + ia[5]*TPL
        return _orig(self, mdl, p, ia)
    B.CL.model = model_w
    _starts = V.starts

    def starts(mdl, nla_sol):
        if mdl == "NLAw":
            return [np.array([nla_sol[0], nla_sol[1], 1.0])]
        if mdl == "TATTw":
            return [np.concatenate([s, [1.0]]) for s in _starts("TATT", nla_sol)]
        return _starts(mdl, nla_sol)
    V.starts = starts

    logf = open(os.path.join(HERE, f"mask_scan{TAG}_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    out, masks_out = [], {}
    for fam, pexp in FAMILIES.items():
        for K0 in K0S:
            kuv = K0*Dz2**(-pexp)
            m = mask_for(kuv)
            masks_out[f"{fam}_{K0:g}"] = m
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            Lw = np.linalg.inv(np.linalg.cholesky(C))
            kz = {z: float(np.interp(z, z2, kuv)) for z in (0.5, 1.0, 2.0)}
            log(f"[{fam} K0 {K0:g}: k_UV(z=0.5,1,2) = {kz[0.5]:.3f}, {kz[1.0]:.3f}, {kz[2.0]:.3f}; {int(m.sum())} points  ({time.time() - t0:.0f}s)]")
            row = dict(family=fam, K0=K0, pexp=pexp, npts=int(m.sum()), kuv_z05=kz[0.5], kuv_z1=kz[1.0], kuv_z2=kz[2.0])
            for case in ("A", "B"):
                free, prior = RT.CASES[case]
                rn = RT.fit(cl, "NLA", d_fl, Ci, Lw, m, free, prior, log, Jcells_S=Jcells_S)
                rt = RT.fit(cl, "TATT", d_fl, Ci, Lw, m, free, prior, log,
                            nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=Jcells_S)
                for mdl, r in (("NLA", rn), ("TATT", rt)):
                    row[f"{mdl}_{case}"] = dict(FoB=r["FoB"], FoB_sig=r.get("FoB_sig"), chi2=r["chi2_bestfit"],
                                               dOm=r["dOm"]/r["sOm"], dS8=r["dS8"]/r["sS8"], at_bound=r["at_bound"],
                                               ia=[r["best"][n] for n in B.IAP[mdl]])
                if case == "A":
                    for mdl in ("NLAw", "TATTw"):
                        rw = RT.fit(cl, mdl, d_fl, Ci, Lw, m, free, prior, log, nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]))
                        row[f"{mdl}_A"] = dict(FoB=rw["FoB"], chi2=rw["chi2_bestfit"], dOm=rw["dOm"]/rw["sOm"],
                                               dS8=rw["dS8"]/rw["sS8"], at_bound=rw["at_bound"], ia=[rw["best"][n] for n in B.IAP[mdl]])
            out.append(row)
            json.dump(out, open(os.path.join(HERE, f"mask_scan{TAG}.json"), "w"), indent=1)
    if not TAG:
        np.savez(os.path.join(HERE, "masks_scan.npz"), z2=z2, D=Dz2, **masks_out)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
