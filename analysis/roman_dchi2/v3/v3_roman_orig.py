#!/usr/bin/env python3
"""Roman xi_+- with redshift-dependent IR/UV masks built from the FLAMINGO
cuts of v3_flamingo.py.

k_UV(z): the per-snapshot cut (core snapshots, interpolated linearly in ln k
between snapshots, flat outside), times a scale factor f. Masks follow the
make_uv_scale_cuts.py recipe (GG + NLA baseline, alpha 0.05, P truncated
above k_UV(z)), exactly as the production K0/D(z) masks.

For each mask (cut shape x f):
  * Delta chi2_Roman of the FLAMINGO truth against the model with the
    per-snapshot amplitudes refitted at k_max = f x cut(z_s) (N_s fixed,
    bounded), through the sim tables;
  * figure of bias: data = GG + FLAMINGO, model = native NLA / TATT with
    power-law evolution, Omega_m and A_s free (bias_v2.fit), with the
    FLAMINGO-noise uncertainty;
  * at f = 1 and 2 also with the free white II template.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B                                              # noqa: E402
import bias_v2 as V                                                    # noqa: E402
from cosmolike_xi_ia import KW, KMIN, LN_FLOOR, write_set, mask_rows   # noqa: E402
import fit_tatt_flamingo as F                                          # noqa: E402

FACTORS = (0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
SHAPES = {"NLA_GoF": ("NLA", "GoF"), "TATT_GoF": ("TATT", "GoF")}


def main():
    data = sys.argv[1]
    t0 = time.time()
    os.chdir(COCOA)
    v3 = np.load(os.path.join(HERE, "v3.npz"))
    fits = np.load("../v2/fits.npz")
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    zs, sns, kd, kg = v3["zs"], v3["sns"], fits["k"][0], v3["kgrid"]
    core = v3["core"]
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]
    logf = open(os.path.join(HERE, "roman_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    # ---------------- mask machinery (as cosmolike_xi_ia.py) ----------------
    cl.ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    k = 10.0**cl.lk/h
    z2 = cl.z2
    lnPL = np.log(cl.PKL.P(z2, 10.0**cl.lk)) + np.log(h**3)
    lnPNL = np.log(cl.PKNL.P(z2, 10.0**cl.lk)) + np.log(h**3)
    lnPLcb = np.log(cl.PKLcb.P(z2, 10.0**cl.lk)) + np.log(h**3)
    zg = cl.z1[cl.z1 <= z2[-1]]
    G = np.sqrt(cl.PKL.P(zg, 0.05)/cl.PKL.P(0, 0.05))*(1 + zg)
    G = np.asarray(G/(np.sqrt(cl.PKL.P(z2[-1], 0.05)/cl.PKL.P(0, 0.05))*(1 + z2[-1]))).ravel()

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

    def kuv_of_z(cut_snap):
        """Per-snapshot cut (core, nan filled from neighbours) -> k_UV on z2."""
        c = np.array(cut_snap, float)
        zc, cc = zs[core], c[core]
        ok = np.isfinite(cc)
        cc = np.interp(zc, zc[ok], cc[ok])
        return np.exp(np.interp(z2, zc, np.log(cc)))

    # ---------------- model tables on the wide grid ----------------
    model = F.TATTModel(halofit="takahashi")
    kk = np.clip(KW, 1.001e-4, 25.0)
    Kr = [model.kernels(z, kk) for z in zs]
    Pm = [model.Pnl(z, kk) for z in zs]
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])

    def model_xi(mdl, jsel):
        """xi_IA of the model with the per-snapshot refit amplitudes at k_max index jsel[s]."""
        PdE, PEE = [], []
        for s in range(len(zs)):
            x = v3[f"{mdl}_x"][s, jsel[s]]
            pr = model.predict(zs[s], kk, x[0], x[1], x[2], 1.0, K=Kr[s], PNL=Pm[s], Ns=0.0)
            PdE.append(pr["dE"]/Pm[s]*PNLw[s])
            PEE.append(pr["EE"]/Pm[s]*PNLw[s])
        p = write_set("run/tables_v3", f"{mdl}_{jsel[0]}_{jsel[6]}", sns, np.array(PdE), np.array(PEE))
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
        return cl.xi_sim() - xi_gg

    def q(v, m):
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    # Jacobian x sigma for the noise propagation (as bias_v2)
    J_cells = np.load("analysis/J_cells.npy")
    T, bk, Pf, err = fits["T"], fits["b"], fits["PNL"], fits["err"]
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

    results = []
    masks_out = {}
    for shape, (mdl_cut, crit) in SHAPES.items():
        cut = v3[f"cut_{mdl_cut}_{crit}"]
        base = kuv_of_z(cut)
        for fct in FACTORS:
            kuv = fct*base
            m = mask_for(kuv)
            masks_out[f"{shape}_x{fct:g}"] = m
            kz1 = float(np.exp(np.interp(1.0, z2, np.log(kuv))))
            kz05 = float(np.exp(np.interp(0.5, z2, np.log(kuv))))
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            Lw = np.linalg.inv(np.linalg.cholesky(C))
            log(f"[{shape} x {fct:g}: k_UV(z=0.5) {kz05:.3f}, k_UV(z=1) {kz1:.3f}; {int(m.sum())} points  ({time.time() - t0:.0f}s)]")
            row = dict(shape=shape, factor=fct, kuv_z05=kz05, kuv_z1=kz1, npts=int(m.sum()))
            # per-snapshot refit amplitudes at k_max = f x cut(z_s)
            jsel = [int(np.argmin(np.abs(kg - fct*(c if np.isfinite(c) else np.nanmedian(cut[core]))))) for c in cut]
            for mdl in ("NLA", "TATT"):
                xi_m = model_xi(mdl, jsel)
                row[f"dchi2_{mdl}"] = q(xi_m - R["xi_flam"], m)
                r = V.fit(cl, mdl, d_fl, Ci, Lw, m, ["omegam", "As_1e9"], log, Jcells_S=Jcells_S)
                row[f"FoB_{mdl}"] = r["flat"]["FoB"]
                row[f"FoBsig_{mdl}"] = r["flat"]["FoB_sig"]
                row[f"chi2_{mdl}"] = r["chi2_bestfit"]
                row[f"dS8_{mdl}"] = r["dS8"]/r["flat"]["sS8"]
                row[f"dOm_{mdl}"] = r["dOm"]/r["flat"]["sOm"]
                row[f"ia_{mdl}"] = [r["best"][n] for n in B.IAP[mdl]]
                if fct in (1.0, 2.0):
                    rw = V.fit(cl, mdl + "w", d_fl, Ci, Lw, m, ["omegam", "As_1e9"], log)
                    row[f"FoBw_{mdl}"] = rw["flat"]["FoB"]
                    row[f"chi2w_{mdl}"] = rw["chi2_bestfit"]
            log(f"    Delta chi2_Roman (FLAMINGO vs refit model): NLA {row['dchi2_NLA']:.2f}, TATT {row['dchi2_TATT']:.2f}")
            results.append(row)
            json.dump(results, open(os.path.join(HERE, "roman_results.json"), "w"), indent=1)
    np.savez(os.path.join(HERE, "masks_v3.npz"), z2=z2, **masks_out,
             **{f"kuv_{shape}": kuv_of_z(v3[f"cut_{mc}_{cr}"]) for shape, (mc, cr) in SHAPES.items()})
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
