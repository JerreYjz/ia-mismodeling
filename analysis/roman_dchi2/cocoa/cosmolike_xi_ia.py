#!/usr/bin/env python3
"""IA part of Roman xi_+- computed with cosmolike itself (roman_real interface
+ the sim-IA patch), for FLAMINGO and for the NLA / TATT fits.

Every IA spectrum, FLAMINGO's and the models', enters cosmolike the same way:
as a sim_IA table (IA_code = 3: P_dE for GI, P_EE for II), already in shear
units at mu = 0 (no conversion inside cosmolike), per snapshot, on one k grid
1e-4 - 300 h/Mpc, interpolated by sim_IA bilinearly in (z, ln k), linear in P.
So truth and models share the Limber integration, the C_l -> xi transform,
the bin averaging and the n(z): only the input spectra differ. (The sim path
carries no IA B modes; their size is estimated separately in python.)

Tables written (dir per set, files PdE0_sn<sn>_nfold1.dat, PEE0_...):
  FLAMINGO, main ("flam"): smoothed spectra (fits.npz) on 0.05 <= k <= 17.7,
      P_dE = P_hE / b(k), P_EE minus the white shape noise (high-k plateau of
      P_BB); outside the data the ratio to P_NL is held at its edge value.
  FLAMINGO variants: "flam_zero" (nothing above 17.7), "flam_simIA" (only
      the measured rows, sim_IA's own extrapolation: power law below, linear
      in P above), "flam_nsnla" (white level from the NLA-side fit of the
      notes), "flam_nons" (no white subtraction), "flam_raw" (measured, not
      smoothed).
  models: NLA_j, TATT_j, the fits of fits.npz (to the smoothed spectra,
      0.05 <= k <= k_max_j), full model spectra at all k.
  residuals ("res_<model>_<j>_<i>"): model_j - FLAMINGO on 0.05 <= k <= K_i
      only (zero elsewhere), white offset of P_EE re-minimised: the same
      residual whose FLAMINGO Delta chi2 is Delta chi2_FL(k_max_j; K_i).
Masks: IR/UV (alpha 0.05, monotonic rows) exactly as make_uv_scale_cuts.py:
  GG + NLA baseline (A1 0.6, eta -1.5), P_lin and P_NL truncated above
  k_UV(z) (ln P - 70); families k_UV(z) = K (n = 0) and K / D(z) (n = 1).

    python cosmolike_xi_ia.py --fits ../v2/fits.npz --spectra ../spectra.npz \
        --data <cocoa_roman_real>/data --out run
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
from cosmolike_driver import Cosmolike  # noqa: E402

KMIN, NSKG = 0.05, 50          # data k_min; wide grid points per decade
KW = np.logspace(-4, np.log10(300.0), int(NSKG*np.log10(300.0/1e-4)) + 1)
LN_FLOOR = -70.0
ALPHA = 0.05
NTH = 15


def write_set(root, name, sns, PdE, PEE, k=KW):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for s, sn in enumerate(sns):
        for tag, P in (("PdE0", PdE[s]), ("PEE0", PEE[s])):
            np.savetxt(os.path.join(d, f"{tag}_sn{sn}_nfold1.dat"),
                       np.column_stack([k, P]), fmt="%.8e")
    return d


def extend_ratio(kd, R, kw=KW, hi="ratio"):
    """Ratio table on the data grid (k >= KMIN used) -> wide grid: linear in
    ln k inside, edge value below KMIN, edge value ('ratio') or 0 ('zero')
    above the last data k."""
    m = kd >= KMIN
    out = np.interp(np.log(kw), np.log(kd[m]), R[m])
    if hi == "zero":
        out[kw > kd[-1]] = 0.0
    return out


def mask_rows(frac):
    keep = frac < ALPHA
    out = keep.copy()
    for r in range(0, keep.size, NTH):
        fail = np.where(~keep[r:r + NTH])[0]
        if fail.size:
            out[r:r + fail.max() + 1] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--spectra", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="run")
    ap.add_argument("--keep-tables", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tabroot = os.path.join(a.out, "tables")
    d = np.load(a.fits)
    v1 = np.load(a.spectra)
    zs, sns, kd, kg = d["zs"], d["sns"], d["k"][0], d["kgrid"]
    nsn, nK = len(zs), len(kg)

    t0 = time.time()
    cl = Cosmolike(os.path.join(HERE, "lib"), a.data)
    h = cl.h
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])            # wide grid, h units
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])            # data grid
    print(f"cosmolike + CAMB ready ({time.time() - t0:.1f}s), sigma8 {cl.sigma8:.4f}", flush=True)

    import fit_tatt_flamingo as F
    model = F.TATTModel(halofit="takahashi")
    kk = np.clip(KW, 1.001e-4, 25.0)                # FAST-PT kernels tabulated 1e-4 - 30
    Kr, Pm = [], []
    for z in zs:
        Kr.append(model.kernels(z, kk))
        Pm.append(model.Pnl(z, kk))

    def model_ratios(mdl, s, x):
        a1, a2, bTA = (x[0], 0.0, 0.0) if mdl == "NLA" else (x[0], x[1], x[2])
        pr = model.predict(zs[s], kk, a1, a2, bTA, 1.0, K=Kr[s], PNL=Pm[s], Ns=0.0)
        return pr["dE"]/Pm[s], pr["EE"]/Pm[s]

    # ------------------------------------------------------------------
    # FLAMINGO truth tables (ratios to P_NL on the data grid -> wide grid)
    T, b, Pfit = d["T"], d["b"], d["PNL"]                  # observed units; TATTModel P_NL
    Ns_hk, Ns_nla = d["Ns_true"], v1["Ns_nla"]
    y = d["y"]
    sets = {}

    def truth_set(name, src, Ns, hi="ratio", rows_only=False):
        PdE, PEE = [], []
        for s in range(nsn):
            rdE = src[s, 0]/(b[s]*Pfit[s])
            rEE = (src[s, 1] - Ns[s])/Pfit[s]
            if rows_only:
                m = kd >= KMIN
                PdE.append(rdE[m]*PNLd[s][m])
                PEE.append(rEE[m]*PNLd[s][m])
            else:
                PdE.append(extend_ratio(kd, rdE, hi=hi)*PNLw[s])
                PEE.append(extend_ratio(kd, rEE, hi=hi)*PNLw[s])
        k = kd[kd >= KMIN] if rows_only else KW
        sets[name] = write_set(tabroot, name, sns, np.array(PdE), np.array(PEE), k)

    truth_set("flam", T, Ns_hk)
    truth_set("flam_zero", T, Ns_hk, hi="zero")
    truth_set("flam_simIA", T, Ns_hk, rows_only=True)
    truth_set("flam_nsnla", T, Ns_nla)
    truth_set("flam_nons", T, np.zeros(nsn))
    truth_set("flam_raw", y, Ns_hk)

    # model tables
    for mdl in ("NLA", "TATT"):
        for j in range(nK):
            x = d[f"{mdl}_x_true"][:, j]
            PdE, PEE = [], []
            for s in range(nsn):
                rdE, rEE = model_ratios(mdl, s, x[s])
                PdE.append(rdE*PNLw[s])
                PEE.append(rEE*PNLw[s])
            sets[f"{mdl}_{j}"] = write_set(tabroot, f"{mdl}_{j}", sns, np.array(PdE), np.array(PEE))

    # residual tables, matched k range, white offset re-minimised
    err = d["err"]
    for mdl in ("NLA", "TATT"):
        for j in range(nK):
            x = d[f"{mdl}_x_true"][:, j]
            for i, K in enumerate(kg):
                PdE, PEE = [], []
                for s in range(nsn):
                    rdE_m, rEE_m = model_ratios(mdl, s, x[s])
                    # residual in observed units on the data grid
                    m_dE = np.interp(np.log(kd), np.log(KW), rdE_m)*Pfit[s]*b[s]
                    m_EE = np.interp(np.log(kd), np.log(KW), rEE_m)*Pfit[s] + x[s][-1]
                    m_BB = d[f"{mdl}_R"][s, j, 2]*Pfit[s] + x[s][-1]
                    sel = (kd >= KMIN) & (kd <= K*(1 + 1e-9))
                    rE = m_EE - T[s, 1]
                    rB = m_BB - T[s, 2]
                    w = np.concatenate([1/err[s, 1][sel]**2, 1/err[s, 2][sel]**2])
                    c = np.sum(w*np.concatenate([rE[sel], rB[sel]]))/np.sum(w)
                    dE = (m_dE - T[s, 0])/(b[s]*Pfit[s])*PNLd[s]      # P_dE residual, cosmolike P_NL
                    dEE = (rE - c)/Pfit[s]*PNLd[s]
                    keep = sel.astype(float)
                    PdE.append(np.interp(np.log(KW), np.log(kd), dE*keep, left=0, right=0))
                    PEE.append(np.interp(np.log(KW), np.log(kd), dEE*keep, left=0, right=0))
                sets[f"res_{mdl}_{j}_{i}"] = write_set(tabroot, f"res_{mdl}_{j}_{i}", sns,
                                                       np.array(PdE), np.array(PEE))
    print(f"tables written ({len(sets)} sets, {time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------
    # cosmolike runs
    xi_gg = cl.xi_native(ia_model=0, A1=[0.0, 0.0])
    out = {"zs": zs, "sns": sns, "kgrid": kg, "xi_gg": xi_gg, "kw": KW}
    for name, path in sets.items():
        cl.use_sim_tables(path, sns, zs, 1)
        out[f"xi_{name}"] = cl.xi_sim() - xi_gg
    print(f"xi for all sets ({time.time() - t0:.0f}s)", flush=True)

    # native cosmolike NLA / TATT (power law) and the same through sim tables:
    # validates the table conventions against cosmolike's own IA code
    x = (1 + zs)/1.62
    for mdl, par in (("NLA", dict(A1=[3.8, 0.57], A2=[0.0, 0.0], BTA=[0.0])),
                     ("TATT", dict(A1=[3.8, 0.57], A2=[-1.08, -0.71], BTA=[0.18]))):
        native = cl.xi_native(ia_model=0 if mdl == "NLA" else 1, **par) - xi_gg
        PdE, PEE = [], []
        for s in range(nsn):
            a1 = par["A1"][0]*x[s]**par["A1"][1]
            a2 = par["A2"][0]*x[s]**par["A2"][1]
            pr = model.predict(zs[s], kk, a1, a2, par["BTA"][0], 1.0, K=Kr[s], PNL=Pm[s], Ns=0.0)
            PdE.append(pr["dE"]/Pm[s]*PNLw[s])
            PEE.append(pr["EE"]/Pm[s]*PNLw[s])
        p = write_set(tabroot, f"check_{mdl}", sns, np.array(PdE), np.array(PEE))
        cl.use_sim_tables(p, sns, zs, 1)
        out[f"check_{mdl}_native"] = native
        out[f"check_{mdl}_tables"] = cl.xi_sim() - xi_gg
        r = (out[f"check_{mdl}_tables"] - native)
        print(f"check {mdl}: |tables - native| / |native IA|: median "
              f"{np.median(np.abs(r)/np.maximum(np.abs(native), 1e-30)):.4f}", flush=True)

    # ------------------------------------------------------------------
    # IR/UV masks (make_uv_scale_cuts.py recipe), n = 0 and n = 1 families
    cl.ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    hh = cl.h
    k = 10.0**cl.lk/hh                                       # h/Mpc on cosmolike's grid
    z2 = cl.z2
    lnPL = np.log(cl.PKL.P(z2, 10.0**cl.lk)) + np.log(hh**3)
    lnPNL = np.log(cl.PKNL.P(z2, 10.0**cl.lk)) + np.log(hh**3)
    lnPLcb = np.log(cl.PKLcb.P(z2, 10.0**cl.lk)) + np.log(hh**3)
    zg = cl.z1[cl.z1 <= z2[-1]]
    G = np.sqrt(cl.PKL.P(zg, 0.05)/cl.PKL.P(0, 0.05))*(1 + zg)
    G = np.asarray(G/(np.sqrt(cl.PKL.P(z2[-1], 0.05)/cl.PKL.P(0, 0.05))*(1 + z2[-1]))).ravel()
    Dz = np.sqrt(cl.PKL.P(z2, 1e-3)/cl.PKL.P(0, 1e-3)).ravel()

    def dv_with(lnL, lnN):
        cl.ci.set_cosmology(omegam=cl.cosmo["omegam"], omegab=cl.cosmo["omegab"],
                            omegan2=cl.omegan2, H0=cl.cosmo["H0"],
                            log10k_2D=cl.lk - np.log10(hh), z_2D=z2,
                            lnP_linear=lnL.flatten(order="F"),
                            lnP_linear_cb=lnPLcb.flatten(order="F"),
                            lnP_nonlinear=lnN.flatten(order="F"), G=G, z_G=zg, z_1D=cl.z1,
                            chi=cl.res.comoving_radial_distance(cl.z1)*hh)
        cl._nuisance([0.6, -1.5], [0.0], [0.0])
        return cl.xi_vector()

    full = dv_with(lnPL, lnPNL)
    out["xi_mask_baseline"] = full
    for fam, n in (("n0", 0.0), ("n1", 1.0)):
        Kfam = kg if fam == "n0" else np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8])
        masks = []
        for K in Kfam:
            kc = K*Dz**(-n)
            keep = k[None, :] <= kc[:, None]
            ir = dv_with(np.where(keep, lnPL, lnPL + LN_FLOOR), np.where(keep, lnPNL, lnPNL + LN_FLOOR))
            uv = full - ir
            den = np.abs(uv) + np.abs(ir)
            frac = np.where(den > 0, np.abs(uv)/np.where(den > 0, den, 1), 0)
            masks.append(mask_rows(frac))
        out[f"masks_{fam}"] = np.array(masks)
        out[f"K_{fam}"] = Kfam
        print(f"masks {fam}: kept {[int(m.sum()) for m in masks]}", flush=True)
    np.savez(os.path.join(a.out, "cosmolike_xi_ia.npz"), **out)
    if not a.keep_tables:
        shutil.rmtree(tabroot)
    print(f"wrote {a.out}/cosmolike_xi_ia.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
