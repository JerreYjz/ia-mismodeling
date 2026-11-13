#!/usr/bin/env python3
"""Roman scale cuts for the halo model with the full 2-halo physics ("HM"):
red centrals NLA + blue centrals tidal torquing + density weighting (cosmolike
TATT: A1, eta1, A2, eta2, b_TA) + red satellites (Fortuna+21 1-halo, b = -2,
core 0.06, cap 0.3, f_1h window: a1h, eta1h; templates2_halo.npz).

Same machinery and settings as the NLA / TATT production scan (v3/mask_scan.py
with roman_test.fit): IR/UV masks k_UV(z) = K0/D(z) (D1) and K0 (flat), data =
GG + FLAMINGO, every parameter free, case A (Omega_m, A_s) and B (+ n_s, H0,
Omega_b with the chain priors), IA exact at the fiducial cosmology and the
cosmology response J_c of the full model at the IA solution, FoB noise from
J_cells. NLA and TATT are refitted on the same masks for a like-for-like
comparison; HM starts from the TATT solution. The satellite templates are
held at the fiducial cosmology (the satellite term is small; see log).
    python hm_scan.py <cocoa data dir>    env COV, FAMS (D1,flat), K0S, OUTTAG
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "..", "v3")
COCOA = os.path.join(HERE, "..", "cocoa")
sys.path.insert(0, COCOA)
sys.path.insert(0, V3)
import bias_analysis as B                                             # noqa: E402
import bias_v2 as V                                                   # noqa: E402
from cosmolike_xi_ia import KMIN, LN_FLOOR, mask_rows                 # noqa: E402
import roman_test as RT                                               # noqa: E402

K0S = tuple(float(x) for x in os.environ.get("K0S", "0.15,0.2,0.25,0.3,0.325,0.35,0.375,0.4,0.45,0.5").split(","))
FAMILIES = {"D1": 1.0, "flat": 0.0}
if os.environ.get("FAMS"):
    FAMILIES = {k: v for k, v in FAMILIES.items() if k in os.environ["FAMS"].split(",")}
TAG = os.environ.get("OUTTAG", "_ex1")
MODELS = os.environ.get("MODELS", "NLA,TATT,HM").split(",")


def main():
    data = sys.argv[1]
    t0 = time.time()
    os.chdir(COCOA)
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load(os.environ.get("COV", "analysis/cov_example1_xi.npy"))
    fits = np.load("../v2/fits.npz")
    zs, kd = fits["zs"], fits["k"][0]
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]

    # ---------------- masks (as v3/mask_scan.py) ----------------
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
    S0 = np.load(os.path.join(V3, "masks_scan.npz"))
    m_chk = mask_for(0.35*Dz2**(-1.0))
    assert np.array_equal(m_chk, S0["D1_0.35"].astype(bool)), "mask machinery differs from the production scan"

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

    # ---------------- the HM model ----------------
    T = dict(np.load(os.path.join(HERE, "templates2_halo.npz")))
    tab = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
    xz = (1 + tab["zs"])/1.62
    la = np.log(np.maximum(tab["a_grid"], 1e-6))

    def sat_xi(a1h, e1h):
        """Satellite 1-halo xi: sum_s [a T_GI(|a|) + a^2 T_II(|a|)], odd in a for
        GI (so finite differences at a1h = 0 are two-sided)."""
        a = a1h*xz**e1h
        v = np.zeros(T["GI1h"].shape[-1])
        for s in range(len(xz)):
            aa = min(abs(a[s]), np.exp(la[-1]))
            if aa <= 0:
                continue
            xx = np.log(max(aa, 1e-6))
            j = int(np.clip(np.searchsorted(la, xx), 1, la.size - 1))
            w = np.clip((xx - la[j - 1])/(la[j] - la[j - 1]), 0, 1)
            g = T["GI1h"][s, j - 1]*(1 - w) + T["GI1h"][s, j]*w
            q = T["II1h"][s, j - 1]*(1 - w) + T["II1h"][s, j]*w
            v = v + np.sign(a[s])*aa*g + aa**2*q
        return v

    B.IAP["HM"] = ["A1", "eta1", "A2", "eta2", "bTA", "a1h", "eta1h"]
    V.BOUNDS["a1h"] = (0.0, 1.0)
    V.BOUNDS["eta1h"] = (-5.0, 5.0)
    V.IA_SIG["a1h"] = 1.0/np.sqrt(12)
    V.IA_SIG["eta1h"] = 10.0/np.sqrt(12)
    _orig = B.CL.model

    def model_hm(self, mdl, p, ia):
        if mdl == "HM":
            return _orig(self, "TATT", p, list(ia[:5])) + sat_xi(ia[5], ia[6])
        return _orig(self, mdl, p, ia)
    B.CL.model = model_hm
    _starts = V.starts
    TATT_SOL = {}

    def starts(mdl, nla_sol):
        if mdl == "HM":
            base = TATT_SOL.get("x", np.array([nla_sol[0], nla_sol[1], 0.0, 0.0, 0.2]))
            return [np.concatenate([base, s]) for s in ([1e-3, 0.0], [0.05, 0.0], [0.3, -2.0])]
        return _starts(mdl, nla_sol)
    V.starts = starts

    logf = open(os.path.join(HERE, f"hm_scan{TAG}_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    path = os.path.join(HERE, f"hm_scan{TAG}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["family"], r["K0"]) for r in out}
    for fam, pexp in FAMILIES.items():
        for K0 in K0S:
            if (fam, K0) in done:
                continue
            kuv = K0*Dz2**(-pexp)
            m = mask_for(kuv)
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            Lw = np.linalg.inv(np.linalg.cholesky(C))
            kz = {z: float(np.interp(z, z2, kuv)) for z in (0.5, 1.0, 2.0)}
            log(f"[{fam} K0 {K0:g}: k_UV(z=0.5,1,2) = {kz[0.5]:.3f}, {kz[1.0]:.3f}, {kz[2.0]:.3f}; {int(m.sum())} points  ({time.time() - t0:.0f}s)]")
            row = dict(family=fam, K0=K0, pexp=pexp, npts=int(m.sum()), kuv_z05=kz[0.5], kuv_z1=kz[1.0], kuv_z2=kz[2.0])
            for case in ("B", "A"):
                free, prior = RT.CASES[case]
                rn = RT.fit(cl, "NLA", d_fl, Ci, Lw, m, free, prior, log, Jcells_S=Jcells_S)
                res = {"NLA": rn}
                if "TATT" in MODELS or "HM" in MODELS:
                    rt = RT.fit(cl, "TATT", d_fl, Ci, Lw, m, free, prior, log,
                                nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=Jcells_S)
                    res["TATT"] = rt
                    TATT_SOL["x"] = np.array([rt["best"][n] for n in B.IAP["TATT"]])
                if "HM" in MODELS:
                    res["HM"] = RT.fit(cl, "HM", d_fl, Ci, Lw, m, free, prior, log,
                                       nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=Jcells_S)
                for mdl, r in res.items():
                    row[f"{mdl}_{case}"] = dict(FoB=r["FoB"], FoB_sig=r.get("FoB_sig"), chi2=r["chi2_bestfit"],
                                               dOm=r["dOm"]/r["sOm"], dS8=r["dS8"]/r["sS8"], sS8=r["sS8"],
                                               at_bound=r["at_bound"], ia=[r["best"][n] for n in B.IAP[mdl]])
                out = [o for o in out if not (o["family"] == fam and o["K0"] == K0)] + [row]
                json.dump(out, open(path, "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
