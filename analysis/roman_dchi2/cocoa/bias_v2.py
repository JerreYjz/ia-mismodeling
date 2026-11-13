#!/usr/bin/env python3
"""Figure of bias, revised.

Changes with respect to bias_analysis.py
  1. IA parameters bounded to the chain priors (A1, eta1, A2, eta2 in [-5, 5],
     b_TA in [0, 2]); the Fisher marginalisation uses the flat-prior width
     (range / sqrt 12) for the IA parameters instead of the sigma-50 regulariser.
  2. TATT fitted from several starting points (NLA solution with A2 = b_TA = 0,
     the Part 1a fiducial, two tidal-torquing starts); best chi2 kept; any
     parameter at a bound is flagged.
  3. Nesting test: native TATT with A2 = b_TA = 0 equals native NLA.
  4. B modes. The sim path carries none, so a TATT fit to it is penalised for
     any A2 or b_TA that generates B modes. Variant "EB": FLAMINGO's P_BB
     (white level removed) added to the data through the python Limber B-mode
     xi (validated against cosmolike's own TATT B modes, check_tatt_bb.py).
  5. Truth noise: fits to the raw (unsmoothed) FLAMINGO spectra, and the FoB
     uncertainty propagated from FLAMINGO's error bars through the cell
     Jacobian (analysis/J_cells.npy) and the linearised estimator.
  6. Exact non-linear fit (CAMB + cosmolike at every trial point, no
     linearisation) at one mask as a check.
Case A throughout (Omega_m, A_s free), production masks k_UV = K0/D(z).
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import bias_analysis as B                      # noqa: E402
from cosmolike_xi_ia import KMIN                # noqa: E402

BOUNDS = {"A1": (-5, 5), "eta1": (-5, 5), "A2": (-5, 5), "eta2": (-5, 5), "bTA": (0, 2),
          "omegam": (0.1, 0.9), "As_1e9": (0.5, 5.0), "ns": (0.8, 1.1), "H0": (50, 90), "omegab": (0.03, 0.07)}
IA_SIG = {n: (hi - lo)/np.sqrt(12) for n, (lo, hi) in BOUNDS.items()}
MASKS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def starts(mdl, nla_sol):
    if mdl == "NLA":
        return [np.array(nla_sol), np.array([3.4, 0.9])]
    a1, e1 = nla_sol
    return [np.array([a1, e1, 0.0, 0.0, 0.0]), np.array(B.TATT_FID),
            np.array([a1, e1, 1.0, 0.0, 0.5]), np.array([a1, e1, -1.0, 0.0, 1.0])]


def fit(cl, mdl, d, Ci, Lw, m, free, log, nla_sol=(3.5, 0.68), Jcells_S=None, G_noise=None):
    nia = len(B.IAP[mdl])
    names = free + B.IAP[mdl]
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])
    x_fid = np.concatenate([[B.FID[n] for n in free], np.zeros(nia)])
    nf = len(free)

    def ia_model(ia):
        return cl.model(mdl, B.FID, ia)[m]

    def Jc_at(ia):
        J = np.zeros((m.sum(), nf))
        for i, n in enumerate(free):
            pp, pm = dict(B.FID), dict(B.FID)
            pp[n] += B.STEP[n]
            pm[n] -= B.STEP[n]
            J[:, i] = (cl.model(mdl, pp, ia) - cl.model(mdl, pm, ia))[m]/(2*B.STEP[n])
        return J

    # IA-only, multi-start
    best = None
    for s0 in starts(mdl, nla_sol):
        r0 = least_squares(lambda ia: Lw @ (d[m] - ia_model(ia)), s0, bounds=(lo[nf:], hi[nf:]),
                           x_scale="jac", max_nfev=600)
        c = float(r0.fun @ r0.fun)
        if best is None or c < best[0]:
            best = (c, r0.x)
    chi2_ia, ia0 = best
    x = np.concatenate([[B.FID[n] for n in free], ia0])
    for rep in range(2):
        Jc = Jc_at(x[nf:])

        def resid(x):
            dc = x[:nf] - x_fid[:nf]
            return Lw @ (d[m] - ia_model(x[nf:]) - Jc @ dc)
        r = least_squares(resid, x, bounds=(lo, hi), x_scale="jac", max_nfev=2000)
        x = r.x
    chi2_lin = float(r.fun @ r.fun)
    pc = dict(B.FID)
    pc.update(zip(free, x[:nf]))
    ia = x[nf:]
    res_exact = (d - cl.model(mdl, pc, ia))[m]
    chi2 = float(res_exact @ Ci @ res_exact)
    at_bound = [n for n, v, l, h in zip(names, x, lo, hi) if (v - l) < 1e-3*(h - l) or (h - v) < 1e-3*(h - l)]
    # Fisher at the solution
    Jia = np.zeros((m.sum(), nia))
    for i in range(nia):
        e = np.zeros(nia)
        e[i] = 0.05
        Jia[:, i] = (ia_model(ia + e) - ia_model(ia - e))/0.1
    J = np.hstack([Jc, Jia])
    out = dict(model=mdl, free=free, chi2_ia_only=chi2_ia, chi2_lin=chi2_lin, chi2_bestfit=chi2,
               best=dict(zip(names, x.tolist())), at_bound=at_bound, npts=int(m.sum()))
    s8_fid, s8_fit = cl.s8(B.FID), cl.s8(pc)
    G = np.zeros((2, len(names)))
    G[0, names.index("omegam")] = 1.0
    for i, n in enumerate(free):
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        G[1, i] = (cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n])
    dth = np.array([pc["omegam"] - B.FID["omegam"], s8_fit - s8_fid])
    out.update(dOm=float(dth[0]), dS8=float(dth[1]), S8_fid=float(s8_fid))
    F0 = J.T @ Ci @ J
    for tag, ia_sig in (("flat", [IA_SIG[n] for n in B.IAP[mdl]]), ("reg50", [50.0]*nia)):
        Pinv = np.diag([0.0]*nf + [1/s**2 for s in ia_sig])
        Cov = np.linalg.inv(F0 + Pinv)
        St = G @ Cov @ G.T
        fob = float(np.sqrt(dth @ np.linalg.solve(St, dth)))
        out[tag] = dict(sOm=float(np.sqrt(St[0, 0])), sS8=float(np.sqrt(St[1, 1])),
                        rho=float(St[0, 1]/np.sqrt(St[0, 0]*St[1, 1])), FoB=fob)
        if tag == "flat" and Jcells_S is not None:
            # noise in the truth -> noise in the parameter shift (linearised estimator)
            Best = Cov @ J.T @ Ci                       # d x / d (data)
            Bn = Best @ Jcells_S[m]                       # d x / d u   (u: FLAMINGO residual in sigma units)
            V = G @ Bn @ Bn.T @ G.T                       # cov of the (Omega_m, S8) shift from truth noise
            g = np.linalg.solve(St, dth)/max(fob, 1e-12)
            out[tag].update(FoB_noise_rms=float(np.sqrt(np.trace(np.linalg.solve(St, V)))),
                            FoB_sig=float(np.sqrt(g @ V @ g)))
    f = out["flat"]
    log(f"    {mdl:4s}: chi2 IA-only {chi2_ia:6.2f}, best {chi2:6.2f} (lin {chi2_lin:6.2f}); "
        f"dOm {dth[0]:+.4f} ({dth[0]/f['sOm']:+.2f}s) dS8 {dth[1]:+.4f} ({dth[1]/f['sS8']:+.2f}s); "
        f"FoB {f['FoB']:.2f} (reg50 {out['reg50']['FoB']:.2f})"
        + (f" +- {f['FoB_sig']:.2f} [noise rms {f['FoB_noise_rms']:.2f}]" if "FoB_sig" in f else "")
        + f"; IA {np.round(ia, 3)}" + (f"  AT BOUND {at_bound}" if at_bound else ""))
    return out


def fit_exact(cl, mdl, d, Lw, m, free, log, x0):
    """No linearisation: CAMB + cosmolike at every trial point."""
    names = free + B.IAP[mdl]
    nf = len(free)
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])

    def resid(x):
        p = dict(B.FID)
        p.update(zip(free, x[:nf]))
        return Lw @ (d[m] - cl.model(mdl, p, x[nf:])[m])
    t0 = time.time()
    r = least_squares(resid, np.array(x0), bounds=(lo, hi), x_scale="jac", max_nfev=400,
                      diff_step=[0.02]*nf + [0.02]*(len(names) - nf))
    p = dict(B.FID)
    p.update(zip(free, r.x[:nf]))
    chi2 = float(r.fun @ r.fun)
    log(f"    exact {mdl:4s}: chi2 {chi2:.3f}; x {np.round(r.x, 4)}; {r.nfev} evaluations, {time.time() - t0:.0f}s")
    return dict(model=mdl, chi2=chi2, best=dict(zip(names, r.x.tolist())), dOm=float(p["omegam"] - B.FID["omegam"]),
                dS8=float(cl.s8(p) - cl.s8(B.FID)))


def main():
    data = sys.argv[1]
    t0 = time.time()
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    fits = np.load("../v2/fits.npz")
    zs, sns, kd = fits["zs"], fits["sns"], fits["k"][0]
    cl = B.CL(os.path.join(HERE, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (
        cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    logf = open("analysis/bias_v2_log.txt", "w")
    out = {}

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    def q(v, m):
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    # ---- nesting test
    a = cl.model("NLA", B.FID, [3.5, 0.68]) - xi_gg
    b = cl.model("TATT", B.FID, [3.5, 0.68, 0.0, 0.0, 0.0]) - xi_gg
    log(f"nesting: native TATT(A2=bTA=0) - native NLA: max|diff|/sigma {np.max(np.abs(a - b)/np.sqrt(np.diag(cov))):.2e}, "
        f"chi2(all) {float((a - b) @ np.linalg.solve(cov, a - b)):.2e}")

    # ---- FLAMINGO B modes (python Limber), the TATT best-fit B modes later
    import roman_limber as RL
    import fit_tatt_flamingo as F
    cosmo = RL.Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
    S = RL.RomanShear(cosmo, os.path.join(data, "example1.nz"))
    lo_k = np.geomspace(1e-4, KMIN, 30, endpoint=False)
    hi_k = np.geomspace(kd[-1]*1.02, 2e3, 20)
    mk = kd >= KMIN
    kk = np.concatenate([lo_k, kd[mk], hi_k])
    zero = lambda K, Z: 0*K

    def xbb(RBB):
        t = np.array([np.concatenate([np.full(len(lo_k), r[mk][0]), r[mk], np.zeros(len(hi_k))]) for r in RBB])
        CE, CB = S.cl({"dE": zero, "EE": zero, "BB": RL.ratio_table(zs, kk, t)}, gg=False)
        return S.xi(CE, CB)
    xi_bb_fl = xbb(fits["Rtrue"][:, 2])
    np.save("run/xi_flam_bb.npy", xi_bb_fl)
    log("FLAMINGO B-mode xi (python Limber): chi2 on masks " + ", ".join(
        f"{K0}/D {q(xi_bb_fl, M1[K1.index(K0)]):.2f}" for K0 in MASKS))

    # ---- truth-noise propagation matrix: J_cells (1080 x 520) times sigma per cell (table units)
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

    datasets = {
        "E":    xi_gg + R["xi_flam"],                     # production: smoothed FLAMINGO, E modes
        "EB":   xi_gg + R["xi_flam"] + xi_bb_fl,          # plus FLAMINGO B modes
        "raw":  xi_gg + R["xi_flam_raw"],                 # unsmoothed spectra
    }
    out["fits"] = []
    for K0 in MASKS:
        m = M1[K1.index(K0)]
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        Lw = np.linalg.inv(np.linalg.cholesky(C))
        log(f"mask {K0}/D(z): {int(m.sum())} points  ({time.time() - t0:.0f}s)")
        for dname, d in datasets.items():
            log(f"  data {dname}")
            rn = fit(cl, "NLA", d, Ci, Lw, m, ["omegam", "As_1e9"], log, Jcells_S=Jcells_S)
            rt = fit(cl, "TATT", d, Ci, Lw, m, ["omegam", "As_1e9"], log,
                     nla_sol=(rn["best"]["A1"], rn["best"]["eta1"]), Jcells_S=Jcells_S)
            for r in (rn, rt):
                r.update(data=dname, mask=K0)
                out["fits"].append(r)
            if dname == "E":
                # B modes generated by the TATT best fit (what the E-only data penalise)
                bt = rt["best"]
                x = (1 + zs)/1.62
                mod = F.TATTModel(halofit="takahashi")
                RBB = []
                for s, z in enumerate(zs):
                    pr = mod.predict(z, kd, bt["A1"]*x[s]**bt["eta1"], bt["A2"]*x[s]**bt["eta2"], bt["bTA"], 1.0, Ns=0.0)
                    RBB.append(pr["BB"]/mod.Pnl(z, kd))
                xb = xbb(np.array(RBB))
                rt["bb_chi2_bestfit"] = q(xb, m)
                log(f"    TATT best-fit B modes alone: chi2 {rt['bb_chi2_bestfit']:.3f} on this mask")
        json.dump(out, open("analysis/bias_v2.json", "w"), indent=1)

    # ---- exact non-linear fits at 0.35/D, data E and EB
    log("exact non-linear fits (no linearisation), mask 0.35/D")
    out["exact"] = []
    m = M1[K1.index(0.35)]
    Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
    for dname in ("E", "EB"):
        for mdl in ("NLA", "TATT"):
            r0 = [r for r in out["fits"] if r["data"] == dname and r["mask"] == 0.35 and r["model"] == mdl][0]
            x0 = [r0["best"][n] for n in ["omegam", "As_1e9"] + B.IAP[mdl]]
            re = fit_exact(cl, mdl, datasets[dname], Lw, m, ["omegam", "As_1e9"], log, x0)
            re.update(data=dname, mask=0.35, lin_dOm=r0["dOm"], lin_dS8=r0["dS8"], lin_chi2=r0["chi2_bestfit"])
            log(f"      linearised: chi2 {r0['chi2_bestfit']:.3f}, dOm {r0['dOm']:+.4f}, dS8 {r0['dS8']:+.4f}; "
                f"exact: dOm {re['dOm']:+.4f}, dS8 {re['dS8']:+.4f}")
            out["exact"].append(re)
    json.dump(out, open("analysis/bias_v2.json", "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")
    logf.close()


if __name__ == "__main__":
    main()
