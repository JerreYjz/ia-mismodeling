#!/usr/bin/env python3
"""Roman-level quantities needed to compare scale-cut strategies.

1. Constraining power of every mask used so far (three K0 families, the
   production K0/D list, the simulation-shaped masks x f): Fisher sigma(Omega_m),
   sigma(S8), FoM = 1/sqrt(det Sigma_(Om,S8)), cases A and B, NLA and TATT
   (IA marginalised with the flat-prior width, as the Roman test). Validated
   against roman_test.json.
2. Linear bias estimator per mask, B = G Cov J^T C^-1 (IA free):
   a. the bias from the white II term alone (template run/xi_white_only.npz,
      amplitude 1 = the measured excess), in sigma and in absolute S8;
   b. tolerances: the bias and Delta chi2 from a coherent fractional error
      eps in the FLAMINGO GI and/or II spectra above k_x (step) or growing as
      ln(k/k_x) above k_x (ramp), propagated with the cell Jacobian J_cells;
      eps_max(k_x) for FoB = 0.3 and for Delta chi2 = 1.
3. Theory-error covariance instead of a cut: C_th = J_cells Sigma J_cells^T
   with Sigma from the measured model-error envelope (run in part 'theory').
    python roman_strategies.py <cocoa data dir> [part: fisher|theory|all]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
V3 = os.path.join(HERE, "..", "v3")
sys.path.insert(0, COCOA)
sys.path.insert(0, V3)
import bias_analysis as B                              # noqa: E402
import bias_v2 as V                                    # noqa: E402
from cosmolike_xi_ia import KMIN                       # noqa: E402

NA = ["omegam", "As_1e9"]


def load_masks():
    out = {}
    s = np.load(os.path.join(V3, "masks_scan.npz"))
    for k in s.files:
        if k not in ("z2", "D"):
            out[f"scan:{k}"] = s[k].astype(bool)
    R = np.load(os.path.join(COCOA, "run", "cosmolike_xi_ia.npz"))
    for K0, m in zip(R["K_n1"], R["masks_n1"]):
        out[f"n1:{K0:g}"] = m.astype(bool)
    z = np.load(os.path.join(V3, "masks_v3.npz"))
    for k in z.files:
        if "GoF_x" in k:
            out[f"simshape:{k}"] = z[k].astype(bool)
    return out


def main():
    data = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    os.chdir(COCOA)
    cov = np.load("analysis/cov_gauss_d3a.npy")
    R = np.load("run/cosmolike_xi_ia.npz")
    rt = json.load(open(os.path.join(V3, "roman_test.json")))
    ref = {t: [x for x in rt if x["test"] == f"{t}_on_FLAMINGO" and x["mask"] == "K0/D 0.3" and x["case"] == "A"][0]
           for t in ("NLA", "TATT")}
    IA_REF = {t: [ref[t]["best"][n] for n in B.IAP[t]] for t in ref}
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    masks = load_masks()

    # ---------------- Jacobians at the fiducial cosmology, reference IA ----------------
    J = {}
    for mdl in ("NLA", "TATT"):
        ia = np.array(IA_REF[mdl])
        cols = []
        for n in B.CP:
            pp, pm = dict(B.FID), dict(B.FID)
            pp[n] += B.STEP[n]
            pm[n] -= B.STEP[n]
            cols.append((cl.model(mdl, pp, ia) - cl.model(mdl, pm, ia))/(2*B.STEP[n]))
        for i in range(len(ia)):
            e = np.zeros(len(ia))
            e[i] = 0.05
            cols.append((cl.model(mdl, B.FID, ia + e) - cl.model(mdl, B.FID, ia - e))/0.1)
        J[mdl] = np.array(cols).T
        print(f"Jacobian {mdl} done ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(HERE, "jacobians.npz"), NLA=J["NLA"], TATT=J["TATT"])
    gS8 = []
    for n in B.CP:
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        gS8.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
    gS8 = np.array(gS8)
    s8fid = cl.s8(B.FID)

    TPL = np.load("run/xi_white_only.npz")["ratio_ratio"]
    J_cells = np.load("analysis/J_cells.npy")
    fits = np.load("../v2/fits.npz")
    zs, kd = fits["zs"], fits["k"][0]
    T, bk, Pf, err, Ns = fits["T"], fits["b"], fits["PNL"], fits["err"], fits["Ns_true"]
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    cells = [(s, x, i) for s in range(len(zs)) for x in range(2) for i in nodes]
    Tc = np.array([T[s, 0, i]*PNLd[s, i]/(bk[s, i]*Pf[s, i]) if x == 0 else (T[s, 1, i] - Ns[s])*PNLd[s, i]/Pf[s, i]
                   for (s, x, i) in cells])
    sc = np.array([err[s, 0, i]*PNLd[s, i]/(bk[s, i]*Pf[s, i]) if x == 0 else err[s, 1, i]*PNLd[s, i]/Pf[s, i]
                   for (s, x, i) in cells])
    cs = np.array([s for (s, x, i) in cells])
    cx = np.array([x for (s, x, i) in cells])
    ck = np.array([kd[i] for (s, x, i) in cells])
    chk = J_cells @ Tc
    big = np.abs(R["xi_flam"]) > 0.05*np.abs(R["xi_flam"]).max()
    print(f"J_cells . T vs xi_flam (|xi| > 5% of max): median ratio {np.median(chk[big]/R['xi_flam'][big]):.3f}", flush=True)

    def setup(mdl, case, m):
        names = (NA if case == "A" else B.CP) + B.IAP[mdl]
        idx = [B.CP.index(n) for n in (NA if case == "A" else B.CP)] + [5 + i for i in range(len(B.IAP[mdl]))]
        Jm = J[mdl][m][:, idx]
        C = cov[np.ix_(m, m)]
        Ci = np.linalg.inv(C)
        pr = [0.0]*2 if case == "A" else [0.0, 0.0] + [1/B.PRIOR_B[n]**2 for n in ("ns", "H0", "omegab")]
        pr += [1/V.IA_SIG[n]**2 for n in B.IAP[mdl]]
        F = Jm.T @ Ci @ Jm + np.diag(pr)
        Cov = np.linalg.inv(F)
        nc = 2 if case == "A" else 5
        G = np.zeros((2, len(names)))
        G[0, 0] = 1.0
        G[1, :nc] = gS8[[B.CP.index(n) for n in names[:nc]]]
        St = G @ Cov @ G.T
        Blin = G @ Cov @ Jm.T @ Ci                        # 2 x n: data residual -> (dOm, dS8)
        return dict(C=C, Ci=Ci, St=St, Blin=Blin, Jm=Jm, Cov=Cov)

    def fob(St, d):
        return float(np.sqrt(d @ np.linalg.solve(St, d)))

    out = dict(IA_REF=IA_REF, s8fid=float(s8fid))
    if part in ("fisher", "all"):
        fis = {}
        for key, m in masks.items():
            row = dict(npts=int(m.sum()))
            for mdl in ("NLA", "TATT"):
                for case in ("A", "B"):
                    S = setup(mdl, case, m)
                    St = S["St"]
                    dw = S["Blin"] @ TPL[m]
                    row[f"{mdl}_{case}"] = dict(sOm=float(np.sqrt(St[0, 0])), sS8=float(np.sqrt(St[1, 1])),
                                                FoM=float(1/np.sqrt(np.linalg.det(St))),
                                                white_dOm=float(dw[0]), white_dS8=float(dw[1]), white_FoB=fob(St, dw))
            fis[key] = row
        out["fisher"] = fis
        # validation against roman_test.json
        val = []
        for x in rt:
            if x["test"] in ("NLA_on_FLAMINGO", "TATT_on_FLAMINGO") and x["mask"].startswith("K0/D"):
                K0 = float(x["mask"].split()[-1])
                mdl = x["test"].split("_")[0]
                f = fis[f"n1:{K0:g}"][f"{mdl}_{x['case']}"]
                val.append((x["mask"], mdl, x["case"], x["sS8"], f["sS8"], x["sOm"], f["sOm"]))
        out["validation"] = val
        for v in val:
            print(f"  {v[0]:10s} {v[1]:4s} {v[2]}: sS8 fit {v[3]:.4f} fisher {v[4]:.4f}; sOm {v[5]:.4f} / {v[6]:.4f}")
        print(f"fisher done ({time.time() - t0:.0f}s)", flush=True)

        # ---------------- tolerances ----------------
        tol = {}
        kxs = kd[(kd >= 0.09) & (kd <= 1.3)]
        for key in ("n1:0.2", "n1:0.3", "n1:0.4", "scan:flat_0.3", "scan:flat_0.5"):
            m = masks[key]
            tol[key] = {}
            for mdl in ("NLA", "TATT"):
                S = setup(mdl, "A", m)
                P = np.eye(int(m.sum())) - S["Jm"] @ np.linalg.solve(S["Jm"].T @ S["Ci"] @ S["Jm"], S["Jm"].T @ S["Ci"])
                rows = []
                for kx in kxs:
                    for shape in ("step", "ramp"):
                        for which in ("GI", "II", "both"):
                            sel = (ck >= kx*(1 - 1e-9)) & ((cx == 0) if which == "GI" else (cx == 1) if which == "II" else True)
                            amp = np.where(sel, 1.0 if shape == "step" else np.log(np.maximum(ck/kx, 1.0)), 0.0)
                            v = (J_cells @ (amp*Tc))[m]
                            d = S["Blin"] @ v
                            dchi = float(v @ S["Ci"] @ v)
                            vr = P @ v
                            dchi_r = float(vr @ S["Ci"] @ vr)
                            fb = fob(S["St"], d)
                            rows.append(dict(kx=float(kx), shape=shape, which=which, FoB1=fb, dS8_1=float(d[1]/np.sqrt(S["St"][1, 1])),
                                             dchi2_1=dchi, dchi2_refit_1=dchi_r,
                                             eps_FoB03=0.3/fb if fb > 0 else np.inf,
                                             eps_dchi1=1/np.sqrt(dchi) if dchi > 0 else np.inf,
                                             eps_dchi1_refit=1/np.sqrt(dchi_r) if dchi_r > 0 else np.inf))
                tol[key][mdl] = rows
        out["tolerance"] = tol
        print(f"tolerances done ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(HERE, "roman_strategies.json"), "w"), indent=1)

    if part in ("theory", "all"):
        import roman_test as RT
        d_fl = cl.model("NLA", B.FID, [0.0, 0.0]) + R["xi_flam"]
        # model-error envelope from the simulation: |M/T - 1| of the fit to k <= anchor,
        # max over core snapshots, made monotonic in k (cumulative max), per probe
        sys.path.insert(0, HERE)
        import simfit as SF
        env = {}
        for mdl, ka in (("NLA", 0.1615), ("TATT", 0.2908)):
            dev = np.zeros((2, len(kd)))
            for s in np.where(SF.core)[0]:
                o = SF.fit(s, SF.y[s], mdl, ka)
                M = SF.pred(s, *o["x"])
                dev[0] = np.maximum(dev[0], np.abs(M[0]/T[s, 0] - 1))
                dev[1] = np.maximum(dev[1], np.abs((M[1] - Ns[s])/(T[s, 1] - Ns[s]) - 1))
            dev = np.maximum.accumulate(dev, axis=1)
            env[mdl] = dev
        out["envelope"] = {mdl: env[mdl].tolist() for mdl in env}
        lnk = np.log(ck)
        corr_k = np.exp(-0.5*(lnk[:, None] - lnk[None, :])**2/0.5**2)
        corr_z = np.exp(-0.5*(zs[cs][:, None] - zs[cs][None, :])**2/0.5**2)
        same_x = (cx[:, None] == cx[None, :])
        theory = []
        logf = open(os.path.join(HERE, "roman_theory_log.txt"), "w")

        def log(s):
            print(s, flush=True)
            logf.write(s + "\n")
            logf.flush()
        for mdl in ("NLA", "TATT"):
            amp = np.array([env[mdl][x, i] for (s, x, i) in cells])*np.abs(Tc)
            Sig = np.outer(amp, amp)*corr_k*corr_z*same_x
            Cth = J_cells @ Sig @ J_cells.T
            for key in ("n1:0.3", "n1:0.4", "n1:0.5", "scan:flat_0.5"):
                m = masks[key]
                for lab, Cm in (("stat", cov), ("stat+theory", cov + Cth)):
                    C = Cm[np.ix_(m, m)]
                    Ci = np.linalg.inv(C)
                    Lw = np.linalg.inv(np.linalg.cholesky(C))
                    log(f"[{mdl} {key} {lab}: {int(m.sum())} pts ({time.time() - t0:.0f}s)]")
                    r = RT.fit(cl, mdl, d_fl, Ci, Lw, m, NA, {}, log, nla_sol=(IA_REF["NLA"][0], IA_REF["NLA"][1]))
                    theory.append(dict(model=mdl, mask=key, cov=lab, npts=int(m.sum()), FoB=r["FoB"], dOm=r["dOm"], dS8=r["dS8"],
                                       sOm=r["sOm"], sS8=r["sS8"], chi2=r["chi2_bestfit"], ia=[r["best"][n] for n in B.IAP[mdl]],
                                       at_bound=r["at_bound"]))
                    json.dump(theory, open(os.path.join(HERE, "roman_theory.json"), "w"), indent=1)
        out["theory"] = theory
        prev = json.load(open(os.path.join(HERE, "roman_strategies.json"))) if os.path.exists(os.path.join(HERE, "roman_strategies.json")) else {}
        prev.update({k: v for k, v in out.items() if k in ("envelope", "theory")})
        json.dump(prev, open(os.path.join(HERE, "roman_strategies.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
