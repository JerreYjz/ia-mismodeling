#!/usr/bin/env python3
"""Roman figure of bias for the halo-model IA (and NLA through the same path).

The IA part of xi_+- is built from per-snapshot, per-component templates
computed with cosmolike's sim_IA tables at the fiducial cosmology (xi is
linear in the tables):
  T_GI2h[s] : P_dE = -C1_s f_rc,s P_NL f_2h        (amplitude A_s)
  T_II2h[s] : P_EE =  C1_s^2 f_rc,s^2 P_NL f_2h    (A_s^2)
  T_GI1h[s, a] : P_dE = -f_1h S_dI(a)              (a_s)
  T_II1h[s, a] : P_EE =  f_1h S_II(a)              (a_s^2)
  xi_IA = sum_s [A_s T_GI2h + A_s^2 T_II2h + a_s T_GI1h(a_s) + a_s^2 T_II1h(a_s)],
  A_s = A1 x_s^eta1, a_s = a1h x_s^eta1h, x = (1+z)/1.62,
with C1_s = 0.01389 Om / D(z_s) and P_NL from cosmolike (as the native NLA).
NLA through the same path: f_rc = f_2h = 1, no 1-halo term ("NLA_tab").
Cosmology: xi = xi_GG(fid) + J_c (p_c - p_fid) + xi_IA, J_c from the
lensing-only model; the IA templates are held at the fiducial cosmology
(approximation quantified by comparing NLA_tab with the native NLA fits).
Fit: least squares over cosmology (case A: Omega_m, A_s; case B: + n_s, H0,
Omega_b with Gaussian priors) and IA (A1, eta1 in [-5, 5]; a1h in [0, 1],
eta1h in [-5, 5]); Fisher with IA flat-prior widths; FoB in (Omega_m, S8);
FLAMINGO-noise uncertainty through J_cells.
    python halo_roman.py <cocoa data dir> [cov npy] [tag]
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
COCOA = os.path.join(HERE, "..", "cocoa")
V3 = os.path.join(HERE, "..", "v3")
sys.path.insert(0, COCOA)
sys.path.insert(0, HERE)
import bias_analysis as B                                 # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set           # noqa: E402
import halo_ia as H                                       # noqa: E402

BOUNDS = {"A1": (-5, 5), "eta1": (-5, 5), "a1h": (0, 1), "eta1h": (-5, 5), "A2": (-5, 5), "eta2": (-5, 5), "bTA": (0, 2)}
# the halo model's A1 is the amplitude of the red centrals, A1_NLA / f_rc with
# f_rc = 0.63 to 0.79 over the core: FLAMINGO needs A1 = 5 to 9, outside the
# NLA chain range, so the halo models get A1 in [-10, 10]
BOUNDS_HALO_A1 = (-10, 10)
IA_SIG = {n: (hi - lo)/np.sqrt(12) for n, (lo, hi) in BOUNDS.items()}
IAP = {"NLA_tab": ["A1", "eta1"], "halo": ["A1", "eta1", "a1h", "eta1h"], "halo_nowin": ["A1", "eta1", "a1h", "eta1h"]}


def templates(cl, tab, nowin, tag):
    """Per-snapshot templates (xi_sim - xi_gg) on the full 1080 vector."""
    cache = os.path.join(HERE, f"templates2_{tag}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    zs = tab["zs"]
    sns = np.load("../v2/fits.npz")["sns"]
    h = cl.h
    cl.at(B.FID)
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    k_all = tab["k_all"]
    idx = np.searchsorted(k_all, KW)
    assert np.allclose(k_all[idx], KW)
    nsn, na = len(zs), len(tab["a_grid"])
    f1, f2 = H.f1h(KW, nowin), H.f2h(KW, nowin)
    out = dict(GI2h=np.zeros((nsn, 1080)), II2h=np.zeros((nsn, 1080)),
               GI1h=np.zeros((nsn, na, 1080)), II1h=np.zeros((nsn, na, 1080)))
    zero = np.zeros((nsn, len(KW)))

    def run(PdE, PEE):
        """xi_sim(tables) - xi_sim(zero tables), both in the same state: the
        baseline is the sim path's own lensing-only vector, so templates sum
        exactly (subtracting the native xi_gg leaves a tiny common offset in
        every template, which the amplitude-weighted sum amplifies)."""
        p = write_set("run/tables_halo", f"tpl_{tag}", sns, PdE, PEE)
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
        v = cl.xi_sim()
        p0 = write_set("run/tables_halo", f"tpl_{tag}_zero", sns, zero, zero)
        cl.use_sim_tables(p0, sns, [float(z) for z in zs], 1)
        return v - cl.xi_sim()
    for s, z in enumerate(zs):
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))
        C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
        P = cl.PKNL.P(z, KW*h)*h**3
        frc = 1.0 if tag == "nla" else tab["hod"][s, 5]
        w2 = np.ones_like(KW) if tag == "nla" else f2
        a = zero.copy(); a[s] = -C1*frc*P*w2
        out["GI2h"][s] = run(a, zero)
        b = zero.copy(); b[s] = (C1*frc)**2*P*w2
        out["II2h"][s] = run(zero, b)
        if tag != "nla":
            for ia in range(na):
                a = zero.copy(); a[s] = -f1*tab["SdI"][s, ia][idx]
                out["GI1h"][s, ia] = run(a, zero)
                b = zero.copy(); b[s] = f1*tab["SII"][s, ia][idx]
                out["II1h"][s, ia] = run(zero, b)
        print(f"  templates {tag}: snapshot {s} done", flush=True)
    np.savez(cache, **out)
    return out


class IAModel:
    def __init__(self, T, tab, mdl):
        self.T, self.mdl = T, mdl
        self.x = (1 + tab["zs"])/1.62
        self.la = np.log(np.maximum(tab["a_grid"], 1e-6))

    def xi(self, ia):
        A = ia[0]*self.x**ia[1]
        v = A @ self.T["GI2h"] + (A**2) @ self.T["II2h"]
        if self.mdl != "NLA_tab":
            a = np.clip(ia[2]*self.x**ia[3], 0.0, np.exp(self.la[-1]))
            for s in range(len(self.x)):
                if a[s] <= 0:
                    continue
                xx = np.log(max(a[s], 1e-6))
                j = int(np.clip(np.searchsorted(self.la, xx), 1, self.la.size - 1))
                w = np.clip((xx - self.la[j - 1])/(self.la[j] - self.la[j - 1]), 0, 1)
                g = self.T["GI1h"][s, j - 1]*(1 - w) + self.T["GI1h"][s, j]*w
                q = self.T["II1h"][s, j - 1]*(1 - w) + self.T["II1h"][s, j]*w
                v = v + a[s]*g + a[s]**2*q
        return v


def fit(cl, M, d, xi_gg, Jc_all, gS8, cov, m, case, Jcells_S, log):
    nc = 2 if case == "A" else 5
    cp = B.CP[:nc]
    names = cp + IAP[M.mdl]
    Jc = Jc_all[:, :nc][m]
    C = cov[np.ix_(m, m)]
    Ci = np.linalg.inv(C)
    Lw = np.linalg.inv(np.linalg.cholesky(C))
    psig = np.array([np.inf, np.inf] + ([B.PRIOR_B[n] for n in ("ns", "H0", "omegab")] if case == "B" else []))
    bnd = dict(BOUNDS)
    if M.mdl.startswith("halo"):
        bnd["A1"] = BOUNDS_HALO_A1
    lo = np.array([-np.inf]*nc + [bnd[n][0] for n in IAP[M.mdl]])
    hi = np.array([np.inf]*nc + [bnd[n][1] for n in IAP[M.mdl]])
    base = (d - xi_gg)[m]

    def resid(x):
        r = Lw @ (base - Jc @ x[:nc] - M.xi(x[nc:])[m])
        pr = (x[:nc]/psig)[np.isfinite(psig)]
        return np.concatenate([r, pr])
    starts = getattr(M, "starts", None) or [[3.5, 0.7]] if M.mdl == "NLA_tab" else getattr(M, "starts", None) or [[5.3, 0.7, 1e-3, 0.0], [5.3, 0.7, 0.05, 0.0], [5.3, 0.7, 0.3, 0.0], [5.3, 0.7, 0.9, 2.0], [5.3, 0.7, 0.01, -3.0], [3.5, 0.7, 0.02, 0.0]]
    best = None
    for s0 in starts:
        x0 = np.concatenate([np.zeros(nc), s0])
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    res = (base - Jc @ x[:nc] - M.xi(x[nc:])[m])
    chi2 = float(res @ Ci @ res)
    nia = len(IAP[M.mdl])
    Jia = np.zeros((int(m.sum()), nia))
    for i in range(nia):
        e = np.zeros(nia)
        st = 0.02 if IAP[M.mdl][i] != "a1h" else max(1e-3, 0.05*abs(x[nc + i]))
        e[i] = st
        xp, xm_ = x[nc:] + e, x[nc:] - e
        xm_[i] = max(xm_[i], lo[nc + i]) if np.isfinite(lo[nc + i]) else xm_[i]
        Jia[:, i] = (M.xi(xp)[m] - M.xi(xm_)[m])/(xp[i] - xm_[i])
    J = np.hstack([Jc, Jia])
    Pinv = np.diag([0.0 if not np.isfinite(v) else 1/v**2 for v in psig] + [12/(bnd[n][1] - bnd[n][0])**2 for n in IAP[M.mdl]])
    Cov = np.linalg.inv(J.T @ Ci @ J + Pinv)
    G = np.zeros((2, len(names)))
    G[0, 0] = 1.0
    G[1, :nc] = gS8[:nc]
    St = G @ Cov @ G.T
    pc = dict(B.FID)
    for i, n in enumerate(cp):
        pc[n] = B.FID[n] + x[i]
    dth = np.array([x[0], cl.s8(pc) - cl.s8(B.FID)])
    fob = float(np.sqrt(dth @ np.linalg.solve(St, dth)))
    Bn = G @ Cov @ J.T @ Ci @ Jcells_S[m]
    g = np.linalg.solve(St, dth)/max(fob, 1e-12)
    fsig = float(np.sqrt(g @ (Bn @ Bn.T) @ g))
    at_b = [n for n, v, l, u in zip(names, x, lo, hi) if np.isfinite(l) and ((v - l) < 1e-3*(u - l) or (u - v) < 1e-3*(u - l))]
    out = dict(model=M.mdl, case=case, npts=int(m.sum()), FoB=fob, FoB_sig=fsig, chi2=chi2,
               dOm=float(dth[0]/np.sqrt(St[0, 0])), dS8=float(dth[1]/np.sqrt(St[1, 1])), sS8=float(np.sqrt(St[1, 1])),
               ia=x[nc:].tolist(), at_bound=at_b)
    log(f"    {M.mdl:10s} case {case}: chi2 {chi2:6.2f} dOm {out['dOm']:+.2f} dS8 {out['dS8']:+.2f} FoB {fob:.2f} +- {fsig:.2f} "
        f"IA {np.round(x[nc:], 3)}" + (f" AT BOUND {at_b}" if at_b else ""))
    return out


def main():
    data = sys.argv[1]
    covf = sys.argv[2] if len(sys.argv) > 2 else "analysis/cov_example1_xi.npy"
    tag = sys.argv[3] if len(sys.argv) > 3 else "example1"
    t0 = time.time()
    os.chdir(COCOA)
    cov = np.load(covf)
    tab = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
    R = np.load("run/cosmolike_xi_ia.npz")
    cl = B.CL(os.path.join(COCOA, "lib"), data)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL, cl.PKLcb, cl.sigma8)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    d_fl = xi_gg + R["xi_flam"]
    logf = open(os.path.join(HERE, f"halo_roman_{tag}_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    # lensing-only cosmology Jacobian and S8 gradient
    Jc = []
    gS8 = []
    for n in B.CP:
        pp, pm = dict(B.FID), dict(B.FID)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        Jc.append((cl.model("NLA", pp, [0.0, 0.0]) - cl.model("NLA", pm, [0.0, 0.0]))/(2*B.STEP[n]))
        gS8.append((cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n]))
    Jc = np.array(Jc).T
    gS8 = np.array(gS8)
    log(f"Jacobian done ({time.time() - t0:.0f}s)")
    T = {"NLA_tab": templates(cl, tab, False, "nla"), "halo": templates(cl, tab, False, "halo"),
         "halo_nowin": templates(cl, tab, True, "halo_nowin")}
    log(f"templates done ({time.time() - t0:.0f}s)")
    models = {k: IAModel(T[k], tab, k) for k in T}
    # validation 1: NLA templates vs native NLA
    nat = cl.model("NLA", B.FID, [3.5, 0.68]) - xi_gg
    tpl = models["NLA_tab"].xi([3.5, 0.68])
    sd = np.sqrt(np.diag(cov))
    log(f"validation: NLA templates vs native NLA, max |diff|/sigma = {np.max(np.abs(tpl - nat)/sd):.3e}, chi2 = {float((tpl - nat) @ np.linalg.solve(cov, tpl - nat)):.3e}")
    # validation 2: full halo table set vs template sum
    sns = np.load("../v2/fits.npz")["sns"]
    zs = tab["zs"]
    ia = [4.0, 0.5, 0.05, 0.0]
    x = (1 + zs)/1.62
    PdE = np.zeros((len(zs), len(KW)))
    PEE = np.zeros_like(PdE)
    idx = np.searchsorted(tab["k_all"], KW)
    la = np.log(np.maximum(tab["a_grid"], 1e-6))
    for s, z in enumerate(zs):
        Dz = np.sqrt(cl.PKL.P(z, 0.05)/cl.PKL.P(0.0, 0.05))
        C1 = B.FID["omegam"]*H.C1RHOCRIT/Dz
        P = cl.PKNL.P(z, KW*h)*h**3
        A = ia[0]*x[s]**ia[1]
        a = ia[2]*x[s]**ia[3]
        j = int(np.clip(np.searchsorted(la, np.log(a)), 1, la.size - 1))
        w = (np.log(a) - la[j - 1])/(la[j] - la[j - 1])
        Sd = tab["SdI"][s, j - 1][idx]*(1 - w) + tab["SdI"][s, j][idx]*w
        Si = tab["SII"][s, j - 1][idx]*(1 - w) + tab["SII"][s, j][idx]*w
        frc = tab["hod"][s, 5]
        PdE[s] = -(A*C1*frc*P*H.f2h(KW) + a*H.f1h(KW)*Sd)
        PEE[s] = (A*C1*frc)**2*P*H.f2h(KW) + a**2*H.f1h(KW)*Si
    p = write_set("run/tables_halo", "check", sns, PdE, PEE)
    cl.at(B.FID)
    cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
    full = cl.xi_sim()
    p0 = write_set("run/tables_halo", "check_zero", sns, 0*PdE, 0*PEE)
    cl.use_sim_tables(p0, sns, [float(z) for z in zs], 1)
    full = full - cl.xi_sim()
    tsum = models["halo"].xi(ia)
    log(f"validation: halo full tables vs template sum, max |diff|/sigma = {np.max(np.abs(full - tsum)/sd):.3e}, chi2 = {float((full - tsum) @ np.linalg.solve(cov, full - tsum)):.3e}")
    # 1-halo share of the IA signal at this point
    log(f"1-halo share: chi2 of the 1-halo part alone (full vector) = {float((tsum - models['halo'].xi([4.0, 0.5, 0.0, 0.0])) @ np.linalg.solve(cov, tsum - models['halo'].xi([4.0, 0.5, 0.0, 0.0]))):.3e}")
    # noise propagation
    fits = np.load("../v2/fits.npz")
    kd = fits["k"][0]
    J_cells = np.load("analysis/J_cells.npy")
    bk, Pf, err = fits["b"], fits["PNL"], fits["err"]
    nodes = np.where(kd >= KMIN)[0]
    PNLd = np.array([cl.PKNL.P(z, kd*h)*h**3 for z in zs])
    sig = []
    for s in range(len(zs)):
        for xx in range(2):
            conv = PNLd[s]/(bk[s]*Pf[s]) if xx == 0 else PNLd[s]/Pf[s]
            sig.extend((err[s, xx]*conv)[nodes])
    Jcells_S = J_cells*np.array(sig)[None, :]
    S = np.load(os.path.join(V3, "masks_scan.npz"))
    Mz = np.load(os.path.join(V3, "masks_v3.npz"))
    masks = [(f"K0/D {K0:g}", S[f"D1_{K0:g}"].astype(bool), dict(family="D1", K0=K0)) for K0 in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)]
    masks += [(f"const {K0:g}", S[f"flat_{K0:g}"].astype(bool), dict(family="flat", K0=K0)) for K0 in (0.2, 0.3, 0.4, 0.5)]
    masks += [(f"GoF {sh}", Mz[f"{sh}_GoF_x1"].astype(bool), dict(family="GoF", shape=sh)) for sh in ("NLA", "TATT")]
    out = []
    for lab, m, meta in masks:
        log(f"[{lab}: {int(m.sum())} pts ({time.time() - t0:.0f}s)]")
        for case in ("A", "B"):
            for k, M in models.items():
                r = fit(cl, M, d_fl, xi_gg, Jc, gS8, cov, m, case, Jcells_S, log)
                r.update(mask=lab, **meta)
                out.append(r)
        json.dump(out, open(os.path.join(HERE, f"halo_roman_{tag}.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
