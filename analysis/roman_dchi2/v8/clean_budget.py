#!/usr/bin/env python3
"""The IA error budget of Roman cosmic shear, with one fixed definition.

Question. How wrong may the FLAMINGO IA power spectra be, as a function of
k, before Roman xi_+- moves Omega_m or S8 by more than 0.3 of its
marginalised error?

Fixed choices.
  Truths      S1:   the stitched FLAMINGO truth (NLA below k_t(z), the spline
                    of the measured spectra above; v3/stitched_truths.npz R_S1)
              prod: the production FLAMINGO truth (the spline of the measured
                    spectra at k >= 0.05, held below; R_prod)
              both at the FLAMINGO D3A cosmology, GG + IA injected through
              sim_IA (cosmolike with the sim-IA patches, untrimmed tables).
  Model       model agnostic: the truth itself, with an IA amplitude A and a
              power-law evolution ((1+z)/1.62)^eta free (GI ~ a(z), II ~ a(z)^2)
              and Omega_m, A_s free. It reproduces the data exactly, so every
              shift below is caused by the perturbation alone.
  Criterion   |dOmega_m| / sigma(Omega_m) < 0.3 and |dS8| / sigma(S8) < 0.3.
  Cuts        none (all 1080 points) and the production mask k_UV = 0.3/D(z).
  Covariance  example1_cov.
  Perturbation  P_X(k, z) -> P_X(k, z) [1 + eps f(k)], the same at every z, for
              X = GI, II, or both, with three smooth shapes f, all of one sign:
                bump at k_c:   exp(-ln^2(k/k_c) / (2 * 0.3^2))   (one k range)
                below k_c:     1 / (1 + exp( ln(k/k_c)/0.25))     (all larger scales)
                above k_c:     1 / (1 + exp(-ln(k/k_c)/0.25))     (all smaller scales)
              for k_c at the 13 nodes from 0.01 to 10 h/Mpc.
  Tolerance   eps_tol = 0.3 / max_i |a_i|, a_i the shift of parameter i per
              unit eps in units of its sigma, by linear response at the
              fiducial (exact first order, since the model fits the data).
              Capped at 1 (100%).
Responses are exact: f is written on 61 tents in ln k (20 per decade, one
cosmolike run each); xi is linear in the injected tables.
Checks: full Gauss-Newton refits of the model at selected tolerances, which
should land at 0.3 sigma in the limiting parameter.
Models: NLA and TATT (cosmolike native, power-law evolution) fitted to each
truth, each cut, case A (roman_test.fit): their actual shifts, and their
errors Delta_X = P_X^model/P_X^truth - 1 at the core snapshots.
Output: clean_budget.json, clean_budget_log.txt.

    python clean_budget.py <cocoa so dir> <cocoa_roman_real/data>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V7 = os.path.join(HERE, "..", "v7")
sys.path.insert(0, V7)
import tolerance_nomask as T                                       # noqa: E402
from tolerance_bands import B, RT, KW, write_set, extend_full, NODES  # noqa: E402
import model_errors as ME                                          # noqa: E402

TOL = 0.3
FINE = np.logspace(-2, 1, 61)
BUMP_W, STEP_W = 0.3, 0.25
KDENSE = np.logspace(-2, 1, 31)
ZCORE = (0.2, 2.0)
TRUTHS = ("S1", "prod")
CUTS = ("none", "K0/D 0.3")


class Truth(T.Template):
    """The truth-as-model of tolerance_nomask, for a chosen truth variant."""

    def __init__(self, cl, st, variant):
        super().__init__(cl, st)
        R = st[f"R_{variant}"]
        self.RdE = np.array([extend_full(self.kd, R[s, 0]) for s in range(len(self.zs))])
        self.REE = np.array([extend_full(self.kd, R[s, 1]) for s in range(len(self.zs))])
        self.R = R


def tents_on(k, nodes):
    x, xn = np.log(k), np.log(nodes)
    H = np.zeros((len(nodes), len(k)))
    for j in range(len(nodes)):
        e = np.zeros(len(nodes))
        e[j] = 1.0
        H[j] = np.interp(x, xn, e)
    return H


def shapes():
    """{(shape, j): f(FINE)} for the three shapes at every node."""
    lnF = np.log(FINE)
    out = {}
    for j, kc in enumerate(NODES):
        x = lnF - np.log(kc)
        out[("bump", j)] = np.exp(-x**2/(2*BUMP_W**2))
        out[("below", j)] = 1/(1 + np.exp(np.clip(x/STEP_W, -60, 60)))
        out[("above", j)] = 1/(1 + np.exp(np.clip(-x/STEP_W, -60, 60)))
    return out


def gn_refit(model, cl, d, Ci, m, free, iap, x0, iters=4):
    """Gauss-Newton refit of the truth-as-model on the points m."""
    x = dict(x0)
    names = free + iap
    Pi = T.prior_inv(free, iap, {})
    for _ in range(iters):
        mv = model(T.full_p({k: x[k] for k in free}), {k: x[k] for k in iap})[m]
        J = T.jacobian(model, free, iap, x)[m]
        g = J.T @ Ci @ (d[m] - mv) - Pi @ np.array([x[n] - x0[n] for n in names])
        step = np.linalg.solve(J.T @ Ci @ J + Pi, g)
        for n, s in zip(names, step):
            x[n] += s
    return x


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    logf = open(os.path.join(HERE, "clean_budget_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])                   # before any sim table
    T.GG["fid"] = gg
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    zs, sns, kd = st["zs"], st["sns"], st["k"]
    zl = [float(z) for z in zs]
    core = [s for s, z in enumerate(zs) if ZCORE[0] - 1e-9 <= z <= ZCORE[1] + 1e-9]
    cov = np.load("analysis/cov_example1_xi.npy")
    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {"none": np.ones(1080, bool), "K0/D 0.3": M1[K1.index(0.3)]}
    Hf = tents_on(KW, FINE)
    shp = shapes()
    tm = ME.F.TATTModel(halofit="takahashi")
    free_t, iap = ["omegam", "As_1e9"], ["A", "eta"]
    x_fid = {**{n: B.FID[n] for n in free_t}, "A": 1.0, "eta": 0.0}
    s8f = cl.s8(dict(B.FID))
    out = dict(about=__doc__, nodes_h_per_Mpc=NODES.tolist(), k_dense=KDENSE.tolist(),
               z_core=[float(zs[s]) for s in core], tolerance=TOL, truths={})

    for tr in TRUTHS:
        model = Truth(cl, st, tr)
        P = model.pnl(B.FID)
        PdE, PEE = model.RdE*P, model.REE*P
        d = model(dict(B.FID), {"A": 1.0, "eta": 0.0})
        ia = d - gg
        zero = np.zeros_like(PdE)
        fine = {"GI": np.array([T.TB_xi(cl, model, PdE*Hf[m], zero, f"c_{tr}_GI_{m}")
                                for m in range(len(FINE))]),
                "II": np.array([T.TB_xi(cl, model, zero, PEE*Hf[m], f"c_{tr}_II_{m}")
                                for m in range(len(FINE))])}
        check = np.max(np.abs((fine["GI"] + fine["II"]).sum(0) - ia))/np.max(np.abs(ia))
        log(f"[{time.time() - t0:.0f}s] truth {tr}: 122 tent responses; sum vs xi_IA {check:.1e}")
        J = T.jacobian(model, free_t, iap, x_fid)
        G = T.s8_row(cl, free_t, iap, x_fid)
        Pi = T.prior_inv(free_t, iap, {})
        tout = dict(partition_check=check, cuts={}, models={})
        for cut, m in masks.items():
            Ci = np.linalg.inv(cov[np.ix_(m, m)])
            Jm = J[m]
            Finv = np.linalg.inv(Jm.T @ Ci @ Jm + Pi)
            St = G @ Finv @ G.T
            sig = np.sqrt(np.diag(St))
            Proj = (G @ Finv @ Jm.T @ Ci)/sig[:, None]              # (2, npts)
            cres = dict(sigma=sig.tolist(), corr=float(St[0, 1]/np.prod(sig)), tol={}, limiting={},
                        shift_per_unit={})
            for shape in ("bump", "below", "above"):
                for X in ("GI", "II", "both"):
                    tol, lim, a_all = [], [], []
                    for j in range(len(NODES)):
                        f = shp[(shape, j)]
                        r = (f @ fine["GI"] if X in ("GI", "both") else 0) + \
                            (f @ fine["II"] if X in ("II", "both") else 0)
                        a = Proj @ r[m]
                        a_all.append(a.tolist())
                        tol.append(float(min(TOL/max(np.max(np.abs(a)), 1e-300), 1.0)))
                        lim.append(["Omega_m", "S8"][int(np.argmax(np.abs(a)))])
                    cres["tol"][f"{shape}|{X}"] = tol
                    cres["limiting"][f"{shape}|{X}"] = lim
                    cres["shift_per_unit"][f"{shape}|{X}"] = a_all
            # full refits at selected tolerances
            ver = []
            for shape, X, j in (("below", "GI", 4), ("below", "II", 4), ("above", "GI", 8),
                                ("above", "II", 8), ("bump", "GI", 2), ("bump", "GI", 8)):
                eps = cres["tol"][f"{shape}|{X}"][j]
                if eps >= 1.0:
                    continue
                fk = shp[(shape, j)] @ Hf                           # f on the wide grid
                fg = 1 + eps*fk if X in ("GI", "both") else 1.0
                fi = 1 + eps*fk if X in ("II", "both") else 1.0
                dd = gg + T.TB_xi(cl, model, PdE*fg, PEE*fi, f"c_{tr}_ver")
                x = gn_refit(model, cl, dd, Ci, m, free_t, iap, x_fid)
                sh = [(x["omegam"] - B.FID["omegam"])/sig[0],
                      (cl.s8(T.full_p({k: x[k] for k in free_t})) - s8f)/sig[1]]
                lin = [eps*v for v in cres["shift_per_unit"][f"{shape}|{X}"][j]]
                ver.append(dict(shape=shape, X=X, k=float(NODES[j]), eps=eps, full=sh, linear=lin))
                log(f"  {tr} {cut:9s} verify {shape:5s} {X} k {NODES[j]:.3f} eps {eps:.4f}: full "
                    f"({sh[0]:+.3f}, {sh[1]:+.3f}) linear ({lin[0]:+.3f}, {lin[1]:+.3f})")
            cres["verify"] = ver
            tout["cuts"][cut] = cres
            for shape in ("bump", "below", "above"):
                for X in ("GI", "II"):
                    log(f"  {tr} {cut:9s} {shape:5s} {X}: " +
                        " ".join(f"{v:.3f}" for v in cres["tol"][f"{shape}|{X}"]))
            # NLA and TATT fitted to this truth with this cut
            Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(m, m)]))
            nla_sol = (3.5, 0.68)
            for mdl in ("NLA", "TATT"):
                r = RT.fit(cl, mdl, d, Ci, Lw, m, *RT.CASES["A"], lambda s: None, nla_sol=nla_sol)
                if mdl == "NLA":
                    nla_sol = (r["best"]["A1"], r["best"]["eta1"])
                err = {"GI": [], "II": []}
                for s in core:
                    Rt = np.array([np.interp(np.log(KDENSE), np.log(kd), model.R[s, i]) for i in (0, 1)])
                    Rm = ME.model_ratio(tm, float(zs[s]), KDENSE, r["best"], mdl)
                    err["GI"].append((Rm[0]/Rt[0] - 1).tolist())
                    err["II"].append((Rm[1]/Rt[1] - 1).tolist())
                tout["models"][f"{cut}|{mdl}"] = dict(
                    best=r["best"], chi2=r["chi2_bestfit"], npts=r["npts"],
                    dOm_over_sigma=r["dOm"]/r["sOm"], dS8_over_sigma=r["dS8"]/r["sS8"],
                    FoB=r["FoB"], error=err)
                log(f"  {tr} {cut:9s} {mdl:4s} fit: chi2 {r['chi2_bestfit']:.1f}; dOm "
                    f"{r['dOm']/r['sOm']:+.2f} sigma, dS8 {r['dS8']/r['sS8']:+.2f} sigma")
        out["truths"][tr] = tout
        json.dump(out, open(os.path.join(HERE, "clean_budget.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
