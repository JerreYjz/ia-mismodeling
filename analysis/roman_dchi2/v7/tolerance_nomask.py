#!/usr/bin/env python3
"""IA tolerance bands with no scale cut: the error budget of the IA spectra.

tolerance_bands.py refits cosmolike's NLA / TATT, which cannot describe the
stitched FLAMINGO truth on every scale (chi2 ~ 2000 with all 1080 points), so
its linear response is only valid on cut masks. Here the model is the truth
itself with free amplitudes, so it reproduces the data exactly at the
fiducial (chi2 = 0) and the linear response is exact to first order:

    P_dE(k, z; p) = a(z)   R_dE(k, z) P_NL(k, z; cosmology)
    P_EE(k, z; p) = a(z)^2 R_EE(k, z) P_NL(k, z; cosmology),
    a(z) = A ((1 + z)/1.62)^eta,         fiducial A = 1, eta = 0,

with R_X = P_X/P_NL the stitched FLAMINGO ratios (v3/stitched_truths.npz,
variant S1: NLA below k_t(z), spline of the measured spectra above). The
cosmology is varied through CAMB as in bias_analysis.CL. The IA amplitude
mimics NLA's A1 and eta1, so the question becomes: given a model with the
right shape and free amplitude, how much may the shape be wrong, scale by
scale, before (Omega_m, S8) moves by 0.3 sigma?

Variants (all with the example1 covariance and every one of the 1080 points;
the K0/D(z) masks are kept for comparison):
    A        Omega_m, A_s, A, eta free
    B        + n_s, H0, Omega_b with Gaussian priors (roman_test.CASES)
    A_fixIA  Omega_m, A_s free, IA amplitude fixed (no IA freedom at all)
Gaussian priors on A and eta with the width of the chain's flat priors on
A1 and eta1 (range/sqrt 12; A1 in [-5, 5] relative to A1 ~ 3.4, eta in [-5, 5]).

Perturbations, bands and the worst-case box are those of tolerance_bands.py.
--verify: Gauss-Newton refits of the full model (cosmology through CAMB, IA
tables rebuilt at every step) at band edges, no mask, case A.

    python tolerance_nomask.py <cocoa so dir> <cocoa_roman_real/data> [--verify]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tolerance_bands as TB                       # noqa: E402  (sets the import paths)
from tolerance_bands import B, RT, KW, write_set, extend_full, tents, joint_band, whiten, worst_fob  # noqa: E402

FOB_MAX = 0.3
SIG_IA = {"A": 10.0/np.sqrt(12)/3.4, "eta": 10.0/np.sqrt(12)}
STEP_IA = {"A": 0.02, "eta": 0.05}
X0 = 1.62
GG = {}
MASKS = ("none", "K0/D 0.2", "K0/D 0.3", "K0/D 0.4", "K0/D 0.5")
VARIANTS = {"A": (["omegam", "As_1e9"], ["A", "eta"], {}),
            "B": (B.CP, ["A", "eta"], B.PRIOR_B),
            "A_fixIA": (["omegam", "As_1e9"], [], {})}


class Template:
    """The stitched truth as a model: cosmology + IA amplitude (A, eta)."""

    def __init__(self, cl, st):
        self.cl, self.st = cl, st
        self.zs, self.sns, self.kd = st["zs"], st["sns"], st["k"]
        self.zl = [float(z) for z in self.zs]
        R = st["R_S1"]
        self.RdE = np.array([extend_full(self.kd, R[s, 0]) for s in range(len(self.zs))])
        self.REE = np.array([extend_full(self.kd, R[s, 1]) for s in range(len(self.zs))])
        self.n = 0

    def pnl(self, p):
        cl = self.cl.at(p)
        h = cl.h
        return np.array([cl.PKNL.P(z, KW*h)*h**3 for z in self.zs])

    def tables(self, p, ia, name):
        P = self.pnl(p)
        a = ia.get("A", 1.0)*((1 + self.zs)/X0)**ia.get("eta", 0.0)
        return write_set(TB.TABLES, name, self.sns, a[:, None]*self.RdE*P, (a**2)[:, None]*self.REE*P)

    def __call__(self, p, ia):
        """Full xi_+- (GG + IA) at cosmology p and IA amplitude ia."""
        path = self.tables(p, ia, "template")
        cl = self.cl.at(p)
        cl.use_sim_tables(path, self.sns, self.zl, 1)
        self.n += 1
        return cl.xi_sim()


def full_p(cosmo):
    p = dict(B.FID)
    p.update(cosmo)
    return p


def jacobian(model, free, iap, x):
    """d xi / d (free cosmology + IA), central differences at x (dict)."""
    names = free + iap
    J = np.zeros((1080, len(names)))
    for i, n in enumerate(names):
        step = B.STEP[n] if n in B.STEP else STEP_IA[n]
        xp, xm = dict(x), dict(x)
        xp[n] += step
        xm[n] -= step
        cp = {k: xp[k] for k in free}
        cm = {k: xm[k] for k in free}
        J[:, i] = (model(full_p(cp), {k: xp[k] for k in iap})
                   - model(full_p(cm), {k: xm[k] for k in iap}))/(2*step)
    return J


def s8_row(cl, free, iap, x):
    names = free + iap
    G = np.zeros((2, len(names)))
    G[0, names.index("omegam")] = 1.0
    for i, n in enumerate(free):
        xp, xm = dict(x), dict(x)
        xp[n] += B.STEP[n]
        xm[n] -= B.STEP[n]
        G[1, i] = (cl.s8(full_p({k: xp[k] for k in free}))
                   - cl.s8(full_p({k: xm[k] for k in free})))/(2*B.STEP[n])
    return G


def prior_inv(free, iap, prior):
    v = [prior.get(n, np.inf) for n in free] + [SIG_IA[n] for n in iap]
    return np.diag([0.0 if not np.isfinite(s) else 1/s**2 for s in v])


def gauss_newton(model, cl, d, Ci, free, iap, prior, iters=4):
    """Refit of the full model to d (no mask), Gauss-Newton with the Jacobian
    rebuilt at every step and Gaussian priors centred on the fiducial."""
    x0 = {**{n: B.FID[n] for n in free}, "A": 1.0, "eta": 0.0}
    x = dict(x0)
    names = free + iap
    Pi = prior_inv(free, iap, prior)
    for _ in range(iters):
        m = model(full_p({k: x[k] for k in free}), {k: x[k] for k in iap})
        J = jacobian(model, free, iap, x)
        dx_prior = np.array([x[n] - x0[n] for n in names])
        g = J.T @ Ci @ (d - m) - Pi @ dx_prior
        step = np.linalg.solve(J.T @ Ci @ J + Pi, g)
        for n, s in zip(names, step):
            x[n] += s
    return x


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    verify = "--verify" in sys.argv
    t0 = time.time()
    os.chdir(TB.COCOA)
    logf = open(os.path.join(HERE, "tolerance_nomask_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    st = np.load(os.path.join(TB.ROOT, "v3", "stitched_truths.npz"))
    model = Template(cl, st)
    fid_x = {**{n: B.FID[n] for n in B.CP}, "A": 1.0, "eta": 0.0}
    # GG first: once sim-IA tables are loaded, a native call with every IA
    # amplitude zero still returns the sim IA (calls with nonzero amplitudes
    # are unaffected), so GG must be computed before any table is loaded
    GG["fid"] = cl.model("NLA", B.FID, [0.0, 0.0])
    xi_gg = GG["fid"]
    d = model(dict(B.FID), {"A": 1.0, "eta": 0.0})
    PNLw = model.pnl(B.FID)
    xi_truth_tb = xi_gg + TB_xi(cl, model, model.RdE*PNLw, model.REE*PNLw, "S1check")
    log(f"[{time.time() - t0:.0f}s] template at the fiducial vs the truth: "
        f"max|diff|/max|xi| = {np.max(np.abs(d - xi_truth_tb))/np.max(np.abs(d)):.1e}")

    H = tents(KW)
    zero = np.zeros_like(PNLw)
    resp = {"GI": [], "II": []}
    for j in range(len(TB.NODES)):
        resp["GI"].append(TB_xi(cl, model, model.RdE*PNLw*H[j], zero, f"GI_{j}"))
        resp["II"].append(TB_xi(cl, model, zero, model.REE*PNLw*H[j], f"II_{j}"))
    resp = {X: np.array(v) for X, v in resp.items()}
    resp["both"] = resp["GI"] + resp["II"]

    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {"none": np.ones(len(d), bool)}
    masks.update({f"K0/D {k}": M1[K1.index(k)] for k in (0.2, 0.3, 0.4, 0.5)})
    cov = np.load("analysis/cov_example1_xi.npy")

    out = dict(about=__doc__, nodes_h_per_Mpc=TB.NODES.tolist(), fob_max=FOB_MAX,
               sig_ia=SIG_IA, results=[])
    jac = {}
    for var, (free, iap, prior) in VARIANTS.items():
        key = (tuple(free), tuple(iap))
        if key not in jac:
            jac[key] = (jacobian(model, free, iap, fid_x), s8_row(cl, free, iap, fid_x))
            log(f"[{time.time() - t0:.0f}s] Jacobian {var}: {len(free) + len(iap)} parameters")
        J, G = jac[key]
        Pi = prior_inv(free, iap, prior)
        for mask, m in masks.items():
            C = cov[np.ix_(m, m)]
            Ci = np.linalg.inv(C)
            Jm = J[m]
            F = Jm.T @ Ci @ Jm + Pi
            Finv = np.linalg.inv(F)
            St = G @ Finv @ G.T
            W = whiten(St)
            Best = Finv @ Jm.T @ Ci
            row = dict(variant=var, mask=mask, npts=int(m.sum()), free=free + iap,
                       sigma_Om=float(np.sqrt(St[0, 0])), sigma_S8=float(np.sqrt(St[1, 1])),
                       Sigma=St.tolist(), bands={})
            for X in ("GI", "II", "both"):
                dth = G @ Best @ resp[X][:, m].T
                L = W @ dth
                single = np.minimum(FOB_MAX/np.linalg.norm(L, axis=0), 1.0)
                joint, s = joint_band(L)
                row["bands"][X] = dict(single=single.tolist(), joint=joint.tolist(), share=s,
                                       L=L.T.tolist(), dtheta_per_unit_eps=dth.T.tolist())
                log(f"  {var:7s} {mask:9s} {X:4s} single " + " ".join(f"{v:.3f}" for v in single))
                log(f"  {var:7s} {mask:9s} {X:4s} joint  " + " ".join(f"{v:.3f}" for v in joint))
                if verify and var == "A" and mask == "none" and X in ("GI", "II"):
                    row["bands"][X]["verify"] = verify_edges(model, cl, d, Ci, free, iap, prior,
                                                            St, resp[X], L, single, joint, log)
            out["results"].append(row)
            json.dump(out, open(os.path.join(HERE, "tolerance_nomask.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s, {model.n} model evaluations)")


def TB_xi(cl, model, pde, pee, name):
    """xi_IA of given tables at the fiducial cosmology (GG removed)."""
    p = write_set(TB.TABLES, name, model.sns, pde, pee)
    cl.at(B.FID)
    cl.use_sim_tables(p, model.sns, model.zl, 1)
    return cl.xi_sim() - GG["fid"]


def verify_edges(model, cl, d, Ci, free, iap, prior, St, resp, L, single, joint, log):
    s8f = cl.s8(dict(B.FID))
    out = []

    def fob_of(x):
        dth = np.array([x["omegam"] - B.FID["omegam"],
                        cl.s8(full_p({k: x[k] for k in free})) - s8f])
        return float(np.sqrt(dth @ np.linalg.solve(St, dth)))
    j = int(np.argmax(np.linalg.norm(L, axis=0)))
    cases = []
    for sgn in (+1, -1):
        eps = np.zeros(len(single))
        eps[j] = sgn*single[j]
        cases.append(("single", eps))
    _, signs = worst_fob(joint, L)
    for sgn in (+1, -1):
        cases.append(("joint corner", sgn*signs*joint))
    for kind, eps in cases:
        x = gauss_newton(model, cl, d + eps @ resp, Ci, free, iap, prior)
        out.append(dict(kind=kind, node=j if kind == "single" else None, eps=eps.tolist(),
                        fob_full=fob_of(x), fob_linear=float(np.linalg.norm(L @ eps))))
        log(f"      verify {kind} (node {j}): full {out[-1]['fob_full']:.3f}, "
            f"linear {out[-1]['fob_linear']:.3f}")
    return out


if __name__ == "__main__":
    main()
