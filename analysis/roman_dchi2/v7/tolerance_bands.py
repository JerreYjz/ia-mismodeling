#!/usr/bin/env python3
"""How much may the IA power spectra be wrong, scale by scale, before Roman
cosmic shear is biased by more than 0.3 sigma in (Omega_m, S8)?

Truth.  The FLAMINGO D3A cosmology; xi_+- = GG + IA with the stitched
        FLAMINGO truth S1 (v3/stitched_test.py): per snapshot, NLA (fitted on
        0.05 <= k <= 0.3) below the transition scale k_t(z), the smoothing
        spline of the FLAMINGO spectra above it, joined by a logistic in ln k
        (width 0.30); P_dE (GI) and P_EE (II) injected through sim_IA.
Model.  cosmolike's native NLA (A1, eta1) or TATT (A1, eta1, A2, eta2, b_TA),
        power-law evolution, refitted with Omega_m and A_s free (case A; case B
        adds n_s, H0, Omega_b with Gaussian priors), as v3/roman_test.py.
Perturbation basis.  Nodes k_j, log spaced; tent functions h_j(k) in ln k
        (linear interpolation between nodes, held flat beyond the end nodes),
        so the h_j sum to one at every k. A perturbation is
            P_X(k, z) -> P_X(k, z) [1 + sum_j eps_j h_j(k)],  X = GI, II or both,
        the same at every redshift. xi is linear in the injected tables, so
        the response to node j, dxi_j = xi[P_X h_j], is exact (one cosmolike
        run per node).
Linear response.  At the best fit of the model to the unperturbed truth,
            dtheta_j = G (F^-1 J^T C^-1) dxi_j,   theta = (Omega_m, S8),
        F = J^T C^-1 J + priors (IA with the flat-prior width, as roman_test),
        G = d(Omega_m, S8)/d(parameters). Whitened by the marginalised
        (Omega_m, S8) covariance Sigma, L_j = Sigma^-1/2 dtheta_j, and the
        figure of bias of a perturbation eps is FoB(eps) = |sum_j eps_j L_j|.
Bands.  single node: |eps_j| < 0.3/|L_j|, the tolerance at node j when every
        other node is exact. joint: a box |eps_j| <= b_j inside which EVERY
        combination of signs and sizes keeps FoB < 0.3. Over a box the largest
        FoB is exact and cheap in two dimensions,
            max_box |sum eps_j L_j| = max_phi sum_j b_j |u(phi) . L_j|,
        and the box is chosen with b_j = min(s/|L_j|, 1) (every node allowed
        the same share of the budget, capped at 100%), s by bisection.
        The response is linear, so the bands are symmetric in sign; the full
        non-linear refits of --verify test that.
Outputs (this folder): tolerance_bands.json, tolerance_bands_log.txt.

    python tolerance_bands.py <cocoa so dir> <cocoa_roman_real/data> [--verify]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
COCOA = os.path.join(ROOT, "cocoa")
sys.path.insert(0, COCOA)
sys.path.insert(0, os.path.join(ROOT, "v3"))
import bias_analysis as B                             # noqa: E402
import bias_v2 as V                                   # noqa: E402
import roman_test as RT                               # noqa: E402
from cosmolike_xi_ia import KW, write_set             # noqa: E402
from stitched_test import extend_full                 # noqa: E402

# perturbation nodes [h/Mpc]: log spaced, about 4 per decade
NODES = np.logspace(np.log10(0.01), np.log10(10.0), 13)
FOB_MAX = 0.3
SPECTRA = ("GI", "II", "both")
MODELS = ("NLA", "TATT")
# the masks: the production family k_UV = K0/D(z). Every point (no cut) is
# not used: there the model misfits the truth (chi2 ~ 2000) and the
# Gauss-Newton response, which drops the residual x Hessian term, fails.
MASK_K0 = (0.2, 0.3, 0.4, 0.5)
MASKS = tuple(f"K0/D {k}" for k in MASK_K0)
VERIFY_MASK = "K0/D 0.3"
TABLES = os.path.join(COCOA, "run", "tables_v7")


def tents(k, nodes=NODES):
    """h_j(k): linear interpolation weights in ln k, flat beyond the ends;
    shape (len(nodes), len(k)); columns sum to one."""
    x, xn = np.log(k), np.log(nodes)
    H = np.zeros((len(nodes), len(k)))
    for j in range(len(nodes)):
        e = np.zeros(len(nodes))
        e[j] = 1.0
        H[j] = np.interp(x, xn, e)     # np.interp holds the end values flat
    return H


def worst_fob(b, L, nphi=3601):
    '''Largest FoB over the box |eps_j| <= b_j: max_phi sum_j b_j |u.L_j|.
    Returns the value and the worst-case sign pattern.'''
    phi = np.linspace(0.0, np.pi, nphi)
    U = np.vstack([np.cos(phi), np.sin(phi)])          # (2, nphi)
    proj = U.T @ L                                     # (nphi, n)
    tot = np.abs(proj) @ b
    i = int(np.argmax(tot))
    return float(tot[i]), np.sign(proj[i])


def joint_band(L, fob=FOB_MAX):
    norm = np.linalg.norm(L, axis=0)
    def band(s):
        return np.minimum(s/np.maximum(norm, 1e-300), 1.0)
    if worst_fob(np.ones(L.shape[1]), L)[0] <= fob:
        return np.ones(L.shape[1]), 1.0
    lo, hi = 0.0, fob
    while worst_fob(band(hi), L)[0] < fob:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if worst_fob(band(mid), L)[0] < fob else (lo, mid)
    return band(lo), lo


def linear_response(cl, mdl, best, free, prior, Ci, m):
    """F^-1 J^T C^-1, G and Sigma(Omega_m, S8) at the best fit (as RT.fit)."""
    nia = len(B.IAP[mdl])
    names = free + B.IAP[mdl]
    pc = dict(B.FID)
    pc.update({n: best[n] for n in free})
    ia = np.array([best[n] for n in B.IAP[mdl]])
    J = np.zeros((m.sum(), len(names)))
    for i, n in enumerate(free):
        pp, pm = dict(pc), dict(pc)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        J[:, i] = (cl.model(mdl, pp, ia) - cl.model(mdl, pm, ia))[m]/(2*B.STEP[n])
    for i in range(nia):
        e = np.zeros(nia)
        e[i] = 0.05
        J[:, len(free) + i] = (cl.model(mdl, pc, ia + e) - cl.model(mdl, pc, ia - e))[m]/0.1
    psig = [prior.get(n, np.inf) for n in free]
    Pinv = np.diag([0.0 if not np.isfinite(v) else 1/v**2 for v in psig]
                   + [1/V.IA_SIG[n]**2 for n in B.IAP[mdl]])
    Cov = np.linalg.inv(J.T @ Ci @ J + Pinv)
    G = np.zeros((2, len(names)))
    G[0, names.index("omegam")] = 1.0
    for i, n in enumerate(free):
        pp, pm = dict(pc), dict(pc)
        pp[n] += B.STEP[n]
        pm[n] -= B.STEP[n]
        G[1, i] = (cl.s8(pp) - cl.s8(pm))/(2*B.STEP[n])
    St = G @ Cov @ G.T
    return Cov @ J.T @ Ci, G, St


def whiten(St):
    w, U = np.linalg.eigh(St)
    return U @ np.diag(w**-0.5) @ U.T


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    verify = "--verify" in sys.argv
    t0 = time.time()
    os.chdir(COCOA)
    logf = open(os.path.join(HERE, "tolerance_bands_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    st = np.load(os.path.join(ROOT, "v3", "stitched_truths.npz"))
    zs, sns, kd = st["zs"], st["sns"], st["k"]
    zl = [float(z) for z in zs]
    h = cl.h
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])
    R = st["R_S1"]
    PdE = np.array([extend_full(kd, R[s, 0])*PNLw[s] for s in range(len(zs))])
    PEE = np.array([extend_full(kd, R[s, 1])*PNLw[s] for s in range(len(zs))])
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])

    def xi_ia(name, pde, pee):
        p = write_set(TABLES, name, sns, pde, pee)
        cl.at(B.FID)
        cl.use_sim_tables(p, sns, zl, 1)
        return cl.xi_sim() - xi_gg

    xi_truth = xi_ia("S1", PdE, PEE)
    d = xi_gg + xi_truth
    # node responses: exact, xi linear in the tables
    H = tents(KW)
    zero = np.zeros_like(PdE)
    resp = {"GI": [], "II": []}
    for j in range(len(NODES)):
        resp["GI"].append(xi_ia(f"GI_{j}", PdE*H[j], zero))
        resp["II"].append(xi_ia(f"II_{j}", zero, PEE*H[j]))
    resp = {X: np.array(v) for X, v in resp.items()}
    resp["both"] = resp["GI"] + resp["II"]
    # check: the tents sum to one, so the node responses sum to xi_IA itself
    log(f"[{time.time() - t0:.0f}s] node responses; sum over nodes vs xi_IA(truth): "
        f"max|diff|/max|xi| = {np.max(np.abs(resp['both'].sum(0) - xi_truth))/np.max(np.abs(xi_truth)):.1e}")

    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {f"K0/D {k}": M1[K1.index(k)] for k in MASK_K0}
    covs = {"example1": np.load("analysis/cov_example1_xi.npy"), "gauss": np.load("analysis/cov_gauss_d3a.npy")}

    out = dict(about=__doc__, nodes_h_per_Mpc=NODES.tolist(), fob_max=FOB_MAX, k_tents=KW.tolist(),
               results=[], checks={})
    # end-to-end check of the fit machinery on this build: the stitched test
    # (Gaussian covariance, K0 = 0.3/D, case A) gave NLA 0.281 and TATT 0.254
    m = masks["K0/D 0.3"]
    C = covs["gauss"][np.ix_(m, m)]
    Ci, Lw = np.linalg.inv(C), np.linalg.inv(np.linalg.cholesky(C))
    chk = {}
    for mdl in MODELS:
        r = RT.fit(cl, mdl, d, Ci, Lw, m, *RT.CASES["A"], log)
        chk[mdl] = r["FoB"]
    out["checks"]["stitched_test_K0_0.3_gauss"] = dict(this_build=chk, cloud_build=dict(NLA=0.281, TATT=0.254))

    for mask in MASKS:
        m = masks[mask]
        C = covs["example1"][np.ix_(m, m)]
        Ci, Lw = np.linalg.inv(C), np.linalg.inv(np.linalg.cholesky(C))
        for case in ("A", "B"):
            free, prior = RT.CASES[case]
            nla_sol = (3.5, 0.68)
            for mdl in MODELS:
                log(f"[{time.time() - t0:.0f}s] {mask} ({int(m.sum())} points), case {case}, {mdl}")
                r = RT.fit(cl, mdl, d, Ci, Lw, m, free, prior, log, nla_sol=nla_sol)
                if mdl == "NLA":
                    nla_sol = (r["best"]["A1"], r["best"]["eta1"])
                Best, G, St = linear_response(cl, mdl, r["best"], free, prior, Ci, m)
                W = whiten(St)
                row = dict(mask=mask, npts=int(m.sum()), case=case, model=mdl, baseline=r,
                           sigma_Om=float(np.sqrt(St[0, 0])), sigma_S8=float(np.sqrt(St[1, 1])),
                           bands={})
                for X in SPECTRA:
                    dth = G @ Best @ resp[X][:, m].T            # (2, nodes)
                    L = W @ dth
                    single = np.minimum(FOB_MAX/np.linalg.norm(L, axis=0), 1.0)
                    joint, s = joint_band(L)
                    row["bands"][X] = dict(single=single.tolist(), joint=joint.tolist(), share=s,
                                           dtheta_per_unit_eps=dth.T.tolist(), L=L.T.tolist())
                    log(f"    {X:4s} single " + " ".join(f"{v:.3f}" for v in single))
                    log(f"    {X:4s} joint  " + " ".join(f"{v:.3f}" for v in joint))
                    if verify and mask == VERIFY_MASK and case == "A" and X in ("GI", "II"):
                        row["bands"][X]["verify"] = verify_band(cl, mdl, d, Ci, Lw, m, free, prior, log,
                                                                r, St, resp[X], L, single, joint, nla_sol)
                out["results"].append(row)
                json.dump(out, open(os.path.join(HERE, "tolerance_bands.json"), "w"), indent=1)
    log(f"done ({time.time() - t0:.0f}s)")


def verify_band(cl, mdl, d, Ci, Lw, m, free, prior, log, r0, St, resp, L, single, joint, nla_sol):
    """Full non-linear refits: each sign at the single-node band edge for
    three nodes, and the worst-case corner of the joint box, both signs. The
    FoB is that of the shift from the unperturbed best fit, in the
    unperturbed Sigma; linear response predicts 0.3 at every edge."""
    def shift_fob(r):
        dth = np.array([r["dOm"] - r0["dOm"], r["dS8"] - r0["dS8"]])
        return float(np.sqrt(dth @ np.linalg.solve(St, dth)))
    out = []
    nodes = [j for j in np.argsort(-np.linalg.norm(L, axis=0))[:3]]
    for j in nodes:
        for sgn in (+1, -1):
            eps = np.zeros(len(single))
            eps[j] = sgn*single[j]
            r = RT.fit(cl, mdl, d + eps @ resp, Ci, Lw, m, free, prior, lambda s: None, nla_sol=nla_sol)
            out.append(dict(kind="single", node=int(j), eps=float(eps[j]), fob_full=shift_fob(r),
                            fob_linear=float(np.linalg.norm(L @ eps))))
            log(f"      verify single node {j} eps {eps[j]:+.3f}: full {out[-1]['fob_full']:.3f}, "
                f"linear {out[-1]['fob_linear']:.3f}")
    _, signs = worst_fob(joint, L)
    for sgn in (+1, -1):
        eps = sgn*signs*joint
        r = RT.fit(cl, mdl, d + eps @ resp, Ci, Lw, m, free, prior, lambda s: None, nla_sol=nla_sol)
        out.append(dict(kind="joint corner", eps=eps.tolist(), fob_full=shift_fob(r),
                        fob_linear=float(np.linalg.norm(L @ eps))))
        log(f"      verify joint corner {sgn:+d}: full {out[-1]['fob_full']:.3f}, linear {out[-1]['fob_linear']:.3f}")
    return out


if __name__ == "__main__":
    main()
