#!/usr/bin/env python3
"""Checks of ia_fisher.py, one PASS or FAIL line each (as verify_prior.py of
notes/correlated_bias_prior_v2).

Toys (always): the tents and the spline; the tridiagonal inverse of the
exponential kernel; the projected precision (annihilates U, rank n - 2,
semidefinite, the null-space form, the Woodbury limit, the matched identity
P Sigma P = P); the coverages and the box probability against Monte Carlo;
the closed-form eta of a rigid and of a marginalised fit against a Monte
Carlo of whole experiments (data noise, Gaussian-prior truths, random IA
errors, a drift); eta = 1 in the matched case.
On the base (when ia_base_<truth>.npz exists): the stored checks; a
constant residual reproduces dxi/dA; the factor form against the projected
precision; eta against a Monte Carlo on the real response.

    python verify_ia_fisher.py
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ia_fisher as IF                                             # noqa: E402

RESULTS = []
ETA_STEP = 0.05                    # STEP_IA["eta"] of v7/tolerance_nomask.py, used for J


def check(name, value, tol, less=True):
    ok = value <= tol if less else value >= tol
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} ({'<=' if less else '>='} {tol:.1e})")


def toy_base(rng, nz=4, nn=6, nd=60):
    """A base with the structure of the real one: 4 template columns, then
    (X, s, j) knobs on a coarse tent grid, with dxi/dA and dxi/deta built
    from the knobs exactly as the template makes them."""
    fine = np.logspace(-2, 1, 9)
    nodes = np.logspace(-2, 1, nn)
    zs = np.linspace(0.2, 2.0, nz)
    nf = len(fine)
    RX = {X: rng.standard_normal((nz, nf, nd)) for X in IF.XS}
    lnx = np.log((1 + zs)/IF.X0)
    JA = RX["GI"].sum((0, 1)) + 2*RX["II"].sum((0, 1))
    Je = (lnx[:, None]*(RX["GI"].sum(1) + 2*RX["II"].sum(1))).sum(0)
    J = np.column_stack([rng.standard_normal(nd), rng.standard_normal(nd), JA, Je])
    B = np.hstack([J] + [RX[X].reshape(nz*nf, -1).T for X in IF.XS])
    b = SimpleNamespace(zs=zs, fine=fine, B=B, nb=B.shape[1], ij=np.arange(4), npar=4,
                        g=np.array([[1.0, 0, 0, 0], [0.4, 0.6, 0, 0]]),
                        iknob={X: 4 + q*nz*nf + np.arange(nz*nf).reshape(nz, nf)
                               for q, X in enumerate(IF.XS)})
    b.knob_map = lambda Xs=IF.XS, nodes=nodes: IF.Base.knob_map(b, Xs, nodes)
    A = rng.standard_normal((nd, nd))
    cov = A @ A.T/nd + np.eye(nd)
    return b, nodes, cov


def experiments(fit, Rmap, S_true, mu_delta, rng, nmc, cov):
    """Monte Carlo of whole experiments: the truth draws A, eta from their
    priors and the residual from N(mu, S_true); the data add noise; the fit
    is the linear estimator. Returns the mean of the squared pulls."""
    D = fit.M
    nres = Rmap.shape[1]
    Ld = np.linalg.cholesky(cov)
    Lr = IF.factor(S_true) if np.any(S_true) else np.zeros((nres, 1))
    Ci = np.linalg.inv(cov)
    Bm = fit.base_B
    pulls = np.zeros((nmc, 2))
    for t in range(nmc):
        th = np.zeros(D.shape[1])
        th[2] = rng.standard_normal()*IF.SIG_IA["A"]
        th[3] = rng.standard_normal()*IF.SIG_IA["eta"]
        delta = mu_delta + Lr @ rng.standard_normal(Lr.shape[1])
        if D.shape[1] > 4:                     # marginalised: the residual is a parameter
            th_true_res = np.linalg.lstsq(fit.Lfit, delta, rcond=None)[0]
            th[4:] = th_true_res
            data = Bm @ (D[:, :4] @ th[:4]) + Bm @ (Rmap @ delta)
        else:
            data = Bm @ (D @ th) + Bm @ (Rmap @ delta)
        data = data + Ld @ rng.standard_normal(len(data))
        est = fit.Finv @ (D.T @ (Bm.T @ (Ci @ data)))
        pulls[t] = fit.g @ (est - th)/fit.sig
    return (pulls**2).mean(0)


def toys():
    rng = np.random.default_rng(1)
    # tents and spline
    k = np.logspace(-3, 1.5, 400)
    check("tents sum to 1", float(np.max(np.abs(IF.tents_on(k, IF.FINE).sum(0) - 1))), 1e-12)
    S = IF.spline_basis(IF.FINE)
    check("spline of a constant is constant", float(np.max(np.abs(S.sum(0) - 1))), 1e-12)
    lin = np.log(IF.NODES) @ S
    check("spline of ln k is ln k inside", float(np.max(np.abs(lin - np.log(IF.FINE)))), 1e-10)
    # the exponential kernel: tridiagonal inverse on uneven nodes
    x = np.sort(rng.uniform(0, 2, 7))
    K = np.exp(-np.abs(x[:, None] - x[None, :])/0.5)
    Q = np.linalg.inv(K)
    off = np.abs(Q[np.abs(np.subtract.outer(range(7), range(7))) > 1]).max()
    check("exp kernel: precision is tridiagonal", float(off/np.abs(Q).max()), 1e-12)
    r = np.exp(-np.diff(x)/0.5)
    check("exp kernel: off-diagonal -r/(1-r^2)", float(np.max(np.abs(np.diag(Q, 1) + r/(1 - r**2)))), 1e-9)
    # the projection
    zs = np.array([0.2, 0.4, 0.75, 1.0, 1.5])
    nodes = np.logspace(-2, 1, 6)
    Sg = IF.residual_cov(nodes, zs, 0.7, 0.4, "exp", IF.XS, 0.3)
    U = IF.template_directions(zs, IF.XS, len(nodes))
    P = IF.projected_precision(Sg, U)
    check("projection annihilates the template", float(np.abs(P @ U).max()/np.abs(P).max()), 1e-10)
    w = np.linalg.eigvalsh(P)
    check("projection is semidefinite", float(max(-w.min(), 0)/w.max()), 1e-10)
    check("projection has rank n - 2", float(abs(np.sum(w > 1e-9*w.max()) - (len(w) - 2))), 0)
    check("matched identity P Sigma P = P", float(np.abs(P @ Sg @ P - P).max()/np.abs(P).max()), 1e-8)
    Nn = np.linalg.svd(U.T)[2][2:].T                       # basis of the complement of U
    Pn = Nn @ np.linalg.solve(Nn.T @ Sg @ Nn, Nn.T)
    check("null-space form", float(np.abs(Pn - P).max()/np.abs(P).max()), 1e-8)
    W = 1e3
    Pw = np.linalg.inv(Sg + U @ np.diag([W**2, W**2]) @ U.T)
    check("Woodbury limit (W = 1e3)", float(np.abs(Pw - P).max()/np.abs(P).max()), 1e-4)
    # coverages and box against Monte Carlo
    mu, V = 0.4, 1.7
    z = mu + np.sqrt(V)*rng.standard_normal(400000)
    check("coverage_1d vs MC", float(abs(IF.coverage_1d(mu, V) - np.mean(np.abs(z) < 1))), 3e-3)
    Cn = np.array([[1.0, 0.6], [0.6, 1.0]])
    m2, S2 = np.array([0.3, -0.5]), np.array([[1.4, 0.2], [0.2, 0.8]])
    x2 = rng.multivariate_normal(m2, S2, 400000)
    q = -2*np.log(1 - IF.ONE_SIGMA)
    mc = np.mean(np.einsum("ni,ij,nj->n", x2, np.linalg.inv(Cn), x2) <= q)
    check("coverage_2d vs MC", float(abs(IF.coverage_2d(Cn, m2, S2) - mc)), 3e-3)
    check("coverage_2d of the fit's own Gaussian is 68.3%",
          float(abs(IF.coverage_2d(Cn, np.zeros(2), Cn) - IF.ONE_SIGMA)), 1e-7)
    mb, Sb = np.array([0.05, -0.1]), np.array([[0.02, 0.012], [0.012, 0.03]])
    xb = rng.multivariate_normal(mb, Sb, 400000)
    check("box_probability vs MC", float(abs(IF.box_probability(mb, Sb)
                                           - np.mean(np.all(np.abs(xb) < IF.TOL, 1)))), 3e-3)
    Sr = np.outer([0.1, 0.2], [0.1, 0.2])
    xr = mb + rng.standard_normal(400000)[:, None]*np.array([0.1, 0.2])
    check("box_probability, rank one, vs MC", float(abs(IF.box_probability(mb, Sr)
                                                      - np.mean(np.all(np.abs(xr) < IF.TOL, 1)))), 3e-3)
    # a whole toy analysis
    b, nodes, cov = toy_base(rng)
    mask = np.ones(cov.shape[0], bool)
    b.cov = cov
    rigid, G = IF.rigid_fit(b, mask)
    Rmap = b.knob_map(IF.XS, nodes)
    Kt = IF.residual_cov(nodes, b.zs, 0.8, 0.5, "exp", IF.XS, 0.0, sigma=0.03)
    mu_d = 0.02*np.sin(np.arange(Rmap.shape[1]))
    A = rigid.shift(Rmap)
    mu_z = rigid.shift(Rmap @ mu_d)
    sc = IF.scores(rigid, A, Kt, mu_z)
    rigid.base_B = b.B
    msq = experiments(rigid, Rmap, Kt, mu_d, rng, 20000, cov)
    for i, p in enumerate(IF.SCORED):
        check(f"rigid fit: eta^2 of {p} vs MC (rel.)", float(abs(sc[p]["eta"]**2 - msq[i])/msq[i]), 0.03)
    smax, lim, rms = IF.rms_budget(A, Kt/0.03**2)
    sc2 = IF.scores(rigid, A, smax**2*Kt/0.03**2)
    check("rms_budget puts the limiting eta at sqrt(1 + 0.3^2)",
          float(abs(sc2[lim]["eta"] - IF.ETA_MAX)), 1e-9)
    s95 = IF.box_budget(A, Kt/0.03**2)
    check("box_budget gives 95% inside", float(abs(IF.box_probability(np.zeros(2), s95**2*A @ (Kt/0.03**2) @ A.T)
                                                   - 0.95)), 1e-8)
    # marginalised, matched: eta = 1 for both parameters, no drift
    fm = IF.marginalised_fit(b, G, Rmap, Kt)
    Am = fm.shift(Rmap)
    scm = IF.scores(fm, Am, Kt)
    check("matched marginalised fit: eta = 1",
          float(max(abs(scm[p]["eta"] - 1) for p in IF.SCORED)), 1e-8)
    fm.base_B = b.B
    fm.Lfit = IF.factor(Kt)
    msq = experiments(fm, Rmap, Kt, np.zeros(Rmap.shape[1]), rng, 20000, cov)
    check("matched marginalised fit: eta^2 = 1 vs MC (worst)", float(np.max(np.abs(msq - 1))), 0.03)
    # a prior wider than the truth: eta <= 1
    fw = IF.marginalised_fit(b, G, Rmap, 4*Kt)
    scw = IF.scores(fw, fw.shift(Rmap), Kt)
    check("prior wider than the truth: eta <= 1", float(max(scw[p]["eta"] for p in IF.SCORED)), 1.0)
    # factor form against the projected precision
    Kf = IF.residual_cov(nodes, b.zs, 0.8, 0.5, "exp", IF.XS, 0.0, sigma=0.05)
    f1 = IF.marginalised_fit(b, G, Rmap, Kf)
    f2 = IF.marginalised_fit_projected(b, G, Rmap, Kf, IF.template_directions(b.zs, IF.XS, len(nodes)))
    check("factor form vs projected precision: sigma (rel.)", float(np.max(np.abs(f1.sig/f2.sig - 1))), 0.02)


def on_base():
    common = os.path.join(HERE, "ia_base_common.npz")
    for tr in ("S1", "prod"):
        path = os.path.join(HERE, f"ia_base_{tr}.npz")
        if not os.path.exists(path):
            print(f"skip  base checks of {tr}: no {os.path.basename(path)}")
            continue
        base = IF.Base(path, common)
        for name, v in json.loads(str(np.load(path)["checks"])).items():
            if name == "sum_vs_deta":
                # J's eta column is a central difference (step ETA_STEP), so it
                # differs from the exact ln x response by O(step^2 (2 ln x)^2/6);
                # the base must reproduce that same difference
                print(f"info  {tr}: exact d/deta vs J's central difference: {v:.3e}")
                continue
            check(f"{tr}: stored check {name}", v, 1e-6)
        d = np.load(path)
        lx = np.log((1 + base.zs)/IF.X0)
        GI, II = d["R_GI"].sum(1), d["R_II"].sum(1)
        fd = ((np.sinh(ETA_STEP*lx)/ETA_STEP)[:, None]*GI
              + (np.sinh(2*ETA_STEP*lx)/ETA_STEP)[:, None]*II).sum(0)
        check(f"{tr}: base reproduces J's eta column (central difference)",
              float(np.max(np.abs(fd - base.J[:, 3]))/np.max(np.abs(base.J[:, 3]))), 1e-6)
        Rmap = base.knob_map(IF.XS)
        U = IF.template_directions(base.zs, IF.XS)
        dA = base.B @ (Rmap @ U[:, 0])
        check(f"{tr}: constant residual reproduces dxi/dA",
              float(np.max(np.abs(dA - base.J[:, 2]))/np.max(np.abs(base.J[:, 2]))), 1e-6)
        rng = np.random.default_rng(3)
        for cut, mask in base.masks.items():
            rigid, G = IF.rigid_fit(base, mask)
            K = IF.residual_cov(IF.NODES, base.zs, 1.0, 0.3, "exp", IF.XS, 0.0, sigma=0.02)
            A = rigid.shift(Rmap)
            sc = IF.scores(rigid, A, K)
            z = A @ (IF.factor(K) @ rng.standard_normal((IF.factor(K).shape[1], 20000)))
            for i, p in enumerate(IF.SCORED):
                mc = 1 + np.mean(z[i]**2)
                check(f"{tr} {cut}: eta^2 of {p} vs MC of the residual (rel.)",
                      float(abs(sc[p]["eta"]**2 - mc)/mc), 0.03)
            f1 = IF.marginalised_fit(base, G, Rmap, K)
            f2 = IF.marginalised_fit_projected(base, G, Rmap, K, U)
            check(f"{tr} {cut}: factor form vs projected precision, sigma (rel.)",
                  float(np.max(np.abs(f1.sig/f2.sig - 1))), 0.02)


if __name__ == "__main__":
    toys()
    on_base()
    print(f"{sum(RESULTS)} of {len(RESULTS)} checks pass")
    sys.exit(0 if all(RESULTS) else 1)
