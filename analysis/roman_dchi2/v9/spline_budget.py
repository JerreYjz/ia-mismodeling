#!/usr/bin/env python3
"""The IA error budget with a cardinal spline basis (Miranda, Hu & Dvorkin 2015, Sec. II B).

arXiv:1411.5956 writes a deviation of a function of ln s as
    delta G'(ln s) = sum_i p_i B_i(ln s),   B_i(ln s_j) = delta_ij,
a natural cubic spline through unit vectors at nodes s_i (the grid extended
beyond the range of interest so the basis has negligible support outside),
fits the coefficients p_i with everything else, and takes the principal
components of their covariance C_ij = <p_i p_j> - <p_i><p_j> = sum_a S_ia
sigma_a^2 S_ja. Here the function is the fractional error of one injected
FLAMINGO IA spectrum (GI or II), the same at every redshift:

    P_X(k, z) -> P_X(k, z) [1 + sum_i p_i B_i(ln k)],

with nodes n per decade (n = 2, 3) on k_i = 0.2 x 10^(j/n), free for 0.2 <= k_i
<= 2 h/Mpc and held at zero elsewhere on the grid (which runs past 0.01 and
10 h/Mpc). B_i is written on the 61 tents of the v9 base (exact: xi is linear
in the tables). The analysis model is the truth itself with Omega_m, A_s and
the IA amplitude A and evolution eta free (A, eta with the chain-width
Gaussian priors), as in v9 (rigid fit). Three budgets per truth, cut and
spectrum:

  node      the tolerance of one coefficient: |p_i| at which Omega_m or S8
            moves by TOL = 0.3 of its error, the others zero
            (p_i^max = TOL / max_p |a_pi|, a the shift per unit p in sigma);
  rms       independent coefficients of rms sigma: the rms shift is
            sigma |a_p|, so sigma_max = TOL / max_p |a_p|;
  free      the coefficients fitted with the cosmology (the paper's choice),
            with a Gaussian prior of width s_p (flat: s_p = inf): the error
            of (Omega_m, S8) against the rigid fit, the posterior errors of
            the p_i, and the principal components of their covariance.

Checks: B_i(k_j) = delta_ij; the rigid-fit errors against the bands data of
the note; a one-sign Gaussian bump through this code against the old bump
band (bands_prod, bands_s1) where it is linear (< 10%); the shift of one
node against a direct least-squares fit in data space.

Output: spline_budget.json, spline_budget_log.txt.
    python spline_budget.py        (cocoa Python, seconds)
"""
import json
import os
import sys

import numpy as np
import scipy.linalg
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ia_fisher as IF                                             # noqa: E402

NOTE = os.path.join(HERE, "..", "..", "..", "notes", "ia_error_budget", "plots")
K_LO, K_HI, K_REF = 0.2, 2.0, 0.2
DENSITIES = (2, 3)
TRUTHS = ("S1", "EB", "prod")
PRIORS = (0.01, 0.03, 0.1, 0.3, 1.0, np.inf)
XSETS = {"GI": ("GI",), "II": ("II",), "GI+II": ("GI", "II")}
KBUMP = np.logspace(np.log10(0.05), np.log10(8.0), 41)
KBAND = np.logspace(np.log10(0.03), 1, 241)


def grid(n):
    """Nodes n per decade through K_REF, past both ends of the tents, and the free ones."""
    j0 = int(np.floor(n*np.log10(IF.FINE[0]/K_REF))) - 1
    j1 = int(np.ceil(n*np.log10(IF.FINE[-1]/K_REF))) + 1
    nodes = K_REF*10.0**(np.arange(j0, j1 + 1)/n)
    free = np.where((nodes >= K_LO*(1 - 1e-9)) & (nodes <= K_HI*(1 + 1e-9)))[0]
    return nodes, free


def cardinal(k, nodes, free):
    """(n_free, len(k)): the natural cubic cardinal splines in ln k of the free nodes."""
    out = []
    for i in free:
        e = np.zeros(len(nodes))
        e[i] = 1.0
        out.append(CubicSpline(np.log(nodes), e, bc_type="natural")(np.log(k)))
    return np.array(out)


def spline_map(base, Xs, Bk):
    """R (nb, len(Xs) * n_free): coefficient (X, i) -> base knobs, same at every z."""
    nz = len(base.zs)
    nf = Bk.shape[0]
    R = np.zeros((base.nb, len(Xs)*nf))
    for q, X in enumerate(Xs):
        for s in range(nz):
            R[base.iknob[X][s][:, None], q*nf + np.arange(nf)[None, :]] = Bk.T
    return R


def rigid(base, G):
    M = np.zeros((base.nb, base.npar))
    M[base.ij, base.ij] = 1.0
    P = IF.template_prior(base.npar)
    return IF.Fit(G, M, P, P, base.g.copy())


def freed(base, G, R, s_p):
    """The rigid fit with the spline coefficients R free, Gaussian prior s_p."""
    M0 = np.zeros((base.nb, base.npar))
    M0[base.ij, base.ij] = 1.0
    M = np.hstack([M0, R])
    n = M.shape[1]
    P = np.zeros((n, n))
    P[:base.npar, :base.npar] = IF.template_prior(base.npar)
    if np.isfinite(s_p):
        P[base.npar:, base.npar:] = np.eye(n - base.npar)/s_p**2
    g = np.hstack([base.g, np.zeros((2, n - base.npar))])
    return IF.Fit(G, M, P, P, g)


def main():
    logf = open(os.path.join(HERE, "spline_budget_log.txt"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    c3 = np.load(os.path.join(HERE, "ia_base_3x2_common.npz"))
    out = dict(about=__doc__.split("\n\n")[0], tol=IF.TOL, k_range=[K_LO, K_HI], priors=[float(p) for p in PRIORS],
               fine=IF.FINE.tolist(), densities={}, truths={})
    kplot = np.logspace(-2, 1, 301)
    for n in DENSITIES:
        nodes, free = grid(n)
        Bn = cardinal(nodes[free], nodes, free)
        assert np.allclose(Bn, np.eye(len(free)), atol=1e-12), "not cardinal"
        out["densities"][str(n)] = dict(nodes=nodes.tolist(), free=nodes[free].tolist(),
                                        k=kplot.tolist(), basis=cardinal(kplot, nodes, free).tolist())
        log(f"{n} per decade: free nodes {np.round(nodes[free], 3).tolist()}, grid {nodes[0]:.4f} to {nodes[-1]:.1f}")

    for tr in TRUTHS:
        base = IF.Base(os.path.join(HERE, f"ia_base_{tr}.npz"), os.path.join(HERE, "ia_base_common.npz"))
        cuts = {"none": base.masks["none"], "K0/D 0.3": base.masks["K03"], "K0/D 0.2": c3["xi|0.2"].astype(bool)}
        out["truths"][tr] = {}
        for cn, mask in cuts.items():
            G = IF.gram(base.B, base.cov, mask)
            fit = rigid(base, G)
            res = dict(npts=int(mask.sum()), sigma=fit.sig.tolist(), dens={}, bump={})
            for X in IF.XS:              # the one-sign Gaussian bump of the old section, for comparison
                bump = []
                for k_c in KBUMP:
                    f = np.exp(-np.log(base.fine/k_c)**2/(2*0.3**2))
                    a = fit.shift(spline_map(base, (X,), f[None, :]))[:, 0]
                    bump.append(float(min(IF.TOL/np.max(np.abs(a)), IF.SIGMA_CAP)))
                res["bump"][X] = bump
            for n in DENSITIES:
                nodes, free = grid(n)
                Bk = cardinal(base.fine, nodes, free)
                rd = {}
                for xn, Xs in XSETS.items():
                    R = spline_map(base, Xs, Bk)
                    A = fit.shift(R)                                   # (2, n_coef) sigma per unit p
                    r = dict(shift=A.tolist())
                    if len(Xs) == 1:
                        t = np.array([min(IF.TOL/np.max(np.abs(A[:, i])), IF.SIGMA_CAP) for i in range(A.shape[1])])
                        r["node_tol"] = t.tolist()
                        r["rms_tol"] = float(min(IF.TOL/np.max(np.linalg.norm(A, axis=1)), IF.SIGMA_CAP))
                        # the band: every coefficient wrong at once, independently, with an rms
                        # proportional to its own one-node tolerance, scaled together until the
                        # rms shift of Omega_m or S8 is TOL; sigma_delta(k)^2 = sum_i sigma_i^2 B_i(k)^2
                        lam = IF.TOL/np.max(np.sqrt((A**2*t[None, :]**2).sum(1)))
                        sig_i = lam*t
                        Bp = cardinal(KBAND, *grid(int(n)))
                        r["band_lambda"] = float(lam)
                        r["band_sigma_i"] = sig_i.tolist()
                        r["band_68"] = np.sqrt((sig_i[:, None]**2*Bp**2).sum(0)).tolist()
                    fr = []
                    for s_p in PRIORS:
                        f = freed(base, G, R, s_p)
                        Cp = f.Finv[base.npar:, base.npar:]
                        w, S = np.linalg.eigh(Cp)
                        o = np.argsort(w)
                        fr.append(dict(prior=float(s_p), sigma=f.sig.tolist(), ratio=(f.sig/fit.sig).tolist(),
                                       sigma_p=np.sqrt(np.diag(Cp)).tolist(),
                                       pc_sigma=np.sqrt(np.clip(w[o], 0, None)).tolist(),
                                       pc_vec=S[:, o].T.tolist(), dropped=f.dropped))
                    r["free"] = fr
                    rd[xn] = r
                res["dens"][str(n)] = rd
                for xn in ("GI", "II"):
                    r = rd[xn]
                    fl = r["free"][-1]
                    log(f"{tr} {cn} ({res['npts']} pts) {n}/dec {xn}: node tol "
                        f"{[round(100*t, 1) for t in r['node_tol']]} %, rms tol {100*r['rms_tol']:.1f}%; "
                        f"freed (flat): sigma(Om, S8) x {np.round(fl['ratio'], 3).tolist()}, sigma(p) "
                        f"{[round(100*s, 1) for s in fl['sigma_p']]} %, PC sigma {[round(100*s, 1) for s in fl['pc_sigma']]} %")
                fl = rd["GI+II"]["free"]
                log(f"{tr} {cn} {n}/dec GI+II freed: sigma(S8) x " + ", ".join(
                    f"{p['prior']:g}: {p['ratio'][1]:.3f}" for p in fl))
            out["truths"][tr][cn] = res

        # checks on this truth
        ref = json.load(open(os.path.join(NOTE, "bands_prod" if tr == "prod" else "bands_s1", "data.json"))) \
            if tr in ("prod", "S1") else None
        if ref is not None:
            for cn, key in (("none", "none"), ("K0/D 0.3", "K0/D 0.3")):
                mask = cuts[cn]
                fit = rigid(base, IF.gram(base.B, base.cov, mask))
                sref = np.array(ref["sigma"][key])
                kc = np.array(ref["k"])
                worst = 0.0
                for X in IF.XS:
                    tol = []
                    for k_c in kc:
                        f = np.exp(-np.log(base.fine/k_c)**2/(2*0.3**2))
                        a = fit.shift(spline_map(base, (X,), f[None, :]))[:, 0]
                        tol.append(IF.TOL/np.max(np.abs(a)))
                    old = np.array(ref["bands"][key][f"bump|{X}"]["up"])
                    ok = (old < 0.1) & (np.array(tol) < 0.1)
                    if ok.any():
                        worst = max(worst, float(np.max(np.abs(np.array(tol)[ok]/old[ok] - 1))))
                log(f"check {tr} {cn}: sigma(Om, S8) {np.round(fit.sig, 5).tolist()} vs note "
                    f"{np.round(sref, 5).tolist()}; Gaussian bump vs old band where < 10%: max rel diff {worst:.1e}")
    # check: one node against a direct least-squares fit (S1, no cut, 3 per decade, GI, node 0.43)
    base = IF.Base(os.path.join(HERE, "ia_base_S1.npz"), os.path.join(HERE, "ia_base_common.npz"))
    mask = base.masks["none"]
    nodes, free = grid(3)
    Bk = cardinal(base.fine, nodes, free)
    R = spline_map(base, ("GI",), Bk)
    dd = (base.B @ R[:, 1])[mask]
    L = scipy.linalg.cholesky(base.cov[np.ix_(mask, mask)], lower=True)
    J = base.B[mask][:, :base.npar]
    P = IF.template_prior(base.npar)
    Aw = np.vstack([scipy.linalg.solve_triangular(L, J, lower=True), np.diag(np.sqrt(np.diag(P)))])
    bw = np.concatenate([scipy.linalg.solve_triangular(L, dd, lower=True), np.zeros(base.npar)])
    p = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    fit = rigid(base, IF.gram(base.B, base.cov, mask))
    log(f"check: node {nodes[free][1]:.3f} GI, direct lstsq {np.round(base.g @ p/fit.sig, 4).tolist()} vs "
        f"{np.round(out['truths']['S1']['none']['dens']['3']['GI']['shift'][0][1], 4)}, "
        f"{np.round(out['truths']['S1']['none']['dens']['3']['GI']['shift'][1][1], 4)}")
    out["k_bump"] = KBUMP.tolist()
    json.dump(out, open(os.path.join(HERE, "spline_budget.json"), "w"))
    figure_data(out)


def figure_data(out):
    """notes/ia_error_budget/plots/spline_{basis,band}/data.json."""
    src = "Producer: analysis/roman_dchi2/v9/spline_budget.py (spline_budget.json). "
    d = {n: {k: v for k, v in out["densities"][n].items()} for n in out["densities"]}
    for n in d:
        d[n]["k"] = out["densities"][n]["k"]
    os.makedirs(os.path.join(NOTE, "spline_basis"), exist_ok=True)
    json.dump(dict(about=src + "Natural cubic cardinal splines in ln k, B_i(k_j) = delta_ij, nodes n per "
                   "decade through 0.2 h/Mpc, free (drawn) for 0.2 <= k_i <= 2, zero at the other grid nodes.",
                   k_range=out["k_range"], densities=d),
              open(os.path.join(NOTE, "spline_basis", "data.json"), "w"), indent=1)
    cuts = ("none", "K0/D 0.3", "K0/D 0.2")
    band = {}
    for tr, T in out["truths"].items():
        band[tr] = {cn: {n: {X: dict(b68=T[cn]["dens"][n][X]["band_68"], lam=T[cn]["dens"][n][X]["band_lambda"],
                                     sigma_i=T[cn]["dens"][n][X]["band_sigma_i"])
                             for X in ("GI", "II")} for n in T[cn]["dens"]} for cn in cuts}
    # the model errors: NLA and TATT fitted to each truth at the fiducial cosmology, no cut
    # (v8/fid_fits.py, stored as drifts in ia_base_<truth>.npz), over the core snapshots
    models = {}
    for tr in out["truths"]:
        dz = np.load(os.path.join(HERE, f"ia_base_{tr}.npz"))
        if "drift|none|NLA|GI" not in dz.files:
            continue
        zs = dz["zs"]
        core = (zs >= 0.2 - 1e-9) & (zs <= 2.0 + 1e-9)
        models[tr] = {m: {X: dict(mean=dz[f"drift|none|{m}|{X}"][core].mean(0).tolist(),
                                  lo=dz[f"drift|none|{m}|{X}"][core].min(0).tolist(),
                                  hi=dz[f"drift|none|{m}|{X}"][core].max(0).tolist()) for X in ("GI", "II")}
                      for m in ("NLA", "TATT")}
    os.makedirs(os.path.join(NOTE, "spline_band"), exist_ok=True)
    json.dump(dict(about=src + "The error band on the spline basis: every coefficient p_i wrong at once, "
                   "independently, with an rms sigma_i proportional to its one-node tolerance and scaled (lam) "
                   "until the rms shift of Omega_m or S8 is 0.3 sigma; the error delta(k) = sum p_i B_i(k) then "
                   "has rms b68(k) = sqrt(sum sigma_i^2 B_i(k)^2) (fraction), its 68% band; the 95% band is "
                   "1.96 b68. Same at every z; rigid fit; Roman xi_+- with example1_cov.",
                   k=KBAND.tolist(), nodes={n: out["densities"][n]["free"] for n in out["densities"]},
                   k_range=out["k_range"], truths=band,
                   models_about="P^model/P^truth - 1 of NLA and TATT fitted to the truth at the fiducial "
                   "cosmology without a cut (IA parameters only; v8/fid_fits.py), mean and range over the "
                   "snapshots 0.2 <= z <= 2, on k_models (the 61 tents)", k_models=IF.FINE.tolist(),
                   models=models),
              open(os.path.join(NOTE, "spline_band", "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
