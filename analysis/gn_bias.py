"""Parameter bias from IA mismodeling by iterated Fisher (Gauss-Newton) fits.

Replaces the Cobaya BOBYQA minimizer for the scale-cut scan. For noiseless
synthetic data near the truth, the MAP of the analysis model solves

    theta <- theta + (F + P)^-1 [ D^T C^-1 (d - m(theta)) - P (theta - mu) ]

with D = dm/dtheta (5-point stencil), F = D^T C^-1 D, and P, mu the
Gaussian prior precisions and means (photo-z 0.002, shear m 0.005). Each
iteration costs 4 data vectors per free parameter; the derivative calls
are independent and run in parallel over MPI ranks. A step that lowers
the posterior is halved (up to 5 times) before it is accepted.

The truth data vector is computed by THIS pipeline (TATT at the truth)
rather than read from the Cobaya-printed .modelvector, so model and truth
share every numerical setting and the matched-model fit returns the truth
exactly. The script also reports chi2 between the two truth vectors as a
consistency diagnostic: it should be << 1.

Everything runs with cosmolike's TATT model; the NLA fit is TATT with
A2 = eta2 = bTA = 0, which reduces identically to NLA in cosmolike.

Output: MAP, chi2_min, S8 and Omega_m shifts, sigmas from (F + P)^-1 at
the MAP, the 1D and 2D shifts in sigma, and a JSON file with everything.

Run inside cocoa from projects/roman_real, e.g. (one command):

  mpirun -n 8 --oversubscribe --bind-to none python gn_bias.py \\
      --dataset tatt_kuv0p5.dataset --model nla --out gn_nla_cs_kuv0p5.json

Options: --model tatt for the matched reference; --probe 3x2pt / 2x2pt;
--fix-nuisance to fix photo-z and m at the truth (faster, for the scan).

External CMB (Option A, a recentred Planck parameter prior):
  --ext-cov planck18_cov.txt   frees H0, ns, Omega_b and adds the Gaussian
  prior N(truth, C_ext) on every parameter named in the file (getdist
  .covmat format: first line '# name1 name2 ...', then the matrix). Make
  the file with make_ext_cov.py. --ext-cov-scale s multiplies C_ext by s
  (e.g. 0.25 for a CMB experiment with half Planck's errors).
"""
import argparse
import functools
import json

import numpy as np

import cosmolike_roman_real_interface as ci
import cosmolike_roman_real_notebook_wrappers as nw
from fisher_sigma import (TRUTH, PROBES, HAS_SHAPES, BOUNDS, parameter_list,
                          make_dv, S8_cov, two_d_sigma, install_lowk_powerlaw,
                          match_likelihood_settings)

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None
RANK = COMM.Get_rank() if COMM else 0
SIZE = COMM.Get_size() if COMM else 1


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def truth_theta(names):
    """The truth of the TATT data vector for any parameter list."""
    tr = {"As_1e9": TRUTH["As_1e9"], "omegam": TRUTH["omegam"],
          "H0": TRUTH["H0"], "ns": TRUTH["ns"], "omegab": TRUTH["omegab"],
          "A1": TRUTH["A1"], "eta1": TRUTH["eta1"], "A2": TRUTH["A2"],
          "eta2": TRUTH["eta2"], "bTA": TRUTH["bTA"]}
    from fisher_sigma import B1_TRUTH
    for i in range(8):
        tr[f"DZ{i+1}"] = 0.0
        tr[f"M{i+1}"] = 0.0
        tr[f"B1_{i+1}"] = B1_TRUTH[i]
    return np.array([tr[n] for n in names], dtype=float)


def ext_precision(path, names, scale=1.0, diag_prior=None):
    """Precision matrix (len(names) square) of an external Gaussian prior.

    The file's covariance is marginal over everything it does not list, so
    taking the sub-block of the parameters we vary and inverting it is the
    correct marginal prior. Parameters not in the file get no prior here.
    """
    with open(path) as f:
        head = f.readline()
    if not head.startswith("#"):
        raise SystemExit(f"{path}: first line must be '# name1 name2 ...'")
    fnames = head[1:].split()
    C = np.atleast_2d(np.loadtxt(path))*scale
    if C.shape != (len(fnames), len(fnames)):
        raise SystemExit(f"{path}: {len(fnames)} names but a {C.shape} matrix")
    use = [n for n in fnames if n in names]
    if not use:
        raise SystemExit(f"{path}: none of {fnames} is a free parameter")
    if diag_prior is not None:
        dbl = [n for n in use if diag_prior[names.index(n)] > 0]
        if dbl:
            raise SystemExit(f"{dbl} have both --prior-* and --ext-cov priors")
    idx = [fnames.index(n) for n in use]
    Pin = np.linalg.inv(C[np.ix_(idx, idx)])
    P = np.zeros((len(names), len(names)))
    pos = [names.index(n) for n in use]
    P[np.ix_(pos, pos)] = Pin
    say(f"external prior on {use} from {path} (scale {scale:g}); "
        f"sigmas {np.sqrt(np.diag(C[np.ix_(idx, idx)])).round(5).tolist()}")
    return P


def parallel_derivatives(dv, theta, names, steps):
    """D[:, j] = dm/dtheta_j, 4 evaluations per column, spread over ranks."""
    coeffs = (-1/12, 8/12, -8/12, 1/12)
    shifts = (2, 1, -1, -2)
    tasks = [(j, s, c) for j in range(len(theta))
             for s, c in zip(shifts, coeffs)]
    import time
    mine = []
    todo = tasks[RANK::SIZE]
    t0 = time.time()
    for k, t in enumerate(todo):
        j, s, c = t
        th = theta.copy()
        th[j] += s*steps[j]
        mine.append((j, c*dv(th, names)/steps[j]))
        if RANK == 0:
            el = time.time() - t0
            print(f"    derivatives: {k+1}/{len(todo)} on rank 0 "
                  f"({el/(k+1):.1f} s each, ~{el/(k+1)*(len(todo)-k-1)/60:.1f}"
                  f" min left in this pass)", flush=True)
    allres = COMM.allgather(mine) if COMM else [mine]
    D = None
    for chunk in allres:
        for j, col in chunk:
            if D is None:
                D = np.zeros((col.size, len(theta)))
            D[:, j] += col
    return D


def evaluate(dv, theta, names):
    """One data vector, computed on rank 0 and broadcast."""
    v = dv(theta, names) if RANK == 0 else None
    return COMM.bcast(v, root=0) if COMM else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", choices=["nla", "tatt"], default="nla")
    ap.add_argument("--probe", choices=PROBES, default="xi",
                    help="xi, gammat, wtheta, 2x2pt (gammat+wtheta), xi_ggl, "
                         "xi_gg, 3x2pt; the dataset's mask must keep that "
                         "probe's entries")
    ap.add_argument("--free-cosmo", action="store_true",
                    help="also vary H0, ns, Omega_b (flat priors of the yamls)")
    ap.add_argument("--prior-ns", type=float, default=None,
                    help="Gaussian sigma on ns (with --free-cosmo)")
    ap.add_argument("--prior-H0", type=float, default=None)
    ap.add_argument("--prior-omegab", type=float, default=None)
    ap.add_argument("--ext-cov", default=None,
                    help="external (CMB) parameter covariance, centred on "
                         "the truth; implies --free-cosmo")
    ap.add_argument("--ext-cov-scale", type=float, default=1.0)
    ap.add_argument("--fix-nuisance", action="store_true",
                    help="fix photo-z shifts and shear m at the truth")
    ap.add_argument("--halofit", default="mead2020")
    ap.add_argument("--limber-gs", type=int, default=0, choices=[0, 1],
                    help="gamma_t projection: 0 = non-Limber below l=150 "
                         "(the roman_real yaml default), 1 = Limber (the "
                         "old GN behaviour)")
    ap.add_argument("--emul", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=100000)
    ap.add_argument("--camb-kmax", type=float, default=10.0)
    ap.add_argument("--camb-boost", type=float, default=1.1)
    ap.add_argument("--k-per-logint", type=int, default=15)
    ap.add_argument("--path", default=None)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="stop when every |step| < tol * sigma")
    ap.add_argument("--out", default=None, help="JSON with the results")
    ap.add_argument("--truth-from-file", action="store_true",
                    help="fit the .dataset's data vector (Cobaya-printed) "
                         "instead of this pipeline's TATT truth")
    ap.add_argument("--lowk-powerlaw", type=float, default=None,
                    help="replace P(k) below this k [h/Mpc] by a power law "
                         "(as Cobaya's extrap_kmin does), e.g. 3e-4")
    a = ap.parse_args()
    if a.ext_cov:
        a.free_cosmo = True

    nw.cnu.get_camb_cosmology = functools.partial(
        nw.cnu.get_camb_cosmology, halofit_version=a.halofit)
    install_lowk_powerlaw(a.lowk_powerlaw)
    cfg = dict(data_file=a.dataset, lmax=a.lmax, non_linear_emul=a.emul,
               IA_model=1, IA_code=0, IA_redshift_evolution=3)
    if a.path:
        cfg["path"] = a.path
    nw.configure(**cfg)
    nw.init_cosmolike(CLprobe=a.probe, with_data=True)
    match_likelihood_settings(adopt_limber_gs=a.limber_gs)
    say(f"gamma_t projection: {'Limber' if a.limber_gs else 'non-Limber below l=150 (as the Cobaya yamls)'}")
    invcov = np.asarray(ci.get_inv_cov_masked(), dtype=np.float64)
    mask = np.asarray(ci.get_mask(), dtype=float)

    # truth vector from this pipeline, and the Cobaya-printed one
    tatt_params = parameter_list("tatt", a.probe)
    tnames = [p[0] for p in tatt_params]
    d = evaluate(make_dv("tatt", a.probe, a), truth_theta(tnames), tnames)
    ndata = int(np.count_nonzero(d))
    say(f"probe {a.probe}: {ndata} data points survive the mask")
    if ndata == 0:
        raise SystemExit("the mask removes every entry of this probe: make "
                         "the mask with make_uv_scale_cuts.py --probe 3x2pt "
                         "(or the matching probe) and point the .dataset at it")
    if a.probe not in HAS_SHAPES:
        say("note: w(theta) has no IA, so the NLA and TATT fits coincide; "
            "this run only gives sigmas")
    d_file = np.asarray(ci.get_dv_masked(), dtype=np.float64)*(d != 0)
    r = d - d_file
    say(f"truth check: chi2(pipeline truth vs data_file) = "
        f"{float(r @ invcov @ r):.4g}  (should be << 1)")
    if a.truth_from_file:
        # fit the Cobaya-printed data vector instead of this pipeline's
        # truth: the shifts are then what a Cobaya analysis of that file
        # would find with this pipeline's model (a cross-check of the
        # truth mismatch). The TATT fit no longer starts at chi2 = 0; its
        # shift is the part caused by the mismatch alone.
        d = d_file
        say("fitting the data_file vector (--truth-from-file)")

    params = parameter_list(a.model, a.probe, a.free_cosmo,
                            {"ns": a.prior_ns, "H0": a.prior_H0,
                             "omegab": a.prior_omegab})
    if a.fix_nuisance:
        params = [p for p in params
                  if not (p[0].startswith("DZ") or p[0].startswith("M"))]
    names = [p[0] for p in params]
    steps = np.array([p[2] for p in params])
    prior_prec = np.array([0.0 if p[3] is None else 1/p[3]**2 for p in params])
    mu = truth_theta(names)  # Gaussian priors are centred on the truth (0)
    P = np.diag(prior_prec)
    if a.ext_cov:
        P = P + ext_precision(a.ext_cov, names, a.ext_cov_scale,
                              diag_prior=prior_prec)

    dv_fit = make_dv(a.model, a.probe, a)
    if a.fix_nuisance:
        base = dv_fit

        def dv_fit(theta, nm):  # pad the fixed nuisances with zeros
            full = list(nm) + [f"DZ{i+1}" for i in range(8)] \
                            + [f"M{i+1}" for i in range(8)]
            return base(np.r_[theta, np.zeros(16)], full)

    def logpost(theta, m):
        res = d - m
        chi2 = float(res @ invcov @ res)
        dth = theta - mu
        pr = float(dth @ P @ dth)
        return chi2, chi2 + pr

    theta = truth_theta(names)  # start at the truth
    m = evaluate(dv_fit, theta, names)
    chi2, post = logpost(theta, m)
    say(f"model={a.model} probe={a.probe} free={len(names)} ranks={SIZE}")
    say(f"iter 0: chi2 = {chi2:.5f}")

    for it in range(1, a.max_iter + 1):
        D = parallel_derivatives(dv_fit, theta, names, steps)
        F = D.T @ invcov @ D
        A = F + P
        g = D.T @ invcov @ (d - m) - P @ (theta - mu)
        if np.linalg.cond(A) > 1e14:
            say("warning: (F + P) is nearly singular: a parameter is "
                "unconstrained by this probe (e.g. b1 vs A_s in gammat "
                "alone); fit gammat together with wtheta (2x2pt)")
        step = np.linalg.solve(A, g)
        sig = np.sqrt(np.diag(np.linalg.inv(A)))
        lo = np.array([BOUNDS.get(n, (-np.inf, np.inf))[0] for n in names])
        hi = np.array([BOUNDS.get(n, (-np.inf, np.inf))[1] for n in names])
        lam = 1.0
        for _ in range(6):
            trial = np.clip(theta + lam*step, lo, hi)  # flat prior edges
            mt = evaluate(dv_fit, trial, names)
            c2, pt = logpost(trial, mt)
            if pt <= post + 1e-10:
                break
            lam *= 0.5
        theta, m, chi2, post = trial, mt, c2, pt
        rel = np.max(np.abs(lam*step)/sig)
        say(f"iter {it}: chi2 = {chi2:.5f}  max|step|/sigma = {rel:.2e}"
            f"  (damping {lam:g})")
        if rel < a.tol:
            break

    # sigmas at the MAP (the last iteration's derivatives: its final step
    # was below tol*sigma, so they are taken at the MAP to that precision)
    C = np.linalg.inv(D.T @ invcov @ D + P)
    if RANK != 0:
        return
    iO = names.index("omegam")
    S8_t, _ = S8_cov(names, truth_theta(names), C)
    S8_m, C2 = S8_cov(names, theta, C)
    at_edge = [n for n, v in zip(names, theta) if n in BOUNDS
               and (np.isclose(v, BOUNDS[n][0]) or np.isclose(v, BOUNDS[n][1]))]
    if at_edge:
        print(f"warning: MAP at a flat-prior edge for {at_edge}: the "
              f"Gaussian sigmas there are not reliable")
    sS8, sOm = np.sqrt(np.diag(C2))
    dlt = np.array([S8_m - S8_t, theta[iO] - TRUTH["omegam"]])
    q, n2 = two_d_sigma(dlt, C2)

    print("\nMAP:")
    for n, v, s in zip(names, theta, np.sqrt(np.diag(C))):
        print(f"  {n:7s} = {v:+.5f}  (sigma {s:.4f})")
    print(f"\nchi2_min = {chi2:.4f}")
    print(f"S8 = {S8_m:.5f} (truth {S8_t:.5f}),  dS8 = {dlt[0]:+.5f} "
          f"= {dlt[0]/sS8:+.2f} sigma   [sigma(S8) = {sS8:.5f}]")
    print(f"Om = {theta[iO]:.5f} (truth {TRUTH['omegam']}),  dOm = "
          f"{dlt[1]:+.5f} = {dlt[1]/sOm:+.2f} sigma   [sigma(Om) = {sOm:.5f}]")
    print(f"2D: dchi2 = {q:.3f} -> {n2:.2f} sigma  "
          f"({'PASS' if n2 < 0.3 else 'FAIL'} at 0.3 sigma)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(dataset=a.dataset, model=a.model, probe=a.probe,
                           fix_nuisance=a.fix_nuisance,
                           free_cosmo=a.free_cosmo,
                           ext_cov=a.ext_cov, ext_cov_scale=a.ext_cov_scale,
                           truth_from_file=a.truth_from_file,
                           limber_gs=a.limber_gs, names=names,
                           map=theta.tolist(), sigma=np.sqrt(np.diag(C)).tolist(),
                           chi2_min=chi2, S8=S8_m, S8_truth=S8_t,
                           dS8=dlt[0], dOm=dlt[1], sigma_S8=sS8,
                           sigma_Om=sOm, rho=C2[0, 1]/(sS8*sOm),
                           shift_2d_sigma=n2), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
