"""Fisher-forecast sigma(S8), sigma(Omega_m) and their correlation at one
scale-cut mask, for the IA mismodeling minimizer runs.

It reproduces the minimizer setup: the same .dataset (so the same mask and
covariance), the same IA model and truth, the same free parameters
(n_s, H0, Omega_b, w, w0+wa, mnu fixed) and the same Gaussian priors on
the photo-z shifts (0.002) and shear calibrations (0.005). The Fisher matrix
F = D^T C^-1 D + priors uses 5-point-stencil derivatives with absolute,
per-parameter steps; S8 errors come from the Jacobian of
S8 = sigma8 (Omega_m/0.3)^0.5 (sigma8 from CAMB).

With --dS8 and --dOm (MAP minus truth from min_read.py) it also prints the
2D shift in sigma, the number to compare with 0.3.

Run inside cocoa from projects/roman_real:

  python fisher_sigma.py --dataset tatt_kuv0p5.dataset --model nla \\
      --dS8 -0.00273 --dOm -0.00446
  python fisher_sigma.py --dataset tatt_kuv0p5.dataset --model tatt

Notes:
  * A Fisher error is the Gaussian approximation around the truth; for
    the final cuts confirm with one MCMC (same yaml, sampler: mcmc).
  * Report sigma from the model whose shift you quote (NLA) and say so;
    --model tatt gives the (larger) TATT errors for comparison.
  * lens = source photo-z, as in the yamls: one shift per bin moves both.
"""
import argparse
import functools

import numpy as np

import cosmolike_roman_real_interface as ci
import cosmolike_roman_real_notebook_wrappers as nw

# truth of the TATT data vector (yaml override block)
TRUTH = dict(As_1e9=2.1, omegam=0.3, omegab=0.04, H0=67.32, ns=0.96605,
             w=-1.0, w0pwa=-1.0,
             A1=3.80, eta1=0.57, A2=-1.08, eta2=-0.71, bTA=0.18)
B1_TRUTH = [1.18, 1.40, 1.55, 1.71, 1.90, 2.15, 2.52, 2.44]
PRIOR_DZ, PRIOR_M = 0.002, 0.005
# flat prior ranges of the yamls (DES Y3 / cocoa standard)
BOUNDS = {"As_1e9": (0.5, 5.0), "omegam": (0.1, 0.9), "H0": (55.0, 91.0),
          "ns": (0.87, 1.07), "omegab": (0.03, 0.07), "A1": (-8.0, 8.0),
          "eta1": (-5.0, 5.0), "A2": (-5.0, 5.0), "eta2": (-5.0, 5.0),
          "bTA": (-2.0, 2.0)}


PROBES = ["xi", "gammat", "wtheta", "2x2pt", "xi_ggl", "xi_gg", "3x2pt"]
HAS_SHAPES = {"xi", "gammat", "2x2pt", "xi_ggl", "xi_gg", "3x2pt"}  # IA, m
HAS_LENS = {"gammat", "wtheta", "2x2pt", "xi_ggl", "xi_gg", "3x2pt"}  # b1


def match_likelihood_settings(adopt_limber_gs=0, adopt_limber_gg=0,
                              nonlimber_boost=1.0):
    """Projection settings the Cobaya likelihood sets and the notebook
    wrappers do not (_cosmolike_prototype_base.py, combo_*.yaml defaults).

    The one that matters: cosmolike's own default is Limber gamma_t
    (adopt_limber_gs = 1), while the roman_real yamls use the exact
    non-Limber projection below l = 150 (adopt_limber_gs: 0). Without this
    call the GN truth and model are Limber gamma_t and disagree with every
    Cobaya-printed data vector and chain at large theta, most in the
    lens-source overlap pairs (the IA term). Call after nw.init_cosmolike.
    """
    ci.init_nonlimber_accuracy_boost(nonlimber_boost=float(nonlimber_boost))
    ci.init_adopt_limber_gs(adopt_limber_gs=int(adopt_limber_gs))
    ci.init_adopt_limber_gg(adopt_limber_gg=int(adopt_limber_gg))
    ci.init_include_HOD_GX(include_HOD_GX=0)
    ci.init_include_halo_IA(include_halo_IA=0)


def install_lowk_powerlaw(kcut_h=None, verbose=False):
    """Replace the P(k) tables below kcut_h [h/Mpc] by a power law.

    cnu.get_camb_cosmology fills the k grid (down to 10^-4.99 /Mpc) with
    CAMB's interpolator, which below CAMB's lowest computed k only has
    spline extrapolation. The Cobaya likelihood instead asks for
    extrap_kmin = 1e-6, i.e. a power law in that range. Lensing and
    clustering integrals never reach those k, but C-FAST-PT transforms
    the whole P_lin table for the one-loop (TATT) IA terms. With kcut_h
    set, lnP_lin, lnP_nl and lnP_cb below kcut_h are replaced, z by z,
    by the log-log line through the table at kcut_h and 2 kcut_h.
    kcut_h = None leaves the tables unchanged.
    """
    if kcut_h is None:
        return
    orig = nw.cnu.get_camb_cosmology

    def wrapped(*args, **kw):
        out = list(orig(*args, **kw))
        lk = np.asarray(out[0], dtype=float)
        nk, nz = lk.size, np.asarray(out[1]).size
        lo = lk < np.log10(kcut_h)
        if lo.any():
            j0 = int(np.argmax(~lo))
            j1 = int(np.searchsorted(lk, lk[j0] + np.log10(2.0)))
            for i in (2, 3, 9):
                if out[i] is None or len(out[i]) != nk*nz:
                    continue
                P = np.array(out[i], dtype=float).reshape(nk, nz)  # [k, z]
                if verbose:
                    s_lo = (P[1] - P[0])/((lk[1] - lk[0])*np.log(10))
                    print(f"  table {i}: dlnP/dlnk at k={10**lk[0]:.2e} h/Mpc "
                          f"was {s_lo[0]:+.3f} (z=0); power law from "
                          f"k={10**lk[j0]:.2e}")
                s = (P[j1] - P[j0])/(lk[j1] - lk[j0])  # per z, ln P per log10 k
                P[lo] = P[j0][None, :] + (lk[lo] - lk[j0])[:, None]*s[None, :]
                out[i] = P.ravel()
        return tuple(out)

    nw.cnu.get_camb_cosmology = wrapped


def parameter_list(model, probe, free_cosmo=False, cosmo_priors=None):
    """(name, fiducial, absolute step, gaussian prior sigma or None).

    Only parameters the probe responds to are free: w(theta) alone has no
    IA and no shear calibration (a free IA amplitude with no prior would
    make the Fisher matrix singular), xi alone has no galaxy bias.
    """
    cosmo_priors = cosmo_priors or {}
    p = [("As_1e9", TRUTH["As_1e9"], 0.02, None),
         ("omegam", TRUTH["omegam"], 0.003, None)]
    if free_cosmo:  # flat priors (BOUNDS) unless a Gaussian sigma is given
        p += [("H0", TRUTH["H0"], 0.5, cosmo_priors.get("H0")),
              ("ns", TRUTH["ns"], 0.005, cosmo_priors.get("ns")),
              ("omegab", TRUTH["omegab"], 0.001, cosmo_priors.get("omegab"))]
    if probe in HAS_SHAPES:
        p += [("A1", TRUTH["A1"], 0.05, None),
              ("eta1", TRUTH["eta1"], 0.05, None)]
        if model == "tatt":
            p += [("A2", TRUTH["A2"], 0.05, None),
                  ("eta2", TRUTH["eta2"], 0.1, None),
                  ("bTA", TRUTH["bTA"], 0.02, None)]
    p += [(f"DZ{i+1}", 0.0, 0.0005, PRIOR_DZ) for i in range(8)]
    if probe in HAS_SHAPES:
        p += [(f"M{i+1}", 0.0, 0.001, PRIOR_M) for i in range(8)]
    if probe in HAS_LENS:
        p += [(f"B1_{i+1}", B1_TRUTH[i], 0.01, None) for i in range(8)]
    return p


def make_dv(model, probe, args):
    def dv(theta, names):
        t = dict(zip(names, theta))
        dz = [t.get(f"DZ{i+1}", 0.0) for i in range(8)]
        m = [t.get(f"M{i+1}", 0.0) for i in range(8)]
        A1 = [t.get("A1", TRUTH["A1"]), t.get("eta1", TRUTH["eta1"])] + [0]*6
        tatt = model == "tatt" and "A2" in t
        A2 = ([t["A2"], t["eta2"]] + [0]*6) if tatt else [0]*8
        BTA = ([t["bTA"]] + [0]*7) if tatt else [0]*8
        kw = dict(M=m, shear_photoz_bias=dz, A1=A1, A2=A2, BTA=BTA)
        if probe in HAS_LENS:
            kw.update(lens_photoz_bias=dz,
                      B1=[t[f"B1_{i+1}"] for i in range(8)],
                      B2=nw.ZEROS8, B_MAG=nw.ZEROS8, B3nl=nw.ZEROS8,
                      BK=nw.ZEROS8, PM=nw.PM_FID)
        nw._set_state(t["omegam"], t.get("omegab", TRUTH["omegab"]),
                      t.get("H0", TRUTH["H0"]), t.get("ns", TRUTH["ns"]),
                      t["As_1e9"], TRUTH["w"], TRUTH["w0pwa"],
                      1.0, args.camb_kmax, args.k_per_logint,
                      args.camb_boost, 1.0, 0, args.emul, **kw)
        return np.array(ci.compute_data_vector_masked(), dtype=np.float64)
    return dv


def fisher(dv, params, invcov):
    names = [p[0] for p in params]
    theta0 = np.array([p[1] for p in params], dtype=float)
    cols = []
    for j, (name, _, step, _) in enumerate(params):
        acc = 0.0
        for s, c in zip((2, 1, -1, -2), (-1/12, 8/12, -8/12, 1/12)):
            th = theta0.copy()
            th[j] += s*step
            acc = acc + c*dv(th, names)
        cols.append(acc/step)
        print(f"  d/d{name:7s} done ({j+1}/{len(params)})", flush=True)
    D = np.column_stack(cols)
    F = D.T @ (invcov @ D)
    for j, (_, _, _, prior) in enumerate(params):
        if prior is not None:
            F[j, j] += 1.0/prior**2
    return F


def s8_gradient(point):
    """S8 and dS8/dp for every cosmological parameter in point.

    point: dict with As_1e9 and omegam, optionally H0, ns, omegab (the
    others are taken at the truth). Finite differences in CAMB, except
    A_s (sigma8 ~ sqrt(A_s) exactly).
    """
    import camb
    full = {k: point.get(k, TRUTH[k]) for k in
            ("As_1e9", "omegam", "H0", "ns", "omegab")}

    def S8_of(q):
        h = q["H0"]/100
        pars = camb.CAMBparams()
        omnuh2 = nw.mnu*(3.046/3)**0.75/94.0708
        pars.set_cosmology(H0=q["H0"], ombh2=q["omegab"]*h*h,
                           omch2=(q["omegam"] - q["omegab"])*h*h - omnuh2,
                           mnu=nw.mnu, num_massive_neutrinos=1)
        pars.InitPower.set_params(As=q["As_1e9"]*1e-9, ns=q["ns"])
        pars.set_matter_power(redshifts=[0.0], kmax=10.0)
        s8 = camb.get_results(pars).get_sigma8_0()
        return s8*np.sqrt(q["omegam"]/0.3)

    S8 = S8_of(full)
    grad = {"As_1e9": S8/(2*full["As_1e9"])}
    steps = {"omegam": 0.003, "H0": 0.5, "ns": 0.005, "omegab": 0.001}
    for k in point:
        if k in steps:
            up, dn = dict(full), dict(full)
            up[k] += steps[k]
            dn[k] -= steps[k]
            grad[k] = (S8_of(up) - S8_of(dn))/(2*steps[k])
    return S8, grad


def s8_jacobian(halofit, omegam, As_1e9):
    """Backward-compatible: (S8, dS8/dAs_1e9, dS8/dOmega_m)."""
    S8, g = s8_gradient({"As_1e9": As_1e9, "omegam": omegam})
    return S8, g["As_1e9"], g["omegam"]


def S8_cov(names, theta, C):
    """S8 and the 2x2 covariance of (S8, Omega_m) from the full C."""
    cosmo = [n for n in names if n in ("As_1e9", "omegam", "H0", "ns",
                                       "omegab")]
    point = {n: theta[names.index(n)] for n in cosmo}
    S8, g = s8_gradient(point)
    J = np.zeros((2, len(names)))
    for n in cosmo:
        J[0, names.index(n)] = g[n]
    J[1, names.index("omegam")] = 1.0
    return S8, J @ C @ J.T


def two_d_sigma(d, C):
    from scipy.stats import chi2, norm
    q = float(d @ np.linalg.solve(C, d))
    return q, norm.ppf(0.5 + 0.5*chi2.cdf(q, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help=".dataset of the minimizer runs (mask + covariance)")
    ap.add_argument("--model", choices=["nla", "tatt"], default="nla")
    ap.add_argument("--probe", choices=PROBES, default="xi")
    ap.add_argument("--halofit", default="mead2020")
    ap.add_argument("--limber-gs", type=int, default=0, choices=[0, 1],
                    help="gamma_t projection: 0 = non-Limber below l=150 "
                         "(the roman_real yaml default), 1 = Limber (the "
                         "old GN behaviour)")
    ap.add_argument("--emul", type=int, default=2, help="non_linear_emul")
    ap.add_argument("--lmax", type=int, default=100000)
    ap.add_argument("--camb-kmax", type=float, default=10.0)
    ap.add_argument("--camb-boost", type=float, default=1.1)
    ap.add_argument("--k-per-logint", type=int, default=15)
    ap.add_argument("--path", default=None)
    ap.add_argument("--free-cosmo", action="store_true",
                    help="also vary H0, ns, Omega_b (flat priors of the yamls)")
    ap.add_argument("--prior-ns", type=float, default=None,
                    help="Gaussian sigma on ns (with --free-cosmo)")
    ap.add_argument("--prior-H0", type=float, default=None)
    ap.add_argument("--prior-omegab", type=float, default=None)
    ap.add_argument("--dS8", type=float, default=None, help="MAP - truth")
    ap.add_argument("--dOm", type=float, default=None, help="MAP - truth")
    ap.add_argument("--save", default=None, help="write F and names to .npz")
    a = ap.parse_args()

    # the wrappers call cnu.get_camb_cosmology without a halofit choice:
    # pin it to the one the data vector and chains use
    nw.cnu.get_camb_cosmology = functools.partial(
        nw.cnu.get_camb_cosmology, halofit_version=a.halofit)

    cfg = dict(data_file=a.dataset, lmax=a.lmax, non_linear_emul=a.emul,
               IA_model=1 if a.model == "tatt" else 0, IA_code=0,
               IA_redshift_evolution=3)
    if a.path:
        cfg["path"] = a.path
    nw.configure(**cfg)
    nw.init_cosmolike(CLprobe=a.probe, with_data=True)
    match_likelihood_settings(adopt_limber_gs=a.limber_gs)
    invcov = np.asarray(ci.get_inv_cov_masked(), dtype=np.float64)

    params = parameter_list(a.model, a.probe, a.free_cosmo,
                            {"ns": a.prior_ns, "H0": a.prior_H0,
                             "omegab": a.prior_omegab})
    print(f"Fisher: model={a.model} probe={a.probe} dataset={a.dataset} "
          f"({len(params)} free parameters, {4*len(params)} data vectors)")
    F = fisher(make_dv(a.model, a.probe, a), params, invcov)
    C = np.linalg.inv(F)
    names = [p[0] for p in params]
    theta0 = np.array([p[1] for p in params])
    S8, C2 = S8_cov(names, theta0, C)
    sS8, sOm = np.sqrt(np.diag(C2))
    rho = C2[0, 1]/(sS8*sOm)

    print(f"\nS8 (truth) = {S8:.5f}")
    print(f"sigma(S8)  = {sS8:.5f}  ({100*sS8/S8:.2f}%)")
    print(f"sigma(Om)  = {sOm:.5f}")
    print(f"rho        = {rho:+.3f}")
    print(f"\nfor min_read.py:  sS8, sOm, rho = {sS8:.6g}, {sOm:.6g}, {rho:.4f}")
    for n in ("H0", "ns", "omegab", "A1", "eta1", "A2", "eta2", "bTA"):
        if n in names:
            i = names.index(n)
            print(f"sigma({n}) = {np.sqrt(C[i, i]):.4f}")
    if a.dS8 is not None and a.dOm is not None:
        d = np.array([a.dS8, a.dOm])
        q, ns = two_d_sigma(d, C2)
        print(f"\nshift: dS8 = {a.dS8:+.5f} ({a.dS8/sS8:+.2f} sigma), "
              f"dOm = {a.dOm:+.5f} ({a.dOm/sOm:+.2f} sigma)")
        print(f"2D: dchi2 = {q:.3f}  ->  {ns:.2f} sigma  "
              f"({'PASS' if ns < 0.3 else 'FAIL'} at 0.3 sigma)")
    if a.save:
        np.savez(a.save, F=F, names=np.array(names), C2=C2)


if __name__ == "__main__":
    main()
