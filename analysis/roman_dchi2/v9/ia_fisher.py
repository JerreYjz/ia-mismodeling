#!/usr/bin/env python3
"""The Fisher machinery of the IA error budget (numpy only).

The method of lssty1xdesi notes/correlated_bias_prior_v2 and
notes/fisher_machinery, for the IA spectra of Roman xi_+- (README.md has the
dictionary and the flow chart).

Coordinates.
  Base knob (X, s, j): a fractional error of P_X (X = GI, II) equal to the
    tent j of FINE (61 tents in ln k, 0.01 to 10 h/Mpc, flat outside) at
    snapshot s alone. Base column: the xi response to it, exact (xi is linear
    in the IA tables). The base also holds the 4 stencil columns of the
    template, J = dxi/d(Omega_m, A_s, A, eta) (for 3x2pt also the 8 linear
    galaxy biases b_i, flat; ia_base_3x2.py).
  Residual delta: the fractional error of P_X at the NODES (31 in ln k) and
    at every snapshot, ordered (X, s, i). R maps it to base knobs through
    the cubic spline in ln k: knob (X, s, j) = sum_i delta(X, s, i) S[i, j].
  Kernel: <delta delta^T> = sigma^2 Kx (x) Kz (x) Kk, with Kk in ln k
    (exponential, the kernel of the bias-prior notes, or squared
    exponential, the kernel of v8), Kz exponential in z (inf: the same error
    at every z) and Kx the GI-II correlation rho.
  Scored parameters: Omega_m and S8, in units of their marginalised error.

Fits.
  rigid:        cosmology (Omega_m, A_s, flat) + template (A, eta, Gaussian
                SIG_IA): the analogue of the growth-tied prior.
  marginalised: rigid + the residual, with a Gaussian prior of covariance
                Sigma_fit, written in factor form delta = L zeta,
                zeta ~ N(0, 1) (any Sigma_fit, even singular). The
                projected-precision form of the notes (template directions
                uncharged) is in projected_precision() and is compared with
                the factor form in verify_ia_fisher.py.

Scores of one fit under random truths delta = mu + N(0, Sigma_true),
exact (the notes' appendix "The test statistic"):
  K = mu_z mu_z^T + A Sigma_true A^T + N,   A = response, N = noise,
  eta_p = sqrt(K_pp), the 1 sigma coverage of each parameter, and of the
  pair, eta_2 = sqrt(tr(Cn^-1 K)/2) and the 2d coverage of the shifted
  Gaussian.
"""
import numpy as np
import scipy.linalg
from scipy import integrate, optimize, stats
from scipy.interpolate import CubicSpline
from scipy.special import erf

TOL = 0.3
ETA_MAX = float(np.sqrt(1.0 + TOL**2))
SIGMA_CAP = 1.0                       # 100%: no limit
PINV_RTOL = 1e-12
ONE_SIGMA = float(erf(1.0/np.sqrt(2.0)))
FINE = np.logspace(-2, 1, 61)
NODES = np.logspace(-2, 1, 31)
X0 = 1.62
SIG_IA = {"A": 10.0/np.sqrt(12)/3.4, "eta": 10.0/np.sqrt(12)}     # tolerance_nomask.SIG_IA
TEMPLATE = ("omegam", "As_1e9", "A", "eta")
SCORED = ("omegam", "S8")
XS = ("GI", "II")


# ---------------------------------------------------------------- grids
def tents_on(k, nodes):
    """H (len(nodes), len(k)): linear tents in ln k, flat outside (sum = 1)."""
    x, xn = np.log(k), np.log(nodes)
    return np.array([np.interp(x, xn, e) for e in np.eye(len(nodes))])


def spline_basis(k, nodes=NODES):
    """S (len(nodes), len(k)): f(k) = delta @ S, the cubic spline in ln k,
    flat outside the nodes (v8 correlated_budget.spline_basis)."""
    x = np.clip(np.log(k), np.log(nodes[0]), np.log(nodes[-1]))
    return np.array([CubicSpline(np.log(nodes), e)(x) for e in np.eye(len(nodes))])


def kernel_k(nodes, ell, kind="exp"):
    """Correlation in ln k: exp(-|d|/ell) ('exp') or exp(-d^2/ell^2) ('sqexp')."""
    d = np.abs(np.log(nodes)[:, None] - np.log(nodes)[None, :])
    if kind == "exp":
        return np.exp(-d/ell)
    if kind == "sqexp":
        return np.exp(-d**2/ell**2)
    raise ValueError(kind)


def kernel_z(zs, Lz):
    """Correlation in z: exp(-|dz|/Lz); Lz = inf gives the same error at every z."""
    zs = np.asarray(zs, float)
    if not np.isfinite(Lz):
        return np.ones((len(zs), len(zs)))
    return np.exp(-np.abs(zs[:, None] - zs[None, :])/Lz)


def kernel_x(Xs, rho):
    n = len(Xs)
    return np.full((n, n), rho) + (1.0 - rho)*np.eye(n)


def residual_cov(nodes, zs, ell, Lz, kind="exp", Xs=XS, rho=0.0, sigma=1.0):
    """Sigma = sigma^2 Kx (x) Kz (x) Kk on the residual (X, s, i)."""
    return sigma**2*np.kron(kernel_x(Xs, rho), np.kron(kernel_z(zs, Lz), kernel_k(nodes, ell, kind)))


def window(nodes, zs, Xs, kc=None, zc=None, wk=0.3, wz=0.25):
    """Weights w(X, s, i) of an error confined around k_c (Gaussian of width wk
    in ln k) and z_c (Gaussian of width wz in z); None = no window."""
    w = np.ones((len(zs), len(nodes)))
    if kc is not None:
        w = w*np.exp(-np.log(nodes/kc)**2/(2*wk**2))[None, :]
    if zc is not None:
        w = w*np.exp(-(np.asarray(zs) - zc)**2/(2*wz**2))[:, None]
    return np.tile(w.ravel(), len(Xs))


def factor(S, rtol=1e-12):
    """L with L L^T = S, from the eigenvectors (S may be singular): the
    columns with eigenvalue below rtol of the largest are dropped."""
    w, V = np.linalg.eigh(0.5*(S + S.T))
    keep = w > rtol*w.max()
    return V[:, keep]*np.sqrt(w[keep])


def template_directions(zs, Xs=XS, n_nodes=len(NODES)):
    """U (n_res, 2): the residuals the template makes, dA and deta:
    delta_GI = dA + deta ln x(z), delta_II = 2 (dA + deta ln x(z))."""
    lnx = np.log((1 + np.asarray(zs))/X0)
    cols = []
    for v in (np.ones_like(lnx), lnx):
        cols.append(np.concatenate([(1.0 if X == "GI" else 2.0)*np.repeat(v, n_nodes) for X in Xs]))
    return np.column_stack(cols)


def projected_precision(S, U):
    """P = Q - Q U (U^T Q U)^-1 U^T Q, Q = S^-1 (the notes' Eq. memproj):
    charges only what the template cannot make."""
    Q = np.linalg.inv(S)
    QU = Q @ U
    return Q - QU @ np.linalg.solve(U.T @ QU, QU.T)


# ---------------------------------------------------------------- the base
class Base:
    """The base of one truth: ia_base_<truth>.npz and ia_base_common.npz."""

    def __init__(self, path, common):
        d = np.load(path, allow_pickle=False)
        c = np.load(common, allow_pickle=False)
        self.truth = str(d["truth"])
        self.zs = d["zs"]
        self.fine = d["fine"]
        self.xi0 = d["xi0"]
        self.xi_ia = d["xi_ia"]
        self.J = d["J"]                                   # (ndv, npar): TEMPLATE, then any nuisance
        self.g = d["g_s8"]                                # (2, npar): Omega_m and S8 rows
        self.npar = self.J.shape[1]                       # 4 (xi_+-) or 12 (3x2pt: + 8 biases)
        self.R = {X: d[f"R_{X}"] for X in XS}             # (nz, 61, 1080)
        self.drift = {k[6:]: d[k] for k in d.files if k.startswith("drift|")}
        self.cov = c["cov"]
        self.masks = {str(n): c[f"mask|{n}"] for n in c["cuts"]}
        nz, nf = len(self.zs), len(self.fine)
        self.B = np.hstack([self.J] + [self.R[X].reshape(nz*nf, -1).T for X in XS])
        self.nb = self.B.shape[1]
        self.ij = np.arange(self.npar)
        self.iknob = {X: self.npar + q*nz*nf + np.arange(nz*nf).reshape(nz, nf)
                      for q, X in enumerate(XS)}

    def knob_map(self, Xs=XS, nodes=NODES):
        """R (nb, n_res): residual (X, s, i) at the nodes -> base knobs."""
        S = spline_basis(self.fine, nodes)                # (n_nodes, 61)
        nz, nn = len(self.zs), len(nodes)
        R = np.zeros((self.nb, len(Xs)*nz*nn))
        for q, X in enumerate(Xs):
            for s in range(nz):
                R[self.iknob[X][s][:, None], q*nz*nn + s*nn + np.arange(nn)[None, :]] = S.T
        return R

    def drift_knobs(self, name):
        """c (nb,): a known drift on the tents, name = '<cut>|<model>', as the
        equivalent data change. The stored drift is Delta = P^model/P^truth - 1;
        an analysis model wrong by Delta moves the fit as the data moving by
        -Delta, hence the sign (checked against the full v8 fits)."""
        c = np.zeros(self.nb)
        for X in XS:
            c[self.iknob[X].ravel()] = -self.drift[f"{name}|{X}"].ravel()
        return c


def gram(B, cov, mask):
    """G_K = B_K^T C_KK^-1 B_K by Cholesky and forward substitution."""
    L = scipy.linalg.cholesky(cov[np.ix_(mask, mask)], lower=True)
    W = scipy.linalg.solve_triangular(L, B[mask], lower=True)
    return W.T @ W


def stable_inverse(F):
    """Pseudo-inverse of a symmetric Fisher matrix (fit_system of the notes):
    unit diagonal, eigen-decomposition, directions below PINV_RTOL dropped.
    Returns (F^-1, number dropped)."""
    d = np.diag(F)
    # a parameter the data do not see (the biases on xi_+- alone) is a null direction
    s = np.where(d > 0, 1.0/np.sqrt(np.where(d > 0, d, 1.0)), 1.0)
    w, V = np.linalg.eigh(F*s[:, None]*s[None, :])
    keep = w > PINV_RTOL*w.max()
    Finv = (s[:, None]*V[:, keep]) @ np.diag(1.0/w[keep]) @ (V[:, keep].T*s[None, :])
    return Finv, int((~keep).sum())


# ---------------------------------------------------------------- fits
class Fit:
    """One Fisher system: F = M^T G M + P, scored on Omega_m and S8.

    G: the Gram of the cut; M (nb, npar): the model on the base; P (npar,
    npar): the prior precision; P0: the priors kept in the noise term (all
    but the residual's); g (2, npar): d(Omega_m, S8)/d(params)."""

    def __init__(self, G, M, P, P0, g):
        self.G, self.M, self.g = G, M, g
        Fd = M.T @ G @ M
        self.F = Fd + P
        self.F0 = Fd + P0
        self.Finv, self.dropped = stable_inverse(self.F)
        self.C2 = g @ self.Finv @ g.T                      # (Omega_m, S8) covariance
        self.sig = np.sqrt(np.diag(self.C2))
        self.Cn = self.C2/np.outer(self.sig, self.sig)     # in sigma units
        self.N = (g @ self.Finv @ self.F0 @ self.Finv @ g.T)/np.outer(self.sig, self.sig)

    def shift(self, C):
        """(2, k): the shift of (Omega_m, S8) in sigma per base vector (columns of C)."""
        C = np.asarray(C)
        dp = self.Finv @ (self.M.T @ (self.G @ C))
        return (self.g @ dp)/self.sig[:, None] if C.ndim == 2 else (self.g @ dp)/self.sig


def template_prior(npar):
    """Prior precision of the rigid parameters: Omega_m, A_s flat, A and eta
    Gaussian (SIG_IA), any further nuisance (the 3x2pt galaxy biases) flat."""
    P = np.zeros((npar, npar))
    P[2, 2], P[3, 3] = 1/SIG_IA["A"]**2, 1/SIG_IA["eta"]**2
    return P


def rigid_fit(base, mask):
    G = gram(base.B, base.cov, mask)
    M = np.zeros((base.nb, base.npar))
    M[base.ij, base.ij] = 1.0
    P = template_prior(base.npar)
    g = base.g.copy()
    return Fit(G, M, P, P, g), G


def marginalised_fit(base, G, Rmap, S_fit):
    """Rigid + the residual with prior N(0, S_fit) in factor form."""
    L = factor(S_fit)
    n0 = base.npar
    M0 = np.zeros((base.nb, n0))
    M0[base.ij, base.ij] = 1.0
    M = np.hstack([M0, Rmap @ L])
    r = L.shape[1]
    P0 = np.zeros((n0 + r, n0 + r))
    P0[:n0, :n0] = template_prior(n0)
    P = P0.copy()
    P[n0:, n0:] += np.eye(r)
    g = np.hstack([base.g, np.zeros((2, r))])
    return Fit(G, M, P, P0, g)


def marginalised_fit_projected(base, G, Rmap, S_fit, U):
    """Rigid + the residual with the projected precision of the notes
    (S_fit must be invertible): the check of the factor form."""
    Pr = projected_precision(S_fit, U)
    n = Pr.shape[0]
    n0 = base.npar
    M0 = np.zeros((base.nb, n0))
    M0[base.ij, base.ij] = 1.0
    M = np.hstack([M0, Rmap])
    P0 = np.zeros((n0 + n, n0 + n))
    P0[:n0, :n0] = template_prior(n0)
    P = P0.copy()
    P[n0:, n0:] += Pr
    g = np.hstack([base.g, np.zeros((2, n))])
    return Fit(G, M, P, P0, g)


# ---------------------------------------------------------------- scores
def coverage_1d(mu, V):
    """P(|zeta| < 1), zeta ~ N(mu, V) (the notes' Eq. coverage)."""
    r = np.sqrt(2.0*V)
    return 0.5*(erf((1.0 - mu)/r) + erf((1.0 + mu)/r))


def coverage_2d(Cn, mu, S):
    """Mass of N(mu, S) inside the fit's 68.3% ellipse x^T Cn^-1 x <= q.

    Whitened by Cn the ellipse is a circle of radius sqrt(q); rotated to the
    eigenbasis of the whitened S the two coordinates are independent, and
    the mass is one quadrature of a normal times an interval probability."""
    q = -2.0*np.log(1.0 - ONE_SIGMA)
    Lc = np.linalg.cholesky(Cn)
    Li = np.linalg.inv(Lc)
    T = Li @ S @ Li.T
    m = Li @ mu
    lam, Q = np.linalg.eigh(0.5*(T + T.T))
    lam = np.clip(lam, 1e-300, None)
    u = Q.T @ m
    rq = np.sqrt(q)

    def integrand(x):
        h = np.sqrt(max(q - x*x, 0.0))
        dens = np.exp(-0.5*(x - u[0])**2/lam[0])/np.sqrt(2*np.pi*lam[0])
        return dens*(stats.norm.cdf((h - u[1])/np.sqrt(lam[1]))
                     - stats.norm.cdf((-h - u[1])/np.sqrt(lam[1])))
    val, _ = integrate.quad(integrand, -rq, rq, limit=200)
    return float(val)


def scores(fit, A, S_true, mu_z=None):
    """eta and coverage of one fit under truths mu + N(0, S_true).

    A (2, n_res): the response to the residual in sigma; mu_z (2,): the
    shift of the known drift (None: no drift)."""
    mu = np.zeros(2) if mu_z is None else np.asarray(mu_z, float)
    S = A @ S_true @ A.T
    V = S + fit.N
    K = np.outer(mu, mu) + V
    out = {}
    for i, p in enumerate(SCORED):
        out[p] = dict(eta=float(np.sqrt(K[i, i])), coverage=float(coverage_1d(mu[i], V[i, i])),
                      drift=float(abs(mu[i])), scatter=float(np.sqrt(S[i, i])))
    Ci = np.linalg.inv(fit.Cn)
    out["pair"] = dict(eta=float(np.sqrt(np.trace(Ci @ K)/2.0)),
                       coverage=coverage_2d(fit.Cn, mu, V),
                       drift=float(np.sqrt(mu @ Ci @ mu)))
    return out


# ---------------------------------------------------------------- budgets
def rms_budget(A, Kunit, mu_z=None):
    """The largest sigma with eta_p <= sqrt(1 + TOL^2) for both p, the rigid
    fit's noise term being 1: sigma^2 a_p K a_p^T + mu_p^2 <= TOL^2.

    Returns (sigma_max capped at SIGMA_CAP, the limiting parameter, the rms
    shift per unit sigma of each parameter)."""
    mu = np.zeros(2) if mu_z is None else np.abs(np.asarray(mu_z, float))
    s = np.sqrt(np.clip(np.einsum("pi,ij,pj->p", A, Kunit, A), 0.0, None))
    room = np.clip(TOL**2 - mu**2, 0.0, None)
    with np.errstate(divide="ignore"):
        lim = np.where(s > 0, np.sqrt(room)/s, np.inf)
    i = int(np.argmin(lim))
    return float(min(lim[i], SIGMA_CAP)), SCORED[i], s.tolist()


def box_probability(mu, S2):
    """P(|z_1| < TOL and |z_2| < TOL), z ~ N(mu, S2), exact."""
    S2 = 0.5*(S2 + S2.T)
    d = np.sqrt(np.clip(np.diag(S2), 0, None))
    if np.all(d == 0):
        return float(np.all(np.abs(mu) < TOL))
    if np.any(d == 0) or abs(S2[0, 1])/np.prod(d) > 1 - 1e-10:
        # rank one: z = mu + t v, t ~ N(0, 1); the interval of t inside the box
        w, V = np.linalg.eigh(S2)
        v = V[:, -1]*np.sqrt(w[-1])
        lo, hi = -np.inf, np.inf
        for i in range(2):
            if abs(v[i]) < 1e-300:
                if abs(mu[i]) >= TOL:
                    return 0.0
                continue
            a, b = sorted(((-TOL - mu[i])/v[i], (TOL - mu[i])/v[i]))
            lo, hi = max(lo, a), min(hi, b)
        return float(max(stats.norm.cdf(hi) - stats.norm.cdf(lo), 0.0))
    # standardised: scipy's CDF returns NaN when the box edges sit ~1e4 widths
    # out, so work in units of each width and clip the edges at 40 (P = 1 in
    # double precision beyond)
    r = S2[0, 1]/np.prod(d)
    lo = np.clip((-TOL - mu)/d, -40.0, 40.0)
    hi = np.clip((TOL - mu)/d, -40.0, 40.0)
    if np.any(hi <= lo):
        return 0.0
    cov = [[1.0, r], [r, 1.0]]
    c = [stats.multivariate_normal.cdf([x, y], mean=np.zeros(2), cov=cov, abseps=1e-10, releps=1e-10)
         for x, y in ((hi[0], hi[1]), (lo[0], hi[1]), (hi[0], lo[1]), (lo[0], lo[1]))]
    return float(min(max(c[0] - c[1] - c[2] + c[3], 0.0), 1.0))


def box_budget(A, Kunit, mu_z=None, q=0.95):
    """The rms sigma at which a fraction q of the errors keeps both shifts
    below TOL (the v8 criterion), exact; capped at SIGMA_CAP."""
    mu = np.zeros(2) if mu_z is None else np.asarray(mu_z, float)
    S1 = A @ Kunit @ A.T
    if box_probability(mu, S1*SIGMA_CAP**2) >= q:
        return float(SIGMA_CAP)
    if box_probability(mu, S1*1e-12) < q:
        return 0.0
    f = lambda ls: box_probability(mu, S1*np.exp(2*ls)) - q
    return float(np.exp(optimize.brentq(f, np.log(1e-6), np.log(SIGMA_CAP), xtol=1e-10)))


def fom_pair(fit):
    """1/sqrt(det C(Omega_m, S8))."""
    return float(1.0/np.sqrt(np.linalg.det(fit.C2)))
