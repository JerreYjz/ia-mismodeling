"""Limber cosmic-shear xi_+- for the Roman real-space setup, with IA spectra.

Standalone Python (CAMB + numpy) following cocoa_roman_real's layout:
8 source bins (example1.nz), log angular bins (default 15, 2.5-250 arcmin),
data vector [xi_+ | xi_-], each block npair (i <= j, i-major) x ntheta.

Limber, k = (l + 1/2)/chi, curved-sky spin-2 prefactor
F(l) = (l+2)(l+1)l(l-1)/(l+1/2)^4 on every term:
    C_GG^ij = F int dchi q_i q_j / chi^2 P_NL
    C_GI^ij = F int dchi (q_i n_j + n_i q_j) / chi^2 P_dE
    C_II^ij = F int dchi n_i n_j / chi^2 P_EE       (E mode)
    C_BB^ij = F int dchi n_i n_j / chi^2 P_BB       (B mode)
q_i the lensing efficiency, n_i(chi) = n_i(z) dz/dchi; units h/Mpc.
xi_+(theta) = int l dl/(2 pi) J0(l theta) (C_E + C_B),
xi_-(theta) = int l dl/(2 pi) J4(l theta) (C_E - C_B)   (flat-sky Hankel),
averaged over each angular bin with weight theta, with a Gaussian taper
exp[-(l/6e4)^2] against ringing from white C_l residuals.

IA spectra enter as ratios R_X(k, z) = P_X(k, z) / P_NL(k, z), X = dE, EE,
BB (callables of 2-D k, z arrays), so FLAMINGO-based and model spectra use
the same machinery.

Gaussian covariance (flat sky, e.g. Joachimi et al. 2008):
    Cov[xi_a^ij(t1), xi_b^kl(t2)] = 1/(2 pi A) int l dl Jbar_a(l t1) Jbar_b(l t2)
        x [ (Ehat_ik Ehat_jl + Ehat_il Ehat_jk) +- (Bhat_ik Bhat_jl + ...) ]
with Ehat = C_E + N delta, Bhat = C_B + N delta, N = sigma_e^2 / n
(sigma_e per component); + for ++ and --, - for +-. The pure noise-noise
part is replaced by its exact discrete form
    (d_ik d_jl + d_il d_jk) 2 sigma_e^4 / (A n_i n_j pi (t_hi^2 - t_lo^2))
on the diagonal of the ++ and -- blocks.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv

C_HMPC = 2997.92458          # c/H0 in Mpc/h
ARCMIN = np.pi/180/60


class Cosmology:
    """CAMB background and P_NL (h/Mpc units)."""

    def __init__(self, h, Om, Ob, ns, As=2.1e-9, sigma8=None, mnu=0.06, w=-1.0,
                 halofit="takahashi", zmax=4.5):
        import camb
        pars = camb.CAMBparams()
        nmass = 1 if mnu > 0 else 0
        pars.set_cosmology(H0=100*h, ombh2=Ob*h*h, omch2=(Om - Ob)*h*h - mnu/93.14,
                           mnu=mnu, num_massive_neutrinos=nmass)
        pars.set_dark_energy(w=w)
        pars.InitPower.set_params(ns=ns, As=As)
        pars.set_matter_power(redshifts=list(np.linspace(zmax, 0, 60)), kmax=60.0)
        if sigma8 is not None:
            r0 = camb.get_results(pars)
            As = As*(sigma8/r0.get_sigma8_0())**2
            pars.InitPower.set_params(ns=ns, As=As)
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params(halofit_version=halofit)
        self.res = camb.get_results(pars)
        self.h, self.Om = h, Om
        self.Pnl = self.res.get_matter_power_interpolator(
            nonlinear=True, hubble_units=True, k_hunit=True,
            extrap_kmax=1e4)
        self.Plin = self.res.get_matter_power_interpolator(
            nonlinear=False, hubble_units=True, k_hunit=True,
            extrap_kmax=1e4)
        p0 = self.Plin.P(0.0, 1e-3)
        self.D = lambda z: np.sqrt(self.Plin.P(z, 1e-3)/p0)

    def chi(self, z):
        return self.res.comoving_radial_distance(z)*self.h          # Mpc/h

    def dchidz(self, z):
        return 2.99792458e5/self.res.hubble_parameter(z)*self.h      # Mpc/h


def ratio_table(zs, k, R):
    """Callable R(K, Z) from a table R[nz, nk] on snapshot redshifts zs and
    wavenumbers k: linear in z and ln k, constant outside the table."""
    zs, lk = np.asarray(zs, float), np.log(k)
    o = np.argsort(zs)
    f = RegularGridInterpolator((zs[o], lk), np.asarray(R)[o], bounds_error=False,
                                fill_value=None)

    def call(K, Z):
        zz = np.clip(Z, zs.min(), zs.max())
        kk = np.clip(np.log(K), lk[0], lk[-1])
        pts = np.stack([np.ravel(zz), np.ravel(kk)], -1)
        return f(pts).reshape(np.shape(K))
    return call


class RomanShear:
    """Kernels, Limber C_l and binned xi_+- for the Roman source sample."""

    def __init__(self, cosmo, nz_file, ntheta=15, tmin=2.5, tmax=250.0,
                 nzgrid=400, nell=260, lmax=1.5e5, nell_dense=8000, ltaper=6e4):
        self.ltaper = ltaper
        self.c = cosmo
        d = np.loadtxt(nz_file)
        # cocoa_roman_real reads the z column as left bin edges
        # (photoz_zmid = 0): the n(z) values sit at the bin midpoints
        d[:, 0] += 0.5*(d[1, 0] - d[0, 0])
        self.nbin = d.shape[1] - 1
        z = np.linspace(0.005, d[-1, 0], nzgrid)
        nz = np.array([np.interp(z, d[:, 0], d[:, i]) for i in range(1, self.nbin + 1)])
        nz /= np.trapezoid(nz, z, axis=1)[:, None]
        self.z = z
        self.chi = cosmo.chi(z)
        self.n = nz/cosmo.dchidz(z)                              # per (Mpc/h)
        a = 1.0/(1.0 + z)
        q = np.zeros_like(self.n)
        for m in range(len(z)):
            sel = slice(m, None)
            fac = (self.chi[sel] - self.chi[m])/self.chi[sel]
            q[:, m] = np.trapezoid(self.n[:, sel]*fac, self.chi[sel], axis=1)
        self.q = q*1.5*cosmo.Om/C_HMPC**2*(self.chi/a)[None, :]
        self.pairs = [(i, j) for i in range(self.nbin) for j in range(i, self.nbin)]
        self.ell = np.geomspace(2.0, lmax, nell)
        self.set_theta(ntheta, tmin, tmax, nell_dense)
        L, Z = np.meshgrid(self.ell, self.z, indexing="ij")
        self.K = (L + 0.5)/self.chi[None, :]
        self.Z = Z
        self.P = np.array([cosmo.Pnl.P(z, self.K[i], grid=False)
                           for i in range(len(self.ell))])

    def set_theta(self, ntheta, tmin, tmax, nell_dense=8000):
        self.ntheta = ntheta
        self.edges = np.geomspace(tmin, tmax, ntheta + 1)*ARCMIN
        self.ell_d = np.geomspace(2.0, self.ell[-1], nell_dense)
        self.J0, self.J4 = self._bessel(self.ell_d)
        self.hw = self.ell_d**2*np.gradient(np.log(self.ell_d))/(2*np.pi)*self.taper(self.ell_d)

    def taper(self, ell):
        """Gaussian taper exp[-(l/l_t)^2] in the Hankel integrals: removes the
        ringing of white (shot-noise-like) C_l residuals at the hard l_max;
        l_t = 6e4 is far above the l ~ 1e4 probed at 2.5 arcmin."""
        return np.exp(-(ell/self.ltaper)**2) if self.ltaper else np.ones_like(ell)

    def _bessel(self, ell, nsub=48):
        """theta-weighted bin averages of J0 and J4: [ntheta, nell]."""
        J0, J4 = [], []
        for t1, t2 in zip(self.edges[:-1], self.edges[1:]):
            t = np.linspace(t1, t2, nsub)
            w = t/np.trapezoid(t, t)
            x = np.outer(t, ell)
            J0.append(np.trapezoid(w[:, None]*jv(0, x), t, axis=0))
            J4.append(np.trapezoid(w[:, None]*jv(4, x), t, axis=0))
        return np.array(J0), np.array(J4)

    @staticmethod
    def prefactor(l):
        return (l + 2)*(l + 1)*l*(l - 1)/(l + 0.5)**4

    def cl(self, ratio=None, kcut=None, gg=True):
        """C_E, C_B [npair, nell] on self.ell.
        ratio: dict of callables R_X(K, Z), X in dE, EE, BB (None: no IA).
        kcut: callable z -> k (h/Mpc): every spectrum zeroed above it (IR)."""
        P = self.P
        if kcut is not None:
            P = np.where(self.K <= np.asarray(kcut(self.z))[None, :], P, 0.0)
        F = self.prefactor(self.ell)
        w = 1.0/self.chi**2
        R = {X: ratio[X](self.K, self.Z)*P for X in ("dE", "EE", "BB")} if ratio else None
        CE = np.zeros((len(self.pairs), len(self.ell)))
        CB = np.zeros_like(CE)
        for p, (i, j) in enumerate(self.pairs):
            integ = np.zeros_like(P)
            if gg:
                integ = integ + self.q[i]*self.q[j]*w*P
            if R is not None:
                integ = integ + (self.q[i]*self.n[j] + self.n[i]*self.q[j])*w*R["dE"]
                integ = integ + self.n[i]*self.n[j]*w*R["EE"]
                CB[p] = F*np.trapezoid(self.n[i]*self.n[j]*w*R["BB"], self.chi, axis=1)
            CE[p] = F*np.trapezoid(integ, self.chi, axis=1)
        return CE, CB

    def _dense(self, C, ell):
        lnl, lnd = np.log(self.ell), np.log(ell)
        return np.array([np.interp(lnd, lnl, c) for c in C])

    def xi(self, CE, CB):
        """Binned data vector [xi_+ (npair x ntheta) | xi_- (npair x ntheta)]."""
        E, B = self._dense(CE, self.ell_d), self._dense(CB, self.ell_d)
        xp = ((E + B)*self.hw) @ self.J0.T
        xm = ((E - B)*self.hw) @ self.J4.T
        return np.concatenate([xp.ravel(), xm.ravel()])

    def gaussian_cov(self, CE, CB, area_deg2=2415.0, n_arcmin2=41.3/8, sigma_e=0.30,
                     nell=6000):
        A = area_deg2*(np.pi/180)**2
        nsr = n_arcmin2/ARCMIN**2
        N = sigma_e**2/nsr
        ld = np.geomspace(2.0, self.ell[-1], nell)
        E, B = self._dense(CE, ld), self._dense(CB, ld)
        J0, J4 = self._bessel(ld)
        hw = ld**2*np.gradient(np.log(ld))/(2*np.pi*A)*self.taper(ld)
        idx = {pr: p for p, pr in enumerate(self.pairs)}
        npair, nt = len(self.pairs), self.ntheta
        off = npair*nt
        cov = np.zeros((2*off, 2*off))
        for p, (i, j) in enumerate(self.pairs):
            for q in range(p, npair):
                k, l = self.pairs[q]
                prs = [(i, k), (j, l), (i, l), (j, k)]
                dl = [float(a == b) for a, b in prs]
                e = [E[idx[(min(a, b), max(a, b))]] + N*d for (a, b), d in zip(prs, dl)]
                b = [B[idx[(min(a, b), max(a, b))]] + N*d for (a, b), d in zip(prs, dl)]
                nn = N*N*(dl[0]*dl[1] + dl[2]*dl[3])
                SE = e[0]*e[1] + e[2]*e[3] - nn
                SB = b[0]*b[1] + b[2]*b[3] - nn
                Cpp = (J0*((SE + SB)*hw)) @ J0.T
                Cmm = (J4*((SE + SB)*hw)) @ J4.T
                Cpm = (J0*((SE - SB)*hw)) @ J4.T          # xi_+^{ij} x xi_-^{kl}
                Cmp = (J4*((SE - SB)*hw)) @ J0.T          # xi_-^{ij} x xi_+^{kl}
                a0, b0 = p*nt, q*nt
                cov[a0:a0 + nt, b0:b0 + nt] = Cpp
                cov[off + a0:off + a0 + nt, off + b0:off + b0 + nt] = Cmm
                cov[a0:a0 + nt, off + b0:off + b0 + nt] = Cpm
                cov[off + a0:off + a0 + nt, b0:b0 + nt] = Cmp
                if q != p:
                    cov[b0:b0 + nt, a0:a0 + nt] = Cpp.T
                    cov[off + b0:off + b0 + nt, off + a0:off + a0 + nt] = Cmm.T
                    cov[off + b0:off + b0 + nt, a0:a0 + nt] = Cpm.T
                    cov[b0:b0 + nt, off + a0:off + a0 + nt] = Cmp.T
        npairs = A*nsr*nsr*np.pi*(self.edges[1:]**2 - self.edges[:-1]**2)
        for p, (i, j) in enumerate(self.pairs):
            v = (2.0 if i == j else 1.0)*2*sigma_e**4/npairs
            for blk in (0, off):
                s = blk + p*nt
                cov[s:s + nt, s:s + nt] += np.diag(v)
        return 0.5*(cov + cov.T)


def uv_mask(xi_full, xi_ir, alpha=0.05, ntheta=15):
    """IR/UV mask (arXiv:2603.13031): keep |UV| / (|UV| + |IR|) < alpha; per
    data-vector row (pair, xi_+-) drop every theta below the largest failing
    theta, as make_uv_scale_cuts.py does."""
    uv = xi_full - xi_ir
    den = np.abs(uv) + np.abs(xi_ir)
    frac = np.divide(np.abs(uv), den, out=np.zeros_like(den), where=den > 0)
    keep = frac < alpha
    out = keep.copy()
    for r in range(0, keep.size, ntheta):
        fail = np.where(~keep[r:r + ntheta])[0]
        if fail.size:
            out[r:r + fail.max() + 1] = False
    return out


class RomanFourier(RomanShear):
    """E-mode shear bandpowers in cocoa_roman_fourier's layout: 15 bands of
    integer multipoles, edges rint(geomspace(30, 4001, 16)), band power =
    sum (2l+1) C_l / sum (2l+1) over l in [first, last]; data vector
    36 pairs x 15 bands (shear block of the 3x2pt vector).
    Gaussian covariance, diagonal in bands:
        Cov = sum_l (2l+1) [Chat_ik Chat_jl + Chat_il Chat_jk] / ((sum (2l+1))^2 f_sky)
    Chat = C_E + N delta, N = sigma_e^2 / n."""

    def __init__(self, cosmo, nz_file, **kw):
        super().__init__(cosmo, nz_file, **kw)
        e = np.rint(np.geomspace(30, 4001, 16)).astype(int)
        self.bfirst, self.blast = e[:-1], e[1:] - 1
        self.nband = len(self.bfirst)
        self.lint = np.arange(self.bfirst[0], self.blast[-1] + 1)
        self.bw = [(self.lint >= f) & (self.lint <= l) for f, l in zip(self.bfirst, self.blast)]
        self.lcen = np.sqrt(self.bfirst*self.blast)

    def _int(self, C):
        return self._dense(C, self.lint)

    def bandpowers(self, CE):
        E = self._int(CE)
        w = 2*self.lint + 1.0
        return np.concatenate([[np.sum(w[m]*e[m])/np.sum(w[m]) for m in self.bw] for e in E])

    def gaussian_cov(self, CE, area_deg2=2415.0, n_arcmin2=41.3/8, sigma_e=0.30):
        fsky = area_deg2/41252.96
        N = sigma_e**2/(n_arcmin2/ARCMIN**2)
        E = self._int(CE)
        w = 2*self.lint + 1.0
        idx = {pr: p for p, pr in enumerate(self.pairs)}
        npair, nb = len(self.pairs), self.nband
        cov = np.zeros((npair*nb, npair*nb))
        for p, (i, j) in enumerate(self.pairs):
            for q in range(p, npair):
                k, l = self.pairs[q]
                prs = [(i, k), (j, l), (i, l), (j, k)]
                c = [E[idx[(min(a, b), max(a, b))]] + N*(a == b) for a, b in prs]
                S = c[0]*c[1] + c[2]*c[3]
                v = np.array([np.sum(w[m]*S[m])/np.sum(w[m])**2/fsky for m in self.bw])
                cov[p*nb + np.arange(nb), q*nb + np.arange(nb)] = v
                cov[q*nb + np.arange(nb), p*nb + np.arange(nb)] = v
        return cov


def uv_mask_fourier(c_full, c_ir, alpha=0.05, nband=15):
    """IR/UV mask for bandpowers: per pair keep the bands below the first
    band with |UV| / (|UV| + |IR|) >= alpha."""
    uv = c_full - c_ir
    den = np.abs(uv) + np.abs(c_ir)
    frac = np.divide(np.abs(uv), den, out=np.zeros_like(den), where=den > 0)
    keep = frac < alpha
    out = keep.copy()
    for r in range(0, keep.size, nband):
        fail = np.where(~keep[r:r + nband])[0]
        if fail.size:
            out[r + fail.min():r + nband] = False
    return out
