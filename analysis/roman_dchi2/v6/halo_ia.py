#!/usr/bin/env python3
"""Halo-model intrinsic alignments (Fortuna et al. 2021) in Python, following
cocoa-cosmolike-core/cosmolike/halo.c (main, 43effe0) term by term.

Units: k in h/Mpc, lengths in Mpc/h, masses in M_sun/h, spectra in (Mpc/h)^3.

Ingredients (halo.c defaults)
  sigma(M, z)   linear CDM + baryon power (CAMB, FLAMINGO D3A), top hat
  f(nu)         Tinker et al. 2010, Delta = 200 mean, z-evolution frozen at
                z = 3; alpha fixed by int b(nu) f(nu) dnu = 1
  b(nu)         Tinker et al. 2010
  c(M, z)       Bhattacharya et al. 2013: c = 9.0 nu^-0.29 D_cb^1.15
  u(k|M)        NFW transform, r_Delta = (3 M / (4 pi 200 rho_m))^(1/3)
  HOD           N_c = [1 + erf((lgM - lgM_min)/sigma)]/2,
                N_s = N_c ((M - M_0)/M_1)^alpha above M_0; f_c = 1
  red fractions f_red = 1 for centrals and satellites (one population: the
                FLAMINGO shape sample has no colour split here; the red
                fraction's normalisation is absorbed by the amplitudes)

Satellite alignment (F21 Eq. 20, halo.c): gamma_bar(r) = a_1h g(r),
  g(r) = [max(r, r_e)/r_vir]^-2,  r_e = max(r_floor, r_vir sqrt(|a_1h|/gamma_max)),
  r_floor = 0.06 Mpc/h, gamma_max = 0.3.
Kernel: gamma_hat(k|M) = sum_{l=2,4,6} P_l K_l(k r_s)/m(c),
  K_l(t) = int_0^c g(x) x (1+x)^-2 j_l(t x) dx,  P_l = -15/8, -15/8, -273/128.
1-halo spectra (F21 Eqs. 17-18):
  S_dI(k) = int dlnM dn/dlnM (M/rho_m) u(k|M) (N_s/n_g) |gamma_hat|
  S_II(k) = int dlnM dn/dlnM (N_s/n_g)^2 gamma_hat^2
  P_dI^1h = a_1h f_1h S_dI,  P_II^1h = a_1h^2 f_1h S_II
2-halo (cosmo2D.c): NLA for centrals times f_rc and f_2h,
  P_dE = -[a_1 f_rc C1 P_NL f_2h + P_dI^1h],
  P_EE = a_1^2 f_rc^2 C1^2 P_NL f_2h + P_II^1h,   C1 = 0.01389 Omega_m / D(z)
Windows (F21 App. B): f_1h = 1 - exp[-(k/4)^2], f_2h = exp[-(k/6)^2]
(variant "nowin": f_1h = f_2h = 1).
"""
import numpy as np
import camb
from scipy.special import sici, spherical_jn, erf
from scipy.integrate import cumulative_trapezoid, trapezoid

H0, OMB, OMM, NS, AS, MNU, TAU = 68.1, 0.0486, 0.306, 0.967, 2.099e-9, 0.06, 0.0697186
h = H0/100.0
RHO_CRIT = 2.77536627e11          # h^2 M_sun / Mpc^3  -> (M_sun/h)/(Mpc/h)^3
OMNU = MNU/93.14/h**2
RHO_M = RHO_CRIT*OMM
RHO_CB = RHO_CRIT*(OMM - OMNU)
DELTA, DELTA_C = 200.0, 1.686
C1RHOCRIT = 0.01389
R_FLOOR, GAMMA_MAX = 0.06, 0.3
K1H, K2H = 4.0, 6.0
PL = {2: -1.875, 4: -1.875, 6: -2.1328125}


class Linear:
    """CAMB linear P_cb(k, z) (h units) and growth."""

    def __init__(self, zs):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=OMB*h**2, omch2=(OMM - OMB - OMNU)*h**2, mnu=MNU, num_massive_neutrinos=1, tau=TAU)
        pars.InitPower.set_params(As=AS, ns=NS)
        zz = sorted(set([0.0] + [float(z) for z in zs]), reverse=True)
        pars.set_matter_power(redshifts=zz, kmax=200.0)
        pars.NonLinear = camb.model.NonLinear_none
        res = camb.get_results(pars)
        self.kh, self.z, self.pk = res.get_linear_matter_power_spectrum(var1="delta_nonu", var2="delta_nonu", hubble_units=True, k_hunit=True)
        _, _, self.pk_tot = res.get_linear_matter_power_spectrum(var1="delta_tot", var2="delta_tot", hubble_units=True, k_hunit=True)
        self.z = np.array(self.z)

    def Ptot(self, z, k):
        """Total-matter linear P(k) at z on any k grid (log interpolation)."""
        i = int(np.argmin(np.abs(self.z - z)))
        assert abs(self.z[i] - z) < 1e-6
        return np.exp(np.interp(np.log(k), np.log(self.kh), np.log(self.pk_tot[i])))

    def P(self, z):
        i = int(np.argmin(np.abs(self.z - z)))
        assert abs(self.z[i] - z) < 1e-6
        return self.kh, self.pk[i]


def sigma2_R(kh, pk, R):
    lk = np.log(kh)
    x = np.outer(R, kh)
    W = 3*(np.sin(x) - x*np.cos(x))/x**3
    return trapezoid(kh[None, :]**3*pk[None, :]*W**2/(2*np.pi**2), lk, axis=1)


def tinker_f(nu, z, alpha=1.0):
    zz = min(z, 3.0)
    beta, phi = 0.589*(1 + zz)**0.20, -0.729*(1 + zz)**-0.08
    eta, gamma = -0.243*(1 + zz)**0.27, 0.864*(1 + zz)**-0.01
    return alpha*(1 + (beta*nu)**(-2*phi))*nu**(2*eta)*np.exp(-gamma*nu**2/2)


def tinker_b(nu):
    y = np.log10(DELTA)
    A = 1.0 + 0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4/y)**4)
    c = 2.4
    return 1 - A*nu**a/(nu**a + DELTA_C**a) + B*nu**b + C*nu**c


def nfw_u(k, rs, c):
    """u(k|M) normalised to 1 at k -> 0; k [nk], rs, c scalars."""
    t = k*rs
    si1, ci1 = sici(t)
    si2, ci2 = sici((1 + c)*t)
    mc = np.log(1 + c) - c/(1 + c)
    return (np.sin(t)*(si2 - si1) - np.sin(c*t)/((1 + c)*t) + np.cos(t)*(ci2 - ci1))/mc


class Halo:
    """Halo-model tables at one redshift. Mass grid in ln M; k grid given."""

    def __init__(self, lin, z, k, nM=160, lgM=(9.0, 16.0), nx=2500, kmax_1h=100.0):
        self.z, self.k = z, np.asarray(k, float)
        kh, pk = lin.P(z)
        _, pk0 = lin.P(0.0)
        self.lnM = np.linspace(lgM[0]*np.log(10), lgM[1]*np.log(10), nM)
        M = np.exp(self.lnM)
        self.M = M
        R = (3*M/(4*np.pi*RHO_CB))**(1/3)
        s2 = sigma2_R(kh, pk, R)
        s20 = sigma2_R(kh, pk0, R)
        self.nu = DELTA_C/np.sqrt(s2)
        self.Dcb = float(np.sqrt(np.median(s2/s20)))
        dlnnu = np.gradient(np.log(self.nu), self.lnM)
        # alpha from int b f dnu = 1 on a wide nu grid
        nug = np.geomspace(1e-3, 30, 6000)
        alpha = 1.0/trapezoid(tinker_b(nug)*tinker_f(nug, z), nug)
        self.f = tinker_f(self.nu, z, alpha)
        self.dndlnM = (RHO_CB/M)*self.f*self.nu*dlnnu        # (h/Mpc)^3
        self.b = tinker_b(self.nu)
        self.c = 9.0*self.nu**-0.29*self.Dcb**1.15
        self.rvir = (3*M/(4*np.pi*DELTA*RHO_M))**(1/3)
        self.rs = self.rvir/self.c
        self.mc = np.log(1 + self.c) - self.c/(1 + self.c)
        self.u = np.array([nfw_u(self.k, rs, c) for rs, c in zip(self.rs, self.c)])   # [nM, nk]
        self.kmax_1h, self.nx = kmax_1h, nx
        self.kin = self.k <= kmax_1h

    def gamma_hat_grid(self, a1h_grid):
        """gamma_hat(k|M) for every |a_1h| of a grid: [n_a, nM, nk_in]. r_e
        depends on |a_1h|; the radial integrals are cumulative arrays per halo,
        read at x_in = min(x_e, c) (linear interpolation on a fine log grid)."""
        kk = self.k[self.kin]
        out = np.zeros((len(a1h_grid), len(self.M), kk.size))
        for iM in range(len(self.M)):
            c, rs, rv = self.c[iM], self.rs[iM], self.rvir[iM]
            x = np.concatenate([[0.0], np.geomspace(1e-5*c, c, self.nx)])
            tx = np.outer(kk*rs, x)
            xs = np.where(x > 0, x, 1.0)
            Ksum = []
            F1, F2 = {}, {}
            for l in (2, 4, 6):
                jl = spherical_jn(l, tx)
                F1[l] = cumulative_trapezoid(x[None, :]/(1 + x[None, :])**2*jl, x, axis=1, initial=0.0)
                c2 = cumulative_trapezoid(np.where(x[None, :] > 0, jl/xs[None, :]/(1 + x[None, :])**2, 0.0), x, axis=1, initial=0.0)
                F2[l] = c2[:, -1:] - c2
            for ia, a1h in enumerate(a1h_grid):
                re = max(R_FLOOR, rv*np.sqrt(abs(a1h)/GAMMA_MAX))
                xe = re/rs
                xin = min(xe, c)
                j = int(np.clip(np.searchsorted(x, xin), 1, x.size - 1))
                w = (xin - x[j - 1])/(x[j] - x[j - 1])
                K = 0.0
                for l in (2, 4, 6):
                    f1 = F1[l][:, j - 1]*(1 - w) + F1[l][:, j]*w
                    f2 = (F2[l][:, j - 1]*(1 - w) + F2[l][:, j]*w) if xe < c else 0.0
                    K = K + PL[l]*c**2*(f1/xe**2 + f2)
                out[ia, iM] = K/self.mc[iM]
        return out

    def set_hod(self, lgMmin, sig, lgM1, lgM0=None, alpha=1.0):
        lgM = self.lnM/np.log(10)
        M0 = 10**(lgMmin if lgM0 is None else lgM0)
        self.Nc = 0.5*(1 + erf((lgM - lgMmin)/sig))
        self.Ns = np.where(self.M > M0, self.Nc*np.clip((self.M - M0)/10**lgM1, 0, None)**alpha, 0.0)
        Nt = self.Nc + self.Ns
        self.ng = trapezoid(self.dndlnM*Nt, self.lnM)
        self.f_rc = trapezoid(self.dndlnM*self.Nc, self.lnM)/self.ng
        self.f_sat = 1 - self.f_rc
        self.bg = trapezoid(self.dndlnM*self.b*Nt, self.lnM)/self.ng
        self.bm = trapezoid(self.dndlnM*self.b*self.M/RHO_M, self.lnM)   # mass-weighted bias (consistency, ~1)
        return self

    def S_grid(self, a1h_grid):
        """S_dI, S_II [n_a, nk] on the k grid for every |a_1h| of the grid
        (zero above kmax_1h). Needs set_hod first."""
        G = self.gamma_hat_grid(a1h_grid)
        w = self.dndlnM[None, :, None]
        Ns = (self.Ns/self.ng)[None, :, None]
        SdI = np.zeros((len(a1h_grid), self.k.size))
        SII = np.zeros_like(SdI)
        SdI[:, self.kin] = trapezoid(w*(self.M[None, :, None]/RHO_M)*self.u[None, :, self.kin]*Ns*np.abs(G), self.lnM, axis=1)
        SII[:, self.kin] = trapezoid(w*Ns**2*G**2, self.lnM, axis=1)
        return SdI, SII


    def strict_grid(self, a1h_grid):
        """Strict halo model pieces (no F21 windows, 2-halo on P_lin):
        S_dI, S_II [n_a, nk] (1-halo, as S_grid), Q [n_a, nk] = int dlnM
        dn/dlnM b(M) (N_s/n_g) |gamma_hat| (satellite 2-halo kernel), and
        I1m [nk] = int dlnM dn/dlnM b (M/rho_cb) u + (1 - same at u = 1)
        (matter 2-halo integral, normalised to 1 at k -> 0: the mass below
        the grid is put in the lowest bin with u = 1)."""
        G = self.gamma_hat_grid(a1h_grid)
        w = self.dndlnM[None, :, None]
        Ns = (self.Ns/self.ng)[None, :, None]
        SdI = np.zeros((len(a1h_grid), self.k.size))
        SII = np.zeros_like(SdI)
        Q = np.zeros_like(SdI)
        SdI[:, self.kin] = trapezoid(w*(self.M[None, :, None]/RHO_M)*self.u[None, :, self.kin]*Ns*np.abs(G), self.lnM, axis=1)
        SII[:, self.kin] = trapezoid(w*Ns**2*G**2, self.lnM, axis=1)
        Q[:, self.kin] = trapezoid(w*self.b[None, :, None]*Ns*np.abs(G), self.lnM, axis=1)
        wb = self.dndlnM*self.b*self.M/RHO_CB
        I0 = trapezoid(wb, self.lnM)
        I1m = trapezoid(wb[:, None]*self.u, self.lnM, axis=0) + (1 - I0)
        self.I0 = I0
        return SdI, SII, Q, I1m


def f1h(k, nowin=False):
    return np.ones_like(k) if nowin else -np.expm1(-(k/K1H)**2)


def f2h(k, nowin=False):
    return np.ones_like(k) if nowin else np.exp(-(k/K2H)**2)
