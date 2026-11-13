r"""Cobaya theory: sigma8 and S8 from any theory that provides the linear P(k).

For emulator chains whose power-spectrum emulator gives P_lin and P_NL but
no sigma8. It asks the provider (e.g. emulmps) for the linear total-matter
P(k, z=0) on Cobaya's standard Pk_grid and computes

    sigma^2(R) = int dlnk Delta^2(k) W^2(kR),  Delta^2 = k^3 P / (2 pi^2)
    W(x)       = 3 (sin x - x cos x) / x^3,     R = 8 Mpc/h
    S8         = sigma8 (Omega_m / 0.3)^0.5

Two methods (yaml option `method`):

  fftlog (default)  Python port of cosmolike's FFTLog mass variance,
                    sigma2_fields_build() in cosmolike/cosmo3D.c of
                    CosmoLike/cocoa-cosmolike-core (the "MASS VARIANCE OF
                    TOTAL MATTER AND COLD MATTER" section). Same bias 0.8,
                    k range 1e-7..1e25 h/Mpc with power-law continuation,
                    Mellin kernel of W^2 and high-frequency taper.
  trapz             direct trapezoid integral in lnk on the provider's grid.

Cobaya units: k in 1/Mpc, P in Mpc^3. Both methods convert to h/Mpc and
(Mpc/h)^3 internally, as cosmolike does, so R = 8 Mpc/h exactly.

yaml (raw docstring):
  theory:
    sigma8_from_pk:
      python_path: ./projects/roman_real/theories
      provides: [sigma8, S8]
      # method: fftlog      # or trapz
  params:
    sigma8: {latex: \sigma_8}
    S8:     {latex: S_8}
"""
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.interpolate import CubicSpline
from scipy.special import loggamma
from cobaya.theory import Theory


def _tophat(x):
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    small = x < 1e-3
    xs = x[small]
    w[small] = 1.0 - xs**2/10.0 + xs**4/280.0
    xl = x[~small]
    w[~small] = 3.0*(np.sin(xl) - xl*np.cos(xl))/xl**3
    return w


def mellin_tophat2(s):
    """int_0^inf dx x^(s-1) W(x)^2 for complex s, 0 < Re s < 4.

    (9 pi/2) Gamma(4-s) Gamma(s/2) / [2^(4-s) Gamma((5-s)/2)^2 Gamma(4-s/2)],
    evaluated with log-Gammas as in cosmo3D.c.
    """
    s = np.asarray(s, dtype=complex)
    ln = (np.log(4.5*np.pi) + loggamma(4.0 - s) + loggamma(s/2.0)
          - (4.0 - s)*np.log(2.0) - 2.0*loggamma((5.0 - s)/2.0)
          - loggamma(4.0 - s/2.0))
    return np.exp(ln)


def sigma2_fftlog(k, pk, R, bias=0.8, kmin=1e-7, kmax=1e25, dlnk=0.01):
    """sigma^2(R) by FFTLog, following cosmolike's sigma2_fields_build.

    k [h/Mpc], pk [(Mpc/h)^3] on any increasing grid; R [Mpc/h], scalar or
    array. Steps (cosmo3D.c numbering):
      1a  uniform lnk grid from kmin to kmax; lnP is splined inside the
          supplied range and continued linearly in lnk (power law) outside,
          using the edge pair of nodes, like p_lin in cosmolike
      3a  g = k^(-bias) Delta^2 = k^(3-bias) P / (2 pi^2); zero-pad to an
          FFT-friendly length
      1c  Mellin kernel of W^2 at s = bias + i eta, tapered above 75% of
          the Nyquist frequency with t - sin(2 pi t)/(2 pi)
      3b  one forward FFT, multiply by the kernel and the phase of the two
          grid origins, one inverse FFT
      3c  divide by N (done by irfft here) and multiply by R^(-bias)
    cosmolike samples lnR on the FFT grid and splines to its mass nodes;
    here the lnR origin is put at each requested R, so the value is read
    at node 0 and no spline is needed.
    """
    lnk_in = np.log(np.asarray(k, dtype=float))
    lnp_in = np.log(np.asarray(pk, dtype=float))
    first = int(np.floor((np.log(kmin) - lnk_in[0])/dlnk))
    last = int(np.ceil((np.log(kmax) - lnk_in[0])/dlnk))
    ninput = last - first + 1
    lnk = lnk_in[0] + (first + np.arange(ninput))*dlnk

    # 1a: interpolate inside, power-law continuation outside
    lnp = np.empty(ninput)
    inside = (lnk >= lnk_in[0]) & (lnk <= lnk_in[-1])
    lnp[inside] = CubicSpline(lnk_in, lnp_in)(lnk[inside])
    lo, hi = lnk < lnk_in[0], lnk > lnk_in[-1]
    slo = (lnp_in[1] - lnp_in[0])/(lnk_in[1] - lnk_in[0])
    shi = (lnp_in[-1] - lnp_in[-2])/(lnk_in[-1] - lnk_in[-2])
    lnp[lo] = lnp_in[0] + slo*(lnk[lo] - lnk_in[0])
    lnp[hi] = lnp_in[-1] + shi*(lnk[hi] - lnk_in[-1])

    # 3a: g on the FFT grid, zero-padded
    nfft = next_fast_len(ninput + (ninput % 2), real=True)
    nfft += nfft % 2
    g = np.zeros(nfft)
    g[:ninput] = np.exp(lnp + (3.0 - bias)*lnk)/(2.0*np.pi**2)

    # 1c: Mellin kernel and taper
    m = np.arange(nfft//2 + 1)
    eta = 2.0*np.pi*m/(nfft*dlnk)
    kern = mellin_tophat2(bias + 1j*eta)
    frac = 2.0*m/nfft
    t = np.clip((1.0 - frac)/0.25, 0.0, 1.0)
    kern = kern*np.where(frac > 0.75, t - np.sin(2.0*np.pi*t)/(2.0*np.pi), 1.0)

    # 3b/3c: sigma^2(R) = R^-b (1/N) sum_m c_m M(s_m) e^{-i eta (lnk0+lnR)}
    c = rfft(g)
    R = np.atleast_1d(np.asarray(R, dtype=float))
    out = np.empty(R.size)
    for i, r in enumerate(R):
        x = c*kern*np.exp(-1j*eta*(lnk[0] + np.log(r)))
        # irfft sums exp(+i...) and divides by N: pass the conjugate
        out[i] = irfft(np.conj(x), n=nfft)[0]*r**(-bias)
    if np.any(out <= 0):
        raise ValueError("nonpositive FFTLog variance")
    return out if out.size > 1 else float(out[0])


def sigma2_trapz(k, pk, R):
    lnk = np.log(k)
    f = k**3*pk*_tophat(k*R)**2/(2.0*np.pi**2)
    return float(np.trapezoid(f, lnk))


class sigma8_from_pk(Theory):
    # options (override in the yaml block)
    method = "fftlog"      # fftlog (cosmolike port) or trapz
    k_max = 5.0            # 1/Mpc asked of the provider
    var_pair = ("delta_tot", "delta_tot")   # total matter, as CAMB's sigma8
    omegam_name = "omegam"
    fftlog_dlnk = 0.01

    def initialize(self):
        self.var_pair = tuple(self.var_pair)
        if self.method not in ("fftlog", "trapz"):
            raise ValueError("method must be fftlog or trapz")

    def get_requirements(self):
        return {"H0": None, self.omegam_name: None,
                "Pk_grid": {"z": [0.0], "k_max": self.k_max,
                            "nonlinear": False,
                            "vars_pairs": [self.var_pair]}}

    def get_can_provide_params(self):
        return ["sigma8", "S8"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        k, z, pk = self.provider.get_Pk_grid(var_pair=self.var_pair,
                                             nonlinear=False)
        k, z, pk = np.asarray(k), np.asarray(z), np.asarray(pk)
        if pk.shape != (z.size, k.size):
            pk = pk.T                      # some providers give P[k, z]
        if np.min(np.abs(z)) > 1e-3:
            raise ValueError(f"provider has no z=0 row (z min {z.min()})")
        p0 = pk[np.argmin(np.abs(z))]
        h = self.provider.get_param("H0")/100.0
        om = self.provider.get_param(self.omegam_name)
        kh, ph = k/h, p0*h**3              # h/Mpc, (Mpc/h)^3
        if self.method == "fftlog":
            s2 = sigma2_fftlog(kh, ph, 8.0, dlnk=self.fftlog_dlnk)
        else:
            s2 = sigma2_trapz(kh, ph, 8.0)
        s8 = float(np.sqrt(s2))
        state["derived"] = {"sigma8": s8, "S8": float(s8*np.sqrt(om/0.3))}
