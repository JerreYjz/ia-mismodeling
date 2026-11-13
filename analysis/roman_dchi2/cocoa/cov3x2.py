#!/usr/bin/env python3
"""Gaussian covariance of the Roman real-space 3x2pt vector (flat sky).

Layout as driver3x2 (2115 entries). Fields: lens galaxies g_i (8 bins, linear
bias b_i, same n(z) as the sources) and source shear E_j (8 bins). Limber
spectra from roman_limber.RomanShear kernels (CAMB P_NL at D3A):
    C_EE^ij = F  int dchi q_i q_j / chi^2 P,   F = (l+2)(l+1)l(l-1)/(l+1/2)^4
    C_gE^ij = F^1/2 int dchi n_i b_i q_j / chi^2 P
    C_gg^ij = int dchi n_i n_j b_i b_j / chi^2 P
Noise: N_E = sigma_e^2 / n_s (sigma_e per component), N_g = 1 / n_l.
For two real-space statistics X_a^{(f1 f2)}(theta_1) = int l dl/2pi J_a(l theta_1) C_{f1 f2}
and X_b^{(f3 f4)}(theta_2):
    Cov = 1/(2 pi A) int l dl Jbar_a(l theta_1) Jbar_b(l theta_2)
          [ Chat_{f1 f3} Chat_{f2 f4} + Chat_{f1 f4} Chat_{f2 f3} ]
with J_0 (xi_+, w), J_4 (xi_-), J_2 (gamma_t), Chat = C + N delta, Jbar the
theta-weighted bin average, Gaussian taper exp[-(l/6e4)^2]. For xi_+- the
B-mode noise enters as in roman_limber.gaussian_cov (SE +- SB). The pure
noise-noise term is removed from the integrand and put back in its exact
discrete form on the diagonal of each auto block:
    X_nn x 2 / (A pi (theta_hi^2 - theta_lo^2)).
    python cov3x2.py <cocoa data dir>   -> analysis/cov3x2_gauss_d3a.npy
"""
import os
import sys

import numpy as np
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bias_analysis as B                              # noqa: E402
from roman_limber import RomanShear, ARCMIN            # noqa: E402
from driver3x2 import B1_FID                           # noqa: E402

AREA_DEG2, N_SRC, N_LENS, SIGMA_E = 2415.0, 41.3/8, 41.3/8, 0.30
GGL_EXCLUDE = {(6, 0), (7, 0), (7, 1)}


def cov3x2(S, bias=B1_FID, nell=6000, ratio=None):
    """S: RomanShear with .q, .n, .chi, .K, .P, .ell, .edges, .pairs.
    ratio: IA ratio tables (dE, EE, BB callables) added to the signal spectra
    (GI + II in C_EE, B modes in C_BB, GI in C_gE), as the xi covariance."""
    ell = S.ell
    F = S.prefactor(ell)
    w = 1.0/S.chi**2
    nb = S.nbin
    RdE = ratio["dE"](S.K, S.Z)*S.P if ratio else None
    # Limber spectra on S.ell
    CEE = {}
    CgE = {}
    Cgg = {}
    CBB = {}
    if ratio:
        CEt, CBt = S.cl(ratio, gg=False)
    for i in range(nb):
        for j in range(nb):
            if j >= i:
                p = S.pairs.index((i, j))
                CEE[(i, j)] = F*np.trapezoid(S.q[i]*S.q[j]*w*S.P, S.chi, axis=1) + (CEt[p] if ratio else 0.0)
                CBB[(i, j)] = CBt[p] if ratio else np.zeros_like(ell)
                Cgg[(i, j)] = bias[i]*bias[j]*np.trapezoid(S.n[i]*S.n[j]*w*S.P, S.chi, axis=1)
            CgE[(i, j)] = np.sqrt(F)*bias[i]*np.trapezoid(S.n[i]*(S.q[j]*S.P + (S.n[j]*RdE if ratio else 0.0))*w, S.chi, axis=1)
    ld = np.geomspace(2.0, ell[-1], nell)
    dense = lambda c: np.interp(np.log(ld), np.log(ell), c)
    NE = SIGMA_E**2/(N_SRC/ARCMIN**2)
    NG = 1.0/(N_LENS/ARCMIN**2)
    Ehat = {k: dense(v) for k, v in CEE.items()}
    Bhat = {k: dense(v) for k, v in CBB.items()}
    gEhat = {k: dense(v) for k, v in CgE.items()}
    gghat = {k: dense(v) for k, v in Cgg.items()}

    def C(f1, f2, noise=True):
        """Chat between fields ('E', i) / ('g', i), dense in l; B-mode: noise only."""
        (t1, a), (t2, b) = f1, f2
        if t1 == "E" and t2 == "E":
            return Ehat[(min(a, b), max(a, b))] + (NE if (noise and a == b) else 0.0)
        if t1 == "g" and t2 == "g":
            return gghat[(min(a, b), max(a, b))] + (NG if (noise and a == b) else 0.0)
        if t1 == "g":
            return gEhat[(a, b)]
        return gEhat[(b, a)]

    def CB(f1, f2):
        (t1, a), (t2, b) = f1, f2
        return Bhat[(min(a, b), max(a, b))] + (NE if a == b else 0.0)

    def Nn(f1, f2):
        (t1, a), (t2, b) = f1, f2
        if t1 != t2 or a != b:
            return 0.0
        return NE if t1 == "E" else NG
    # bin-averaged Bessel functions
    Js = {}
    for order in (0, 2, 4):
        rows = []
        for t1, t2 in zip(S.edges[:-1], S.edges[1:]):
            t = np.linspace(t1, t2, 48)
            wt = t/np.trapezoid(t, t)
            rows.append(np.trapezoid(wt[:, None]*jv(order, np.outer(t, ld)), t, axis=0))
        Js[order] = np.array(rows)
    A = AREA_DEG2*(np.pi/180)**2
    hw = ld**2*np.gradient(np.log(ld))/(2*np.pi*A)*S.taper(ld)
    nt = S.ntheta
    # blocks: (name, fields (f1, f2), bessel order, spin sign for B handling)
    blocks = []
    for (i, j) in S.pairs:
        blocks.append(("xip", (("E", i), ("E", j)), 0))
    for (i, j) in S.pairs:
        blocks.append(("xim", (("E", i), ("E", j)), 4))
    for l in range(nb):
        for s in range(nb):
            if (l, s) not in GGL_EXCLUDE:
                blocks.append(("gt", (("g", l), ("E", s)), 2))
    for i in range(nb):
        blocks.append(("w", (("g", i), ("g", i)), 0))
    n = nt*len(blocks)
    assert n == 2115, n
    cov = np.zeros((n, n))
    disc = 2.0/(A*np.pi*(S.edges[1:]**2 - S.edges[:-1]**2))
    for p, (na, (f1, f2), oa) in enumerate(blocks):
        for q in range(p, len(blocks)):
            nb_, (f3, f4), ob = blocks[q]
            SE = C(f1, f3)*C(f2, f4) + C(f1, f4)*C(f2, f3)
            nn = Nn(f1, f3)*Nn(f2, f4) + Nn(f1, f4)*Nn(f2, f3)
            SE = SE - nn
            if na in ("xip", "xim") and nb_ in ("xip", "xim"):
                SB = CB(f1, f3)*CB(f2, f4) + CB(f1, f4)*CB(f2, f3) - nn
                sgn = 1.0 if na == nb_ else -1.0
                integ = SE + sgn*SB
            else:
                integ = SE
            blk = (Js[oa]*(integ*hw)) @ Js[ob].T
            if q == p:
                blk = blk + np.diag(nn*disc)
            a0, b0 = p*nt, q*nt
            cov[a0:a0 + nt, b0:b0 + nt] = blk
            if q != p:
                cov[b0:b0 + nt, a0:a0 + nt] = blk.T
    return 0.5*(cov + cov.T), blocks


def main():
    data = sys.argv[1]
    from roman_limber import Cosmology
    cosmo = Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
    S = RomanShear(cosmo, os.path.join(data, "example1.nz"))
    import roman_limber as RL
    d = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
    zs, kd, Rt = d["zs"], d["k"][0], d["Rtrue"]
    lo = np.geomspace(1e-4, 0.05, 30, endpoint=False)
    hi = np.geomspace(kd[-1]*1.02, 2e3, 20)
    m = kd >= 0.05
    kk = np.concatenate([lo, kd[m], hi])
    out = np.zeros((len(zs), 3, len(kk)))
    for si in range(len(zs)):
        for x in range(3):
            r = Rt[si, x, m]
            out[si, x] = np.concatenate([np.full(len(lo), r[0]), r, np.full(len(hi), r[-1])])
    ratio = {X: RL.ratio_table(zs, kk, out[:, i]) for i, X in enumerate(("dE", "EE", "BB"))}
    cov, blocks = cov3x2(S, ratio=ratio)
    np.save(os.path.join(HERE, "analysis", "cov3x2_gauss_d3a.npy"), cov)
    old = np.load(os.path.join(HERE, "analysis", "cov_gauss_d3a.npy"))
    r = np.diag(cov)[:1080]/np.diag(old)
    print(f"xi block diagonal vs stored xi covariance: median ratio {np.median(r):.4f}, range {r.min():.3f}-{r.max():.3f}")
    ev = np.linalg.eigvalsh(cov)
    print(f"min eigenvalue / max: {ev.min()/ev.max():.2e} (positive definite: {ev.min() > 0})")
    print("done")


if __name__ == "__main__":
    main()
