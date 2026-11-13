#!/usr/bin/env python3
"""Central-galaxy 1-halo IA from triaxial (elliptical NFW) host halos.

Physics. A halo's density is rho_NFW(r_e) with r_e^2 = r^T (1 - E) r, E a
small symmetric traceless tensor (E_ii = 2 ln of the axis lengths, abc = 1).
To first order the quadrupole is delta rho = -(1/2) (rhat E rhat) d rho/d ln r,
whose Fourier transform (integrating by parts, the truncation at r_vir
included) is
    delta rho(k)/M = -(khat E khat) u_E(k|M),
    u_E(k|M) = (k r_s / 2 m(c)) int_0^c x^2/(1+x)^2 j_1(k r_s x) dx     (~ k^2 at low k).
The central's intrinsic shape follows its halo, gamma_c = a_c [E]_proj
(a_c: alignment strength x misalignment x shape-convention, free). For
random halo orientations <E_ij E_kl> = (sigma_E^2/2)(d_ik d_jl + d_il d_jk -
2/3 d_ij d_kl), the cross spectrum carries the linear-alignment angular
factor (1 - mu^2), so the FLAMINGO convention (3/2 x monopole = mu = 0) is
the k-perp value:
    P_dE^{1h,c}(k) = -a_c Pc(k),
    Pc_m(k) = (1/2) int dlnM dn/dlnM (N_c/n_g) (M/rho_m) sigma_E^2(M) u_E(k|M)   (matter)
    Pc_g(k) = (1/2) int dlnM dn/dlnM (N_s/n_g^2) sigma_E^2(M) u_E(k|M)           (galaxy positions:
              satellites tracing the elliptical NFW; enters P_hE directly)
Halo shapes from the literature (Allgood et al. 2006): s = c/a = 0.54
(M/M*)^-0.05, M* at nu = 1, q = b/a = (1 + s)/2; sigma_E^2 = (4/5) sum_i
(ln a_i)^2. Variant "flat": sigma_E^2 fixed to its value at 10^13.5 M_sun/h.
The central II 1-halo term is zero (one central per halo; its self-pair is
shape noise).

Outputs halo_central_tables.npz: Pc_m, Pc_mflat, Pc_g [nsn, nk_all], sigE2
[nsn, nM], lgM, plus I1m, Plin from halo_strict_tables.npz.
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import halo_ia as H          # noqa: E402

ST = dict(np.load(os.path.join(HERE, "halo_strict_tables.npz")))
SIG_LGM, ALPHA = 0.25, 1.0


def j1(y):
    y = np.where(y < 1e-3, 1e-3, y)
    return np.sin(y)/y**2 - np.cos(y)/y


def j1_small(y):
    return np.where(y < 1e-3, y/3.0, j1(y))


def u_E(k, rs, c, nx=6000):
    x = np.concatenate([[0.0], np.geomspace(1e-5*c, c, nx)])
    t = k*rs
    w = x**2/(1 + x)**2
    I = trapezoid(w[None, :]*j1_small(np.outer(t, x)), x, axis=1)
    mc = np.log(1 + c) - c/(1 + c)
    return t/(2*mc)*I


def sigma_E2(M, nu, lnM):
    """Allgood et al. 2006 axis ratios -> first-order tensor variance."""
    lnMstar = np.interp(0.0, np.log(nu), lnM)          # nu = 1
    s = np.clip(0.54*np.exp(-0.05*(np.log(M) - lnMstar)), 0.3, 0.95)
    q = (1 + s)/2
    la = -(np.log(q) + np.log(s))/3
    lb, lc = np.log(q) + la, np.log(s) + la
    return 0.8*(la**2 + lb**2 + lc**2), s, q


def main():
    t0 = time.time()
    zs, k_all = ST["zs"], ST["k_all"]
    lin = H.Linear(list(zs))
    nsn = len(zs)
    Pm, Pmf, Pg = (np.zeros((nsn, k_all.size)) for _ in range(3))
    S2 = []
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=10)
        lgMmin, lgM1 = ST["hod"][s, :2]
        hm.set_hod(lgMmin, SIG_LGM, lgM1, alpha=ALPHA)
        assert abs(hm.ng/ST["hod"][s, 3] - 1) < 1e-3, (hm.ng, ST["hod"][s, 3])
        UE = np.array([u_E(k_all, rs, c) for rs, c in zip(hm.rs, hm.c)])        # [nM, nk]
        se2, sax, qax = sigma_E2(hm.M, hm.nu, hm.lnM)
        se2f = np.interp(13.5*np.log(10), hm.lnM, se2)*np.ones_like(se2)
        w = hm.dndlnM[:, None]
        Pm[s] = 0.5*trapezoid(w*(hm.Nc/hm.ng*hm.M/H.RHO_M*se2)[:, None]*UE, hm.lnM, axis=0)
        Pmf[s] = 0.5*trapezoid(w*(hm.Nc/hm.ng*hm.M/H.RHO_M*se2f)[:, None]*UE, hm.lnM, axis=0)
        Pg[s] = 0.5*trapezoid(w*(hm.Ns/hm.ng**2*se2)[:, None]*UE, hm.lnM, axis=0)
        S2.append(se2)
        i13 = int(np.argmin(np.abs(hm.lnM - 13.5*np.log(10))))
        ik = [int(np.argmin(np.abs(k_all - kk))) for kk in (0.1, 0.3, 1.0, 3.0)]
        print(f"z {z:4.2f}: lgM* {np.interp(0.0, np.log(hm.nu), hm.lnM)/np.log(10):5.2f}  s,q(1e13.5) {sax[i13]:.2f},{qax[i13]:.2f}"
              f"  sigE2 {se2[i13]:.3f}  Pc_m(0.1,0.3,1,3) {np.array2string(Pm[s, ik], precision=3)}"
              f"  Pc_g {np.array2string(Pg[s, ik], precision=3)}  [{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(HERE, "halo_central_tables.npz"), zs=zs, k_all=k_all, kd=ST["kd"], Pc_m=Pm, Pc_mflat=Pmf,
             Pc_g=Pg, sigE2=np.array(S2), lgM=hm.lnM/np.log(10), I1m=ST["I1m"], Plin=ST["Plin"])


if __name__ == "__main__":
    main()
