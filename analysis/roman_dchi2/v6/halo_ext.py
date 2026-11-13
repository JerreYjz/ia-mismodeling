#!/usr/bin/env python3
"""Strict halo-model IA with three extra satellite freedoms (on top of
halo_strict.py: 2-halo on P_lin, satellites in the 2-halo term, no windows):

  1. radial slope b of the satellite alignment, gamma_bar(r) = a1h (r/r_vir)^b
     (F21 / halo.c: b = -2), capped at gamma_max = 0.3 with a core below
     r_e = max(r_floor, r_vir (a1h/gamma_max)^(1/b)) (b < 0); for b = 0 the
     amplitude is min(a1h, gamma_max) everywhere;
  2. satellite concentration c_sat = f_c c_DM (satellites distributed as NFW
     with c_sat inside r_vir; the matter profile in S_dI keeps c_DM);
  3. HOD satellite slope alpha, with lg M_min, lg M_1 re-solved for every alpha
     to FLAMINGO's n_g and b_g (so f_rc, f_sat follow).

Kernel in y = r/r_vir (so the Bessel table j_l(k r_vir y) is shared by all
f_c, b, a1h): W(y) = y/(1 + c_sat y)^2, N = int_0^1 W dy,
  gamma_hat = sum_l P_l [ g_c int_0^{y_e} W j_l dy + int_{y_e}^1 W y^b j_l dy ] / N,
g_c = y_e^b (b < 0). For f_c = 1, b = -2, alpha = 1 this is halo_strict.py's
kernel (checked at run time).

Grids: alpha in ALPHAS, f_c in FCS, b in BS, a1h in A_GRID.
Output halo_ext_tables_<part>.npz: SdI, SII, Q [s, alpha, f_c, b, a, k], I1m,
Plin [s, k], hod [s, alpha, (lgMmin, lgM1, ok, n_g, b_g, f_rc, f_sat)].
    python halo_ext.py <part> <nparts>     (snapshots split across processes)
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import fsolve, brentq
from scipy.special import spherical_jn
from scipy.integrate import cumulative_trapezoid, trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
sys.path.insert(0, os.path.join(HERE, "..", "cocoa"))
import halo_ia as H          # noqa: E402
import halo_sim as HS        # noqa: E402
import simfit as SF          # noqa: E402
from cosmolike_xi_ia import KW   # noqa: E402

ALPHAS = np.array([0.75, 1.0, 1.25, 1.5])
FCS = np.array([0.25, 0.5, 0.75, 1.0])
BS = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0])
A_GRID = HS.A_GRID
NY = 1200


def calibrate(hm, ngt, bgt, alpha):
    def eqs(p):
        hm.set_hod(p[0], HS.SIG_LGM, p[1], alpha=alpha)
        return [np.log(hm.ng/ngt), hm.bg/bgt - 1]
    best = None
    for x0 in ([11.5, 12.8], [11.0, 12.5], [12.0, 13.2], [10.5, 12.0], [11.5, 13.5], [11.5, 12.2]):
        sol = fsolve(eqs, x0)
        r = np.abs(eqs(sol))
        if best is None or r.sum() < best[1].sum():
            best = (sol, r)
    sol, r = best
    ok = r.max() < 1e-3 and sol[1] > sol[0]
    if not ok:
        f = lambda lg: (hm.set_hod(lg, HS.SIG_LGM, lg + 1.2, alpha=alpha).ng - ngt)
        lg = brentq(f, 9.5, 14.0)
        sol = np.array([lg, lg + 1.2])
    hm.set_hod(sol[0], HS.SIG_LGM, sol[1], alpha=alpha)
    return sol, ok


def kernel_halo(kk, rvir, cdm, out_shape):
    """gamma_hat [n_fc, n_b, n_a, nk] for one halo."""
    y = np.concatenate([[0.0], np.geomspace(1e-5, 1.0, NY)])
    J = {l: spherical_jn(l, np.outer(kk*rvir, y)) for l in (2, 4, 6)}
    yf = H.R_FLOOR/rvir
    G = np.zeros(out_shape)
    ys = np.where(y > 0, y, 1.0)
    for i, fc in enumerate(FCS):
        c = fc*cdm
        W = y/(1 + c*y)**2
        N = trapezoid(W, y)
        Cin = {l: cumulative_trapezoid(W[None, :]*J[l], y, axis=1, initial=0.0) for l in J}
        for jb, b in enumerate(BS):
            Wb = np.where(y > 1e-4, W*ys**b, 0.0)
            Cout = {l: cumulative_trapezoid(Wb[None, :]*J[l], y, axis=1, initial=0.0) for l in J}
            for ia, a in enumerate(A_GRID):
                fac = 1.0
                if b < 0:
                    ycap = (H.GAMMA_MAX/a)**(1.0/b) if a > 0 else 0.0
                    ye = max(yf, ycap)
                    gc = ye**b
                else:
                    ye, gc = yf, 1.0
                    fac = min(1.0, H.GAMMA_MAX/a) if a > 0 else 1.0
                yin = min(ye, 1.0)
                j = int(np.clip(np.searchsorted(y, yin), 1, y.size - 1))
                w = (yin - y[j - 1])/(y[j] - y[j - 1])
                K = 0.0
                for l, Pl in H.PL.items():
                    cin = Cin[l][:, j - 1]*(1 - w) + Cin[l][:, j]*w
                    if ye < 1.0:
                        cout = Cout[l][:, -1] - (Cout[l][:, j - 1]*(1 - w) + Cout[l][:, j]*w)
                    else:
                        cout = 0.0
                    K = K + Pl*(gc*cin + cout)
                G[i, jb, ia] = fac*K/N
    return G


def main():
    part, nparts = int(sys.argv[1]), int(sys.argv[2])
    t0 = time.time()
    zs, kd = SF.zs, SF.kd
    k_all = np.unique(np.concatenate([kd, KW]))
    lin = H.Linear(list(zs))
    mine = [s for s in range(len(zs)) if s % nparts == part]
    na, nal, nfc, nb = A_GRID.size, ALPHAS.size, FCS.size, BS.size
    shp = (len(mine), nal, nfc, nb, na, k_all.size)
    SdI, SII, Q = np.zeros(shp), np.zeros(shp), np.zeros(shp)
    I1m = np.zeros((len(mine), k_all.size))
    Plin = np.zeros_like(I1m)
    hod = np.zeros((len(mine), nal, 7))
    for im, s in enumerate(mine):
        z = float(zs[s])
        hm = H.Halo(lin, z, k_all, nM=140, nx=10)
        ngt = HS.NGAL[z]*1e6/HS.BOX**3
        sel = (kd >= 0.04) & (kd <= 0.1)
        bgt = float(np.mean(SF.b[s][sel]))
        NsW = []
        for ial, al in enumerate(ALPHAS):
            sol, ok = calibrate(hm, ngt, bgt, al)
            hod[im, ial] = [sol[0], sol[1], ok, hm.ng, hm.bg, hm.f_rc, hm.f_sat]
            NsW.append(hm.Ns/hm.ng)
        NsW = np.array(NsW)                                   # [alpha, nM]
        wb = hm.dndlnM*hm.b*hm.M/H.RHO_CB
        I1m[im] = trapezoid(wb[:, None]*hm.u, hm.lnM, axis=0) + (1 - trapezoid(wb, hm.lnM))
        Plin[im] = lin.Ptot(z, k_all)
        # trapezoid weights in ln M
        dl = np.diff(hm.lnM)
        tw = np.zeros_like(hm.lnM)
        tw[:-1] += dl/2
        tw[1:] += dl/2
        kin = hm.kin
        for iM in range(hm.M.size):
            if NsW[:, iM].max() <= 0:
                continue
            G = kernel_halo(hm.k[kin], hm.rvir[iM], hm.c[iM], (nfc, nb, na, int(kin.sum())))
            aG = np.abs(G)
            n = tw[iM]*hm.dndlnM[iM]
            for ial in range(nal):
                ns = NsW[ial, iM]
                if ns <= 0:
                    continue
                SdI[im, ial][..., kin] += n*(hm.M[iM]/H.RHO_M)*hm.u[iM, kin]*ns*aG
                SII[im, ial][..., kin] += n*ns**2*G**2
                Q[im, ial][..., kin] += n*hm.b[iM]*ns*aG
        print(f"[part {part}] z {z:4.2f}: f_rc(alpha) {np.round(hod[im, :, 5], 3)} solved {hod[im, :, 2].astype(int)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(os.path.join(HERE, f"halo_ext_tables_{part}.npz"), snaps=np.array(mine[:im + 1]), zs=zs, kd=kd, k_all=k_all,
                 a_grid=A_GRID, alphas=ALPHAS, fcs=FCS, bs=BS, SdI=SdI[:im + 1], SII=SII[:im + 1], Q=Q[:im + 1],
                 I1m=I1m[:im + 1], Plin=Plin[:im + 1], hod=hod[:im + 1])


if __name__ == "__main__":
    main()
