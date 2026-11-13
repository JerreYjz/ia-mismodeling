#!/usr/bin/env python3
"""Strictly self-consistent halo-model IA against the FLAMINGO spectra.

Differences from the Fortuna+21 / halo.c model (halo_sim.py):
  * the 2-halo terms use the linear spectrum through the halo-model matter
    2-halo integral, not P_NL (no matter 1-halo power inside the IA 2-halo);
  * satellites contribute to the 2-halo term through the same kernel as in the
    1-halo term;
  * no F21 windows (f_1h = f_2h = 1): the 1-halo / 2-halo split is the halo
    model's own.
Model (cosmolike convention; centrals aligned with the linear tide, amplitude
a1 C1 per central; satellites radially aligned, amplitude a1h):
  g(k)  = a1 f_rc C1 + a1h Q(k)                       (IA field per unit delta_lin)
  P_dE  = -[ I1m(k) P_lin g(k) + a1h S_dI(k) ]
  P_EE  =    P_lin g(k)^2 + a1h^2 S_II(k)
  Q = int dlnM n b (N_s/n_g) |gamma_hat|, I1m = int dlnM n b (M/rho) u (-> 1),
  S_dI, S_II as in halo_ia (1-halo). k -> 0: P_dE -> -a1 f_rc C1 P_lin.
Same HOD calibration, data, probes, fit rule as halo_sim.py.
Outputs: halo_strict_tables.npz, halo_strict_sim.npz, halo_strict_cuts.csv.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
sys.path.insert(0, os.path.join(HERE, "..", "cocoa"))
import halo_ia as H          # noqa: E402
import halo_sim as HS        # noqa: E402
import simfit as SF          # noqa: E402
from cosmolike_xi_ia import KW   # noqa: E402

A_GRID = HS.A_GRID


def tables():
    t0 = time.time()
    zs, kd = SF.zs, SF.kd
    k_all = np.unique(np.concatenate([kd, KW]))
    lin = H.Linear(list(zs))
    old = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
    nsn = len(zs)
    SdI = np.zeros((nsn, A_GRID.size, k_all.size))
    SII, Q = np.zeros_like(SdI), np.zeros_like(SdI)
    I1m = np.zeros((nsn, k_all.size))
    Plin = np.zeros_like(I1m)
    I0 = np.zeros(nsn)
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=1600)
        lgMmin, lgM1 = old["hod"][s, :2]          # same HOD as the F21 run
        hm.set_hod(lgMmin, HS.SIG_LGM, lgM1, alpha=HS.ALPHA)
        SdI[s], SII[s], Q[s], I1m[s] = hm.strict_grid(A_GRID)
        I0[s] = hm.I0
        Plin[s] = lin.Ptot(float(z), k_all)
        print(f"z {z:4.2f}: f_rc {hm.f_rc:.3f}, mass in grid {I0[s]:.3f}, I1m(1)={np.interp(1.0, k_all, I1m[s]):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    tab = dict(zs=zs, kd=kd, k_all=k_all, a_grid=A_GRID, SdI=SdI, SII=SII, Q=Q, I1m=I1m, Plin=Plin, I0=I0, hod=old["hod"])
    np.savez(os.path.join(HERE, "halo_strict_tables.npz"), **tab)
    return tab


def model_factory(tab, s):
    kd, k_all = tab["kd"], tab["k_all"]
    idx = np.searchsorted(k_all, kd)
    assert np.allclose(k_all[idx], kd)
    Sd, Si, Qs = tab["SdI"][s][:, idx], tab["SII"][s][:, idx], tab["Q"][s][:, idx]
    I1, PL = tab["I1m"][s][idx], tab["Plin"][s][idx]
    la = np.log(np.maximum(tab["a_grid"], 1e-6))
    frc = tab["hod"][s, 5]
    C1 = SF.model.cosmo["Om"]*H.C1RHOCRIT/SF.model.D(SF.zs[s])

    def at(T, a):
        x = np.log(max(a, 1e-6))
        j = int(np.clip(np.searchsorted(la, x), 1, la.size - 1))
        w = np.clip((x - la[j - 1])/(la[j] - la[j - 1]), 0, 1)
        return T[j - 1]*(1 - w) + T[j]*w

    def pred(a1, a1h):
        g = a1*frc*C1 + a1h*at(Qs, a1h)
        PdE = -(I1*PL*g + a1h*at(Sd, a1h))
        PEE = PL*g**2 + a1h**2*at(Si, a1h)
        return np.array([SF.b[s]*PdE, PEE + SF.Ns[s], np.zeros_like(kd) + SF.Ns[s]])
    return pred


def fit_all(tab):
    kg = SF.kg
    X = np.zeros((SF.nsn, kg.size, 2))
    C = np.zeros((SF.nsn, kg.size))
    NU = np.zeros_like(C)
    for s in range(SF.nsn):
        pred = model_factory(tab, s)
        for j, k in enumerate(kg):
            X[s, j], C[s, j], NU[s, j] = HS.fit_one(pred, s, k)
    P = chi2_dist.sf(C, NU)
    cuts = np.array([SF.cut_rule(P[s] >= 0.05) for s in range(SF.nsn)])
    core = SF.core
    pc = chi2_dist.sf(C[core].sum(0), NU[core].sum(0))
    out = dict(x=X, chi2=C, dof=NU, p=P, cut=cuts, comb_p=pc, comb_cut=SF.cut_rule(pc >= 0.05))
    np.savez(os.path.join(HERE, "halo_strict_sim.npz"), kgrid=kg, zs=SF.zs, core=core, **{f"strict_{k}": v for k, v in out.items()})
    hs = np.load(os.path.join(HERE, "halo_sim.npz"))
    with open(os.path.join(HERE, "halo_strict_cuts.csv"), "w") as fh:
        fh.write("z,NLA_GoF,TATT_GoF,halo_F21_GoF,halo_strict_GoF\n")
        for s in np.where(core)[0]:
            fh.write(f"{SF.zs[s]:g},{SF.v3['cut_NLA_GoF'][s]:.3f},{SF.v3['cut_TATT_GoF'][s]:.3f},{hs['halo_cut'][s]:.3f},{cuts[s]:.3f}\n")
    print("strict cuts (core):", np.round(cuts[core], 3), " combined cut", out["comb_cut"])
    for s in np.where(core)[0]:
        j = [int(np.argmin(np.abs(kg - kk))) for kk in (0.16, 0.29, 0.52, 1.0, 3.04)]
        print(f"z {SF.zs[s]:4.2f} chi2/nu at k<=0.16,0.29,0.52,1,3: strict " +
              " ".join(f"{C[s, i]/NU[s, i]:6.2f}" for i in j) + " | F21 " +
              " ".join(f"{hs['halo_chi2'][s, i]/hs['halo_dof'][s, i]:6.2f}" for i in j) + " | NLA " +
              " ".join(f"{SF.v3['NLA_chi2'][s, i]/SF.v3['NLA_dof'][s, i]:6.2f}" for i in j) +
              f"  (a1, a1h at 0.29: {X[s, j[1], 0]:.2f}, {X[s, j[1], 1]:.3f})", flush=True)
    return out


if __name__ == "__main__":
    f = os.path.join(HERE, "halo_strict_tables.npz")
    tab = dict(np.load(f)) if (len(sys.argv) > 1 and sys.argv[1] == "fit") else tables()
    fit_all(tab)
