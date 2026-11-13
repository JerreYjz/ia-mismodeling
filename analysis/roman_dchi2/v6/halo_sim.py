#!/usr/bin/env python3
"""Halo-model IA against the FLAMINGO spectra: HOD calibration, S tables,
goodness of fit versus k_max, cuts. Same conventions as v3/v3_flamingo.py
(production truth: measured spectra, N_s fixed to the B plateau, probes hE,
EE, BB, 0.05 <= k <= k_max, FLAMINGO P_NL and b(k) in the model).

Model per snapshot (cosmolike convention, observed units):
  hE = b(k) P_dE,  EE = P_EE + N_s,  BB = N_s
  P_dE = -[a1 f_rc C1 P_NL f_2h + a_1h f_1h S_dI(a_1h)]
  P_EE = a1^2 f_rc^2 C1^2 P_NL f_2h + a_1h^2 f_1h S_II(a_1h),  C1 = 0.01389 Om / D
Free: a1 in [0, 20], a_1h in [0, 3]. Variants: F21 windows ("halo") and
f_1h = f_2h = 1 ("halo_nowin"). NLA (a_1h = 0, f_rc = f_2h = 1) for reference.
Outputs: halo_tables.npz (HOD, S tables on k_all for the a_1h grid),
halo_sim.npz, halo_cuts.csv, halo_hod.csv.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq, least_squares, fsolve
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
sys.path.insert(0, os.path.join(HERE, "..", "cocoa"))
import halo_ia as H          # noqa: E402
import simfit as SF          # noqa: E402
from cosmolike_xi_ia import KW   # noqa: E402

BOX = 681.0
NGAL = {0.0: 2.03, 0.1: 3.69, 0.2: 5.53, 0.3: 5.77, 0.4: 5.85, 0.5: 5.93, 0.6: 5.93, 0.75: 5.88,
        1.0: 5.66, 1.5: 4.81, 2.0: 3.66, 3.0: 1.61, 4.0: 0.53}          # millions (hinfo)
A_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 24)])
SIG_LGM, ALPHA = 0.25, 1.0


def calibrate_hod(hm, ng_target, bg_target):
    """lg M_min, lg M_1 such that n_g and b_g match; sigma, alpha fixed,
    M_0 = M_min. Falls back to n_g only (M_1 = 10^1.2 M_min) if no solution."""
    def eqs(p):
        hm.set_hod(p[0], SIG_LGM, p[1], alpha=ALPHA)
        return [np.log(hm.ng/ng_target), hm.bg/bg_target - 1]
    best = None
    for x0 in ([11.5, 12.8], [11.0, 12.5], [12.0, 13.2], [10.5, 12.0]):
        sol, info, ier, msg = fsolve(eqs, x0, full_output=True)
        r = np.abs(eqs(sol))
        if best is None or r.sum() < best[1].sum():
            best = (sol, r)
    sol, r = best
    ok = r.max() < 1e-3 and sol[1] > sol[0]
    if not ok:
        f = lambda lg: (hm.set_hod(lg, SIG_LGM, lg + 1.2, alpha=ALPHA).ng - ng_target)
        lg = brentq(f, 9.5, 14.0)
        sol = np.array([lg, lg + 1.2])
    hm.set_hod(sol[0], SIG_LGM, sol[1], alpha=ALPHA)
    return sol, ok


def main():
    t0 = time.time()
    zs, kd = SF.zs, SF.kd
    k_all = np.unique(np.concatenate([kd, KW]))
    lin = H.Linear(list(zs))
    tab = dict(zs=zs, kd=kd, k_all=k_all, a_grid=A_GRID)
    nsn = len(zs)
    SdI = np.zeros((nsn, A_GRID.size, k_all.size))
    SII = np.zeros_like(SdI)
    hod = np.zeros((nsn, 7))       # lgMmin, lgM1, ok, n_g, b_g, f_rc, f_sat
    for s, z in enumerate(zs):
        hm = H.Halo(lin, float(z), k_all, nM=140, nx=1600)
        ngt = NGAL[float(z)]*1e6/BOX**3
        sel = (kd >= 0.04) & (kd <= 0.1)
        bgt = float(np.mean(SF.b[s][sel]))
        sol, ok = calibrate_hod(hm, ngt, bgt)
        hod[s] = [sol[0], sol[1], ok, hm.ng, hm.bg, hm.f_rc, hm.f_sat]
        a, b = hm.S_grid(A_GRID)
        SdI[s], SII[s] = a, b
        print(f"z {z:4.2f}: n_g target {ngt:.4f} got {hm.ng:.4f}; b_g target {bgt:.3f} got {hm.bg:.3f} "
              f"(lgMmin {sol[0]:.2f}, lgM1 {sol[1]:.2f}, solved {ok}); f_rc {hm.f_rc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    tab.update(SdI=SdI, SII=SII, hod=hod)
    np.savez(os.path.join(HERE, "halo_tables.npz"), **tab)
    with open(os.path.join(HERE, "halo_hod.csv"), "w") as fh:
        fh.write("z,lgMmin,lgM1,solved,n_g,b_g,f_rc,f_sat,ng_target_inv\n")
        for s, z in enumerate(zs):
            fh.write(f"{z:g},{hod[s,0]:.3f},{hod[s,1]:.3f},{int(hod[s,2])},{hod[s,3]:.5f},{hod[s,4]:.3f},{hod[s,5]:.3f},{hod[s,6]:.3f},{BOX**3/(NGAL[float(z)]*1e6):.1f}\n")
    fit_all(tab)
    print(f"done ({time.time() - t0:.0f}s)")


def model_factory(tab, s, nowin):
    """Observed-unit prediction at snapshot s on the FLAMINGO k nodes."""
    kd, k_all = tab["kd"], tab["k_all"]
    idx = np.searchsorted(k_all, kd)
    assert np.allclose(k_all[idx], kd)
    Sd = tab["SdI"][s][:, idx]
    Si = tab["SII"][s][:, idx]
    la = np.log(np.maximum(tab["a_grid"], 1e-6))
    frc = tab["hod"][s, 5]
    D = SF.model.D(SF.zs[s])
    C1 = SF.model.cosmo["Om"]*H.C1RHOCRIT/D
    P = SF.Pnl[s]
    f1, f2 = H.f1h(kd, nowin), H.f2h(kd, nowin)

    def S_at(a):
        x = np.log(max(a, 1e-6))
        j = int(np.clip(np.searchsorted(la, x), 1, la.size - 1))
        w = np.clip((x - la[j - 1])/(la[j] - la[j - 1]), 0, 1)
        return Sd[j - 1]*(1 - w) + Sd[j]*w, Si[j - 1]*(1 - w) + Si[j]*w

    def pred(a1, a1h):
        sd, si = S_at(a1h)
        PdE = -(a1*frc*C1*P*f2 + a1h*f1*sd)
        PEE = (a1*frc*C1)**2*P*f2 + a1h**2*f1*si
        return np.array([SF.b[s]*PdE, PEE + SF.Ns[s], np.zeros_like(kd) + SF.Ns[s]])
    return pred


def fit_one(pred, s, kmax):
    m = SF.kmask(kmax)

    def resid(x):
        return ((SF.y[s] - pred(*x))/SF.err[s])[:, m].ravel()
    best = None
    for x0 in ([4.0, 1e-3], [4.0, 0.03], [4.0, 0.3], [4.0, 1.0], [2.0, 2.0]):
        r = least_squares(resid, x0, bounds=([0.0, 0.0], [20.0, 3.0]), x_scale=[1.0, 0.1])
        if best is None or r.cost < best.cost:
            best = r
    c2 = float(best.fun @ best.fun)
    dof = 3*int(m.sum()) - 2
    return best.x, c2, dof


def fit_all(tab):
    kg = SF.kg
    out = {}
    for var, nowin in (("halo", False), ("halo_nowin", True)):
        X = np.zeros((SF.nsn, kg.size, 2))
        C = np.zeros((SF.nsn, kg.size))
        NU = np.zeros_like(C)
        for s in range(SF.nsn):
            pred = model_factory(tab, s, nowin)
            for j, k in enumerate(kg):
                X[s, j], C[s, j], NU[s, j] = fit_one(pred, s, k)
        P = chi2_dist.sf(C, NU)
        cuts = np.array([SF.cut_rule(P[s] >= 0.05) for s in range(SF.nsn)])
        core = SF.core
        nuc = NU[core].sum(0)
        pc = chi2_dist.sf(C[core].sum(0), nuc)
        out[var] = dict(x=X, chi2=C, dof=NU, p=P, cut=cuts, comb_p=pc, comb_cut=SF.cut_rule(pc >= 0.05))
        print(var, "cuts (core):", np.round(cuts[core], 3), " combined p:", np.array2string(pc[:7], precision=3), "combined cut", out[var]["comb_cut"], flush=True)
    np.savez(os.path.join(HERE, "halo_sim.npz"), kgrid=kg, zs=SF.zs, core=SF.core,
             **{f"{v}_{k}": out[v][k] for v in out for k in out[v]})
    with open(os.path.join(HERE, "halo_cuts.csv"), "w") as fh:
        fh.write("z,NLA_GoF,TATT_GoF,halo_GoF,halo_nowin_GoF\n")
        for s in np.where(SF.core)[0]:
            fh.write(f"{SF.zs[s]:g},{SF.v3['cut_NLA_GoF'][s]:.3f},{SF.v3['cut_TATT_GoF'][s]:.3f},{out['halo']['cut'][s]:.3f},{out['halo_nowin']['cut'][s]:.3f}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit_all(dict(np.load(os.path.join(HERE, "halo_tables.npz"))))
    else:
        main()
