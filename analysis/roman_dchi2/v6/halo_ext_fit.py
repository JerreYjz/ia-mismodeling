#!/usr/bin/env python3
"""Fits of the extended strict halo model (halo_ext.py) to FLAMINGO.

For every snapshot, k_max and grid point (alpha, f_c, b): least squares over
(a1, a1h), as in halo_strict.py. Two summaries:
  * per-snapshot best grid point ("ext, free per z"): nu = 3 N_k - 5;
  * one grid point shared by all core snapshots ("ext, global"), chosen by the
    core-summed chi2 at K_SEL; nu = 3 N_k - 2 per snapshot.
Writes halo_ext_sim.npz, halo_ext_cuts.csv, halo_ext_strict_tables.npz (the
global grid point in the halo_strict_tables format, for the Roman scan).
"""
import glob
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
import halo_ia as H          # noqa: E402
import simfit as SF          # noqa: E402

K_SEL = float(os.environ.get("K_SEL", "1.0"))


def load():
    parts = sorted(glob.glob(os.path.join(HERE, "halo_ext_tables_*.npz")))
    P = [dict(np.load(p)) for p in parts]
    t = {k: P[0][k] for k in ("zs", "kd", "k_all", "a_grid", "alphas", "fcs", "bs")}
    order = np.concatenate([p["snaps"] for p in P])
    for k in ("SdI", "SII", "Q", "I1m", "Plin", "hod"):
        arr = np.concatenate([p[k] for p in P])
        out = np.zeros((len(t["zs"]),) + arr.shape[1:])
        out[order] = arr
        t[k] = out
    t["have"] = np.isin(np.arange(len(t["zs"])), order)
    return t


def factory(t, s, ial, ifc, ib):
    kd, k_all = t["kd"], t["k_all"]
    idx = np.searchsorted(k_all, kd)
    Sd, Si, Qs = (t[K][s, ial, ifc, ib][:, idx] for K in ("SdI", "SII", "Q"))
    I1, PL = t["I1m"][s][idx], t["Plin"][s][idx]
    la = np.log(np.maximum(t["a_grid"], 1e-6))
    frc = t["hod"][s, ial, 5]
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


def fit_one(pred, s, kmax):
    m = SF.kmask(kmax)

    def resid(x):
        return ((SF.y[s] - pred(*x))/SF.err[s])[:, m].ravel()
    best = None
    for x0 in ([5.0, 1e-3], [5.0, 0.05], [5.0, 0.5]):
        r = least_squares(resid, x0, bounds=([0.0, 0.0], [20.0, 3.0]), x_scale=[1.0, 0.1])
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(best.fun @ best.fun), int(m.sum())


def main():
    t0 = time.time()
    t = load()
    assert t["have"].all(), "tables incomplete"
    kg, core = SF.kg, SF.core
    nal, nfc, nb = len(t["alphas"]), len(t["fcs"]), len(t["bs"])
    X = np.zeros((SF.nsn, nal, nfc, nb, kg.size, 2))
    C = np.full((SF.nsn, nal, nfc, nb, kg.size), np.inf)
    NK = np.zeros(kg.size, int)
    for s in range(SF.nsn):
        for ial in range(nal):
            for ifc in range(nfc):
                for ib in range(nb):
                    pred = factory(t, s, ial, ifc, ib)
                    for j, k in enumerate(kg):
                        X[s, ial, ifc, ib, j], C[s, ial, ifc, ib, j], NK[j] = fit_one(pred, s, k)
        print(f"z {SF.zs[s]:4.2f} done ({time.time() - t0:.0f}s)", flush=True)
    # per-snapshot best grid point
    flat = C.reshape(SF.nsn, -1, kg.size)
    ibest = flat.argmin(1)
    Cfree = flat.min(1)
    nu_free = 3*NK[None, :] - 5
    Pfree = chi2_dist.sf(Cfree, nu_free)
    cut_free = np.array([SF.cut_rule(Pfree[s] >= 0.05) for s in range(SF.nsn)])
    # global grid point (core-summed chi2 at K_SEL)
    jsel = int(np.argmin(np.abs(kg - K_SEL)))
    tot = C[core].sum(0)[..., jsel]
    g = np.unravel_index(int(np.argmin(tot)), tot.shape)
    Cglob = C[:, g[0], g[1], g[2], :]
    nu_g = 3*NK[None, :] - 2
    Pglob = chi2_dist.sf(Cglob, nu_g)
    cut_glob = np.array([SF.cut_rule(Pglob[s] >= 0.05) for s in range(SF.nsn)])
    # reference: strict default grid point alpha = 1, f_c = 1, b = -2
    i0 = (list(t["alphas"]).index(1.0), list(t["fcs"]).index(1.0), list(t["bs"]).index(-2.0))
    C0 = C[:, i0[0], i0[1], i0[2], :]
    gl = dict(alpha=float(t["alphas"][g[0]]), fc=float(t["fcs"][g[1]]), b=float(t["bs"][g[2]]))
    print("global grid point (core chi2 at k <= %.2f):" % kg[jsel], gl)
    np.savez(os.path.join(HERE, "halo_ext_sim.npz"), kgrid=kg, zs=SF.zs, core=core, X=X, C=C, NK=NK, ibest=ibest,
             Cfree=Cfree, Pfree=Pfree, cut_free=cut_free, Cglob=Cglob, Pglob=Pglob, cut_glob=cut_glob, glob=np.array(g),
             C0=C0, k_sel=kg[jsel], alphas=t["alphas"], fcs=t["fcs"], bs=t["bs"])
    shp = (nal, nfc, nb)
    hs = np.load(os.path.join(HERE, "halo_sim.npz"))
    with open(os.path.join(HERE, "halo_ext_cuts.csv"), "w") as fh:
        fh.write("z,NLA,TATT,F21,strict,ext_global,ext_free\n")
        for s in np.where(core)[0]:
            fh.write(f"{SF.zs[s]:g},{SF.v3['cut_NLA_GoF'][s]:.3f},{SF.v3['cut_TATT_GoF'][s]:.3f},{hs['halo_cut'][s]:.3f},"
                     f"{np.load(os.path.join(HERE, 'halo_strict_sim.npz'))['strict_cut'][s]:.3f},{cut_glob[s]:.3f},{cut_free[s]:.3f}\n")
    print("cuts ext global:", np.round(cut_glob[core], 3), " ext free:", np.round(cut_free[core], 3))
    js = [int(np.argmin(np.abs(kg - kk))) for kk in (0.16, 0.29, 0.52, 1.0, 3.04)]
    for s in np.where(core)[0]:
        bi = [np.unravel_index(ibest[s, j], shp) for j in js]
        print(f"z {SF.zs[s]:4.2f} chi2/nu @0.16,0.29,0.52,1,3  strict " + " ".join(f"{C0[s, j]/(3*NK[j] - 2):6.2f}" for j in js) +
              " | global " + " ".join(f"{Cglob[s, j]/(3*NK[j] - 2):6.2f}" for j in js) +
              " | free " + " ".join(f"{Cfree[s, j]/(3*NK[j] - 5):6.2f}" for j in js) +
              " | NLA " + " ".join(f"{SF.v3['NLA_chi2'][s, j]/SF.v3['NLA_dof'][s, j]:6.2f}" for j in js) +
              " | free (alpha,fc,b) @0.52,1: " + ", ".join(f"({t['alphas'][b[0]]:g},{t['fcs'][b[1]]:g},{t['bs'][b[2]]:g})" for b in bi[2:4]) +
              f" a1h(global,@1) {X[s, g[0], g[1], g[2], js[3], 1]:.3f}", flush=True)
    # strict-format tables for the global point
    hod = t["hod"][:, g[0], :]
    np.savez(os.path.join(HERE, "halo_ext_strict_tables.npz"), zs=t["zs"], kd=t["kd"], k_all=t["k_all"], a_grid=t["a_grid"],
             SdI=t["SdI"][:, g[0], g[1], g[2]], SII=t["SII"][:, g[0], g[1], g[2]], Q=t["Q"][:, g[0], g[1], g[2]],
             I1m=t["I1m"], Plin=t["Plin"], hod=hod, glob=np.array([gl["alpha"], gl["fc"], gl["b"]]))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
