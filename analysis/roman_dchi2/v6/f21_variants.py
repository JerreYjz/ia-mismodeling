#!/usr/bin/env python3
"""Fortuna et al. 2021 halo model and variants against the FLAMINGO spectra.

Same data, probes (hE, EE, BB), k >= 0.05, N_s fixed to the B plateau and
FLAMINGO P_NL, b(k) in the model as all other fits (v3/v4/v6).

Variants (per snapshot, every k_max of the grid):
  NLA, TATT     references (refitted here for the probe split)
  F21           Fortuna+21 as published: 2-halo NLA on P_NL with f_2h,
                radial satellites, b = -2, core 0.06, cap 0.3, f_1h window
                (k_1h = 4, k_2h = 6); one population (v6 halo_tables)
  F21_rb        + red/blue centrals as in F21 Eqs. 3-4: the II 2-halo carries
                (f_r A_r)^2 + (f_b A_b)^2 without the red-blue cross term, so
                the II amplitude decouples from the GI one:
                P_EE^2h = r g^2 C1^2 P_NL f_2h,  r = q/g^2 >= 1/2 (any f_r, A_b)
  F21_ext       + satellite slope b, c_sat/c_DM, HOD alpha (grid, profiled)
  F21_win       + window scales k_1h in {0.5,1,2,4,none}, k_2h in {3,6,none}
  F21_all       ext (reduced grid) x windows x red/blue r
  hyb0          blue centrals with tidal torquing (TATT A2 term, b_TA = 0),
                red centrals NLA, red satellites F21 1-halo:
                P = TATT(a1, a2, 0) f_2h + satellites
  hyb           hyb0 with b_TA free (density weighting)
  hyb_ext       hyb with the satellite grid (reduced) and windows
nu = 3 N_k - n_par, discrete grid dimensions counted as parameters.
Outputs f21_variants.npz (chi2, chi2 per probe, parameters, cuts) and a log.
"""
import os
import sys
import time
import itertools

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "v4"))
import halo_ia as H          # noqa: E402
import simfit as SF          # noqa: E402
import halo_ext_fit as XF    # noqa: E402

KD = SF.kd
NSN, KG = SF.nsn, SF.kg
LOG = open(os.path.join(HERE, f"f21_variants_log_{sys.argv[1] if len(sys.argv) > 1 else 'all'}.txt"), "w")


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


V6 = dict(np.load(os.path.join(HERE, "halo_tables.npz")))
# the extended satellite grids (halo_ext_tables_*.npz, 250 MB) are only needed
# by the F21_ext / F21_all / hyb_ext variants; everything else runs without them
EXT = XF.load() if XF.glob.glob(os.path.join(HERE, "halo_ext_tables_*.npz")) else None
IDX6 = np.searchsorted(V6["k_all"], KD)
IDXX = np.searchsorted(EXT["k_all"], KD) if EXT is not None else None
LA = np.log(np.maximum(V6["a_grid"], 1e-6))


def interp_a(T, a):
    x = np.log(max(a, 1e-6))
    j = int(np.clip(np.searchsorted(LA, x), 1, LA.size - 1))
    w = np.clip((x - LA[j - 1])/(LA[j] - LA[j - 1]), 0, 1)
    return T[j - 1]*(1 - w) + T[j]*w


def win1(k1):
    return np.ones_like(KD) if k1 is None else -np.expm1(-(KD/k1)**2)


def win2(k2):
    return np.ones_like(KD) if k2 is None else np.exp(-(KD/k2)**2)


def sat_tables(s, grid=None):
    """(S_dI, S_II) [n_a, nk] on the data k: v6 tables (b = -2, f_c = 1,
    alpha = 1) or an ext grid point (ial, ifc, ib)."""
    if grid is None:
        return V6["SdI"][s][:, IDX6], V6["SII"][s][:, IDX6]
    ial, ifc, ib = grid
    return EXT["SdI"][s, ial, ifc, ib][:, IDXX], EXT["SII"][s, ial, ifc, ib][:, IDXX]


def C1P(s):
    return SF.model.cosmo["Om"]*H.C1RHOCRIT/SF.model.D(SF.zs[s])*SF.Pnl[s]


def out3(s, PdE, PEE, PBB):
    return np.array([SF.b[s]*PdE, PEE + SF.Ns[s], PBB + SF.Ns[s]])


def m_f21(s, grid=None, k1=4.0, k2=6.0, rb=False):
    """Parameters: a (= a1 f_rc, the NLA-like amplitude), a1h[, r]."""
    Sd, Si = sat_tables(s, grid)
    cp = C1P(s)
    f1, f2 = win1(k1), win2(k2)

    def pred(x):
        a, a1h = x[0], x[1]
        r = x[2] if rb else 1.0
        sd, si = interp_a(Sd, a1h), interp_a(Si, a1h)
        PdE = -(a*cp*f2 + a1h*f1*sd)
        PEE = r*a**2*cp*SF.model.cosmo["Om"]*H.C1RHOCRIT/SF.model.D(SF.zs[s])*f2 + a1h**2*f1*si
        return out3(s, PdE, PEE, 0.0*KD)
    lo, hi = [0.0, 0.0] + ([0.5] if rb else []), [20.0, 3.0] + ([20.0] if rb else [])
    starts = [[4.0, 1e-3] + ([1.0] if rb else []), [4.0, 0.05] + ([1.5] if rb else []), [4.0, 0.5] + ([1.0] if rb else [])]
    return pred, (lo, hi), starts


def m_hm2h(s, a2=True):
    """Halo model with the 2-halo physics completed: red centrals NLA, blue
    centrals tidal torquing (A2), density weighting of the observed shape
    field with b_TA = b_g (the HOD/FLAMINGO galaxy bias, not free), F21 red
    satellites. Parameters: a1, [a2], a1h."""
    Sd, Si = sat_tables(s)
    f1, f2 = win1(4.0), win2(6.0)
    K = SF.KER[s]
    bg = float(V6["hod"][s, 4])

    def pred(x):
        a1 = x[0]
        aa2 = x[1] if a2 else 0.0
        a1h = x[-1]
        pr = SF.model.predict(SF.zs[s], KD, a1, aa2, bg, 1.0, K=K, PNL=SF.Pnl[s], Ns=0.0, bk=SF.b[s])
        sd, si = interp_a(Sd, a1h), interp_a(Si, a1h)
        return out3(s, pr["dE"]*f2 - a1h*f1*sd, pr["EE"]*f2 + a1h**2*f1*si, pr["BB"]*f2)
    if a2:
        return pred, ([0.0, -5.0, 0.0], [20.0, 5.0, 3.0]), [[4.0, 0.0, 1e-3], [4.0, -1.0, 0.02], [3.0, -2.0, 0.05], [4.0, 1.0, 0.05]]
    return pred, ([0.0, 0.0], [20.0, 3.0]), [[4.0, 1e-3], [3.0, 0.05]]


def m_hyb(s, grid=None, k1=4.0, k2=6.0, bta=True):
    """Parameters: a1, a2[, bTA], a1h. TATT on P_NL (FLAMINGO), F21 satellites."""
    Sd, Si = sat_tables(s, grid)
    f1, f2 = win1(k1), win2(k2)
    K = SF.KER[s]

    def pred(x):
        a1, a2 = x[0], x[1]
        bt = x[2] if bta else 0.0
        a1h = x[-1]
        pr = SF.model.predict(SF.zs[s], KD, a1, a2, bt, 1.0, K=K, PNL=SF.Pnl[s], Ns=0.0, bk=SF.b[s])
        sd, si = interp_a(Sd, a1h), interp_a(Si, a1h)
        PdE = pr["dE"]*f2 - a1h*f1*sd
        PEE = pr["EE"]*f2 + a1h**2*f1*si
        PBB = pr["BB"]*f2
        return out3(s, PdE, PEE, PBB)
    if bta:
        lo, hi = [0.0, -5.0, 0.0, 0.0], [20.0, 5.0, 2.0, 3.0]
        starts = [[4.0, 0.0, 0.1, 1e-3], [4.0, -1.0, 0.5, 0.02], [4.0, -1.0, 0.3, 0.3], [4.0, 1.0, 0.5, 0.05]]
    else:
        lo, hi = [0.0, -5.0, 0.0], [20.0, 5.0, 3.0]
        starts = [[4.0, 0.0, 1e-3], [4.0, -1.0, 0.02], [4.0, -1.0, 0.3], [4.0, 1.0, 0.05]]
    return pred, (lo, hi), starts


def m_ref(s, mdl):
    def pred(x):
        if mdl == "NLA":
            return SF.pred(s, x[0])
        return SF.pred(s, x[0], x[1], x[2])
    if mdl == "NLA":
        return pred, ([0.0], [20.0]), [[4.0]]
    return pred, ([0.0, -5.0, 0.0], [20.0, 5.0, 2.0]), [[4.0, 0.0, 0.1], [4.0, -1.0, 0.5], [4.0, 1.0, 0.5], [4.0, 0.0, 1.0]]


def fit(pred, bnd, starts, s, kmax, x_prev=None):
    m = SF.kmask(kmax)

    def resid(x):
        return ((SF.y[s] - pred(x))/SF.err[s])[:, m].ravel()
    best = None
    st = list(starts) + ([list(x_prev)] if x_prev is not None else [])
    for x0 in st:
        x0 = np.clip(x0, bnd[0], bnd[1])
        r = least_squares(resid, x0, bounds=bnd, x_scale="jac", max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    rr = ((SF.y[s] - pred(best.x))/SF.err[s])[:, m]
    return best.x, float((rr**2).sum()), (rr**2).sum(1), int(m.sum())


if EXT is not None:
    ALI = list(EXT["alphas"])
    FCI = list(EXT["fcs"])
    BSI = list(EXT["bs"])
    FULL_GRID = list(itertools.product(range(len(ALI)), range(len(FCI)), range(len(BSI))))
    RED_GRID = [(ALI.index(a), FCI.index(f), BSI.index(b)) for a in (1.0, 1.5) for f in (0.25, 1.0) for b in (-2.0, 0.0)]
else:
    FULL_GRID = RED_GRID = []
WINS = [(k1, k2) for k1 in (0.5, 1.0, 2.0, 4.0, None) for k2 in (3.0, 6.0, None)]

VARIANTS = {
    # name: (list of factories (one per discrete grid point), n_par)
    "NLA": (lambda s: [m_ref(s, "NLA")], 1),
    "TATT": (lambda s: [m_ref(s, "TATT")], 3),
    "F21": (lambda s: [m_f21(s)], 2),
    "F21_rb": (lambda s: [m_f21(s, rb=True)], 3),
    "F21_ext": (lambda s: [m_f21(s, grid=g) for g in FULL_GRID], 5),
    "F21_win": (lambda s: [m_f21(s, k1=w[0], k2=w[1]) for w in WINS], 4),
    "F21_all": (lambda s: [m_f21(s, grid=g, k1=w[0], k2=w[1], rb=True) for g in RED_GRID for w in WINS], 8),
    "hm2h_dw": (lambda s: [m_hm2h(s, a2=False)], 2),
    "hm2h": (lambda s: [m_hm2h(s)], 3),
    "hyb0": (lambda s: [m_hyb(s, bta=False)], 3),
    "hyb": (lambda s: [m_hyb(s)], 4),
    "hyb_ext": (lambda s: [m_hyb(s, grid=g, k1=w[0], k2=w[1]) for g in RED_GRID for w in WINS], 9),
}


def main():
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    for name in names:
        path = os.path.join(HERE, f"f21v_{name}.npz")
        res = {}
        fac, npar = VARIANTS[name]
        C = np.zeros((NSN, KG.size))
        CP = np.zeros((NSN, KG.size, 3))
        NK = np.zeros(KG.size, int)
        X = np.full((NSN, KG.size, 10), np.nan)
        G = np.zeros((NSN, KG.size), int)
        for s in range(NSN):
            models = fac(s)
            prev = [None]*len(models)
            for j, kmax in enumerate(KG):
                best = None
                for ig, (pred, bnd, starts) in enumerate(models):
                    x, c2, cp, nk = fit(pred, bnd, starts, s, kmax, prev[ig])
                    prev[ig] = x
                    if best is None or c2 < best[1]:
                        best = (x, c2, cp, nk, ig)
                x, c2, cp, nk, ig = best
                C[s, j], CP[s, j], NK[j], G[s, j] = c2, cp, nk, ig
                X[s, j, :len(x)] = x
        nu = np.broadcast_to(3*NK[None, :] - npar, (NSN, KG.size)).astype(float)
        P = chi2_dist.sf(C, nu)
        cuts = np.array([SF.cut_rule(P[s] >= 0.05) for s in range(NSN)])
        core = SF.core
        pc = chi2_dist.sf(C[core].sum(0), nu[core].sum(0))
        res.update({f"{name}_chi2": C, f"{name}_chi2p": CP, f"{name}_nu": nu, f"{name}_x": X, f"{name}_grid": G,
                    f"{name}_p": P, f"{name}_cut": cuts, f"{name}_combp": pc, f"{name}_npar": npar})
        res["NK"], res["kgrid"], res["zs"], res["core"] = NK, KG, SF.zs, core
        np.savez(path, **res)
        js = [int(np.argmin(np.abs(KG - kk))) for kk in (0.16, 0.29, 0.52, 1.0, 3.04)]
        say(f"{name:8s} (n_par {npar}) cuts core {np.round(cuts[core], 2)}  [{time.time() - t0:.0f}s]")
        for s in np.where(core)[0]:
            say(f"   z {SF.zs[s]:4.2f} chi2/nu @0.16,0.29,0.52,1,3: " + " ".join(f"{C[s, j]/nu[s, j]:7.2f}" for j in js) +
                "  | share hE/EE/BB @0.52: " + "/".join(f"{CP[s, js[2], i]/max(C[s, js[2]], 1e-9):.2f}" for i in range(3)) +
                "  | x@0.52 " + np.array2string(X[s, js[2], :npar - 0], precision=3) + f" grid {G[s, js[2]]}")


if __name__ == "__main__":
    main()
