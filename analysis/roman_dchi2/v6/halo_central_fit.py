#!/usr/bin/env python3
"""FLAMINGO fits of the halo model with a central 1-halo term (halo_central.py).

Same data, probes, k >= 0.05, N_s, FLAMINGO P_NL and b(k), k_max grid and
bookkeeping as f21_variants.py (outputs f21v_<name>.npz in the same format).

  HM        reference (= f21_variants "hyb"): TATT(a1, a2, b_TA) on P_NL x f_2h
            + F21 satellites (a1h)
  HMc_add   HM + central 1-halo GI, b(k) * (-a_c Pc_m); P_NL kept in the
            2-halo (double counts the central-own-halo matter; a_c of either
            sign, diagnostic)
  HMc       central 1-halo consistent: the NLA part of the GI 2-halo on
            I1m(k) P_lin (centrals x matter of other halos), + b(k)(-a_c Pc_m)
            (centrals x own triaxial halo); EE 2-halo NLA part on P_NL
  HMc_lin   HMc with the EE NLA part on P_lin too (strict 2-halo)
  HMc_g     HMc with the galaxy-position version: hE += -a_c Pc_g (central x
            satellites tracing the elliptical halo), not multiplied by b(k)
  HMc_flat  HMc with sigma_E^2 independent of mass
Parameters a1, a2, b_TA, a1h[, a_c]; nu = 3 N_k - n_par.
"""
import os
import sys
import time

import numpy as np
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = [_argv[0], "_import_hc"]
import f21_variants as FV    # noqa: E402
sys.argv = _argv
try:
    os.remove(os.path.join(HERE, "f21_variants_log__import_hc.txt"))
except OSError:
    pass
SF = FV.SF
KD, KG, NSN = SF.kd, SF.kg, SF.nsn
CT = dict(np.load(os.path.join(HERE, "halo_central_tables.npz")))
IDX = np.searchsorted(CT["k_all"], KD)
assert np.allclose(CT["k_all"][IDX], KD)
LOG = open(os.path.join(HERE, f"halo_central_fit_log_{sys.argv[1] if len(sys.argv) > 1 else 'all'}.txt"), "w")


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


GR = dict(np.load(os.path.join(HERE, "halo_central_grid.npz")))
SC = dict(np.load(os.path.join(HERE, "halo_satcoh_tables.npz"))) if os.path.exists(os.path.join(HERE, "halo_satcoh_tables.npz")) else None


def m_hmcs(s, coh_gi=True, radial=True):
    """HMc_lin + satellites coherently aligned with the host (halo_satcoh.py):
    EE = BB += a_c a_s Pcs + a_s^2 Pss, P_dE -= a_s Psm. Parameters a1, a2,
    b_TA, a1h, a_c, a_s (a1h fixed to 0 if not radial)."""
    pred0, (lo, hi), starts0 = m_hmc(s, "HMc_lin")
    Pcs, Pss, Psm = (SC[n][s][IDX] for n in ("Pcs", "Pss", "Psm"))
    bk = SF.b[s]

    def pred(x):
        x5 = np.array(x[:5])
        if not radial:
            x5[3] = 0.0
        ac, as_ = x[4], x[5]
        p = pred0(x5)
        coh = ac*as_*Pcs + as_**2*Pss
        return p + np.array([-bk*as_*Psm if coh_gi else 0*KD, coh, coh])
    lo, hi = lo + [0.0], hi + [2.0]
    starts = [st + [a] for st in starts0[:3] for a in (1e-3, 0.02, 0.1)]
    return pred, (lo, hi), starts


def m_hmc(s, kind="HMc", ip=None, ib=None):
    Sd, Si = FV.sat_tables(s)
    f1, f2 = FV.win1(4.0), FV.win2(6.0)
    K = SF.KER[s]
    PL = CT["Plin"][s][IDX]
    I1 = CT["I1m"][s][IDX]
    Pc = {"HMc_g": CT["Pc_g"], "HMc_flat": CT["Pc_mflat"]}.get(kind, CT["Pc_m"])[s][IDX]
    if ip is not None:
        Pc = GR["Pc"][s, ip, ib]
    P_gi = SF.Pnl[s] if kind == "HMc_add" else I1*PL
    P_ii = PL if kind.startswith("HMc_lin") else SF.Pnl[s]
    bk = SF.b[s]

    def pred(x):
        a1, a2, bt, a1h, ac = x
        pg = SF.model.predict(SF.zs[s], KD, a1, a2, bt, 1.0, K=K, PNL=P_gi, Ns=0.0, bk=bk)
        pi = pg if P_ii is P_gi else SF.model.predict(SF.zs[s], KD, a1, a2, bt, 1.0, K=K, PNL=P_ii, Ns=0.0, bk=bk)
        sd, si = FV.interp_a(Sd, a1h), FV.interp_a(Si, a1h)
        PdE = pg["dE"]*f2 - a1h*f1*sd
        hE = bk*PdE - ac*Pc if kind == "HMc_g" else bk*(PdE - ac*Pc)
        return np.array([hE, pi["EE"]*f2 + a1h**2*f1*si + SF.Ns[s], pi["BB"]*f2 + SF.Ns[s]])
    alo = -300.0 if kind == "HMc_add" else 0.0
    lo, hi = [0.0, -5.0, 0.0, 0.0, alo], [20.0, 5.0, 2.0, 3.0, 300.0]
    starts = [[4.0, -1.0, 0.1, 1e-3, 0.0], [4.0, -1.0, 0.3, 0.02, 10.0], [3.0, -1.5, 0.3, 1e-3, 40.0],
              [4.0, 0.0, 0.5, 0.05, 100.0]] + ([[4.0, -1.5, 0.1, 1e-3, -20.0]] if kind == "HMc_add" else [])
    return pred, (lo, hi), starts


VARIANTS = {k: (lambda s, k=k: [m_hmc(s, k)], 5) for k in ("HMc_add", "HMc", "HMc_lin", "HMc_g", "HMc_flat")}
IP0, IB0 = int(np.where(GR["ps"] == 0)[0][0]), int(np.where(GR["betas"] == 0)[0][0])
def m_hmcw(s, kind="HMc_lin"):
    """Control: HMc (or TATT) + a free white EE = BB constant W (N_s shift)."""
    if kind == "TATT":
        def pred0(x):
            return SF.pred(s, x[0], x[1], x[2])
        lo, hi = [0.0, -5.0, 0.0], [20.0, 5.0, 2.0]
        starts0 = [[4.0, 0.0, 0.1], [4.0, -1.0, 0.5], [4.0, 1.0, 0.5]]
        n0 = 3
    else:
        pred0, (lo, hi), starts0 = m_hmc(s, kind)
        n0 = 5

    def pred(x):
        p = pred0(x[:n0])
        return p + np.array([0*KD, x[n0] + 0*KD, x[n0] + 0*KD])
    W0 = SF.Ns[s]
    return pred, (lo + [-W0], hi + [5*W0]), [st + [w] for st in starts0[:3] for w in (0.0, 0.5*W0)]


GB = dict(np.load(os.path.join(HERE, "halo_gbias_tables.npz"))) if os.path.exists(os.path.join(HERE, "halo_gbias_tables.npz")) else None


def m_nob(s, kind="HMc"):
    """No measured b(k): the galaxy-shape cross from the halo model alone.
    HMc: P_hE = I_g(k) f_2h P_dE^2h(P_lin) - a_c Pc_g (galaxy-position central
         term), EE/BB as HMc_lin, + W.  Parameters a1, a2, bTA, a_c, W.
    TATT: P_hE = b_g P_dE(P_NL) (constant large-scale bias), + W."""
    K = SF.KER[s]
    PL = CT["Plin"][s][IDX]
    f2 = FV.win2(6.0)
    Ig = GB["Ig"][s][IDX]
    bg = float(Ig[0])
    Pcg = CT["Pc_g"][s][IDX]
    W0 = SF.Ns[s]
    if kind == "TATT":
        def pred(x):
            a1, a2, bt, W = x
            p = SF.model.predict(SF.zs[s], KD, a1, a2, bt, 1.0, K=K, PNL=SF.Pnl[s], Ns=0.0, bk=SF.b[s])
            return np.array([bg*p["dE"], p["EE"] + SF.Ns[s] + W, p["BB"] + SF.Ns[s] + W])
        return pred, ([0.0, -5.0, 0.0, -W0], [20.0, 5.0, 2.0, 5*W0]), [[4.0, 0.0, 0.1, 0.0], [4.0, -1.0, 0.5, 0.5*W0], [4.0, 1.0, 0.5, 0.5*W0]]

    def pred(x):
        a1, a2, bt, ac, W = x
        p = SF.model.predict(SF.zs[s], KD, a1, a2, bt, 1.0, K=K, PNL=PL, Ns=0.0, bk=SF.b[s])
        return np.array([Ig*p["dE"]*f2 - ac*Pcg, p["EE"]*f2 + SF.Ns[s] + W, p["BB"]*f2 + SF.Ns[s] + W])
    return pred, ([0.0, -5.0, 0.0, 0.0, -W0], [20.0, 5.0, 2.0, 300.0, 5*W0]), \
        [[4.0, -1.0, 0.3, a, w] for a in (1.0, 10.0, 40.0) for w in (0.0, 0.5*W0)]


VARIANTS["HMc_nob"] = (lambda s: [m_nob(s)], 5)
VARIANTS["TATT_nob"] = (lambda s: [m_nob(s, "TATT")], 4)
VARIANTS["HMc_W"] = (lambda s: [m_hmcw(s)], 6)
VARIANTS["TATT_W"] = (lambda s: [m_hmcw(s, "TATT")], 4)
VARIANTS["HMcs"] = (lambda s: [m_hmcs(s)], 6)
VARIANTS["HMcs_noGI"] = (lambda s: [m_hmcs(s, coh_gi=False)], 6)
VARIANTS["HMcs_norad"] = (lambda s: [m_hmcs(s, radial=False)], 5)
VARIANTS["HMc_lin_p"] = (lambda s: [m_hmc(s, "HMc_lin", ip, IB0) for ip in range(GR["ps"].size)], 6)
VARIANTS["HMc_lin_b"] = (lambda s: [m_hmc(s, "HMc_lin", IP0, ib) for ib in range(GR["betas"].size)], 6)
VARIANTS["HMc_lin_pb"] = (lambda s: [m_hmc(s, "HMc_lin", ip, ib) for ip in range(GR["ps"].size)
                                     for ib in range(GR["betas"].size)], 7)


def main():
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    for name in names:
        fac, npar = VARIANTS[name]
        C = np.zeros((NSN, KG.size))
        CP = np.zeros((NSN, KG.size, 3))
        NK = np.zeros(KG.size, int)
        X = np.full((NSN, KG.size, 10), np.nan)
        Gi = np.zeros((NSN, KG.size), int)
        for s in range(NSN):
            models = fac(s)
            prev = [None]*len(models)
            for j, kmax in enumerate(KG):
                best = None
                for ig, (pred, bnd, starts) in enumerate(models):
                    x, c2, cp, nk = FV.fit(pred, bnd, starts, s, kmax, prev[ig])
                    prev[ig] = x
                    if best is None or c2 < best[1]:
                        best = (x, c2, cp, nk, ig)
                x, c2, cp, nk, ig = best
                C[s, j], CP[s, j], NK[j], Gi[s, j] = c2, cp, nk, ig
                X[s, j, :len(x)] = x
        nu = np.broadcast_to(3*NK[None, :] - npar, (NSN, KG.size)).astype(float)
        P = np.where(nu > 0, chi2_dist.sf(C, np.maximum(nu, 1)), 1.0)
        cuts = np.array([SF.cut_rule(P[s] >= 0.05) for s in range(NSN)])
        core = SF.core
        res = {f"{name}_chi2": C, f"{name}_chi2p": CP, f"{name}_nu": nu, f"{name}_x": X, f"{name}_p": P,
               f"{name}_cut": cuts, f"{name}_npar": npar, f"{name}_grid": Gi, "NK": NK, "kgrid": KG, "zs": SF.zs, "core": core}
        np.savez(os.path.join(HERE, f"f21v_{name}.npz"), **res)
        js = [int(np.argmin(np.abs(KG - kk))) for kk in (0.16, 0.29, 0.52, 1.0, 3.04)]
        say(f"{name:9s} (n_par {npar}) cuts core {np.round(cuts[core], 2)}  [{time.time() - t0:.0f}s]")
        for s in np.where(core)[0]:
            say(f"   z {SF.zs[s]:4.2f} chi2/nu @0.16,0.29,0.52,1,3: " + " ".join(f"{C[s, j]/nu[s, j]:7.2f}" for j in js) +
                "  | share hE/EE/BB @0.52: " + "/".join(f"{CP[s, js[2], i]/max(C[s, js[2]], 1e-9):.2f}" for i in range(3)) +
                "  | x@0.52 " + np.array2string(X[s, js[2], :6], precision=3) +
                "  x@3 " + np.array2string(X[s, js[4], :6], precision=3) + f"  grid@0.52,1,3 {Gi[s, js[2]]},{Gi[s, js[3]]},{Gi[s, js[4]]}")


if __name__ == "__main__":
    main()
