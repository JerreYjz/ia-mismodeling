#!/usr/bin/env python3
"""FLAMINGO truths that keep the close-pair II term (N_s) and its B-mode partner.

flamingo/bmode_noise showed that N_s, the P_BB plateau at 5 to 10 h/Mpc that
every truth subtracts from P_EE, is not shape noise: it is the shape
correlation of close galaxy pairs (mostly central-satellite), with equal E
and B parts. So the II signal of the data is P_EE (Poisson floor only
removed) in E and P_BB in B. These truths keep it:

  NB     S1 with N_s kept and the B modes added, stitched as S1 (weight w):
         E  R_II = R_S1,II + w N_s/P_NL       (= (1 - w) C1^2 A^2 + w T_EE/P_NL)
         B  R_BB = w T_BB/P_NL                 (k >= 0.05; 0 below, as NLA)
  NBall  the same with the added terms at every k (w = 1 for them):
         E  R_S1,II + N_s/P_NL,  B  T_BB/P_NL (k >= 0.05), N_s/P_NL below
GI is that of S1 (P_hE has no shape noise). The sim_IA path carries E modes
only; a B-mode table adds C_BB with the same Limber kernel, so its xi is
(xi_+^E, -xi_-^E) of the same table run as E (xi_+- = int (C_EE +- C_BB)),
and nothing in gamma_t or w.

Written: ia_base_3x2_<truth>.npz and ia_base_<truth>.npz with xi_ia (and its
E and B parts), the vectors fisher_minimize.py reads, and bmode_truth.json:
Delta chi2 of each truth against S1 and EB on the cuts of the note.
Checks: the S1 table through this path against ia_base_3x2_S1.npz; the B
part of the old B-mode excess (P_BB - N_s, unstitched) against the python
Limber B modes of cocoa/run/xi_flam_bb.npy (validated against cosmolike's
own TATT B modes).

    python bmode_truth_3x2.py <cosmolike so dir (interface_sim)> <roman_real/data>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ia_base_3x2 as IB                                           # noqa: E402
from ia_base_3x2 import CL3, Truth3, dv_ia, B, T, NXI, NDV, BN      # noqa: E402
from driver3x2 import B1_FID                                       # noqa: E402

KMIN = 0.05


def chi2(v, cov, m):
    return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.chdir(T.TB.COCOA)
    for i, n in enumerate(BN):
        B.FID[n], B.STEP[n] = B1_FID[i], IB.BIAS_STEP
    cl = CL3(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    cl.at(B.FID)
    gg = cl.dv_native(0, A1=[0.0, 0.0], A2=[0.0, 0.0], BTA=[0.0])

    st = dict(np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz")))
    f = np.load(os.path.join(T.TB.ROOT, "v2", "fits.npz"))
    kd, Tm, PNL, Ns = f["k"][0], f["T"], f["PNL"], f["Ns_true"]
    assert np.allclose(kd, st["k"]) and np.array_equal(f["sns"], st["sns"])
    w, m = st["w"], kd >= KMIN
    ns = Ns[:, None]/PNL                                   # N_s/P_NL on the data grid
    bb = Tm[:, 2]/PNL                                      # smoothed P_BB/P_NL
    out = dict(about=__doc__.split("\n\n")[0], N_s=Ns.tolist(), zs=st["zs"].tolist(), truths={})
    print(f"w at k = 0.05: {np.round(w[:, np.argmin(abs(kd - 0.05))], 2).tolist()}")

    def tables(R, name):
        st[f"R_{name}"] = R
        return Truth3(cl, st, name)

    def xi_of(model, REE):
        """IA vector of E table REE (ratio on the wide grid) with GI of model."""
        P = model.pnl(B.FID)
        return dv_ia(cl, model, model.RdE*P, REE*P, gg)

    def b_part(model, RBB):
        """B-mode vector of the ratio table RBB (wide grid): E run, xi_- negated."""
        P = model.pnl(B.FID)
        v = dv_ia(cl, model, np.zeros_like(P), RBB*P, gg)
        assert np.max(np.abs(v[NXI:])) < 1e-12*np.max(np.abs(v[:NXI])) + 1e-30, "II leaked into gamma_t or w"
        v = v.copy()
        v[NXI//2:NXI] *= -1
        v[NXI:] = 0.0
        return v

    s1 = tables(st["R_S1"], "S1")
    xi_s1 = xi_of(s1, s1.REE)
    ref = {tr: np.load(os.path.join(HERE, f"ia_base_3x2_{tr}.npz"))["xi_ia"] for tr in ("S1", "EB")}
    out["check_S1_rel"] = float(np.max(np.abs(xi_s1 - ref["S1"]))/np.max(np.abs(ref["S1"])))
    print(f"[{time.time() - t0:.0f}s] check: S1 through this path vs ia_base_3x2_S1, max rel "
          f"{out['check_S1_rel']:.1e}")

    defs = {}
    R = st["R_S1"].copy()
    R[:, 1] = st["R_S1"][:, 1] + w*np.where(m, ns, 0.0)
    Rb = np.zeros_like(R)
    Rb[:, 1] = w*np.where(m, bb, 0.0)
    defs["NB"] = (R, Rb)
    R = st["R_S1"].copy()
    R[:, 1] = st["R_S1"][:, 1] + ns
    Rb = np.zeros_like(R)
    Rb[:, 1] = np.where(m, bb, ns)
    defs["NBall"] = (R, Rb)

    c = np.load(os.path.join(HERE, "ia_base_3x2_common.npz"))
    cov = c["cov"]
    wm = lambda wc: np.any([c[f"w|{k:g}"] for k in (0.1, 0.15, 0.2) if k <= wc + 1e-9], axis=0)
    cuts = {
        "cs 0.2": np.concatenate([c["xi|0.2"], np.zeros(NDV - NXI, bool)]),
        "cs 0.35": np.concatenate([c["xi|0.35"], np.zeros(NDV - NXI, bool)]),
        "cs 0.6": np.concatenate([c["xi|0.6"], np.zeros(NDV - NXI, bool)]),
        "3x2 xi0.2 gt0.5": np.concatenate([c["xi|0.2"], c["gt|0.5"] & c["behind"], wm(0.2)]),
        "3x2 xi0.35 gt0.5": np.concatenate([c["xi|0.35"], c["gt|0.5"] & c["behind"], wm(0.2)]),
        "all xi": np.concatenate([np.ones(NXI, bool), np.zeros(NDV - NXI, bool)]),
    }
    for name, (R, Rb) in defs.items():
        mod = tables(R, name)
        modb = tables(Rb, name + "_B")
        e = xi_of(mod, mod.REE)
        b = b_part(modb, modb.REE)
        xi = e + b
        res = dict(dchi2={})
        for cn, cm in cuts.items():
            res["dchi2"][cn] = {f"vs {tr}": chi2(xi - ref[tr], cov, cm) for tr in ("S1", "EB")}
            res["dchi2"][cn]["E part (N_s in E)"] = chi2(e - xi_s1, cov, cm)
            res["dchi2"][cn]["B part"] = chi2(b, cov, cm)
            res["dchi2"][cn]["EB vs S1"] = chi2(ref["EB"] - ref["S1"], cov, cm)
            res["dchi2"][cn]["npts"] = int(cm.sum())
        out["truths"][name] = res
        np.savez(os.path.join(HERE, f"ia_base_3x2_{name}.npz"), truth=name, xi_ia=xi, xi_E=e, xi_B=b,
                 xi0=gg + xi, about=__doc__)
        np.savez(os.path.join(HERE, f"ia_base_{name}.npz"), truth=name, xi_ia=xi[:NXI], xi_E=e[:NXI],
                 xi_B=b[:NXI], about=__doc__)
        print(f"[{time.time() - t0:.0f}s] {name}: " + "; ".join(
            f"{cn} ({d['npts']}): vs S1 {d['vs S1']:.3f}, vs EB {d['vs EB']:.3f}, E {d['E part (N_s in E)']:.3f},"
            f" B {d['B part']:.3f}, EB-S1 {d['EB vs S1']:.3f}" for cn, d in res["dchi2"].items()))

    # check: B modes of the old excess (P_BB - N_s, unstitched) against python Limber
    Rb = np.zeros_like(st["R_S1"])
    Rb[:, 1] = np.where(m, bb - ns, (bb - ns)[:, m][:, :1])
    bx = b_part(tables(Rb, "chkB"), tables(Rb, "chkB").REE)[:NXI]
    py = np.load(os.path.join(T.TB.ROOT, "cocoa", "run", "xi_flam_bb.npy"))
    cv = np.load(os.path.join(T.TB.ROOT, "cocoa", "analysis", "cov_gauss_d3a.npy"))
    R0 = np.load(os.path.join(T.TB.COCOA, "run", "cosmolike_xi_ia.npz"))
    mk = R0["masks_n1"][list(np.round(R0["K_n1"], 4)).index(0.35)].astype(bool)
    chk = dict(cosmolike=chi2(bx, cv, mk), python=chi2(py, cv, mk), diff=chi2(bx - py, cv, mk),
               corr=float(bx[mk] @ py[mk]/np.sqrt((bx[mk] @ bx[mk])*(py[mk] @ py[mk]))),
               ratio_xip=float(np.sum(bx[:NXI//2][mk[:NXI//2]])/np.sum(py[:NXI//2][mk[:NXI//2]])),
               ratio_xim=float(np.sum(bx[NXI//2:][mk[NXI//2:]])/np.sum(py[NXI//2:][mk[NXI//2:]])))
    out["check_bmodes_vs_python_0.35"] = chk
    print(f"check: B modes of P_BB - N_s on the 0.35/D mask, chi2 cosmolike {chk['cosmolike']:.3f}, python "
          f"{chk['python']:.3f}, of the difference {chk['diff']:.3f}, correlation {chk['corr']:.3f}, "
          f"sum ratio xi+ {chk['ratio_xip']:.3f}, xi- {chk['ratio_xim']:.3f}")
    json.dump(out, open(os.path.join(HERE, "bmode_truth.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
