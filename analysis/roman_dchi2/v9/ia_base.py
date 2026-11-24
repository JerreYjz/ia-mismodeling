#!/usr/bin/env python3
"""Stage 1 of the v9 IA error budget: the base of derivative columns.

The fisher_machinery note computes one derivative column per elementary knob
once, and every model, prior, cut and truth afterwards is a product of small
matrices. Here the knobs are the fractional errors of the FLAMINGO IA
spectra on one ln k tent at one snapshot:

    knob (X, s, j):  P_X(k, z_s) -> P_X(k, z_s) [1 + tent_j(k)],   only at z_s,

X = GI, II, j = 0..60 (FINE, 0.01 to 10 h/Mpc, the end tents flat beyond),
s = 0..12 (the snapshots of v3/stitched_truths.npz). Its column is the xi_+-
response, exact (xi is linear in the injected tables): one cosmolike
evaluation each, 2 x 13 x 61 = 1586 per truth. The table directory is reused
(SCRATCH); a column recomputed at the end checks that nothing is stale.

Also stored, per truth (S1, prod):
  J        dxi/d(Omega_m, A_s, A, eta) of the truth-as-model (central
           differences, tolerance_nomask.jacobian), and g_s8, the rows of
           (Omega_m, S8) in those parameters (s8_row);
  xi0      the data vector, GG + xi_IA, and xi_ia;
  drift|<cut>|<model>|<X>   the known drifts: P^model/P^truth - 1 of the NLA
           and TATT best fits of fid_fits.json (fitted with that cut) at every
           snapshot, on FINE (the convention of fid_fits.py; the analysis
           model being wrong by Delta moves the fit as the data moving by
           -Delta, and every score uses the drift squared);
  checks   the tents summed over (X, s) against xi_ia, and against J's A and
           eta columns.
A truth with a B-mode II table (CP, v3/cp_truth.py; bmode_tables.py) adds the
B part (xi_+, -xi_-) to xi_ia, J and the II knobs: a knob of II moves E and
B together (R_II = R_II_E + R_II_B, both parts stored), 3 x 13 x 61 columns.
ia_base_common.npz: the example1 covariance and the masks (none and
k_UV = 0.3/D(z)).

    python ia_base.py <cocoa so dir> <cocoa_roman_real/data> [--truths S1,prod]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(HERE, "..", "v8")
sys.path.insert(0, V8)
sys.path.insert(0, HERE)
import clean_budget as CB                                          # noqa: E402
from clean_budget import T, B, ME, KW                              # noqa: E402
import ia_fisher as IF                                             # noqa: E402
import bmode_tables as BT                                          # noqa: E402

SCRATCH = "v9_scratch"
CUTS = {"none": "none", "K0/D 0.3": "K03"}                         # mask name -> file label
MODELS = ("NLA", "TATT")
FREE_T, IAP = ["omegam", "As_1e9"], ["A", "eta"]


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    truths = CB.TRUTHS
    if "--truths" in sys.argv:
        truths = tuple(sys.argv[sys.argv.index("--truths") + 1].split(","))
    assert np.allclose(CB.FINE, IF.FINE)
    t0 = time.time()
    fid_fits = json.load(open(os.path.join(V8, "fid_fits.json")))
    os.chdir(T.TB.COCOA)
    logf = open(os.path.join(HERE, "ia_base_log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])                       # before any sim table
    T.GG["fid"] = gg
    st = BT.load_truths(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    cov = np.load("analysis/cov_example1_xi.npy")
    R0 = np.load("run/cosmolike_xi_ia.npz")
    M1, K1 = R0["masks_n1"], [round(float(k), 4) for k in R0["K_n1"]]
    masks = {"none": np.ones(1080, bool), "K0/D 0.3": M1[K1.index(0.3)]}
    np.savez(os.path.join(HERE, "ia_base_common.npz"), cov=cov,
             cuts=np.array(list(CUTS.values())),
             **{f"mask|{CUTS[c]}": m for c, m in masks.items()})
    Hf = CB.tents_on(KW, CB.FINE)
    tm = ME.F.TATTModel(halofit="takahashi")
    x_fid = {**{n: B.FID[n] for n in FREE_T}, "A": 1.0, "eta": 0.0}
    log(f"[{time.time() - t0:.0f}s] start: truths {truths}, {2*len(st['zs'])*len(CB.FINE)} "
        "knobs per truth")

    for tr in truths:
        model = CB.Truth(cl, st, tr)
        zs = model.zs
        nz, nf = len(zs), len(CB.FINE)
        P = model.pnl(B.FID)
        PdE, PEE = model.RdE*P, model.REE*P
        zero = np.zeros_like(PdE)
        xi_ia = T.TB_xi(cl, model, PdE, PEE, SCRATCH)
        withb = BT.has_b(st, tr)
        if withb:
            # B-mode II (decision of 2026-11-23): a knob of II moves E and B
            # together, R_II = R_II_E + R_II_B; both parts are stored
            mB = CB.Truth(cl, st, tr + "_B")
            PBB = mB.REE*P
            xi_ia = xi_ia + BT.flip_b(T.TB_xi(cl, mB, zero, PBB, SCRATCH))
            RB = np.zeros((nz, nf, 1080))
        R = {X: np.zeros((nz, nf, 1080)) for X in IF.XS}
        n, ntot, tl = 0, (3 if withb else 2)*nz*nf, time.time()
        for s in range(nz):
            for j in range(nf):
                for X in IF.XS:
                    tab = zero.copy()
                    tab[s] = (PdE if X == "GI" else PEE)[s]*Hf[j]
                    R[X][s, j] = T.TB_xi(cl, model, tab if X == "GI" else zero,
                                         tab if X == "II" else zero, SCRATCH)
                    n += 1
                if withb:
                    tab = zero.copy()
                    tab[s] = PBB[s]*Hf[j]
                    RB[s, j] = BT.flip_b(T.TB_xi(cl, mB, zero, tab, SCRATCH))
                    n += 1
            el = time.time() - tl
            log(f"[{time.time() - t0:.0f}s] {tr}: snapshot {s + 1}/{nz} (z = {zs[s]:g}) done; "
                f"{el/n:.2f} s per column, about {el/n*(ntot - n)/60:.0f} min left")
        # stale-table check: one column again, after every other table was written
        tab = zero.copy()
        tab[nz//2] = PdE[nz//2]*Hf[nf//2]
        again = T.TB_xi(cl, model, tab, zero, SCRATCH)
        stale = float(np.max(np.abs(again - R["GI"][nz//2, nf//2]))/np.max(np.abs(again)))
        extra = {}
        if withb:
            RE = R["II"]
            R["II"] = RE + RB
            extra = dict(R_II_E=RE, R_II_B=RB)
            J = T.jacobian(BT.EBModel(model, mB, CB.Truth(cl, st, "ZERO")), FREE_T, IAP, x_fid)
        else:
            J = T.jacobian(model, FREE_T, IAP, x_fid)
        g = T.s8_row(cl, FREE_T, IAP, x_fid)
        lnx = np.log((1 + zs)/IF.X0)
        tot = R["GI"].sum((0, 1)) + R["II"].sum((0, 1))
        dA = R["GI"].sum((0, 1)) + 2*R["II"].sum((0, 1))
        de = (lnx[:, None]*(R["GI"].sum(1) + 2*R["II"].sum(1))).sum(0)
        scale = np.max(np.abs(xi_ia))
        checks = dict(stale=stale,
                      sum_vs_xi_ia=float(np.max(np.abs(tot - xi_ia))/scale),
                      sum_vs_dA=float(np.max(np.abs(dA - J[:, 2]))/np.max(np.abs(J[:, 2]))),
                      sum_vs_deta=float(np.max(np.abs(de - J[:, 3]))/np.max(np.abs(J[:, 3]))))
        log(f"[{time.time() - t0:.0f}s] {tr}: checks " +
            ", ".join(f"{k} {v:.1e}" for k, v in checks.items()))
        drifts = {}
        for cut, lab in CUTS.items():
            for mdl in MODELS:
                if f"{tr}|{cut}|{mdl}" not in fid_fits["fits"]:      # truths added later (lin)
                    continue
                bp = fid_fits["fits"][f"{tr}|{cut}|{mdl}"]["best"]
                err = {"GI": np.zeros((nz, nf)), "II": np.zeros((nz, nf))}
                for s in range(nz):
                    Rt = np.array([np.interp(np.log(CB.FINE), np.log(model.kd), model.R[s, i])
                                   for i in (0, 1)])
                    Rm = ME.model_ratio(tm, float(zs[s]), CB.FINE, bp, mdl)
                    err["GI"][s] = Rm[0]/Rt[0] - 1
                    err["II"][s] = Rm[1]/Rt[1] - 1
                for X in IF.XS:
                    drifts[f"drift|{lab}|{mdl}|{X}"] = err[X]
        np.savez(os.path.join(HERE, f"ia_base_{tr}.npz"), truth=tr, zs=zs, fine=CB.FINE,
                 xi0=gg + xi_ia, xi_ia=xi_ia, J=J, template=np.array(IF.TEMPLATE), g_s8=g,
                 R_GI=R["GI"], R_II=R["II"], checks=json.dumps(checks), **drifts, **extra)
        log(f"[{time.time() - t0:.0f}s] {tr}: wrote ia_base_{tr}.npz")
    log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
