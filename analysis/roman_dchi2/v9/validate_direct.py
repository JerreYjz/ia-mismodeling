#!/usr/bin/env python3
"""Validation of the v9 linear response by direct injection.

Random IA errors delta(X, s, i) are drawn from the kernel of the drift
section of ia_budget.py (exp, ell = 1, L_z = 0.3, GI and II independent) at
the budget sigma_max of each cut (ia_budget.json, global), written into the
truth tables as P_X(k, z_s) [1 + f(k, z_s)], with f on the tents exactly as
the base represents it (spline at FINE, tents in ln k beyond), injected
through cosmolike, and refitted with the full Gauss-Newton fit of v8
(clean_budget.gn_refit: CAMB at every step, A and eta free). The refitted
shifts of Omega_m and S8 are compared with the linear prediction of
ia_fisher. Production truth only.
Output: validate_direct.json.

    python validate_direct.py <cocoa so dir> <cocoa_roman_real/data> [--n 5]
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
from clean_budget import T, B, KW                                  # noqa: E402
import ia_fisher as IF                                             # noqa: E402
from ia_base import CUTS, FREE_T, IAP                              # noqa: E402

TRUTH = "prod"
KERNEL = dict(ell=1.0, lz=0.3, kind="exp", set="both")
RNG_SEED = 20261117


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 5
    t0 = time.time()
    budget = json.load(open(os.path.join(HERE, "ia_budget.json")))
    base = IF.Base(os.path.join(HERE, f"ia_base_{TRUTH}.npz"), os.path.join(HERE, "ia_base_common.npz"))
    zs = base.zs
    Rmap = base.knob_map(IF.XS)
    K1 = IF.residual_cov(IF.NODES, zs, KERNEL["ell"], KERNEL["lz"], KERNEL["kind"], IF.XS, 0.0)
    L1 = IF.factor(K1)
    S_fine = IF.spline_basis(IF.FINE)
    Hf = CB.tents_on(KW, CB.FINE)
    os.chdir(T.TB.COCOA)
    cl = B.CL(so_dir, data, threads=8)
    cl.cache[(0.306, 2.099, 0.967, 68.1, 0.0486)] = (cl.res, cl.h, cl.omegan2, cl.PKL, cl.PKNL,
                                                     cl.PKLcb, cl.sigma8)
    gg = cl.model("NLA", B.FID, [0.0, 0.0])
    T.GG["fid"] = gg
    st = np.load(os.path.join(T.TB.ROOT, "v3", "stitched_truths.npz"))
    model = CB.Truth(cl, st, TRUTH)
    P = model.pnl(B.FID)
    PdE, PEE = model.RdE*P, model.REE*P
    s8f = cl.s8(dict(B.FID))
    x_fid = {**{q: B.FID[q] for q in FREE_T}, "A": 1.0, "eta": 0.0}
    rng = np.random.default_rng(RNG_SEED)
    nz, nn = len(zs), len(IF.NODES)
    out = dict(about=__doc__, truth=TRUTH, kernel=KERNEL, cuts={})
    for cut, lab in CUTS.items():
        m = base.masks[lab]
        rigid, _ = IF.rigid_fit(base, m)
        row = [r for r in budget["truths"][TRUTH][lab]["global"]
               if r["set"] == KERNEL["set"] and r["kind"] == KERNEL["kind"] and r["ell"] == KERNEL["ell"]
               and r["lz"] == f"{KERNEL['lz']:g}"][0]
        sigma = row["sigma_max"]
        Ci = np.linalg.inv(base.cov[np.ix_(m, m)])
        res = []
        for t in range(n):
            delta = sigma*(L1 @ rng.standard_normal(L1.shape[1]))
            lin = rigid.shift(Rmap @ delta)
            f = {}
            for q, X in enumerate(IF.XS):
                d_nodes = delta[q*nz*nn:(q + 1)*nz*nn].reshape(nz, nn)
                f[X] = (d_nodes @ S_fine) @ Hf                    # (nz, len(KW)), as the base
            xi = T.TB_xi(cl, model, PdE*(1 + f["GI"]), PEE*(1 + f["II"]), "v9_validate")
            dd = gg + xi
            x = CB.gn_refit(model, cl, dd, Ci, m, FREE_T, IAP, x_fid)
            full = [(x["omegam"] - B.FID["omegam"])/rigid.sig[0],
                    (cl.s8(T.full_p({k: x[k] for k in FREE_T})) - s8f)/rigid.sig[1]]
            res.append(dict(sigma=sigma, full=full, linear=lin.tolist(),
                            diff=float(np.max(np.abs(np.array(full) - lin)))))
            print(f"[{time.time() - t0:.0f}s] {cut:9s} #{t}: sigma {sigma:.4f}; full ({full[0]:+.3f}, "
                  f"{full[1]:+.3f}), linear ({lin[0]:+.3f}, {lin[1]:+.3f})", flush=True)
        out["cuts"][lab] = res
        json.dump(out, open(os.path.join(HERE, "validate_direct.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
