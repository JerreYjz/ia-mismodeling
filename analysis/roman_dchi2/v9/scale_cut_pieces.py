#!/usr/bin/env python3
"""Where the NLA and TATT shifts of scale_cut_scan.py come from.

For each truth, cut K_0 and model, the known error of the model fitted with
that cut (best fits of scale_cut_scan.json, the error rebuilt on the tents at
every snapshot as there) is split by spectrum (GI, II) and by range of k
(BINS), and each piece is pushed through the rigid fit of the base alone. The
pieces add up to the total shift. Their quadrature sum (rss) is the shift
the same errors would give if each range had a random sign; it overstates
the harm of a smooth error, because a flat offset split into ranges is no
longer absorbed by the free amplitude.
Output: scale_cut_pieces.json. numpy only (with the v8 model ratios).

    python scale_cut_pieces.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v8"))
sys.path.insert(0, HERE)
import ia_fisher as IF                                             # noqa: E402
from clean_budget import ME                                        # noqa: E402

BINS = [0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 10.0]


def main():
    scan = json.load(open(os.path.join(HERE, "scale_cut_scan.json")))
    R0 = np.load(os.path.join(HERE, "..", "cocoa", "run", "cosmolike_xi_ia.npz"))
    masks = {"none": np.ones(1080, bool)}
    masks.update({f"{float(k):g}": m for k, m in zip(R0["K_n1"], R0["masks_n1"])})
    st = np.load(os.path.join(HERE, "..", "v3", "stitched_truths.npz"))
    tm = ME.F.TATTModel(halofit="takahashi")
    out = dict(about=__doc__, bins=BINS, truths={})
    lo = np.r_[0.0, BINS[1:-1]]                    # the end bins take the flat tents beyond
    hi = np.r_[BINS[1:-1], np.inf]
    for tr, t in scan["truths"].items():
        base = IF.Base(os.path.join(HERE, f"ia_base_{tr}.npz"), os.path.join(HERE, "ia_base_common.npz"))
        zs, kd, R = base.zs, st["k"], st[f"R_{tr}"]
        Rt = np.array([[np.interp(np.log(IF.FINE), np.log(kd), R[s, i]) for i in (0, 1)]
                       for s in range(len(zs))])
        tout = {}
        for K, row in t.items():
            rigid, _ = IF.rigid_fit(base, masks[K])
            kout = {}
            for mdl, v in row["models"].items():
                c = np.zeros(base.nb)
                for s in range(len(zs)):
                    Rm = ME.model_ratio(tm, float(zs[s]), IF.FINE, v["best"], mdl)
                    for i, X in enumerate(IF.XS):
                        c[base.iknob[X][s]] = -(Rm[i]/Rt[s, i] - 1)
                total = rigid.shift(c)
                assert np.allclose(total, v["mu"], atol=1e-6), (tr, K, mdl)
                pieces = {}
                for X in IF.XS:
                    p = []
                    for a, b in zip(lo, hi):
                        sel = np.zeros(base.nb, bool)
                        sel[base.iknob[X][:, (IF.FINE >= a) & (IF.FINE < b)].ravel()] = True
                        p.append(rigid.shift(np.where(sel, c, 0.0)).tolist())
                    pieces[X] = p                               # (n bins, 2): Omega_m, S8
                allp = np.array(pieces["GI"] + pieces["II"])
                assert np.allclose(allp.sum(0), total, atol=1e-8)
                kout[mdl] = dict(total=total.tolist(), pieces=pieces,
                                 rss=np.sqrt((allp**2).sum(0)).tolist(),
                                 largest=np.abs(allp).max(0).tolist())
            tout[K] = kout
        out["truths"][tr] = tout
    json.dump(out, open(os.path.join(HERE, "scale_cut_pieces.json"), "w"), indent=1)
    p = out["truths"]["prod"]["0.3"]["NLA"]
    print("prod K0 0.3 NLA: total", np.round(p["total"], 3), "rss", np.round(p["rss"], 3))


if __name__ == "__main__":
    main()
