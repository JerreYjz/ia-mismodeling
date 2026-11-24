#!/usr/bin/env python3
"""Add the known drifts of a truth to its base, ia_base_<truth>.npz.

ia_base.py stores drift|<cut>|<model>|<X> from v8/fid_fits.json when the fits
exist; for a truth whose base was built before its fits (CP), this adds them
afterwards with the same code: P^model/P^truth - 1 of the NLA and TATT best
fits at every snapshot, on FINE, X = GI, II (the E tables). numpy only.

    python add_drifts.py CP
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v8"))
sys.path.insert(0, HERE)
import clean_budget as CB                                          # noqa: E402
from clean_budget import ME                                        # noqa: E402
import ia_fisher as IF                                             # noqa: E402
import bmode_tables as BT                                          # noqa: E402
from ia_base import CUTS, MODELS                                   # noqa: E402


def main():
    tr = sys.argv[1]
    fid = json.load(open(os.path.join(HERE, "..", "v8", "fid_fits.json")))
    st = BT.load_truths(os.path.join(HERE, "..", "v3", "stitched_truths.npz"))
    path = os.path.join(HERE, f"ia_base_{tr}.npz")
    base = dict(np.load(path))
    zs, kd, R = st["zs"], st["k"], st[f"R_{tr}"]
    assert np.allclose(base["zs"], zs)
    tm = ME.F.TATTModel(halofit="takahashi")
    for cut, lab in CUTS.items():
        for mdl in MODELS:
            bp = fid["fits"][f"{tr}|{cut}|{mdl}"]["best"]
            err = {X: np.zeros((len(zs), len(IF.FINE))) for X in IF.XS}
            for s in range(len(zs)):
                Rt = np.array([np.interp(np.log(CB.FINE), np.log(kd), R[s, i]) for i in (0, 1)])
                Rm = ME.model_ratio(tm, float(zs[s]), CB.FINE, bp, mdl)
                for i, X in enumerate(IF.XS):
                    err[X][s] = Rm[i]/Rt[i] - 1
            for X in IF.XS:
                base[f"drift|{lab}|{mdl}|{X}"] = err[X]
            core = (zs >= 0.2 - 1e-9) & (zs <= 2 + 1e-9)
            print(f"{tr} {lab:4s} {mdl:4s}: " + ", ".join(
                f"{X} mean |err| (core z, k 0.1 to 1) {np.mean(np.abs(err[X][core][:, (IF.FINE >= 0.1) & (IF.FINE <= 1)])):.3f}"
                for X in IF.XS))
    np.savez(path, **base)
    print(f"wrote drifts into {os.path.basename(path)}")


if __name__ == "__main__":
    main()
