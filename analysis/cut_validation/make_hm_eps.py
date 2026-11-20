#!/usr/bin/env python3
"""data.json of plots/hm_eps: where the halo-model correction acts.

The fractional model error eps_X(k, z) = 1 - P_X^model/P_X^truth of NLA, TATT
and their halo-model variants at FLAMINGO's alpha (NLA-F, TATT-F), GI and II,
at three snapshots, on the stitched truth, at the reference point of the
cosmic-shear fit at K0 = 0.35 (D3A, the IA parameters fitted at fixed
cosmology), read from pieces_maps/ of bias_pieces.py. eps > 0: the model is
below the truth. Next to it, the figure of bias of each piece of the split at
K0 = 0.4 (cosmic shear), the cut the recommended one has to beat.
    python3 make_hm_eps.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["NLA", "NLAf", "TATT", "TATTf"]
ZS = [0.5, 1.0, 2.0]


def main():
    R = json.load(open(os.path.join(HERE, "bias_pieces.json")))["runs"]
    out = dict(about=f"Producer: make_hm_eps.py.\n{__doc__}", models=MODELS, zs=ZS, eps={}, pieces={})
    for m in MODELS:
        f = np.load(os.path.join(HERE, "pieces_maps", f"cspcmb_{m}_0.35_S1.npz"))
        c, fine, zs = f["c"], f["fine"], f["zs"]
        out["k"] = fine.tolist()
        for z in ZS:
            s = int(np.argmin(np.abs(zs - z)))
            assert abs(zs[s] - z) < 1e-6
            out["eps"][f"{m}|{z:g}"] = dict(GI=c[s, 0].tolist(), II=c[s, 1].tolist())
        out["pieces"][m] = {q: v["fob"] for q, v in R[f"cs+cmb|{m}|0.4|S1"]["zshape"].items()}
    json.dump(out, open(os.path.join(HERE, "plots", "hm_eps", "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
