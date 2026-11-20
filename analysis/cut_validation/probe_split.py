#!/usr/bin/env python3
"""The 3x2pt bias split by probe, from the linear maps of pieces_maps/.

bias_pieces.py saves, per run, M: the linear map from the masked
data-minus-model vector E to the (Omega_m, S8) shift, with the joint 3x2pt
Fisher matrix. The rows of the masked E are xi_+- first, then gamma_t, then
w, so the shift is the sum of three terms, M_xi E_xi + M_gt E_gt + M_w E_w:
what the cosmic-shear rows, the gamma_t rows and the w rows each contribute
to the 3x2pt bias. The split is exact for the linear shift (it adds up), and
each term is given in sigma with its figure of bias. When a combination of
cuts fails, the term that fails names the probe whose cut is tightened.

For every 3x2pt run in bias_pieces.json (keys 3x2+cmb|MODEL|K0|TRUTH[|gtK]):
    python3 probe_split.py            (writes probe_split.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NXI, NGT = 1080, 915          # xi_+- rows, gamma_t rows; w after them (fisher_minimize.py)


def split(path):
    d = np.load(path)
    m, E, M, C2 = d["mask"], d["E_direct"], d["M"], d["C2"]
    rows = np.arange(len(m))[m]
    blocks = dict(xi=rows < NXI, gt=(rows >= NXI) & (rows < NXI + NGT), w=rows >= NXI + NGT)
    out = {}
    tot = M @ E[m]
    for name, b in blocks.items():
        dp = M[:, b] @ E[m][b]
        out[name] = dict(mu=(dp/d["sig"]).tolist(), fob=float(np.sqrt(dp @ np.linalg.solve(C2, dp))),
                         rows=int(b.sum()))
    out["total"] = dict(mu=(tot/d["sig"]).tolist(), fob=float(np.sqrt(tot @ np.linalg.solve(C2, tot))))
    assert np.allclose(sum(np.array(out[q]["mu"]) for q in blocks), out["total"]["mu"])
    return out


def main():
    R = json.load(open(os.path.join(HERE, "bias_pieces.json")))["runs"]
    out = dict(about=__doc__, runs={})
    for key in sorted(R):
        if not key.startswith("3x2+cmb|"):
            continue
        path = os.path.join(HERE, "pieces_maps", key.replace("|", "_").replace("+", "p") + ".npz")
        if not os.path.exists(path):
            continue
        s = split(path)
        s["fob_map"] = R[key]["fob_map"]
        out["runs"][key] = s
    json.dump(out, open(os.path.join(HERE, "probe_split.json"), "w"), indent=1)
    for k, s in out["runs"].items():
        print(f"{k:34s} MAP {s['fob_map']:.2f} lin {s['total']['fob']:.2f} | xi {s['xi']['fob']:.2f} "
              f"({s['xi']['mu'][0]:+.2f},{s['xi']['mu'][1]:+.2f})  gt {s['gt']['fob']:.2f} "
              f"({s['gt']['mu'][0]:+.2f},{s['gt']['mu'][1]:+.2f})  w {s['w']['fob']:.2f}")


if __name__ == "__main__":
    main()
