#!/usr/bin/env python3
"""Write plots/chi2dof/data.json from v3.npz: chi2_min/nu of NLA and TATT
refitted at every k_max, per core snapshot, and the p = 0.05 equivalent
threshold chi2_0.95(nu)/nu.

    python make_chi2dof_data.py
"""
import json
import os

import numpy as np
from scipy.stats import chi2 as chi2_dist

HERE = os.path.dirname(os.path.abspath(__file__))
NPAR = {"NLA": 1, "TATT": 3}


def main():
    v = np.load(os.path.join(HERE, "v3.npz"))
    core = v["core"]
    out = dict(
        about="Producer: analysis/roman_dchi2/v3/make_chi2dof_data.py from v3/v3.npz (made by v3/v3_flamingo.py). "
              "chi2_min/nu of NLA (a1) and TATT (a1, a2, bTA) refitted to FLAMINGO at every k_max, three probes "
              "(P_hE, P_EE, P_BB), 0.05 <= k <= k_max, N_s fixed to the B plateau; nu = 3 N_k - n_par. "
              "p05 = chi2_0.95(nu)/nu, the p = 0.05 equivalent.",
        kmax=v["kgrid"].tolist(), z=v["zs"][core].tolist(), npar=NPAR, models={})
    for m in NPAR:
        dof = v[f"{m}_dof"][core]
        assert np.all(dof == dof[0]), "nu differs between snapshots"
        out["models"][m] = dict(nu=dof[0].tolist(),
                                chi2nu=(v[f"{m}_chi2"][core]/dof).tolist(),
                                p05=(chi2_dist.isf(0.05, dof[0])/dof[0]).tolist())
    with open(os.path.join(HERE, "plots", "chi2dof", "data.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
