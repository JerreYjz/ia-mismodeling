#!/usr/bin/env python3
"""data.json of plots/truth_eb: the II truth with the white shape noise
subtracted (S1, production) against the measured B-mode subtracted (EB).

At four snapshots, in units of the NLA II spectrum C1^2 P_NL (C1 with the NLA
fit's amplitude, as stitched_truths.npz): the measured (P_EE - N_s)/P_NL and
(P_EE - P_BB)/P_NL with their mode-scatter errors (the error of the
difference adds those of E and B in quadrature), and the truths R_prod, R_S1,
R_EB. Second row: the B-mode excess over the plateau as a fraction of the II
signal of S1, (P_BB - N_s)/(P_EE - N_s), measured and smoothed (fits.npz T).
    ~/cocoa/Cocoa/.local/bin/python truth_eb_data.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZS = [0.3, 1.0, 2.0, 3.0]


def main():
    st = np.load(os.path.join(HERE, "stitched_truths.npz"))
    f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
    kd, y, e, T, PNL, Ns = f["k"][0], f["y"], f["err"], f["T"], f["PNL"], f["Ns_true"]
    out = dict(about=f"Producer: v3/truth_eb_data.py.\n{__doc__}", k=kd.tolist(), zs=ZS, snap={})
    for z in ZS:
        s = int(np.argmin(np.abs(st["zs"] - z)))
        nla = st["C1"][s]**2*PNL[s]
        out["snap"][f"{z:g}"] = dict(
            meas_ns=((y[s, 1] - Ns[s])/nla).tolist(), err_ns=(e[s, 1]/nla).tolist(),
            meas_eb=((y[s, 1] - y[s, 2])/nla).tolist(), err_eb=(np.hypot(e[s, 1], e[s, 2])/nla).tolist(),
            prod=(st["R_prod"][s, 1]/st["C1"][s]**2).tolist(), S1=(st["R_S1"][s, 1]/st["C1"][s]**2).tolist(),
            EB=(st["R_EB"][s, 1]/st["C1"][s]**2).tolist(),
            frac_meas=((y[s, 2] - Ns[s])/(T[s, 1] - Ns[s])).tolist(),
            frac_err=(e[s, 2]/np.abs(T[s, 1] - Ns[s])).tolist(),
            frac_smooth=((T[s, 2] - Ns[s])/(T[s, 1] - Ns[s])).tolist())
    json.dump(out, open(os.path.join(HERE, "plots", "truth_eb", "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
