#!/usr/bin/env python3
"""data.json of plots/truth_lowk: the FLAMINGO II and GI ratios at large
scales against the two Roman truths.

Per snapshot (z = 0.2, 0.5, 1.0, 1.5): the measured ratios with FLAMINGO's
mode-scatter errors (v2/flamingo_truth.load, k < 1.2 h/Mpc, the nfold 1
measurement), R_EE = (P_EE - Ns)/P_NL and R_dE = P_hE/(b P_NL), each divided
by its NLA value (C1^2 and -C1, C1 of v3/stitched_truths.npz: the NLA fit on
0.05 <= k <= 0.3); and the production and stitched (S1) truths on the data
grid, divided the same way; and a measurement-only truth: the same
weighted smoothing spline as the production truth (v2/flamingo_truth.py,
chi2 of the data about it = number of points) but over every measured k
from 0.009 h/Mpc up, so no value is held and no NLA enters; and a
linear-limit truth: per snapshot, the error-weighted mean of the measured
ratio at k <= KLIN (GI and II separately, no NLA amplitude or redshift law),
joined to the production spline above k = 0.085 (its edge point at 0.067 left out) by a logistic weight in
ln k centred at KJ (width 0.3, as the stitch). NLA is 1 on
this scale.
    ~/cocoa/Cocoa/.local/bin/python truth_lowk_data.py
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import UnivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402
import flamingo_truth as FT    # noqa: E402

PS = os.path.join(HERE, "..", "..", "..", "flamingo", "data", "PS")
ZS = [0.2, 0.5, 1.0, 1.5]
KLIN, KJ, WIDTH = 0.07, 0.08, 0.3


def main():
    t = np.load(os.path.join(HERE, "stitched_truths.npz"))
    model = F.TATTModel(halofit="takahashi")
    out = dict(about=f"Producer: truth_lowk_data.py.\n{__doc__}", k_truth=t["k"].tolist(), kmin=FT.KMIN,
               klin=KLIN, kjoin=KJ, panels=[])
    for z in ZS:
        s = int(np.argmin(abs(t["zs"] - z)))
        sn = int(t["sns"][s])
        d = FT.load(PS, sn, model)
        C1 = float(t["C1"][s])
        m = d["k"] < 1.2
        ee = (d["y"]["EE"] - d["Ns"])/d["PNL"]/C1**2
        de = d["y"]["hE"]/(d["b"]*d["PNL"])/(-C1)
        p = dict(z=float(t["zs"][s]), sn=sn, k=d["k"][m].tolist(),
                 EE=ee[m].tolist(), EE_err=(d["err"]["EE"]/d["PNL"]/C1**2)[m].tolist(),
                 dE=de[m].tolist(), dE_err=(d["err"]["hE"]/(d["b"]*d["PNL"])/C1)[m].tolist())
        for v in ("prod", "S1"):
            p[f"{v}_EE"] = (t[f"R_{v}"][s, 1]/C1**2).tolist()
            p[f"{v}_dE"] = (t[f"R_{v}"][s, 0]/(-C1)).tolist()
        p["w"] = t["w"][s].tolist()
        lk = np.log(d["k"])
        for key, y, e in (("EE", (d["y"]["EE"] - d["Ns"])/d["PNL"], d["err"]["EE"]/d["PNL"]),
                          ("dE", d["y"]["hE"]/(d["b"]*d["PNL"]), d["err"]["hE"]/(d["b"]*d["PNL"]))):
            sp = UnivariateSpline(lk, y, w=1.0/e, k=3, s=float(len(lk)))
            p[f"meas_{key}"] = (sp(np.log(t["k"]))/(C1**2 if key == "EE" else -C1)).tolist()
        kt = t["k"]
        w = 1.0/(1.0 + np.exp(-np.log(kt/KJ)/WIDTH))
        for key in ("EE", "dE"):
            y, e = np.array(p[key]), np.array(p[key + "_err"])
            j = np.array(p["k"]) <= KLIN
            c = float(np.sum(y[j]/e[j]**2)/np.sum(1/e[j]**2))
            ce = float(1/np.sqrt(np.sum(1/e[j]**2)))
            chi = float(np.sum(((y[j] - c)/e[j])**2))
            spl = np.where(kt >= 0.085, np.array(p[f"prod_{key}"]), c)
            p[f"lin_{key}"] = ((1 - w)*c + w*spl).tolist()
            p[f"lin_{key}_c"] = [c, ce, chi, int(j.sum())]
            print(f"   linear limit {key}: {c:.3f} +- {ce:.3f}, chi2 {chi:.1f} for {j.sum()} points")
        out["panels"].append(p)
        j = (d["k"] > 0.04) & (d["k"] < 0.2)
        print(f"z={p['z']}: k {np.round(d['k'][j], 3)}")
        print(f"   EE/C1^2 {np.round(ee[j], 2)} +- {np.round((d['err']['EE']/d['PNL']/C1**2)[j], 2)}")
    json.dump(out, open(os.path.join(HERE, "plots", "truth_lowk", "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
