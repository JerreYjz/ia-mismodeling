#!/usr/bin/env python3
"""The linear-limit FLAMINGO truth ("lin"), added to stitched_truths.npz.

The production truth holds the spline of the measured ratios at its edge
value (k = 0.067) on all larger scales, and the stitched truth (S1) puts NLA
there with the amplitude of the NLA fit. Neither uses the measured points
below k = 0.05. The linear-limit truth does, and assumes only that the
ratios R_X = P_X/P_NL are constant on linear scales (the limit every
tidal-alignment model shares), with no NLA amplitude and no redshift law:

    c_X(z_s)  the error-weighted mean of the measured ratio over every
              measured k <= KLIN (7 points: 0.009 to 0.067 h/Mpc, FLAMINGO's
              mode-scatter errors), GI and II separately, per snapshot;
    R_X       (1 - w) c_X + w R_X^prod,  w = 1/(1 + exp(-ln(k/KJ)/WIDTH)),
              with R_X^prod (the production spline) used only at k >= 0.085,
              so its held edge never enters (c_X below).

The measured ratios are those of v2/flamingo_truth.py (GI: P_hE/(b P_NL),
II: (P_EE - Ns)/P_NL, Ns the B-mode plateau, the production convention).
Writes R_lin and lin_c (13, 2, 4: c, its error, chi2 about it, npts) into
stitched_truths.npz (every other key unchanged; the previous file is kept as
stitched_truths_before_lin.npz).
    ~/cocoa/Cocoa/.local/bin/python linear_limit_truth.py
"""
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F  # noqa: E402
import flamingo_truth as FT    # noqa: E402

PS = os.path.join(HERE, "..", "..", "..", "flamingo", "data", "PS")
KLIN, KJ, WIDTH, KSPL = 0.07, 0.08, 0.3, 0.085


def main():
    path = os.path.join(HERE, "stitched_truths.npz")
    st = dict(np.load(path))
    kd, zs, sns = st["k"], st["zs"], st["sns"]
    model = F.TATTModel(halofit="takahashi")
    w = 1/(1 + np.exp(-np.log(kd/KJ)/WIDTH))
    R = np.zeros_like(st["R_prod"])
    C = np.zeros((len(zs), 2, 4))
    for s, sn in enumerate(sns):
        d = FT.load(PS, int(sn), model)
        assert np.allclose(d["k"], kd, rtol=1e-3), "measured grid differs from the truth grid"
        y = [d["y"]["hE"]/(d["b"]*d["PNL"]), (d["y"]["EE"] - d["Ns"])/d["PNL"]]
        e = [d["err"]["hE"]/(d["b"]*d["PNL"]), d["err"]["EE"]/d["PNL"]]
        j = kd <= KLIN
        for i in (0, 1):
            wt = 1/e[i][j]**2
            c = float(np.sum(y[i][j]*wt)/np.sum(wt))
            C[s, i] = [c, 1/np.sqrt(np.sum(wt)), float(np.sum((y[i][j] - c)**2*wt)), j.sum()]
            spl = np.where(kd >= KSPL, st["R_prod"][s, i], c)
            R[s, i] = (1 - w)*c + w*spl
        print(f"z = {zs[s]:g}: GI {C[s, 0, 0]/-st['C1'][s]:.3f} +- {C[s, 0, 1]/st['C1'][s]:.3f} "
              f"(chi2 {C[s, 0, 2]:.1f}/{int(C[s, 0, 3]) - 1}), II {C[s, 1, 0]/st['C1'][s]**2:.3f} +- "
              f"{C[s, 1, 1]/st['C1'][s]**2:.3f} (chi2 {C[s, 1, 2]:.1f}/{int(C[s, 1, 3]) - 1}) in NLA units; "
              f"production held {st['R_prod'][s, 0, 0]/-st['C1'][s]:.3f}, {st['R_prod'][s, 1, 0]/st['C1'][s]**2:.3f}")
    kk = kd <= 10.0      # above 10 h/Mpc II turns negative in every truth (shape-noise subtraction)
    assert np.all(R[:, 0, kk] < 0) and np.all(R[:, 1, kk] > 0), "ratio changes sign"
    bak = os.path.join(HERE, "stitched_truths_before_lin.npz")
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    st["R_lin"], st["lin_c"] = R, C
    np.savez(path, **st)


if __name__ == "__main__":
    main()
