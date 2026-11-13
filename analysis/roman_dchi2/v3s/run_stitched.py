#!/usr/bin/env python3
"""v3 scale-cut analysis (GoF, anchored, deviation, Delta chi2_FL, A(z)) with
the stitched spectra as the truth.

Stitched truth per snapshot (FLAMINGO notes): w(k) = logistic in ln k at
k_t(z), width 0.30; NLA side (-C1, C1^2) from the notes' NLA fit on
0.05 <= k <= 0.3 (spectra.npz, build_spectra.py):
    P_hE  = (1 - w) (-C1 b P_NL)         + w P_hE^sim
    P_EE  = (1 - w) (C1^2 P_NL + N_s)    + w P_EE^sim        (P^sim raw, N_s included)
  data  y : simulation side = measured spectra (for the GoF chi2)
  truth T : simulation side = smoothing spline (noise-free; Delta chi2_FL,
            fractional deviations)
Variants: S2 (main): N_s = notes' value from the NLA + N_s fit on 0.05-0.3;
          S1: N_s = B-mode plateau;
          prod2: production data and truth with the same two probes.
Only P_hE and P_EE are fitted (the stitch has no NLA side for P_BB and the
notes' N_s over-subtracts it); errors = FLAMINGO's measured error bars at
every k; the effective number of degrees of freedom is sum_k w_k^2 per
probe, since the residual variance under the null is w^2 sigma^2.
    python run_stitched.py [S2|S1|prod2 ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v3"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import v3_flamingo as V3                                  # noqa: E402
from fit_stitched_kmax import KT, SIGMA_W, weight         # noqa: E402


def truths():
    f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
    spec = np.load(os.path.join(HERE, "..", "spectra.npz"))
    zs, sns, kd = f["zs"], f["sns"], f["k"][0]
    y, T, b, Pf, NsB = f["y"], f["T"], f["b"], f["PNL"], f["Ns_true"]
    NsN = spec["Ns_nla"]
    C1 = -spec["truth_highk"][:, 0, 0]
    nsn = len(zs)
    out = {"prod2": dict(probes=[0, 1], outdir=os.path.join(HERE, "prod2"), label=" (production, $P_{hE}+P_{EE}$)")}
    for tag, Nsc, lab in (("S2", NsN, " (stitched, notes' $N_s$)"), ("S1", NsB, " (stitched, $N_s$ = B plateau)")):
        ys, Ts, W = np.zeros_like(y), np.zeros_like(T), np.zeros((nsn, kd.size))
        for s in range(nsn):
            w = weight(kd, KT[int(sns[s])], SIGMA_W)
            W[s] = w
            hE_nla = -C1[s]*b[s]*Pf[s]
            EE_nla = C1[s]**2*Pf[s] + Nsc[s]
            for src, dst in ((y, ys), (T, Ts)):
                dst[s, 0] = (1 - w)*hE_nla + w*src[s, 0]
                dst[s, 1] = (1 - w)*EE_nla + w*src[s, 1]
                dst[s, 2] = src[s, 2]
        out[tag] = dict(y=ys, T=Ts, Ns_EE=Nsc, w=W, probes=[0, 1], outdir=os.path.join(HERE, tag), label=lab)
        if tag == "S2":   # noise model (a): the stitched data's own errors w sigma (floor 0.05), standard dof
            err_w = f["err"]*np.maximum(W, 0.05)[:, None, :]
            out["S2w"] = dict(y=ys, T=Ts, Ns_EE=Nsc, err=err_w, probes=[0, 1], outdir=os.path.join(HERE, "S2w"),
                              label=" (stitched, errors $w\\sigma$)")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["S2", "S1", "S2w", "prod2"]
    tr = truths()
    for tag in which:
        print(f"===== {tag} =====", flush=True)
        V3.main(tr[tag])
