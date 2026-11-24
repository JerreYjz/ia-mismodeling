#!/usr/bin/env python3
"""The CP truth: the stitched truth S1 with the close-pair II term kept.

flamingo/bmode_noise showed that N_s, the white P_BB plateau that S1
subtracts from P_EE, is the shape correlation of close galaxy pairs, equal in
E and B. The pair sum (V/N^2) sum c_ij j0(k r_ij) says the term is flat below
k ~ 1 h/Mpc and falls to N_s only at 5 to 10 h/Mpc, so its low-k level N_low
is about twice N_s. NBall (v9/bmode_truth_3x2.py) kept N_s at low k; CP keeps
N_low there instead, with S1's stitch weight w:

  E   R_CP,II  = R_S1,II + w N_s/P_NL + (1 - w) N_low/P_NL
               = (1 - w) [C1^2 A^2 (linear limit) + N_low/P_NL] + w T_EE/P_NL
  B   R_CP,BB  = w T_BB/P_NL + (1 - w) N_low/P_NL
  GI  that of S1 (P_hE has no such term)

N_low(z) is the inverse-variance mean of the measured P_BB over
0.15 <= k <= 0.6 h/Mpc (v2/fits.npz), where the linear-limit IA has no B
modes and the pair term is still flat. It is checked against the pair sum at
the four snapshots where that was computed (z = 0, 1, 2, 3).

Writes v3/cp_truth.npz: R_CP and R_CP_B, (13, 2, 26) on the data grid as
R_<truth> of stitched_truths.npz (column 0 GI, column 1 II; the B table has
GI = 0), and N_low, N_s, zs. numpy only.

    python cp_truth.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "..", "tatt_fit"))
import fit_tatt_flamingo as F                                        # noqa: E402

KLOW = (0.15, 0.6)          # h/Mpc, the range that sets N_low
PAIR_SNS = (78, 58, 38, 18)  # snapshots with the close-pair sum


def main():
    st = np.load(os.path.join(HERE, "stitched_truths.npz"))
    f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
    k, zs = st["k"], st["zs"]
    assert np.allclose(f["k"][0], k) and np.array_equal(f["sns"], st["sns"])
    w, PNL, Ns, T = st["w"], f["PNL"], f["Ns_true"], f["T"]
    y, err = f["y"][:, 2], f["err"][:, 2]                        # measured P_BB and its error
    nz = len(zs)

    sel = (k >= KLOW[0]) & (k <= KLOW[1])
    Nlow = np.array([np.sum(y[s, sel]/err[s, sel]**2)/np.sum(1/err[s, sel]**2) for s in range(nz)])
    Nlow_err = np.array([1/np.sqrt(np.sum(1/err[s, sel]**2)) for s in range(nz)])

    R = st["R_S1"].copy()
    R[:, 1] = st["R_S1"][:, 1] + w*Ns[:, None]/PNL + (1 - w)*Nlow[:, None]/PNL
    Rb = np.zeros_like(R)
    Rb[:, 1] = w*T[:, 2]/PNL + (1 - w)*Nlow[:, None]/PNL

    # checks
    checks = {}
    pair = {}
    for sn in PAIR_SNS:
        d = json.load(open(os.path.join(ROOT, "flamingo", "bmode_noise", "results", f"bmode_noise_sn{sn}.json")))
        z, erms = F.SNAPSHOTS[sn][:2]
        s1 = 1/(2*(1 - erms**2))                                  # grid units -> truth units, two shape legs
        s = int(np.argmin(abs(zs - z)))
        pair[f"{z:g}"] = dict(pair_low=d["pairs"]["pred_EEBB_half"][0]*s1**2, N_low=float(Nlow[s]),
                              N_s=float(Ns[s]))
    checks["pair_vs_Nlow"] = {z: v["pair_low"]/v["N_low"] for z, v in pair.items()}
    # where w -> 1 the E table is T_EE/P_NL (as NBall), and S1 + N_s there
    hi = w > 0.999
    checks["E_vs_NBall_where_w1"] = float(np.max(np.abs(R[:, 1][hi] - (st["R_S1"][:, 1] + Ns[:, None]/PNL)[hi])))
    checks["B_vs_TBB_where_w1"] = float(np.max(np.abs(Rb[:, 1][hi] - (T[:, 2]/PNL)[hi])))
    checks["GI_equals_S1"] = float(np.max(np.abs(R[:, 0] - st["R_S1"][:, 0])))
    checks["Nlow_over_Ns"] = (Nlow/Ns).tolist()
    checks["Nlow_snr"] = (Nlow/Nlow_err).tolist()
    checks["B_positive"] = bool(np.all(Rb[:, 1] > 0))

    np.savez(os.path.join(HERE, "cp_truth.npz"), R_CP=R, R_CP_B=Rb, N_low=Nlow, N_low_err=Nlow_err,
             N_s=Ns, zs=zs, k=k, about=__doc__)
    json.dump(dict(about=__doc__, klow=KLOW, pair=pair, checks=checks), open(os.path.join(HERE, "cp_truth.json"), "w"),
              indent=1)
    print("z      N_s     N_low  (+-)    N_low/N_s")
    for s in range(nz):
        print(f"{zs[s]:4.2f}  {Ns[s]:.4f}  {Nlow[s]:.4f} ({Nlow_err[s]:.4f})  {Nlow[s]/Ns[s]:.2f}")
    print("pair-sum low-k level / N_low:", {z: round(v, 3) for z, v in checks["pair_vs_Nlow"].items()})
    for kk in ("E_vs_NBall_where_w1", "B_vs_TBB_where_w1", "GI_equals_S1", "B_positive"):
        print(kk, checks[kk])


if __name__ == "__main__":
    main()
