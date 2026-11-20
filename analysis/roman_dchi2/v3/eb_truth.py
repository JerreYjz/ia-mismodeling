#!/usr/bin/env python3
"""The E-minus-B FLAMINGO truth ("EB"), added to stitched_truths.npz.

Every other truth takes the II signal as P_EE - N_s, with N_s the B-mode
plateau at 5 to 10 h/Mpc (white shape noise). FLAMINGO's B-mode is not
white: at k = 0.2 to 1 h/Mpc it lies 0.06 to 0.14 above that plateau (5 to
20 sigma), falling with k, and below it above 10 h/Mpc. The same power sits
in P_EE, where the other truths count it as II signal. The halo-model fits
of the FLAMINGO review absorb it with a free white term in P_EE and P_BB at
once. This truth assumes instead that the excess is equal in E and B (as any
noise-like power is) and subtracts the measured B-mode scale by scale:

    II   (P_EE - P_BB)/P_NL with the smoothing splines of v2 (fits.npz "T":
         the weighted cubic splines in ln k of flamingo_truth.py), stitched as
         S1: (1 - w) C1^2 A^2 + w (T_EE - T_BB)/P_NL, the same w(k_t) and the
         same NLA amplitude as S1 (set by GI, which carries no shape noise);
    GI   that of S1 (P_hE has no shape noise).

So R_EB differs from R_S1 only in II above k_t, by -w (T_BB - N_s)/P_NL.
Writes R_EB (13, 2, 26) into stitched_truths.npz (every other key unchanged;
the previous file is kept as stitched_truths_before_eb.npz).
    ~/cocoa/Cocoa/.local/bin/python eb_truth.py
"""
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = os.path.join(HERE, "stitched_truths.npz")
    st = dict(np.load(path))
    f = np.load(os.path.join(HERE, "..", "v2", "fits.npz"))
    kd, T, PNL, Ns = f["k"][0], f["T"], f["PNL"], f["Ns_true"]
    assert np.allclose(kd, st["k"]) and np.array_equal(f["sns"], st["sns"])
    R = st["R_S1"].copy()
    excess = (T[:, 2] - Ns[:, None])/PNL                   # smoothed B-mode excess over the plateau
    R[:, 1] = st["R_S1"][:, 1] - st["w"]*excess
    # S1 is (1 - w) C1sq + w (T_EE - Ns)/PNL above KMIN; check the identity
    m = kd >= 0.05
    s1_hi = (T[:, 1] - Ns[:, None])/PNL
    rebuilt = (1 - st["w"])*st["R_S1"][:, 1, :1] + st["w"]*np.where(m, s1_hi, st["R_S1"][:, 1, :1])
    assert np.allclose(rebuilt, st["R_S1"][:, 1], rtol=1e-6), "S1 is not the expected stitch"
    for s, z in enumerate(st["zs"]):
        c2 = st["C1"][s]**2
        j = [int(np.argmin(np.abs(kd - q))) for q in (0.3, 1.0, 13.0)]
        print(f"z = {z:g}: II/C1^2 S1 -> EB at k = 0.3, 1, 13: "
              + ", ".join(f"{st['R_S1'][s, 1, i]/c2:.2f} -> {R[s, 1, i]/c2:.2f}" for i in j))
    kk = kd <= 17.0
    print("II > 0 at every k:", bool(np.all(R[:, 1, kk] > 0)), "; S1:", bool(np.all(st["R_S1"][:, 1, kk] > 0)))
    bak = os.path.join(HERE, "stitched_truths_before_eb.npz")
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    st["R_EB"] = R
    np.savez(path, **st)


if __name__ == "__main__":
    main()
