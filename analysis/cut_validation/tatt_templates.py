#!/usr/bin/env python3
"""Per-snapshot TATT templates for the freer redshift evolution (fisher_minimize.py).

The freer evolution multiplies both TATT amplitudes by the same g(z),

    a1(z) = A1 x^eta1 g(z),   a2(z) = A2 x^eta2 g(z),   x = (1 + z)/1.62,

with b_TA unchanged. Every GI term of TATT is linear in (a1, a2) and every II
term quadratic, so at snapshot s the GI part changes by (g_s - 1) and the II
part by (g_s^2 - 1) times its native value. To apply this at any amplitudes,
this script writes the data-vector response of each amplitude monomial m at
unit coefficient (the TATT spectra of tatt_fit/fit_tatt_flamingo.TATTModel,
in cosmolike's convention, c1 = 0.01389 Omega_m/D, c2 = c1/D):

    GI:  a1        -c1 P_NL                 II:  a1^2           c1^2 P_NL
         a1 bTA    -c1 K_dE                      a1^2 bTA      2 c1^2 K_dE
         a2         5 c2 K_mix                   a1^2 bTA^2      c1^2 K_ta
                                                 a1 a2        -10 c1 c2 K_mix
                                                 a1 a2 bTA    -10 c1 c2 K_mixEE
                                                 a2^2          25 c2^2 K_tt

plus the B modes of three II terms (a1^2 bTA^2: c1^2 K_taBB, a1 a2 bTA:
-10 c1 c2 K_mixBB, a2^2: 25 c2^2 K_ttB), which enter xi_+ as the E modes do
and xi_- with the opposite sign (C_EE +- C_BB):

    T_X[m, s] = sum_j R_X[s, j] (P_m/P_NL)(k_j, z_s) / R_X^truth(k_j, z_s)

R_X[s, j] is the v9 base column (the response to the truth's P_X on the ln k
tent j at snapshot s) and R_X^truth = P_X^truth/P_NL, as in hm_templates.py,
whose N_GI, N_II are the a1 and a1^2 rows here (checked below). The
templates are held at the D3A (= FLAMINGO) cosmology, Takahashi halofit, and
are exact in the amplitudes. Output: tatt_templates_<probe>_<truth>.npz with
zs, GI_names, II_names, T_GI (3, 13, n), T_II (6, 13, n).

    ~/cocoa/Cocoa/.local/bin/python tatt_templates.py [truth ...]   (default S1 EB lin prod)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "roman_dchi2")
sys.path.insert(0, os.path.join(HERE, "..", "tatt_fit"))
import fit_tatt_flamingo as F                                        # noqa: E402

GI_NAMES = ["a1", "a1 bTA", "a2"]
II_NAMES = ["a1^2", "a1^2 bTA", "a1^2 bTA^2", "a1 a2", "a1 a2 bTA", "a2^2"]
C1RHO, OM = 0.01389, 0.306
NXI = 1080               # xi_+ rows 0:540, xi_- rows 540:1080


def term_ratios(tm, z, k):
    """(3, nk) GI and (6, nk) II monomials at unit coefficient, over P_NL."""
    D = tm.D(z)
    c1, c2 = C1RHO*OM/D, C1RHO*OM/D**2
    K = tm.kernels(z, k)
    P = tm.Pnl(z, k)
    one = np.ones_like(k)
    gi = np.array([-c1*one, -c1*K["dE"]/P, 5*c2*K["mix"]/P])
    ii = np.array([c1**2*one, 2*c1**2*K["dE"]/P, c1**2*K["ta"]/P, -10*c1*c2*K["mix"]/P,
                   -10*c1*c2*K["mixEE"]/P, 25*c2**2*K["tt"]/P])
    bb = np.array([0*one, 0*one, c1**2*K["taBB"]/P, 0*one, -10*c1*c2*K["mixBB"]/P, 25*c2**2*K["ttB"]/P])
    return gi, ii, bb


def main():
    st = np.load(os.path.join(RD, "v3", "stitched_truths.npz"))
    zs = st["zs"]
    tm = F.TATTModel(halofit="takahashi")
    truths = sys.argv[1:] or ["S1", "EB", "lin", "prod"]
    for probe, base in (("cs", "ia_base"), ("3x2", "ia_base_3x2")):
        for truth in truths:
            b = np.load(os.path.join(RD, "v9", f"{base}_{truth}.npz"))
            fine = b["fine"]
            assert np.allclose(b["zs"], zs)
            R = st[f"R_{truth}"]
            RdE = np.array([np.interp(np.log(fine), np.log(st["k"]), R[s, 0]) for s in range(len(zs))])
            REE = np.array([np.interp(np.log(fine), np.log(st["k"]), R[s, 1]) for s in range(len(zs))])
            assert np.all(RdE < 0) and np.all(REE > 0), "truth ratio changes sign: division unsafe"
            gi = np.zeros((3, len(zs), len(fine)))
            ii = np.zeros((6, len(zs), len(fine)))
            bb = np.zeros((6, len(zs), len(fine)))
            for s, z in enumerate(zs):
                gi[:, s], ii[:, s], bb[:, s] = term_ratios(tm, float(z), fine)
            T_GI = np.einsum("msj,sjn->msn", gi/RdE[None], b["R_GI"])
            T_II = np.einsum("msj,sjn->msn", ii/REE[None], b["R_II"])
            # B modes: C_BB enters xi_+ as C_EE does and xi_- with the opposite sign
            sgn = np.zeros(b["R_II"].shape[-1])
            sgn[:NXI//2], sgn[NXI//2:NXI] = 1.0, -1.0
            T_II += np.einsum("msj,sjn->msn", bb/REE[None], b["R_II"])*sgn
            h = np.load(os.path.join(HERE, f"hm_templates_{probe}_{truth}.npz"))
            e1 = np.abs(T_GI[0] - h["N_GI"]).max()/np.abs(h["N_GI"]).max()
            e2 = np.abs(T_II[0] - h["N_II"]).max()/np.abs(h["N_II"]).max()
            out = os.path.join(HERE, f"tatt_templates_{probe}_{truth}.npz")
            np.savez(out, zs=zs, fine=fine, GI_names=GI_NAMES, II_names=II_NAMES, T_GI=T_GI, T_II=T_II,
                     term_GI=gi, term_II=ii, term_BB=bb, about=__doc__)
            # size of each term at the TATT best fit scale (A1 3.4, A2 -1, bTA 0.5, z = 1)
            s1 = int(np.argmin(abs(zs - 1.0)))
            j = [int(np.argmin(abs(fine - k))) for k in (0.1, 1.0)]
            print(f"{probe} {truth}: a1 rows vs hm_templates N_GI {e1:.1e}, N_II {e2:.1e}; "
                  f"z=1 GI terms/P_NL at k 0.1, 1: {np.round(gi[:, s1][:, j], 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
