#!/usr/bin/env python3
"""Columns of the one-loop galaxy-bias model for the cut scans (gbias_perbin.py --oneloop).

The 3x2pt analysis models gamma_t and w with linear bias. Cosmolike also has
the one-loop bias expansion (cosmo2D.c),

    P_gm = b1 P_NL + 1/2 (b2 P_d1d2 + bs2 P_d1s2 + b3 P_d1p3),
    P_gg = b1^2 P_NL + b1 b2 P_d1d2 + b2^2/4 P_d2d2 + b1 bs2 P_d1s2 + ...,

switched on when any b2 != 0. Here b2 is free per lens bin and the tidal
and third-order biases follow b1, bs2 = -4/7 (b1 - 1) and b3 = b1 - 1
(bias_model [0, 0, 1, 1, 0, 0]: the coevolution relations of 1405.1447).

At the reference point (D3A, b1 = B1_FID, b2 = b2_ref, IA off) this writes,
with central differences:
    d_b2[i]   dv/db2_i                                    (2115, 8)
    d_b1[i]   dv/db1_i of the one-loop model minus that of linear bias
              (the one-loop correction to the b1 columns)  (2115, 8)
    dv1       v_oneloop(ref) - v_linear(ref)               (2115,)
b2_ref is by default the Lazeyras et al. (1511.01096) relation b2(b1) at
B1_FID, the natural size of b2 for halos of that bias; --b2ref takes 8
values instead. The b2^2 term makes the model quadratic in b2, so the
linear shifts of the scans hold only near b2_ref: the scans are run at the
b2 fitted from the first pass (a Gauss-Newton step), and the move of b2 in
the scans checks how far they are from it. The cosmology and nuisance
columns stay those of linear bias (the one-loop terms' own cosmology
dependence is neglected: a correction to a correction). Checks printed: the
linear b1 column against the Jacobian of gbias_scan.py, the xi_+- rows
untouched, and the quadratic term (dv at b2_ref +- 1 against the linear
extrapolation).

    python gbias_oneloop.py [--b2ref b1,...,b8] [--out name]   (cocoa Python; writes gbias_oneloop.npz)

The scans use
    python gbias_oneloop.py --b2ref " -0.23,-0.28,-0.32,-0.33,-0.24,0.25,1.19,0.78"
(the b2 fitted, from the Lazeyras reference, at the uniform cuts 0.15).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "..", "roman_dchi2")
sys.path.insert(0, os.path.join(RD, "cocoa"))
sys.path.insert(0, os.path.dirname(HERE))
from driver3x2 import Cosmolike3x2                                   # noqa: E402
from fisher_minimize import B1_FID                                   # noqa: E402

SO = os.path.expanduser("~/cocoa/Cocoa/projects/roman_real/interface")
DATA = os.path.expanduser("~/cocoa/Cocoa/projects/roman_real/data")
NXI, NGT, NDV, NB = 1080, 915, 2115, 8
H_B = 0.02


def lazeyras(b1):
    return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3


def main():
    cwd = os.getcwd()
    cl = Cosmolike3x2(SO, DATA, threads=4)
    os.chdir(cwd)
    ci = cl.ci
    ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    ci.init_bias(bias_model=[0, 0, 1, 1, 0, 0])
    cl.sim = False
    cl.dv_native(0, A1=[0.0, 0.0], A2=[0.0, 0.0], BTA=[0.0])     # set the D3A cosmology once

    def dv(b1, b2):
        ci.set_nuisance_bias(B1=list(b1), B2=list(b2), B_MAG=[0.0]*NB, B3nl=[0.0]*NB, BK=[0.0]*NB)
        return np.asarray(ci.compute_data_vector_masked(), dtype=float)

    a = sys.argv[1:]
    b1 = np.array(B1_FID, float)
    b2r = lazeyras(b1)
    if "--b2ref" in a:
        b2r = np.array([float(x) for x in a[a.index("--b2ref") + 1].split(",")])
    out = a[a.index("--out") + 1] if "--out" in a else "gbias_oneloop.npz"
    zero = np.zeros(NB)
    vlin = dv(b1, zero)
    v1 = dv(b1, b2r)
    d_b2 = np.zeros((NDV, NB))
    d_b1 = np.zeros((NDV, NB))
    d_b1_lin = np.zeros((NDV, NB))
    for i in range(NB):
        e = np.zeros(NB)
        e[i] = 1.0
        d_b2[:, i] = (dv(b1, b2r + H_B*e) - dv(b1, b2r - H_B*e))/(2*H_B)
        g1 = (dv(b1 + H_B*e, b2r) - dv(b1 - H_B*e, b2r))/(2*H_B)
        d_b1_lin[:, i] = (dv(b1 + H_B*e, zero) - dv(b1 - H_B*e, zero))/(2*H_B)
        d_b1[:, i] = g1 - d_b1_lin[:, i]
    dv1 = v1 - vlin
    print(f"b2_ref: {np.round(b2r, 3).tolist()}")
    print(f"check: xi rows untouched, max |dv1[xi]| / max|v[xi]| = "
          f"{np.abs(dv1[:NXI]).max()/np.abs(vlin[:NXI]).max():.1e}, max |d_b2[xi]| {np.abs(d_b2[:NXI]).max():.1e}")
    # quadratic term: v at b2_ref + 1 against the linear extrapolation, relative to the one-loop change
    for i in (0, 7):
        e = np.zeros(NB)
        e[i] = 1.0
        full = dv(b1, b2r + e) - v1
        lin = d_b2[:, i]
        rows = slice(NXI, NDV)
        print(f"check: bin {i + 1}, b2 + 1: |full - linear| / |full| = "
              f"{np.linalg.norm((full - lin)[rows])/np.linalg.norm(full[rows]):.3f} (the b2^2 term)")
    rg = (dv1/vlin)[NXI:NXI + NGT]
    rw = (dv1/vlin)[NXI + NGT:].reshape(NB, 15)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("w: one-loop minus linear at b2_ref, fractional, per lens bin and angle (2.5 to 250 arcmin)")
    print(rw)
    print(f"gamma_t: one-loop minus linear, fractional, range {rg.min():.3f} to {rg.max():.3f}")
    np.savez(os.path.join(HERE, out), d_b2=d_b2, d_b1=d_b1, d_b1_lin=d_b1_lin, dv1=dv1,
             vlin=vlin, b2_ref=b2r, b1=b1, about=__doc__)


if __name__ == "__main__":
    main()
