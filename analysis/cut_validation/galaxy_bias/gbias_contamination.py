#!/usr/bin/env python3
"""The Roman 3x2pt data-vector change from nonlinear galaxy bias.

The 3x2pt vector of cosmolike (driver3x2.py: linear bias b_i = B1_FID, no
magnification, no point mass, Limber for gamma_t and w, the setup of
fisher_minimize.py) has gamma_t proportional to b_i P_NL and w to b_i^2 P_NL.
With the measured bias of flamingo_beta.py, P_gm = b beta_gm P_NL and
P_gg = b^2 beta_gg P_NL, so the data with nonlinear bias are cosmolike run with
the nonlinear power multiplied by beta (the shape of the FLAMINGO galaxies'
bias at each redshift, applied to every lens bin there):

    dD[gamma_t] = D(P_NL beta_gm) - D(P_NL),   dD[w] = D(P_NL beta_gg) - D(P_NL),

xi_+- untouched (dD = 0 there), IA off, D3A. beta is interpolated linearly in
z between the snapshots (held at the end values outside 0 to 4) and in ln k.
Checks printed: D(P_NL) against cosmolike's own vector, and a constant
beta = 1.1 raising gamma_t and w by exactly 10 percent.

Variants (name -> file gbias_dv_<name>.npz, read by fisher_minimize.py --gbias):
    flam      beta_gm on gamma_t and beta_gg on w (the default)
    flamgg    w only (gamma_t rows zero)
    flamgm    gamma_t only (w rows zero)

    python gbias_contamination.py [beta_tables.npz]       (cocoa Python)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "..", "roman_dchi2")
sys.path.insert(0, os.path.join(RD, "cocoa"))
from driver3x2 import Cosmolike3x2                                   # noqa: E402

SO = os.path.expanduser("~/cocoa/Cocoa/projects/roman_real/interface")
DATA = os.path.expanduser("~/cocoa/Cocoa/projects/roman_real/data")
NXI, NGT, NDV = 1080, 915, 2115


def beta_on(tab, name, z2, kh):
    """ln beta on cosmolike's (z2, k) grid, shape (nz, nk)."""
    zt, kt, bt = tab["z"], tab["k"], tab[name]
    lnk = np.log(np.clip(kh, kt[0], kt[-1]))
    onk = np.array([np.interp(lnk, np.log(kt), b) for b in bt])          # (nzt, nk)
    zc = np.clip(z2, zt[0], zt[-1])
    out = np.empty((len(z2), len(kh)))
    for j in range(len(kh)):
        out[:, j] = np.interp(zc, zt, onk[:, j])
    return np.log(out)


def main():
    tab = np.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "beta_tables.npz"))
    cwd = os.getcwd()
    cl = Cosmolike3x2(SO, DATA, threads=4)
    os.chdir(cwd)
    ci, h = cl.ci, cl.h
    ci.init_IA(ia_model=0, ia_redshift_evolution=3, ia_code=0)
    cl.sim = False
    k = 10.0**cl.lk
    kh = k/h
    lnPL = np.log(cl.PKL.P(cl.z2, k)) + np.log(h**3)
    lnPNL = np.log(cl.PKNL.P(cl.z2, k)) + np.log(h**3)
    lnPLcb = np.log(cl.PKLcb.P(cl.z2, k)) + np.log(h**3)
    zg = cl.z1[cl.z1 <= cl.z2[-1]]
    G = np.sqrt(cl.PKL.P(zg, 0.05)/cl.PKL.P(0, 0.05))*(1 + zg)
    G = np.asarray(G/(np.sqrt(cl.PKL.P(cl.z2[-1], 0.05)/cl.PKL.P(0, 0.05))*(1 + cl.z2[-1]))).ravel()

    def dv(lnN):
        ci.set_cosmology(omegam=cl.cosmo["omegam"], omegab=cl.cosmo["omegab"], omegan2=cl.omegan2,
                         H0=cl.cosmo["H0"], log10k_2D=cl.lk - np.log10(h), z_2D=cl.z2,
                         lnP_linear=lnPL.flatten(order="F"), lnP_linear_cb=lnPLcb.flatten(order="F"),
                         lnP_nonlinear=lnN.flatten(order="F"), G=G, z_G=zg, z_1D=cl.z1,
                         chi=cl.res.comoving_radial_distance(cl.z1)*h)
        cl._nuisance([0.0], [0.0], [0.0])
        return np.asarray(ci.compute_data_vector_masked(), dtype=float)

    D0 = dv(lnPNL)
    ref = cl.dv_native(0, A1=[0.0, 0.0], A2=[0.0, 0.0], BTA=[0.0])
    print(f"check: D(P_NL) vs cosmolike's own vector, max |rel diff| "
          f"{np.max(np.abs(D0 - ref)/np.maximum(np.abs(ref), 1e-30)):.2e}")
    D11 = dv(lnPNL + np.log(1.1))
    rel = (D11 - D0)[NXI:]/D0[NXI:]
    print(f"check: beta = 1.1 -> gamma_t and w change by {rel.min():.4f} to {rel.max():.4f} (0.1 expected)")
    Dgm = dv(lnPNL + beta_on(tab, "beta_gm", cl.z2, kh))
    Dgg = dv(lnPNL + beta_on(tab, "beta_gg", cl.z2, kh))
    gt, w = slice(NXI, NXI + NGT), slice(NXI + NGT, NDV)
    for name, use_gt, use_w in (("flam", True, True), ("flamgg", False, True), ("flamgm", True, False)):
        dD = np.zeros(NDV)
        if use_gt:
            dD[gt] = (Dgm - D0)[gt]
        if use_w:
            dD[w] = (Dgg - D0)[w]
        np.savez(os.path.join(HERE, f"gbias_dv_{name}.npz"), dD=dD, D0=D0, z=tab["z"], b1=tab["b1"],
                 source=tab["source"], about=__doc__)
    rg, rw = ((Dgm - D0)/D0)[gt].reshape(-1, 15), ((Dgg - D0)/D0)[w].reshape(-1, 15)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("w: fractional change per lens bin (rows) and angle (2.5 to 250 arcmin)")
    print(rw)
    print("gamma_t: fractional change, the diagonal-ish pairs (lens i, source i+1)")
    pairs = np.asarray(ci.get_gs_redshift_bins()).reshape(-1, 2)
    for p, (l, s) in enumerate(pairs):
        if s == min(l + 1, 7):
            print(int(l), int(s), rg[p])


if __name__ == "__main__":
    main()
