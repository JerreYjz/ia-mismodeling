#!/usr/bin/env python3
"""Where does a white P_EE term land in Roman xi_+-?

Inject only the white II excess X_s (per snapshot, bias_white.py), with the
four combinations of extension outside the measured k range:
  low  = ratio : X/P_NL held at its k = 0.05 value below 0.05 (production)
  low  = white : X constant below 0.05
  high = ratio : X/P_NL held at its 17.7 value above (production)
  high = zero  : nothing above 17.7
and report xi_+- / sigma per theta bin (median over pairs), and Delta chi2 on
the masks. Also the same with the python Limber (Gaussian l taper at 6e4).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bias_analysis as B                                     # noqa: E402
from cosmolike_xi_ia import KW, KMIN, write_set              # noqa: E402
import roman_limber as RL                                     # noqa: E402

MASKS = (0.25, 0.35, 0.5)
THETA = np.geomspace(2.5, 250, 16)
TH = np.sqrt(THETA[:-1]*THETA[1:])


def main():
    data = sys.argv[1]
    R = np.load("run/cosmolike_xi_ia.npz")
    cov = np.load("analysis/cov_gauss_d3a.npy")
    sd = np.sqrt(np.diag(cov))
    M1, K1 = R["masks_n1"], list(R["K_n1"])
    f = np.load("../v2/fits.npz")
    zs, sns, kd = f["zs"], f["sns"], f["k"][0]
    y, err, b, Pf, Ns = f["y"], f["err"], f["b"], f["PNL"], f["Ns_true"]
    cl = B.CL(os.path.join(HERE, "lib"), data)
    h = cl.h
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])
    sel = (kd >= KMIN) & (kd <= 0.5)
    X_meas = (y[:, 1] - Ns[:, None]) - (y[:, 0]/b)**2/Pf
    dX = np.sqrt(err[:, 1]**2 + (2*(y[:, 0]/b)*err[:, 0]/b/Pf)**2)
    Xw = np.array([np.sum(X_meas[s, sel]/dX[s, sel]**2)/np.sum(1/dX[s, sel]**2) for s in range(13)])
    # FLAMINGO's P_NL (fit convention) on KW for the ratio extension
    Pf_w = np.array([np.interp(np.log(KW), np.log(kd), np.log(Pf[s])) for s in range(13)])
    Pf_w = np.exp(Pf_w)
    k_lo, k_hi = kd[kd >= KMIN][0], kd[-1]

    def q(v, m):
        return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))

    def tables(low, high):
        PEE = np.zeros((13, len(KW)))
        for s in range(13):
            ratio = Xw[s]/Pf_w[s]                     # white in observed units = X / P_NL ratio
            r = ratio.copy()
            if low == "ratio":
                r[KW < k_lo] = Xw[s]/np.interp(np.log(k_lo), np.log(kd), Pf[s])
            if high == "ratio":
                r[KW > k_hi] = Xw[s]/Pf[s][-1]
            elif high == "zero":
                r[KW > k_hi] = 0.0
            PEE[s] = r*PNLw[s]
        return PEE
    print("cosmolike, white II excess only. xi/sigma median over bin pairs, per theta bin; Delta chi2 on masks")
    res = {}
    for low in ("ratio", "white"):
        for high in ("ratio", "zero"):
            p = write_set("run/tables_diag", f"white_{low}_{high}", sns, np.zeros((13, len(KW))), tables(low, high))
            cl.at(B.FID)
            cl.use_sim_tables(p, sns, [float(z) for z in zs], 1)
            v = cl.xi_sim() - xi_gg
            res[(low, high)] = v
            r = (v/sd).reshape(2, 36, 15)
            print(f"  low {low:5s} high {high:5s}: chi2 all {q(v, np.ones(1080, bool)):7.2f}; masks "
                  + ", ".join(f"{K0}/D {q(v, M1[K1.index(K0)]):.3f}" for K0 in MASKS))
            print("     xi+/sigma: " + " ".join(f"{np.median(np.abs(r[0, :, t])):.3f}" for t in range(15)))
            print("     xi-/sigma: " + " ".join(f"{np.median(np.abs(r[1, :, t])):.3f}" for t in range(15)))
    print("  theta bins [arcmin]: " + " ".join(f"{t:.1f}" for t in TH))
    np.savez("run/xi_white_only.npz", **{f"{a}_{b}": v for (a, b), v in res.items()})
    # python Limber, same term, Gaussian taper at l = 6e4
    cosmo = RL.Cosmology(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, As=2.099e-9, mnu=0.06)
    S = RL.RomanShear(cosmo, os.path.join(data, "example1.nz"))
    zero = lambda K, Z: 0*K
    for low, high in (("ratio", "ratio"), ("white", "zero")):
        PEE = tables(low, high)
        CE, CB = S.cl({"dE": zero, "EE": RL.ratio_table(zs, KW, PEE/PNLw), "BB": zero}, gg=False)
        v = S.xi(CE, CB)
        r = (v/sd).reshape(2, 36, 15)
        print(f"  python Limber low {low} high {high}: chi2 all {q(v, np.ones(1080, bool)):7.2f}; masks "
              + ", ".join(f"{K0}/D {q(v, M1[K1.index(K0)]):.3f}" for K0 in MASKS))
        print("     xi+/sigma: " + " ".join(f"{np.median(np.abs(r[0, :, t])):.3f}" for t in range(15)))


if __name__ == "__main__":
    main()
