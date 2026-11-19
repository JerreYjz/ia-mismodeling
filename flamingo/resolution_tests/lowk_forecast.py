"""Gaussian forecast of the S/N of P_hE per k bin at low k, z = 0, for the L1 (681 Mpc/h) and L2p8
(1906.8 Mpc/h) boxes. Mode counts are exact on the box's Fourier grid, with the pipeline's binning
(30 log bins, 0.003 to 20 h/Mpc); sigma^2 = (P_hE^2 + P_hh P_EE) / N_modes with shot and shape noise.

Cases: L1_m8 as measured (10.0 < log M* < 10.41); L2p8 if it had the L1_m8 galaxies ("ideal");
L2p8 with the measured L1_m9 numbers (lowmass_L1m9). Also prints the measured L2p8_m9 high-mass
(M* > 4e11) S/N for comparison. Writes lowk_forecast.json.
Needs ia_halofit_v2 (cosmology.py, stitch.py) for the CAMB/halofit P_NL.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/yijiezhu/Downloads/ia_halofit_v2")
from cosmology import Cosmology  # noqa: E402
from stitch import camb_power  # noqa: E402

he = np.loadtxt(os.path.join(HERE, "restest_L1m9", "L2p8_m9", "PS", "PhE0_sn78_nfold1.dat"))
print("measured, L2p8_m9 high mass (M* > 4e11 Msun):")
for r in he[:7]:
    print(f"  k={r[0]:.4f}  -PhE={-r[1]:8.1f} +- {r[2]:7.1f}  S/N={-r[1] / r[2]:5.1f}")
C = Cosmology(h=0.681, Om=0.304611, Ob=0.0486, ns=0.967, sigma8=0.807, linear_source="camb")
Pnl, _ = camb_power(C, [0.0], kmax=1.0)
edges = np.logspace(np.log10(0.003), np.log10(20), 31)
L1, L2 = 681.0, 1906.8
CASES = {  # name: (box, N_gal, bias, a_nla, e_rms); numbers from lowmass_L1m9/results.json
    "L1_m8": (L1, 2.031e6, 1.147, 1.864, 0.250),
    "L2p8, L1_m8 galaxies (ideal)": (L2, 2.031e6 * (L2 / L1) ** 3, 1.147, 1.864, 0.250),
    "L2p8_m9, measured m9 galaxies": (L2, 3.033e6 * (L2 / L1) ** 3, 0.966, 0.844, 0.372),
}
out = {}
for name, (L, ngal, b, anla, erms) in CASES.items():
    n = ngal / L ** 3
    kf = 2 * np.pi / L
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo > 0.07:
            break
        nmax = int(hi / kf) + 1
        g = np.arange(-nmax, nmax + 1)
        kk = kf * np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2)
        nm = np.sum((kk >= lo) & (kk < hi)) / 2
        if nm < 1:
            continue
        k = np.sqrt(lo * hi)
        P = float(Pnl(k, 0.0))
        A = anla * 0.0134 * C.Om
        phe, phh, pee = b * A * P, b * b * P + 1 / n, A * A * P + erms ** 2 / 2 / n
        rows.append((k, nm, phe / np.sqrt((phe ** 2 + phh * pee) / nm)))
    out[name] = {"L": L, "k_f": kf, "k": [r[0] for r in rows], "modes": [r[1] for r in rows],
                 "sn": [r[2] for r in rows]}
    print(f"\n{name}: k_f = {kf:.4f} h/Mpc")
    for k, nm, sn in rows:
        print(f"  k = {k:.4f}  modes = {nm:6.0f}  S/N(P_hE) = {sn:5.1f}")
json.dump(out, open(os.path.join(HERE, "lowk_forecast.json"), "w"), indent=1)
