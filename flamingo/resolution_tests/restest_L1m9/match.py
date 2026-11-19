"""Galaxy-by-galaxy shape match between L1_m8 and L1_m9 (same initial phases), M* > 4.05e11 Msun."""
import json
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

R = "/Users/yijiezhu/ia_power_flamingo/restest_soap_m9_vs_m8/L1_m8/Galaxies/galaxies_78.hdf5"  # not in the repo (galaxy catalogue)
# the L1_m9 catalogue is not in the repo: RESTEST_L1M9_CAT points to it
S = os.environ.get("RESTEST_L1M9_CAT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "L1_m9/Galaxies/galaxies_78.hdf5"))
a, b = h5py.File(R, "r"), h5py.File(S, "r")
L = float(a.attrs["BoxSize"])
pa, pb = a["Position"][:], b["Position"][:]
t = cKDTree(pb, boxsize=L)
d, j = t.query(pa)
ma, mb = a["Mass"][:], b["Mass"][:][j]
ok = (d < 0.2) & (np.abs(np.log10(ma / mb)) < 0.3)
print(f"L1_m8 {len(pa)} galaxies, L1_m9 {len(pb)}; matched {ok.sum()} ({ok.mean():.2f}) within 0.2 Mpc/h and 0.3 dex")
cen = a["IsCentral"][:][ok] == 1
out = {"matched": int(ok.sum()), "n_m8": len(pa), "n_m9": len(pb), "proj": {}}
for proj in "zxy":
    ea = np.stack([a[f"e1{proj}"][:][ok], a[f"e2{proj}"][:][ok]], 1)
    eb = np.stack([b[f"e1{proj}"][:][j][ok], b[f"e2{proj}"][:][j][ok]], 1)
    R_ = np.sum(ea * eb) / np.sum(ea * ea)
    corr = np.sum(ea * eb) / np.sqrt(np.sum(ea * ea) * np.sum(eb * eb))
    th = 0.5 * np.degrees(np.abs(np.angle((ea[:, 0] + 1j * ea[:, 1]) * np.conj(eb[:, 0] + 1j * eb[:, 1]))))
    out["proj"][proj] = {"corr": float(corr), "pa_med": float(np.median(th)), "pa_med_cen": float(np.median(th[cen])), "pa_med_sat": float(np.median(th[~cen]))}
    print(f"  proj {proj}: <e8.e9>/<e8^2> = {R_:.3f}, corr = {corr:.3f}, e_rms m8 {np.sqrt(np.mean(ea**2)):.3f} "
          f"m9 {np.sqrt(np.mean(eb**2)):.3f}, median PA offset {np.median(th):.1f} deg "
          f"(centrals {np.median(th[cen]):.1f}, sats {np.median(th[~cen]):.1f})")
print("log M* matched median: m8 %.2f, m9 %.2f" % (np.log10(np.median(ma[ok]) * 1e10), np.log10(np.median(mb[ok]) * 1e10)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "match.json"), "w"), indent=1)
