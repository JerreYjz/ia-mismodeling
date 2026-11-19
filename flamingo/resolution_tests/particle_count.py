"""Star/DM particle counts vs stellar mass in FLAMINGO SOAP z=0 catalogues (sample of rows)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extraction"))
from common import open_hdf5  # noqa: E402

NROW = 1 << 22
RUNS = {
    "L1_m8": "cosma:/FLAMINGO/L1_m8/L1_m8/SOAP-HBT/halo_properties_0078.hdf5",
    "L1_m9": "cosma:/FLAMINGO/L1_m9/L1_m9/SOAP-HBT/halo_properties_0077.hdf5",
    "L2p8_m9": "cosma:/FLAMINGO/L2p8_m9/L2p8_m9/SOAP-HBT/halo_properties_0078.hdf5",
}
out = {}
for run, path in RUNS.items():
    t0 = time.time()
    try:
        f = open_hdf5(path)
    except Exception as e:  # noqa: BLE001
        print(run, "open failed", e)
        continue
    z = float(np.asarray(f["Header"].attrs["Redshift"]).reshape(-1)[0])
    n = f["BoundSubhalo/StellarMass"].shape[0]
    m = np.asarray(f["BoundSubhalo/StellarMass"][:NROW]) * 1e10
    ns = np.asarray(f["BoundSubhalo/NumberOfStarParticles"][:NROW])
    ndm = np.asarray(f["BoundSubhalo/NumberOfDarkMatterParticles"][:NROW])
    mdm_tot = np.asarray(f["BoundSubhalo/DarkMatterMass"][:NROW]) * 1e10 if "DarkMatterMass" in f["BoundSubhalo"].keys() else None
    cen = np.asarray(f["InputHalos/IsCentral"][:NROW])
    thr = {}
    for ds in ["StellarInertiaTensorNoniterative", "DarkMatterInertiaTensorNoniterative"]:
        a = f["BoundSubhalo/" + ds].attrs
        thr[ds] = {k: str(np.asarray(a[k]).reshape(-1)[0]) for k in a.keys() if "Mask" in k}
    a = f["ProjectedAperture/30kpc/projz/ProjectedStellarInertiaTensor"].attrs if "ProjectedStellarInertiaTensor" in f["ProjectedAperture/30kpc/projz"].keys() else {}
    thr["Proj30"] = {k: str(np.asarray(a[k]).reshape(-1)[0]) for k in a.keys() if "Mask" in k}
    good = ns >= 20
    mstar_p = np.median(m[good] / ns[good])
    mdm_p = np.median(mdm_tot[ndm > 20] / ndm[ndm > 20]) if mdm_tot is not None else np.nan
    rows = []
    for lo in np.arange(9.5, 11.6, 0.25):
        s = (np.log10(np.clip(m, 1, None)) >= lo) & (np.log10(np.clip(m, 1, None)) < lo + 0.25)
        sc = s & (cen == 1)
        rows.append(dict(logM=lo + 0.125, n=int(s.sum()),
                         Nstar_med=float(np.median(ns[s])) if s.any() else np.nan,
                         frac_ge100=float(np.mean(ns[s] >= 100)) if s.any() else np.nan,
                         NDM_cen_med=float(np.median(ndm[sc])) if sc.any() else np.nan))
    out[run] = dict(z=z, nsub_total=int(n), nrow=NROW, mstar_particle=float(mstar_p),
                    mdm_particle=float(mdm_p), logM_at_100=float(np.log10(100 * mstar_p)),
                    thresholds=thr, bins=rows)
    print(f"== {run}  z={z:.2f}  Nsub={n}  m*_part={mstar_p:.3e}  mDM_part={mdm_p:.3e}  "
          f"log M*(100 stars)={np.log10(100 * mstar_p):.2f}  ({time.time() - t0:.0f}s)")
    print("   thresholds:", thr)
    for r in rows:
        print(f"   logM*={r['logM']:.2f} n={r['n']:7d} N*_med={r['Nstar_med']:7.0f} "
              f"f(N*>=100)={r['frac_ge100']:.2f} NDM_cen_med={r['NDM_cen_med']:7.0f}")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
