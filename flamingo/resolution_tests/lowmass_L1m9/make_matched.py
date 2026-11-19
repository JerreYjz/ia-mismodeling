"""Same-galaxy samples: L1_m8 galaxies (10.0 < log M* < 10.41) and their L1_m9 counterparts.

Counterpart = nearest m9 galaxy (wide 9.8-10.7 catalogue) within 0.2 Mpc/h and 0.3 dex in M*; pairs
sharing an m9 galaxy are dropped. Writes L1_m8_match/ and L1_m9_match/ catalogues (all datasets
subset, attributes copied), each read unchanged by galaxy_ds_field.py.
"""
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
M8 = "/Users/yijiezhu/ia_power_flamingo/run_L1_m8_particles/results/Galaxies/galaxies_78.hdf5"
M9 = os.path.join(HERE, "L1_m9/Galaxies/galaxies_78.hdf5")
a, b = h5py.File(M8, "r"), h5py.File(M9, "r")
L = float(a.attrs["BoxSize"])
d, j = cKDTree(b["Position"][:], boxsize=L).query(a["Position"][:])
ok = (d < 0.2) & (np.abs(np.log10(a["Mass"][:] / b["Mass"][:][j])) < 0.3)
uj, cnt = np.unique(j[ok], return_counts=True)
ok &= ~np.isin(j, uj[cnt > 1])
i8 = np.flatnonzero(ok)
i9 = j[ok]
print(f"m8 sample {len(d)}, matched {len(i8)} ({len(i8) / len(d):.3f}); m9 catalogue {b['Mass'].shape[0]}")
for src, idx, name in [(a, i8, "L1_m8_match"), (b, i9, "L1_m9_match")]:
    with h5py.File(os.path.join(HERE, name, "Galaxies", "galaxies_78.hdf5"), "w") as g:
        for k in src:
            g[k] = src[k][()][idx]
        for k, v in src.attrs.items():
            g.attrs[k] = v
        g.attrs["Matched"] = "same-galaxy sample, see make_matched.py"
