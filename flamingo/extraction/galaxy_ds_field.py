#!/usr/bin/env python3
"""Stage 2: galaxy number-density and shape fields.

Reads the galaxy catalogue written by soap_galaxies.py, optionally applies a
tighter mass / central-satellite cut, assigns each galaxy to its grid cell
(NGP, box folded ``nfold`` times) and writes three Fourier-space grids (raw
complex64, (ng, ng, ng//2+1)), with the same names as the TNG pipeline:

    galdens_fixng_ms_sub_red_sn<sn>_nfold<nfold>.bin   delta_g = n/<n> - 1
    e1z_fixng_ms_sub_red_sn<sn>_nfold<nfold>.bin       sum(e1) * ng^3 / N_gal
    e2z_fixng_ms_sub_red_sn<sn>_nfold<nfold>.bin       sum(e2) * ng^3 / N_gal

plus the shape-noise statistics file used by ia_power.py,

    hinfo_fixng_ms_sub_red_sn<sn>_nfold<nfold>.dat

Usage:  python galaxy_ds_field.py <parameterfile> <nfold>

Parameter file tags:
  GalCatDir   directory holding galaxies_<SnapNum>.hdf5 (soap_galaxies.py output)
  OutDirBase  output directory
  SnapNum     snapshot number
  Ngrid       grid size per side
Optional:
  M_min M_max  further mass cut in 10^10 Msun/h (default: keep all)
  Select       all | centrals | satellites (default all)
  ShapeSuffix  read e1z<suffix>/e2z<suffix>, e.g. _2re or _red from particle_shapes.py
               (default: the catalogue's primary e1z/e2z)
"""

import os
import sys

import numpy as np

from common import (fft_r2c_inplace, new_grid, periodic_index, read_parameter,
                    save_grid)

PARAM_TAGS = {"GalCatDir": str, "OutDirBase": str, "SnapNum": int, "Ngrid": int}
OPTIONAL = {"M_min": (float, 0.0), "M_max": (float, np.inf), "Select": (str, "all"),
            "ShapeSuffix": (str, "")}


def read_catalogue(cat_dir, snap_num):
    import h5py

    fname = os.path.join(cat_dir, f"galaxies_{snap_num}.hdf5")
    print(f"read galaxies from {fname}")
    if not os.path.isfile(fname):
        sys.exit(f"{fname} not found; run soap_galaxies.py first.")
    with h5py.File(fname, "r") as f:
        cat = {k: f[k][()] for k in f}
        attrs = dict(f.attrs)
    return cat, attrs


def build_field(ng, flat_cells, values, norm, subtract_one):
    """NGP-deposit ``values`` into a fresh grid, normalise, FFT in place."""
    grid = new_grid(ng)
    np.add.at(grid.reshape(-1), flat_cells, values)
    real = grid[..., :ng]
    real *= np.float32(norm)
    if subtract_one:
        real -= np.float32(1.0)
    return fft_r2c_inplace(grid)


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: galaxy_ds_field.py <parameterfile> <nfold>")
    p = read_parameter(argv[1], PARAM_TAGS, optional=OPTIONAL)
    nfold = int(argv[2])
    if nfold <= 0:
        sys.exit("nfold must be a positive integer")
    sn, ng = p["SnapNum"], p["Ngrid"]

    cat, attrs = read_catalogue(p["GalCatDir"], sn)
    box_size = float(attrs["BoxSize"])            # Mpc/h
    boxlen = box_size / nfold
    print(f"BoxSize = {box_size:g} Mpc/h, boxlen = {boxlen:g} Mpc/h, "
          f"cell = {boxlen / ng:g} Mpc/h, z = {float(attrs['Redshift']):g}")
    print(f"catalogue: {attrs['ShapeField']}, {attrs['MassField']}, Select={attrs['Select']}")

    sfx = p["ShapeSuffix"]
    if sfx:
        if f"e1z{sfx}" not in cat:
            sys.exit(f"catalogue has no e1z{sfx}; ShapeSuffix must match a particle_shapes.py output")
        for name in ("e1z", "e2z", "e1x", "e2x", "e1y", "e2y"):
            cat[name] = cat[f"{name}{sfx}"]
        print(f"using shapes e1z{sfx}/e2z{sfx}")
    ngal = len(cat["Mass"])
    keep = (cat["Mass"] > p["M_min"]) & (cat["Mass"] < p["M_max"])
    if p["Select"] == "centrals":
        keep &= cat["IsCentral"] == 1
    elif p["Select"] == "satellites":
        keep &= cat["IsCentral"] == 0
    elif p["Select"] != "all":
        sys.exit(f"Select must be all, centrals or satellites, got '{p['Select']}'")
    finite = np.isfinite(cat["e1z"]) & np.isfinite(cat["e2z"])
    ngal_nan = int(np.count_nonzero(keep & ~finite))
    used = keep & finite
    ngal_used = int(np.count_nonzero(used))
    print(f"ngal={ngal}  passing cut={int(keep.sum())}  NaN={ngal_nan}  used={ngal_used}")
    if ngal_used == 0:
        sys.exit("No galaxies pass the selection; nothing to grid.")

    pos = cat["Position"][used].astype(np.float64)
    cell = periodic_index(pos * (ng / boxlen) + 1e-10, ng)
    rowlen = 2 * (ng // 2 + 1)
    flat_cells = (cell[:, 0] * ng + cell[:, 1]) * rowlen + cell[:, 2]
    norm = float(ng) ** 3 / ngal_used

    tag = f"_fixng_ms_sub_red_sn{sn}_nfold{nfold}"
    out_dir = p["OutDirBase"]

    # One grid at a time keeps the peak memory at ~1 field.
    ones = np.ones(ngal_used, dtype=np.float32)
    save_grid(build_field(ng, flat_cells, ones, norm, True),
              os.path.join(out_dir, f"galdens{tag}.bin"))
    save_grid(build_field(ng, flat_cells, cat["e1z"][used].astype(np.float32), norm, False),
              os.path.join(out_dir, f"e1z{tag}.bin"))
    save_grid(build_field(ng, flat_cells, cat["e2z"][used].astype(np.float32), norm, False),
              os.path.join(out_dir, f"e2z{tag}.bin"))

    def rms(name):
        e = cat[name][used].astype(np.float64)
        return float(np.sqrt(np.mean(e ** 2))) if np.isfinite(e).all() else 0.0

    hinfo = os.path.join(out_dir, f"hinfo{tag}.dat")
    with open(hinfo, "w") as f:
        f.write("#ngal ngal_nan ngal_used e1z_rms e2z_rms e1x_rms e2x_rms "
                "e1y_rms e2y_rms e1J_rms e2J_rms boxsize \n")
        vals = [rms(n) for n in ("e1z", "e2z", "e1x", "e2x", "e1y", "e2y")] + [0.0, 0.0]
        # boxsize (Mpc/h) last, so ia_power.py can check its config against the catalogue
        f.write(f"{ngal} {ngal_nan} {ngal_used} " + " ".join(f"{v:g}" for v in vals)
                + f" {box_size:.6f}\n")
    print(f"wrote {hinfo}")
    print("Hooray! Done!")


if __name__ == "__main__":
    main(sys.argv)
