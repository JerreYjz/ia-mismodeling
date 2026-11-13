#!/usr/bin/env python3
"""Stage 1: galaxy positions and shapes from a FLAMINGO SOAP catalogue.

Replaces shape_tensor.py of the TNG pipeline. FLAMINGO's SOAP catalogue
already contains stellar inertia tensors, so no particle data is read. For
every subhalo with

    M_min < MassField * h < M_max      (10^10 Msun/h, the TNG convention)
    N_particles >= the SOAP mask threshold of ShapeField (100 for stars)
    [central / satellite cut]

this writes  <OutDirBase>/galaxies_<SnapNum>.hdf5  with

    Position    (N, 3) float64   comoving Mpc/h, wrapped into [0, BoxSize)
    e1z e2z     (N,)   float64   ellipticity projected along z (the LOS of ia_power.py)
    e1x e2x e1y e2y              same along x / y (3D tensors only, else NaN)
    Mass        (N,)   float64   MassField in 10^10 Msun/h
    IsCentral   (N,)   int8
    SOAPIndex   (N,)   int64     row in the SOAP file
    Tensor      (N, 6 or 3)      the SOAP tensor as stored
    Header attrs: BoxSize (Mpc/h), h, Redshift, SnapNum, ShapeField, MassField, ...

Ellipticities follow the TNG pipeline (galaxy_ds_field.py there):
    e1 = (I_aa - I_bb) / (I_aa + I_bb),   e2 = 2 I_ab / (I_aa + I_bb)
with (a, b) = (x, y) for z, (y, z) for x and (z, x) for y. SOAP 3D tensors
are stored as (xx, yy, zz, xy, xz, yz); projected ones (ProjectedAperture/
.../projz) as (xx, yy, xy), with the same (a, b) order as above.

Usage:  python soap_galaxies.py <parameterfile>

Parameter file tags:
  SoapFile     local path, or cosma:/FLAMINGO/<box>/<run>/SOAP-HBT/halo_properties_XXXX.hdf5
  OutDirBase   output directory
  SnapNum      snapshot number (only used in the output file name)
  M_min M_max  mass cut on MassField, in 10^10 Msun/h
Optional:
  ShapeField   default BoundSubhalo/StellarInertiaTensorNoniterative
  MassField    default BoundSubhalo/StellarMass
  Select       all | centrals | satellites          (default all)
  MinParticles extra cut on the ShapeField mask dataset (default 0 = SOAP threshold only)
  MaxRows      only read the first MaxRows subhalos (testing; default 0 = all)
"""

import os
import sys
import time

import numpy as np

from common import (ROWS_PER_READ, attr_scalar, flamingo_header, open_hdf5,
                    read_parameter, read_rows)

PARAM_TAGS = {"SoapFile": str, "OutDirBase": str, "SnapNum": int,
              "M_min": float, "M_max": float}
OPTIONAL = {
    "ShapeField": (str, "BoundSubhalo/StellarInertiaTensorNoniterative"),
    "MassField": (str, "BoundSubhalo/StellarMass"),
    "Select": (str, "all"),
    "MinParticles": (int, 0),
    "MaxRows": (int, 0),
}

# (a, b) axis pairs per projection, as in the TNG pipeline and SOAP's 2D tensors
PROJ_AXES = {"z": (0, 1), "x": (1, 2), "y": (2, 0)}
# index of I_ab in SOAP's flattened 3D tensor (xx, yy, zz, xy, xz, yz)
OFFDIAG = {(0, 1): 3, (1, 2): 5, (2, 0): 4}


def ellipticity_3d(t, proj):
    a, b = PROJ_AXES[proj]
    iaa, ibb, iab = t[:, a], t[:, b], t[:, OFFDIAG[a, b]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (iaa - ibb) / (iaa + ibb), 2.0 * iab / (iaa + ibb)


def ellipticity_2d(t):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (t[:, 0] - t[:, 1]) / (t[:, 0] + t[:, 1]), 2.0 * t[:, 2] / (t[:, 0] + t[:, 1])


def mask_dataset(dset, min_particles):
    """SOAP's own mask for a property: (dataset name, threshold)."""
    a = dset.attrs
    if "Mask Datasets" in a and int(attr_scalar(a, "Masked")):
        names = np.asarray(a["Mask Datasets"]).reshape(-1)
        name = names[0].decode() if isinstance(names[0], bytes) else str(names[0])
        return name, max(int(attr_scalar(a, "Mask Threshold")), min_particles)
    return None, min_particles


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: soap_galaxies.py <parameterfile>")
    p = read_parameter(argv[1], PARAM_TAGS, optional=OPTIONAL)
    select = p["Select"]
    if select not in ("all", "centrals", "satellites"):
        sys.exit(f"Select must be all, centrals or satellites, got '{select}'")

    t0 = time.time()
    f = open_hdf5(p["SoapFile"])
    box_mpc, h, z = flamingo_header(f)
    box = box_mpc * h                                   # Mpc/h
    nsub = int(attr_scalar(f["Header"].attrs, "NumSubhalos_Total"))
    nrows = min(nsub, p["MaxRows"]) if p["MaxRows"] > 0 else nsub
    print(f"BoxSize = {box_mpc:g} cMpc = {box:g} cMpc/h, h = {h}, z = {z:g}, "
          f"{nsub} subhalos (reading {nrows})")

    shape_ds = f[p["ShapeField"]]
    ncomp = shape_ds.shape[1]
    if ncomp == 3:
        if not p["ShapeField"].split("/")[-2].endswith("projz"):
            sys.exit("A projected (3-component) ShapeField must be a projz one: ia_power.py "
                     "uses z as the line of sight.")
    elif ncomp != 6:
        sys.exit(f"{p['ShapeField']} has {ncomp} components; expected 6 (3D) or 3 (projected)")
    mask_name, nmin = mask_dataset(shape_ds, p["MinParticles"])
    print(f"ShapeField = {p['ShapeField']} ({'3D' if ncomp == 6 else 'projected z'})")
    print(f"MassField  = {p['MassField']}, {p['M_min']:g} < M < {p['M_max']:g} [1e10 Msun/h]")
    print(f"particle cut: {mask_name} >= {nmin}" if mask_name else "no SOAP mask on ShapeField")

    out = {k: [] for k in ("Position", "Mass", "IsCentral", "SOAPIndex", "Tensor")}
    for s in range(0, nrows, ROWS_PER_READ):
        e = min(s + ROWS_PER_READ, nrows)
        mass = read_rows(f[p["MassField"]], s, e).astype(np.float64).reshape(-1) * h
        sel = (mass > p["M_min"]) & (mass < p["M_max"])
        if mask_name:
            sel &= read_rows(f[mask_name], s, e).reshape(-1) >= nmin
        cen = None
        if select != "all":
            cen = read_rows(f["InputHalos/IsCentral"], s, e).reshape(-1)
            sel &= (cen == 1) if select == "centrals" else (cen == 0)
        idx = np.flatnonzero(sel)
        if len(idx):
            out["Tensor"].append(read_rows(shape_ds, s, e, idx).astype(np.float64))
            out["Position"].append(read_rows(f["InputHalos/HaloCentre"], s, e, idx).astype(np.float64))
            out["IsCentral"].append((cen[idx] if cen is not None
                                     else read_rows(f["InputHalos/IsCentral"], s, e, idx)).astype(np.int8))
            out["Mass"].append(mass[idx])
            out["SOAPIndex"].append(idx.astype(np.int64) + s)
        print(f"rows {s}-{e}: {len(idx)} selected ({time.time() - t0:.0f} s)", flush=True)

    cat = {k: (np.concatenate(v) if v else np.empty((0,))) for k, v in out.items()}
    tensor = cat["Tensor"].reshape(-1, ncomp)
    pos = np.mod(cat["Position"].reshape(-1, 3) * h, box)   # comoving Mpc -> Mpc/h

    nan = np.full(len(tensor), np.nan)
    if ncomp == 6:
        e = {pr: ellipticity_3d(tensor, pr) for pr in "zxy"}
    else:
        e = {"z": ellipticity_2d(tensor), "x": (nan, nan), "y": (nan, nan)}
    # masked/empty tensors (all zero) give NaN; drop them here
    good = np.isfinite(e["z"][0]) & np.isfinite(e["z"][1])
    if not good.all():
        print(f"WARNING: dropping {np.count_nonzero(~good)} galaxies with a zero/invalid tensor")
    print(f"{np.count_nonzero(good)} galaxies selected "
          f"({np.count_nonzero(cat['IsCentral'][good] == 1)} centrals)")

    import h5py
    os.makedirs(p["OutDirBase"], exist_ok=True)
    fname = os.path.join(p["OutDirBase"], f"galaxies_{p['SnapNum']}.hdf5")
    with h5py.File(fname + ".part", "w") as g:
        g["Position"] = pos[good]
        for pr in "zxy":
            g[f"e1{pr}"] = e[pr][0][good]
            g[f"e2{pr}"] = e[pr][1][good]
        g["Mass"] = cat["Mass"][good]
        g["IsCentral"] = cat["IsCentral"][good]
        g["SOAPIndex"] = cat["SOAPIndex"][good]
        g["Tensor"] = tensor[good]
        a = g.attrs
        a["BoxSize"] = box
        a["h"] = h
        a["Redshift"] = z
        a["SnapNum"] = p["SnapNum"]
        a["SoapFile"] = p["SoapFile"]
        a["ShapeField"] = p["ShapeField"]
        a["MassField"] = p["MassField"]
        a["Select"] = select
        a["M_min"], a["M_max"] = p["M_min"], p["M_max"]
        a["RowsRead"] = nrows
    os.replace(fname + ".part", fname)
    print(f"wrote {fname}. Yatta!")


if __name__ == "__main__":
    main(sys.argv)
