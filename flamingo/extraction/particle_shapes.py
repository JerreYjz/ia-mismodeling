#!/usr/bin/env python3
"""Stage 1 (particle version): TNG-style galaxy shapes from FLAMINGO star particles.

Use this instead of soap_galaxies.py when SOAP has no stellar tensors in the
mass range of interest (L1_m8 SOAP stores them only for M* > 4e11 Msun).

Galaxies are selected from the SOAP catalogue exactly as in soap_galaxies.py
(M_min < MassField * h < M_max in 10^10 Msun/h, NumberOfStarParticles >=
MinStars, optional central/satellite cut). Their shapes are then measured from
every bound star particle (PartType4/GroupNr_bound == InputHalos/HaloCatalogueIndex)
with the tensor of the TNG pipeline (Galaxy_Shape_Tensor/eigen.c, no-iteration
branch):

    I_jk = sum_i m_i r_j r_k / r^2,    r = x_i - HaloCentre (periodic)

i.e. the reduced tensor of all bound stars, no aperture. A particle sitting
exactly on the centre (r = 0) is skipped; in the C code it turned the whole
tensor into NaN. The tensor is stored divided by sum(m), which does not change
the ellipticities.

The same pass also accumulates
  * the unreduced tensor of ALL bound stars, I_jk = sum m r_j r_k / sum m (suffix _all).
    This is what the TNG shape code actually computed (Galaxy_Shape_Tensor/eigen_iteration.c,
    run by run_iteration.sh: its radius cut and 1/r^2 weight are commented out, so the
    iteration does not change the result); it is the default (ShapeDef unreduced_all);
  * the unreduced tensor of the bound stars within 2 R_half,* (suffix _2re).
All three are written; ShapeDef picks the one copied to the unsuffixed e1z/e2z/... read by
galaxy_ds_field.py.

The star particles are streamed in blocks of ChunkRows with NThreads parallel
connections. Partial sums are checkpointed every CheckpointEvery blocks to
<OutDirBase>/shapes_<SnapNum>.ckpt.npz, so an interrupted run resumes where it
stopped. Output: <OutDirBase>/galaxies_<SnapNum>.hdf5 in the soap_galaxies.py
format (read unchanged by galaxy_ds_field.py), plus NStarsUsed per galaxy.

Usage:  python particle_shapes.py <parameterfile>

Parameter file tags:
  SoapFile     cosma:/FLAMINGO/<box>/<run>/SOAP-HBT/halo_properties_XXXX.hdf5 (or local)
  SnapFile     cosma:/FLAMINGO/<box>/<run>/snapshots/flamingo_XXXX/flamingo_XXXX.hdf5
  OutDirBase   output directory
  SnapNum      snapshot number
  M_min M_max  mass cut on MassField, in 10^10 Msun/h
Optional:
  MassField        default BoundSubhalo/StellarMass
  Select           all | centrals | satellites          (default all)
  MinStars         minimum BoundSubhalo/NumberOfStarParticles (default 100)
  NThreads         parallel streams (default 4)
  ChunkRows        star particles per block (default 2^23)
  CheckpointEvery  blocks between checkpoints (default 20)
  MaxParticles     only read the first MaxParticles stars (testing; default 0 = all)
  ShapeDef         unreduced_all | reduced_all | unreduced_2re: tensor for e1z/e2z (default unreduced_all = TNG)
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import (ROWS_PER_READ, attr_scalar, flamingo_header, open_hdf5,
                    read_parameter, read_rows)
from soap_galaxies import ellipticity_3d

PARAM_TAGS = {"SoapFile": str, "SnapFile": str, "OutDirBase": str, "SnapNum": int,
              "M_min": float, "M_max": float}
OPTIONAL = {
    "MassField": (str, "BoundSubhalo/StellarMass"),
    "Select": (str, "all"),
    "MinStars": (int, 100),
    "NThreads": (int, 4),
    "ChunkRows": (int, 1 << 23),
    "CheckpointEvery": (int, 20),
    "MaxParticles": (int, 0),
    "ShapeDef": (str, "unreduced_all"),
}
SHAPE_SUFFIX = {"unreduced_all": "_all", "reduced_all": "_red", "unreduced_2re": "_2re"}
# flattened tensor order, as SOAP: (xx, yy, zz, xy, xz, yz)
PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def select_galaxies(f, p, h):
    """Rows of the SOAP catalogue passing the cuts, with their ids and centres."""
    n = int(attr_scalar(f["Header"].attrs, "NumSubhalos_Total"))
    out = {k: [] for k in ("SOAPIndex", "HaloCatalogueIndex", "Centre", "Mass", "IsCentral",
                           "NStarsSOAP", "RHalf")}
    for s in range(0, n, ROWS_PER_READ):
        e = min(s + ROWS_PER_READ, n)
        mass = read_rows(f[p["MassField"]], s, e).astype(np.float64).reshape(-1) * h
        nstar = read_rows(f["BoundSubhalo/NumberOfStarParticles"], s, e).reshape(-1)
        cen = read_rows(f["InputHalos/IsCentral"], s, e).reshape(-1)
        sel = (mass > p["M_min"]) & (mass < p["M_max"]) & (nstar >= p["MinStars"])
        if p["Select"] != "all":
            sel &= (cen == 1) if p["Select"] == "centrals" else (cen == 0)
        idx = np.flatnonzero(sel)
        out["SOAPIndex"].append(idx.astype(np.int64) + s)
        out["HaloCatalogueIndex"].append(
            read_rows(f["InputHalos/HaloCatalogueIndex"], s, e, idx).astype(np.int64).reshape(-1))
        out["Centre"].append(read_rows(f["InputHalos/HaloCentre"], s, e, idx).astype(np.float64))
        out["Mass"].append(mass[idx])
        out["IsCentral"].append(cen[idx].astype(np.int8))
        out["NStarsSOAP"].append(nstar[idx].astype(np.int64))
        out["RHalf"].append(   # comoving Mpc, like the coordinates
            read_rows(f["BoundSubhalo/HalfMassRadiusStars"], s, e, idx).astype(np.float64).reshape(-1))
        print(f"SOAP rows {s}-{e}: {len(idx)} selected", flush=True)
    gal = {k: np.concatenate(v) for k, v in out.items()}
    gal["Centre"] = gal["Centre"].reshape(-1, 3)
    return gal


def accumulate(acc, gidx, pos, mass, centre, box, rmax2=None):
    """Add the tensor terms of matched particles to the per-galaxy sums.

    acc: dict with 'T' (ngal, 6), 'M' (ngal,), 'N' (ngal,) for the reduced tensor
    and, if rmax2 (per-galaxy squared aperture radius) is given, 'T2', 'M2', 'N2'
    for the unreduced tensor within the aperture; gidx: galaxy index of each
    particle; pos in the units of centre/box (comoving Mpc).
    """
    ngal = len(acc["M"])
    r = pos - centre[gidx]
    r -= box * np.round(r / box)                      # periodic wrap
    r2 = np.einsum("ij,ij->i", r, r)
    if "T3" in acc:                                  # unreduced, all bound stars (TNG)
        m64 = mass.astype(np.float64)
        for c, (i, j) in enumerate(PAIRS):
            acc["T3"][:, c] += np.bincount(gidx, m64 * r[:, i] * r[:, j], ngal)
        acc["M3"] += np.bincount(gidx, m64, ngal)
        acc["N3"] += np.bincount(gidx, minlength=ngal)
    if rmax2 is not None:
        a = r2 < rmax2[gidx]
        ga, ra, ma = gidx[a], r[a], mass[a].astype(np.float64)
        for c, (i, j) in enumerate(PAIRS):
            acc["T2"][:, c] += np.bincount(ga, ma * ra[:, i] * ra[:, j], ngal)
        acc["M2"] += np.bincount(ga, ma, ngal)
        acc["N2"] += np.bincount(ga, minlength=ngal)
    ok = r2 > 0                                       # r = 0 would be 0/0 (NaN in the C code)
    gidx, r, r2, mass = gidx[ok], r[ok], r2[ok], mass[ok].astype(np.float64)
    w = mass / r2
    for c, (a, b) in enumerate(PAIRS):
        acc["T"][:, c] += np.bincount(gidx, w * r[:, a] * r[:, b], ngal)
    acc["M"] += np.bincount(gidx, mass, ngal)
    acc["N"] += np.bincount(gidx, minlength=ngal)


_local = threading.local()


def star_datasets(path):
    """Per-thread handles to the PartType4 datasets (one connection per thread)."""
    if getattr(_local, "path", None) != path:
        f = open_hdf5(path)
        _local.ds = {k: f["PartType4/" + k] for k in ("Coordinates", "Masses", "GroupNr_bound")}
        _local.path = path
    return _local.ds


def read_block(path, s, e, tries=5):
    for t in range(tries):
        try:
            ds = star_datasets(path)
            g = np.asarray(ds["GroupNr_bound"][s:e])
            x = np.asarray(ds["Coordinates"][s:e])
            m = np.asarray(ds["Masses"][s:e])
            return g, x, m
        except Exception as ex:  # network hiccup: reconnect and retry
            print(f"WARNING: block {s}-{e} failed ({ex!r}); retry {t + 1}/{tries}", flush=True)
            _local.path = None
            time.sleep(10 * (t + 1))
    raise RuntimeError(f"block {s}-{e} failed {tries} times")


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: particle_shapes.py <parameterfile>")
    p = read_parameter(argv[1], PARAM_TAGS, optional=OPTIONAL)
    if p["Select"] not in ("all", "centrals", "satellites"):
        sys.exit(f"Select must be all, centrals or satellites, got '{p['Select']}'")
    if p["ShapeDef"] not in SHAPE_SUFFIX:
        sys.exit(f"ShapeDef must be one of {list(SHAPE_SUFFIX)}, got '{p['ShapeDef']}'")
    t0 = time.time()
    os.makedirs(p["OutDirBase"], exist_ok=True)
    sn = p["SnapNum"]
    ckpt = os.path.join(p["OutDirBase"], f"shapes_{sn}.ckpt.npz")

    soap = open_hdf5(p["SoapFile"])
    box_mpc, h, z = flamingo_header(soap)
    snap = open_hdf5(p["SnapFile"])
    zs = float(attr_scalar(snap["Header"].attrs, "Redshift"))
    if abs(zs - z) > 1e-4:
        sys.exit(f"SOAP redshift {z} != snapshot redshift {zs}: wrong file pair")
    nstar_tot = snap["PartType4/GroupNr_bound"].shape[0]
    nread = min(nstar_tot, p["MaxParticles"]) if p["MaxParticles"] > 0 else nstar_tot
    print(f"BoxSize = {box_mpc:g} cMpc, h = {h}, z = {z:g}; {nstar_tot} star particles "
          f"(reading {nread})")
    print(f"selection: {p['M_min']:g} < {p['MassField']} < {p['M_max']:g} [1e10 Msun/h], "
          f"NumberOfStarParticles >= {p['MinStars']}, Select={p['Select']}")

    keys = ("T", "M", "N", "T2", "M2", "N2", "T3", "M3", "N3")
    if os.path.isfile(ckpt):
        c = np.load(ckpt)
        if not all(k in c.files for k in keys + ("gal_RHalf",)):
            sys.exit(f"{ckpt} was written by an older version (missing tensor sums); delete it to restart")
        gal = {k[4:]: c[k] for k in c.files if k.startswith("gal_")}
        acc = {k: c[k] for k in keys}
        done = set(c["done"].tolist())
        print(f"resuming from {ckpt}: {len(done)} blocks done, {len(gal['Mass'])} galaxies")
    else:
        gal = select_galaxies(soap, p, h)
        ngal = len(gal["Mass"])
        acc = {"T": np.zeros((ngal, 6)), "M": np.zeros(ngal), "N": np.zeros(ngal, np.int64),
               "T2": np.zeros((ngal, 6)), "M2": np.zeros(ngal), "N2": np.zeros(ngal, np.int64),
               "T3": np.zeros((ngal, 6)), "M3": np.zeros(ngal), "N3": np.zeros(ngal, np.int64)}
        done = set()
        print(f"{ngal} galaxies selected ({time.time() - t0:.0f} s)")
    order = np.argsort(gal["HaloCatalogueIndex"])
    hid_sorted = gal["HaloCatalogueIndex"][order]
    rmax2 = (2.0 * gal["RHalf"]) ** 2

    def save_ckpt():
        np.savez(ckpt + ".part.npz", **acc,
                 done=np.array(sorted(done), dtype=np.int64),
                 **{"gal_" + k: v for k, v in gal.items()})
        os.replace(ckpt + ".part.npz", ckpt)

    step = p["ChunkRows"]
    blocks = [(s, min(s + step, nread)) for s in range(0, nread, step) if s not in done]
    nblk_tot = -(-nread // step)
    t1, nnew = time.time(), 0
    with ThreadPoolExecutor(p["NThreads"]) as ex:
        futs = [(s, e, ex.submit(read_block, p["SnapFile"], s, e))
                for s, e in blocks[:2 * p["NThreads"]]]
        queue = blocks[2 * p["NThreads"]:]
        while futs:
            s, e, fut = futs.pop(0)
            g, x, m = fut.result()
            if queue:  # keep 2*NThreads blocks in flight
                s2, e2 = queue.pop(0)
                futs.append((s2, e2, ex.submit(read_block, p["SnapFile"], s2, e2)))
            pos = np.clip(np.searchsorted(hid_sorted, g), 0, len(hid_sorted) - 1)
            hit = hid_sorted[pos] == g
            accumulate(acc, order[pos[hit]], x[hit], m[hit], gal["Centre"], box_mpc, rmax2)
            done.add(s)
            nnew += 1
            if nnew % p["CheckpointEvery"] == 0 or not futs:
                save_ckpt()
            el = time.time() - t1
            eta = el / nnew * (nblk_tot - len(done))
            print(f"block {len(done)}/{nblk_tot} (rows {s}-{e}): {np.count_nonzero(hit)} member stars; "
                  f"{el / 60:.1f} min elapsed, ETA {eta / 60:.0f} min", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        tensors = {"_all": acc["T3"] / acc["M3"][:, None], "_red": acc["T"] / acc["M"][:, None],
                   "_2re": acc["T2"] / acc["M2"][:, None]}
    nused = {"_all": acc["N3"], "_red": acc["N"], "_2re": acc["N2"]}
    ells = {s: {pr: ellipticity_3d(t, pr) for pr in "zxy"} for s, t in tensors.items()}
    prim = SHAPE_SUFFIX[p["ShapeDef"]]
    tensor, e = tensors[prim], ells[prim]
    good = np.isfinite(e["z"][0]) & np.isfinite(e["z"][1]) & (nused[prim] >= 3)
    print(f"primary shapes: {p['ShapeDef']}; e_rms(z) unreduced/all (TNG) = "
          f"{np.sqrt(np.nanmean(ells['_all']['z'][0][good] ** 2)):.4f}, reduced/all = "
          f"{np.sqrt(np.nanmean(ells['_red']['z'][0][good] ** 2)):.4f}, unreduced/2R_half = "
          f"{np.sqrt(np.nanmean(ells['_2re']['z'][0][good] ** 2)):.4f}")
    full = nread == nstar_tot
    if full:
        mism = np.count_nonzero(acc["N"] != gal["NStarsSOAP"])
        print(f"star-count check: {mism} of {len(acc['N'])} galaxies differ from SOAP NumberOfStarParticles "
              f"(r = 0 particles are skipped, so a difference of 1 is expected for some)")
        big = np.count_nonzero(np.abs(acc["N"] - gal["NStarsSOAP"]) > 1)
        if big:
            print(f"WARNING: {big} galaxies differ by more than one star particle")
    if not good.all():
        print(f"WARNING: dropping {np.count_nonzero(~good)} galaxies without a valid tensor")
    print(f"{np.count_nonzero(good)} galaxies with shapes "
          f"({np.count_nonzero(gal['IsCentral'][good] == 1)} centrals)")

    import h5py
    box = box_mpc * h
    fname = os.path.join(p["OutDirBase"], f"galaxies_{sn}.hdf5")
    with h5py.File(fname + ".part", "w") as g:
        g["Position"] = np.mod(gal["Centre"][good] * h, box)       # comoving Mpc/h
        for pr in "zxy":
            g[f"e1{pr}"] = e[pr][0][good]
            g[f"e2{pr}"] = e[pr][1][good]
        g["Mass"] = gal["Mass"][good]
        g["IsCentral"] = gal["IsCentral"][good]
        g["SOAPIndex"] = gal["SOAPIndex"][good]
        g["Tensor"] = tensor[good]
        g["NStarsUsed"] = acc["N"][good]
        for s in ("_all", "_red", "_2re"):  # every definition, whichever is primary
            for pr in "zxy":
                g[f"e1{pr}{s}"] = ells[s][pr][0][good]
                g[f"e2{pr}{s}"] = ells[s][pr][1][good]
            g[f"Tensor{s}"] = tensors[s][good]
        g["NStars2re"] = acc["N2"][good]
        g["RHalf"] = gal["RHalf"][good] * h                         # comoving Mpc/h
        a = g.attrs
        a["BoxSize"] = box
        a["h"] = h
        a["Redshift"] = z
        a["SnapNum"] = sn
        a["SoapFile"] = p["SoapFile"]
        a["SnapFile"] = p["SnapFile"]
        a["ShapeField"] = {"unreduced_all": "particles: unreduced tensor, all bound stars (TNG)",
                           "reduced_all": "particles: reduced tensor, all bound stars",
                           "unreduced_2re": "particles: unreduced tensor, bound stars within 2 R_half"}[p["ShapeDef"]]
        a["ShapeDef"] = p["ShapeDef"]
        a["MassField"] = p["MassField"]
        a["Select"] = p["Select"]
        a["M_min"], a["M_max"] = p["M_min"], p["M_max"]
        a["MinStars"] = p["MinStars"]
        a["ParticlesRead"] = nread
    os.replace(fname + ".part", fname)
    if full and os.path.isfile(ckpt):
        os.remove(ckpt)
    print(f"wrote {fname} ({(time.time() - t0) / 60:.1f} min). Yatta!")


if __name__ == "__main__":
    main(sys.argv)
