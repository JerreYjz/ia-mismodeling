#!/usr/bin/env python3
"""Check particle_shapes.py output against an independent per-galaxy calculation.

For galaxies in a galaxies_<sn>.hdf5 written by particle_shapes.py (a full run,
or a MaxParticles test run), this fetches each galaxy's bound stars through the
snapshot's spatial cell index (Cells/...), recomputes the TNG reduced tensor
with a plain loop over particles, and compares the star count with SOAP's
NumberOfStarParticles and the tensor/ellipticities with the catalogue.
Only galaxies whose stars were all read are checked (NStarsUsed close to SOAP).

usage: validate_particle_shapes.py <galaxies_sn.hdf5> [n_galaxies=10]
"""

import sys

import h5py
import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from common import open_hdf5  # noqa: E402


def main(cat_path, ngal=10):
    with h5py.File(cat_path, "r") as c:
        a = dict(c.attrs)
        cat = {k: c[k][()] for k in ("SOAPIndex", "Tensor", "e1z", "e2z", "NStarsUsed")}
        cat["Tensor_2re"] = c["Tensor_2re"][()] if "Tensor_2re" in c else None
        shape_def = a.get("ShapeDef", "reduced_all")
    soap = open_hdf5(a["SoapFile"])
    snap = open_hdf5(a["SnapFile"])
    h = float(a["h"])
    box = float(a["BoxSize"]) / h                                    # comoving Mpc
    nsoap = np.array([int(np.asarray(soap["BoundSubhalo/NumberOfStarParticles"][i:i + 1])[0])
                      for i in cat["SOAPIndex"][:2000]])
    complete = np.flatnonzero(np.abs(cat["NStarsUsed"][:2000] - nsoap) <= 1)
    rng = np.random.default_rng(0)
    pick = rng.choice(complete, min(ngal, len(complete)), replace=False)
    print(f"{len(complete)} of the first 2000 catalogue galaxies were read completely; checking {len(pick)}")

    cent = np.asarray(snap["Cells/Centres"][...])
    cnt = np.asarray(snap["Cells/Counts/PartType4"][...])
    off = np.asarray(snap["Cells/OffsetsInFile/PartType4"][...])
    half = float(np.asarray(snap["Cells/Meta-data"].attrs["size"]).ravel()[0]) / 2
    worst = 0.0
    for k in pick:
        i = int(cat["SOAPIndex"][k])
        c0 = np.asarray(soap["InputHalos/HaloCentre"][i:i + 1])[0].astype(np.float64)
        hid = int(np.asarray(soap["InputHalos/HaloCatalogueIndex"][i:i + 1])[0])
        d = cent - c0
        d -= box * np.round(d / box)
        X, M = [], []
        for j in np.flatnonzero(np.all(np.abs(d) <= half + 1.0, axis=1)):
            s, e = off[j], off[j] + cnt[j]
            sel = np.asarray(snap["PartType4/GroupNr_bound"][s:e]) == hid
            if sel.any():
                X.append(np.asarray(snap["PartType4/Coordinates"][s:e])[sel])
                M.append(np.asarray(snap["PartType4/Masses"][s:e])[sel])
        X, M = np.concatenate(X), np.concatenate(M).astype(np.float64)
        I, msum, n = np.zeros((3, 3)), 0.0, 0
        for x, m in zip(X, M):                                       # eigen.c, no-iteration branch
            r = x - c0
            r -= box * np.round(r / box)
            r2 = r @ r
            if r2 == 0:
                continue
            I += m * np.outer(r, r) / r2
            msum += m
            n += 1
        ref = np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]) / msum
        rel2 = 0.0
        if cat["Tensor_2re"] is not None:                          # unreduced, r < 2 R_half
            rh = float(np.asarray(soap["BoundSubhalo/HalfMassRadiusStars"][i:i + 1])[0])
            r = X - c0
            r -= box * np.round(r / box)
            inside = np.einsum("ij,ij->i", r, r) < (2 * rh) ** 2
            J = np.einsum("n,ni,nj->ij", M[inside], r[inside], r[inside]) / M[inside].sum()
            ref2 = np.array([J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]])
            rel2 = np.abs(ref2 - cat["Tensor_2re"][k]).max() / np.abs(ref2).max()
        r = X - c0
        r -= box * np.round(r / box)
        K = np.einsum("n,ni,nj->ij", M, r, r) / M.sum()                # unreduced, all bound stars
        ref3 = np.array([K[0, 0], K[1, 1], K[2, 2], K[0, 1], K[0, 2], K[1, 2]])
        if shape_def == "unreduced_2re":
            ref = ref2
        elif shape_def == "unreduced_all":
            ref = ref3
        e1 = (ref[0] - ref[1]) / (ref[0] + ref[1])
        e2 = 2 * ref[3] / (ref[0] + ref[1])
        rel = np.abs(ref - cat["Tensor"][k]).max() / np.abs(ref).max()
        worst = max(worst, rel, rel2)
        print(f"SOAP row {i:9d}: stars SOAP {nsoap[k]:5d} / cells {len(M):5d} / catalogue {cat['NStarsUsed'][k]:5d}; "
              f"e1z {e1:+.4f} vs {cat['e1z'][k]:+.4f}, e2z {e2:+.4f} vs {cat['e2z'][k]:+.4f}; "
              f"tensor rel diff {rel:.1e}, 2R_half tensor rel diff {rel2:.1e}")
    ok = worst < 1e-5
    print(("PASS" if ok else "FAIL") + f": worst tensor rel diff {worst:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], *(int(t) for t in sys.argv[2:])))
