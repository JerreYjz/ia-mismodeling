#!/usr/bin/env python3
"""Write a tiny FLAMINGO-SOAP-like catalogue with a known alignment signal.

Same group/dataset names, units (comoving Mpc and 1e10 Msun, no h) and
SOAP mask attributes as the real halo_properties_XXXX.hdf5. Galaxies sit
around a few "clusters"; with ``aligned=1`` each one's major axis points at
its cluster (radial alignment, so P_hE != 0 and P_hB = 0), with
``aligned=0`` orientations are random (P_hE = P_hB = 0).
A fraction of subhalos have < 100 star particles and zeroed (masked)
tensors, as SOAP writes them.

Usage: python make_mock_soap.py <out.hdf5> [aligned=1] [seed=1]
"""

import sys

import h5py
import numpy as np

BOX = 300.0      # comoving Mpc
H = 0.681
NCL = 60
NGAL = 30000


def random_rotation(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    return q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]


def main(out, aligned=1, seed=1):
    rng = np.random.default_rng(seed)
    cl = rng.uniform(0, BOX, (NCL, 3))
    host = rng.integers(0, NCL, NGAL)
    offset = rng.normal(0, 3.0, (NGAL, 3))
    pos = np.mod(cl[host] + offset, BOX)

    # axis lengths a > b > c, major axis along u
    axes2 = np.stack([np.full(NGAL, 9.0), np.full(NGAL, 4.0), np.full(NGAL, 2.0)], 1) * 1e-6
    if aligned:
        u = offset / np.linalg.norm(offset, axis=1, keepdims=True)
        rot = random_rotation(rng, NGAL)
        # build an orthonormal frame whose first vector is u
        v = rot[:, :, 1] - np.einsum("ij,ij->i", rot[:, :, 1], u)[:, None] * u
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        w = np.cross(u, v)
        frame = np.stack([u, v, w], axis=2)
    else:
        frame = random_rotation(rng, NGAL)
    t = np.einsum("nij,nj,nkj->nik", frame, axes2, frame)  # R diag R^T  (physical Mpc^2)
    tensor6 = np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2], t[:, 0, 1], t[:, 0, 2], t[:, 1, 2]], 1)
    tensor2 = np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 0, 1]], 1)

    nstar = rng.integers(20, 2000, NGAL).astype(np.uint32)
    mstar = (nstar * 0.07 * rng.uniform(0.8, 1.2, NGAL)).astype(np.float32)  # 1e10 Msun
    masked = nstar < 100
    tensor6[masked] = 0
    tensor2[masked] = 0
    central = (rng.uniform(size=NGAL) < 0.6).astype(np.int64)

    with h5py.File(out, "w") as f:
        h = f.create_group("Header").attrs
        h["BoxSize"] = np.array([BOX] * 3)
        h["Redshift"] = np.array([0.0])
        h["NumSubhalos_Total"] = np.array([NGAL], dtype=np.int32)
        f.create_group("Cosmology").attrs["h"] = np.array([H])
        ih = f.create_group("InputHalos")
        ih["HaloCentre"] = pos
        ih["IsCentral"] = central
        bs = f.create_group("BoundSubhalo")
        bs["StellarMass"] = mstar
        bs["NumberOfStarParticles"] = nstar
        d = bs.create_dataset("StellarInertiaTensorNoniterative", data=tensor6.astype(np.float32))
        d.attrs["Masked"] = np.int8(1)
        d.attrs["Mask Datasets"] = np.array(["BoundSubhalo/NumberOfStarParticles"], dtype=object)
        d.attrs["Mask Threshold"] = np.int64(100)
        pz = f.create_group("ProjectedAperture/50kpc/projz")
        pz["NumberOfStarParticles"] = nstar
        d = pz.create_dataset("ProjectedStellarInertiaTensorNoniterative", data=tensor2.astype(np.float32))
        d.attrs["Masked"] = np.int8(1)
        d.attrs["Mask Datasets"] = np.array(["ProjectedAperture/50kpc/projz/NumberOfStarParticles"],
                                            dtype=object)
        d.attrs["Mask Threshold"] = np.int64(100)
    print(f"mock SOAP written to {out}: {NGAL} subhalos, {np.count_nonzero(~masked)} unmasked, "
          f"aligned={aligned}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
