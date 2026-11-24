#!/usr/bin/env python3
"""What is the white level left in the FLAMINGO P_EE and P_BB after shape-noise subtraction?

ia_power.py subtracts the analytic shape-noise floor (V/N) <e1^2 + e2^2>/2,
which is exactly the term of each galaxy with itself on an NGP grid. A
plateau N_s is still left in P_BB at 5 to 10 h/Mpc (1 to 6% of the floor),
and the IA truths subtract it again as white noise. Two tests here:

1. Orientation shuffle: every galaxy's ellipticity is rotated by a random
   angle (|e| kept, positions kept), which removes every correlation between
   shapes. The spectra are rebuilt with the same code (galaxy_ds_field.build_field
   and ia_power.calc_power, same floor). If the shuffled P_EE and P_BB are zero
   within their errors, the subtraction is right and the leftover is shape
   correlations between distinct galaxies. Run on all galaxies, and on
   centrals and satellites alone (the satellites carry any 1-halo term).
2. Close pairs: for pairs of distinct galaxies, with c_ij = e1_i e1_j + e2_i e2_j,
   E_i E_j + B_i B_j = c_ij whatever the direction of k, so the pairs add to the
   monopoles
       (P_EE + P_BB)/2 = (V/N^2) sum_{i<j} c_ij j0(k r_ij)
   on top of the floor. Summed over pairs closer than R_MAX, this predicts the
   leftover at k >~ 1/R_MAX from the catalogue alone, without any grid, and
   says which pairs carry it (separation, centrals or satellites, near
   duplicates).

    python bmode_noise.py <sn> [--ng 512] [--folds 1,4,16]
Run with the cocoa Python (h5py, scipy). Writes results/bmode_noise_sn<sn>.json.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extraction"))
from binned import BinnedData                                          # noqa: E402
from galaxy_ds_field import build_field                               # noqa: E402
from ia_power import calc_power                                        # noqa: E402
from common import periodic_index                                     # noqa: E402

CAT = os.path.expanduser("~/ia_power_flamingo/run_L1_m8_particles/results/Galaxies/galaxies_{sn}.hdf5")
KMIN, KMAX, NBIN = 0.003, 20.0, 30          # the binning of ia_power.cfg
R_MAX = 1.0                                 # Mpc/h, largest pair separation summed
R_EDGES = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
SEED = 12345


def load(sn):
    with h5py.File(CAT.format(sn=sn), "r") as f:
        box = float(f.attrs["BoxSize"])
        z = float(f.attrs["Redshift"])
        d = {q: f[q][...] for q in ("Position", "e1z", "e2z", "IsCentral", "Mass", "SOAPIndex")}
    ok = np.isfinite(d["e1z"]) & np.isfinite(d["e2z"])
    d = {q: v[ok] for q, v in d.items()}
    return d, box, z


def spectra(pos, e1, e2, box, nfold, ng):
    """P_hh, P_EE0, P_BB0, P_EB0 of one sample with the pipeline's code and floor."""
    n = len(e1)
    boxlen = box/nfold
    cell = periodic_index(pos*(ng/boxlen) + 1e-10, ng)
    flat = (cell[:, 0]*ng + cell[:, 1])*(2*(ng//2 + 1)) + cell[:, 2]
    norm = float(ng)**3/n
    fields = {"h": build_field(ng, flat, np.ones(n, np.float32), norm, True),
              "e1": build_field(ng, flat, e1.astype(np.float32), norm, False),
              "e2": build_field(ng, flat, e2.astype(np.float32), norm, False)}
    floor = box**3/n*(np.mean(e1**2) + np.mean(e2**2))/2      # as ia_power.read_shot_noise
    sp = {q: BinnedData(NBIN, KMIN, KMAX, True) for q in ("Phh", "PEE0", "PBB0", "PEB0")}
    calc_power(fields, boxlen, nfold, sp, {}, floor)
    out = {}
    for q, bd in sp.items():
        km, pm, sig, num = bd.means()            # the columns ia_power.py writes
        m = num > 0
        out[q] = dict(k=km[m].tolist(), P=pm[m].tolist(), err=sig[m].tolist(), nmodes=num[m].tolist())
    # only k below half the Nyquist of this grid is kept (NGP aliasing above)
    out["k_use_max"] = 0.5*np.pi*ng/boxlen
    out["floor"] = floor
    return out


def pairs(pos, e1, e2, cen, box, n_all):
    """Close-pair sums of c_ij, by separation and pair type, and the j0 prediction."""
    t = cKDTree(pos, boxsize=box)
    ij = t.query_pairs(R_MAX, output_type="ndarray")
    i, j = ij[:, 0], ij[:, 1]
    dx = pos[i] - pos[j]
    dx -= box*np.round(dx/box)
    r = np.sqrt((dx**2).sum(1))
    c = e1[i]*e1[j] + e2[i]*e2[j]
    pref = box**3/n_all**2
    typ = np.where(cen[i] + cen[j] == 2, "cen-cen", np.where(cen[i] + cen[j] == 1, "cen-sat", "sat-sat"))
    shells = []
    for a, b in zip(R_EDGES[:-1], R_EDGES[1:]):
        s = (r >= a) & (r < b)
        row = dict(r_lo=a, r_hi=b, npairs=int(s.sum()), sum_c=float(pref*c[s].sum()),
                   mean_c=float(c[s].mean()) if s.any() else None)
        for q in ("cen-cen", "cen-sat", "sat-sat"):
            row[q] = float(pref*c[s & (typ == q)].sum())
        shells.append(row)
    kk = np.logspace(np.log10(0.05), np.log10(20), 40)
    pred = []
    for k in kk:
        x = k*r
        j0 = np.where(x > 1e-8, np.sin(x)/np.maximum(x, 1e-30), 1.0)
        pred.append(float(pref*(c*j0).sum()))
    # near duplicates: closer than 10 kpc/h with nearly the same ellipticity
    near = r < 0.01
    de = np.hypot(e1[i] - e1[j], e2[i] - e2[j])
    dup = near & (de < 0.05)
    return dict(shells=shells, k=kk.tolist(), pred_EEBB_half=pred, npairs=int(len(r)),
                n_near_10kpc=int(near.sum()), n_near_dup=int(dup.sum()),
                median_de_near=float(np.median(de[near])) if near.any() else None,
                median_de_all=float(np.median(de)))


def main():
    a = sys.argv[1:]
    sn = int(a[0])
    ng = int(a[a.index("--ng") + 1]) if "--ng" in a else 512
    folds = [int(x) for x in a[a.index("--folds") + 1].split(",")] if "--folds" in a else [1, 4, 16]
    t0 = time.time()
    d, box, z = load(sn)
    n = len(d["e1z"])
    cen = d["IsCentral"].astype(int)
    print(f"sn {sn}: z = {z:.3f}, N = {n}, centrals {cen.sum()}, box {box:.1f} Mpc/h, "
          f"unique SOAPIndex {len(np.unique(d['SOAPIndex'])) == n}", flush=True)
    rng = np.random.default_rng(SEED)
    th = rng.uniform(0, np.pi, n)               # rotate the shape by th: e -> e exp(2 i th)
    c2, s2 = np.cos(2*th), np.sin(2*th)
    e1r = d["e1z"]*c2 - d["e2z"]*s2
    e2r = d["e1z"]*s2 + d["e2z"]*c2
    samples = {"real": (np.ones(n, bool), d["e1z"], d["e2z"]),
               "shuffled": (np.ones(n, bool), e1r, e2r),
               "centrals": (cen == 1, d["e1z"], d["e2z"]),
               "satellites": (cen == 0, d["e1z"], d["e2z"]),
               "satellites_shuffled": (cen == 0, e1r, e2r)}
    out = dict(about=__doc__, sn=sn, z=z, N=n, box=box, ng=ng, folds=folds, spectra={})
    for name, (m, e1, e2) in samples.items():
        for nf in folds:
            r = spectra(d["Position"][m], e1[m], e2[m], box, nf, ng)
            out["spectra"][f"{name}|{nf}"] = r
            print(f"  {name:20s} nfold {nf:2d}: floor {r['floor']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    out["pairs"] = pairs(d["Position"], d["e1z"], d["e2z"], cen, box, n)
    out["pairs_shuffled"] = pairs(d["Position"], e1r, e2r, cen, box, n)
    print(f"  pairs within {R_MAX} Mpc/h: {out['pairs']['npairs']}  ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"bmode_noise_sn{sn}.json"), "w"))
    for s in out["pairs"]["shells"]:
        print(f"  r {s['r_lo']:.3f} to {s['r_hi']:.3f}: pairs {s['npairs']:9d}  (V/N^2) sum c = {s['sum_c']:+.4f}"
              f"  [cen-cen {s['cen-cen']:+.4f} cen-sat {s['cen-sat']:+.4f} sat-sat {s['sat-sat']:+.4f}]")
    print(f"  near (<10 kpc/h) {out['pairs']['n_near_10kpc']}, near with |de|<0.05 {out['pairs']['n_near_dup']}")


if __name__ == "__main__":
    main()
