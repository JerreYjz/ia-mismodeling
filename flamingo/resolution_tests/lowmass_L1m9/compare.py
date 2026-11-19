"""Low-mass resolution test, z = 0, 10.0 < log M* < 10.41: L1_m9 (particle shapes, MinStars 10) vs L1_m8.

a_nla from the low-k fit (k = 0.05-0.3, nfold 1) as in restest_soap_m9_vs_m8/compare.py; P_dE ratio
from nfold 1 (k < 0.8) and nfold 4 (k >= 0.8); matched-galaxy shape statistics by m9 star count.
Writes results.json.
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/yijiezhu/Downloads/ia_halofit_v2")
from cosmology import Cosmology  # noqa: E402
from stitch import camb_power  # noqa: E402

C = Cosmology(h=0.681, Om=0.304611, Ob=0.0486, ns=0.967, sigma8=0.807, linear_source="camb")
Pnl, _ = camb_power(C, [0.0], kmax=30.0)
L = 681.0


def spectrum(box, nf):
    d = os.path.join(HERE, box)
    v = open(os.path.join(d, "Galaxy_DS_Field", f"hinfo_fixng_ms_sub_red_sn78_nfold{nf}.dat")).read().split("\n")[1].split()
    ngal, erms = float(v[2]), 0.5 * (float(v[3]) + float(v[4]))
    f = 1 / (2 * (1 - erms ** 2))
    hh = np.loadtxt(os.path.join(d, "PS", f"Phh_sn78_nfold{nf}.dat"))
    he = np.loadtxt(os.path.join(d, "PS", f"PhE0_sn78_nfold{nf}.dat"))
    k = hh[:, 0]
    b = np.sqrt(np.clip(hh[:, 1] - L ** 3 / ngal, 1e-30, None) / Pnl(k, 0.0))
    return k, 1.5 * he[:, 1] / b * f, 1.5 * he[:, 2] / b * f, b, ngal, erms


res, spec = {}, {}
for box in ["L1_m8", "L1_m9", "L1_m8_match", "L1_m9_match"]:
    k1, p1, s1, b1, ng, er = spectrum(box, 1)
    k4, p4, s4, b4, _, _ = spectrum(box, 4)
    m = (k1 >= 0.05) & (k1 <= 0.3)
    P = Pnl(k1[m], 0.0)
    w = 1 / s1[m] ** 2
    A = np.sum(w * p1[m] * P) / np.sum(w * P ** 2)
    sA = 1 / np.sqrt(np.sum(w * P ** 2))
    norm = -0.0134 * C.Om
    lo, hi = k1 < 0.8, k4 >= 0.8
    spec[box] = dict(k=np.r_[k1[lo], k4[hi]], p=np.r_[p1[lo], p4[hi]], s=np.r_[s1[lo], s4[hi]])
    res[box] = dict(ngal=ng, erms=er, bias=float(np.mean(b1[m])), anla=A / norm, err=sA / abs(norm))
    print(f"{box:12s}: N_gal {ng:9.0f}  e_rms {er:.3f}  b {np.mean(b1[m]):.3f}  a_nla {A / norm:.3f} +- {sA / abs(norm):.3f}")
ratios = {}
for tag, (n9, n8) in {"mass cut": ("L1_m9", "L1_m8"), "same galaxies": ("L1_m9_match", "L1_m8_match")}.items():
    r = res[n9]["anla"] / res[n8]["anla"]
    print(f"\n[{tag}] a_nla ratio m9/m8 = {r:.3f};  e_rms ratio = {res[n9]['erms'] / res[n8]['erms']:.3f};  "
          f"bias ratio = {res[n9]['bias'] / res[n8]['bias']:.3f}  (high-mass M*>4e11: 0.877)")
    k8 = spec[n8]["k"]
    assert np.allclose(k8, spec[n9]["k"])
    ratio = spec[n9]["p"] / spec[n8]["p"]
    err = np.abs(ratio) * np.hypot(spec[n9]["s"] / spec[n9]["p"], spec[n8]["s"] / spec[n8]["p"])
    for kk, rr, ee in zip(k8, ratio, err):
        if 0.02 < kk < 10:
            print(f"  k={kk:7.3f}  P_dE ratio {rr:6.3f} +- {ee:5.3f}")
    ratios[tag] = {"k": k8.tolist(), "ratio": ratio.tolist(), "err": err.tolist(),
                   "erms_ratio": res[n9]["erms"] / res[n8]["erms"], "anla_ratio": r}

# same galaxies: shape agreement by m9 star count (rows of the two match catalogues are paired).
# The catalogues are not in the repository (GBs): set LOWMASS_CATDIR to the folder holding
# L1_m8_match/ and L1_m9_match/; without them the previous results.json values are kept.
CATDIR = os.environ.get("LOWMASS_CATDIR", HERE)
f8 = os.path.join(CATDIR, "L1_m8_match/Galaxies/galaxies_78.hdf5")
f9 = os.path.join(CATDIR, "L1_m9_match/Galaxies/galaxies_78.hdf5")
if os.path.isfile(f8) and os.path.isfile(f9):
    a, b = h5py.File(f8, "r"), h5py.File(f9, "r")
    nst = b["NStarsUsed"][:]
    ea = np.stack([a["e1z"][:], a["e2z"][:]], 1)
    eb = np.stack([b["e1z"][:], b["e2z"][:]], 1)
    print()
    match = []
    for lo_, hi_ in [(10, 20), (20, 40), (40, 1e9), (10, 1e9)]:
        s = (nst >= lo_) & (nst < hi_)
        c = np.sum(ea[s] * eb[s]) / np.sqrt(np.sum(ea[s] ** 2) * np.sum(eb[s] ** 2))
        th = 0.5 * np.degrees(np.abs(np.angle((ea[s, 0] + 1j * ea[s, 1]) * np.conj(eb[s, 0] + 1j * eb[s, 1]))))
        match.append(dict(nstar=[lo_, hi_], n=int(s.sum()), corr=float(c), pa_med=float(np.median(th)),
                          erms8=float(np.sqrt(np.mean(ea[s] ** 2))), erms9=float(np.sqrt(np.mean(eb[s] ** 2)))))
        print(f"  N*_m9 {lo_:>3.0f}-{hi_:<5.0f} n={s.sum():7d} corr={c:.3f} median PA offset {np.median(th):4.1f} deg "
              f"e_rms m8 {match[-1]['erms8']:.3f} m9 {match[-1]['erms9']:.3f}")


else:
    print("\nmatched catalogues not found; keeping the matched-shape numbers of results.json")
    match = json.load(open(os.path.join(HERE, "results.json")))["matched"]

out = {"about": "lowmass_L1m9/compare.py: z=0, 10.0<log M*<10.41, unreduced tensor of all bound stars; "
                "L1_m9 MinStars 10, L1_m8 MinStars 100; ng 512, nfold 1 (k<0.8) and 4 (k>=0.8)",
       "summary": res, "ratios": ratios, "matched": match}
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
