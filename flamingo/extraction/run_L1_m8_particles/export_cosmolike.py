#!/usr/bin/env python3
"""Write the FLAMINGO IA spectra in the layout read by sim_IA.c (CosmoLike).

sim_IA.c reads, per snapshot sn and fold n (set_sim_IA_config(dir, nsn, sn, z, nfold)):
    P{EE,BB,dE,dB}{0,2,4}_sn<sn>_nfold<n>.dat,  P{dd,hh,dh}_sn<sn>_nfold<n>.dat   (k P sigma N)
    hinfo_<stem>_sn<sn>_nfold<n>.dat                                           (e_rms, ngal_used)
as raw Pds outputs (distortion units, monopole-style multipoles); it applies the shear, mu = 0 and
NGP-window conversions itself. The FLAMINGO run has no matter field (it would mean ~2.6 TB of
particles per snapshot), so the matter legs come from FLAMINGO's own measured matter power spectrum
(power_spectra/power_matter_<i>.txt, shot-noise subtracted) and a scale-dependent galaxy bias:

    b(k)  = sqrt[(P_hh - V/N) / P_dd]
    P_dE_l ~= P_hE_l / b,  P_dB_l ~= P_hB_l / b,  P_dh = b P_dd,  P_dd = FLAMINGO matter

which assumes deterministic bias (no stochasticity) between galaxies and matter; it is
approximate on small scales. Shapes: results/ (unreduced tensor of all bound stars, the TNG definition of eigen_iteration.c).

usage: python export_cosmolike.py [all]
Writes cosmolike_<set>/ with the files for every snapshot that has nfold 1 and 4 spectra, plus
README_sim_IA.txt with the set_sim_IA_* calls.
"""
import glob
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import open_hdf5  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H, L, NG = 0.681, 681.0, 1024
RUN = "FLAMINGO/L1_m8/L1_m8/"
FOLDS = (1, 4)
SPEC_L = ("PEE", "PBB", "PhE", "PhB")          # files with a multipole index (0, 2, 4)


def remote_text(path):
    f = open_hdf5("cosma:/" + path)
    t = f.open().read()
    return t.decode() if isinstance(t, bytes) else t


def flamingo_pmm(z, cache={}):
    """FLAMINGO measured matter P(k) at redshift z: (k [h/Mpc], P [(Mpc/h)^3])."""
    if "zlist" not in cache:
        lines = [l for l in remote_text(RUN + "output_list_ps.txt").splitlines() if l and l[0] != "#"]
        cache["zlist"] = np.array([float(l) for l in lines])
    i = int(np.argmin(np.abs(cache["zlist"] - z)))
    if abs(cache["zlist"][i] - z) > 1e-3:
        sys.exit(f"no FLAMINGO matter power spectrum at z = {z} (nearest {cache['zlist'][i]})")
    d = np.loadtxt(remote_text(RUN + f"power_spectra/power_matter_{i:04d}.txt").splitlines())
    return d[:, 1] / H, d[:, 2] * H ** 3, i


def main(which="all"):
    res = {"all": "results"}[which]
    ps, gds = os.path.join(HERE, res, "PS"), os.path.join(HERE, res, "Galaxy_DS_Field")
    out = os.path.join(HERE, f"cosmolike_{which}")
    os.makedirs(out, exist_ok=True)
    sns, zs = [], []
    for f16 in sorted(glob.glob(os.path.join(ps, "Phh_sn*_nfold4.dat"))):
        sn = int(os.path.basename(f16)[6:].split("_")[0])
        if not all(os.path.isfile(os.path.join(ps, f"Phh_sn{sn}_nfold{n}.dat")) for n in FOLDS):
            continue
        with h5py.File(os.path.join(HERE, "results", "Galaxies", f"galaxies_{sn}.hdf5"), "r") as g:
            z = round(float(g.attrs["Redshift"]), 4)
        kp, pp, ip = flamingo_pmm(z)
        pmm = lambda k: np.exp(np.interp(np.log(k), np.log(kp), np.log(pp)))
        for n in FOLDS:
            hinfo = os.path.join(gds, f"hinfo_fixng_ms_sub_red_sn{sn}_nfold{n}.dat")
            ngal = float(open(hinfo).read().split("\n")[1].split()[2])
            shutil.copy(hinfo, out)
            hh = np.loadtxt(os.path.join(ps, f"Phh_sn{sn}_nfold{n}.dat"))
            k = hh[:, 0]
            pdd = pmm(k)
            b = np.sqrt(np.clip(hh[:, 1] - L ** 3 / ngal, 1e-30, None) / pdd)
            fmt = "%.6g"
            np.savetxt(os.path.join(out, f"Phh_sn{sn}_nfold{n}.dat"), hh, fmt=fmt)
            np.savetxt(os.path.join(out, f"Pdd_sn{sn}_nfold{n}.dat"),
                       np.column_stack([k, pdd, np.zeros_like(k), hh[:, 3]]), fmt=fmt)
            np.savetxt(os.path.join(out, f"Pdh_sn{sn}_nfold{n}.dat"),
                       np.column_stack([k, b * pdd, hh[:, 2] / (2 * b), hh[:, 3]]), fmt=fmt)
            for tag in SPEC_L:
                for ell in (0, 2, 4):
                    d = np.loadtxt(os.path.join(ps, f"{tag}{ell}_sn{sn}_nfold{n}.dat"))
                    if not np.allclose(d[:, 0], k, rtol=1e-5):
                        sys.exit(f"k grid of {tag}{ell} sn{sn} nfold{n} differs from Phh")
                    if tag in ("PhE", "PhB"):                  # galaxy leg -> matter leg
                        d = d.copy()
                        d[:, 1] /= b
                        d[:, 2] /= b
                        tag_out = "Pd" + tag[2]
                    else:
                        tag_out = tag
                    np.savetxt(os.path.join(out, f"{tag_out}{ell}_sn{sn}_nfold{n}.dat"), d, fmt=fmt)
        sns.append(sn)
        zs.append(z)
        print(f"sn{sn} z={z}: exported nfold {FOLDS} (FLAMINGO power_matter_{ip:04d}; "
              f"b(k=0.5-2) = {np.mean(b[(k > 0.5) & (k < 2)]):.3f})")
    order = np.argsort(zs)
    sns, zs = [sns[i] for i in order], [zs[i] for i in order]
    with open(os.path.join(out, "README_sim_IA.txt"), "w") as f:
        f.write(f"""FLAMINGO L1_m8 IA spectra for sim_IA.c  (shapes: { {'all': 'unreduced tensor, all bound stars (TNG definition)', '2re': 'unreduced tensor within 2 R_half', 'red': 'reduced tensor, all bound stars'}[which] })

Host calls:
  int    sn[] = {{{', '.join(map(str, sns))}}};
  double z[]  = {{{', '.join(f'{x:g}' for x in zs)}}};
  set_sim_IA_config("<this dir>", {len(sns)}, sn, z, 4);          /* nfold = 4 recommended */
  set_sim_IA_window(1, {L}, {NG});                                 /* NOT the TNG default 205 */
  set_sim_IA_erms_from_hinfo("<this dir>", "fixng_ms_sub_red", ...);
  set_sim_IA_shape_convention(1, SIM_IA_LOS_HYBRID);

Why nfold = 4: box 681 Mpc/h, ng 1024. sim_IA.c uses k = {0.5}-{7.0} h/Mpc. nfold 1 has k_Nyq = 4.7 h/Mpc
(unreliable above ~2.4); nfold 4 has k_Nyq = 18.9, so 0.5-7 is well inside its trusted half-Nyquist.
nfold 1 files are included for k < 0.5 checks.

Matter legs (no matter field in this run): P_dd = FLAMINGO's measured matter power spectrum
(power_spectra/power_matter_*.txt), b(k) = sqrt[(P_hh - V/N)/P_dd], P_dE ~= P_hE/b, P_dB ~= P_hB/b,
P_dh = b P_dd. Deterministic-bias approximation; less reliable at k > 1 h/Mpc.
Phh is not shot-noise subtracted (as in the TNG tables); PEE0/PBB0 are shape-noise subtracted.
""")
    print(f"wrote {out} ({len(sns)} snapshots)")


if __name__ == "__main__":
    main(*sys.argv[1:])
