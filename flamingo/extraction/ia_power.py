#!/usr/bin/env python3
"""Stage 3: intrinsic-alignment power spectra from galaxy fields only.

FLAMINGO version of the TNG ``Pds`` stage: there is no matter field. Reads
three Fourier-space grids on an ng^3 mesh (galaxy number density h,
ellipticities e1, e2; from galaxy_ds_field.py) and forms the E/B shape fields

    E =  e1 cos(2 phi) + e2 sin(2 phi),   B = -e1 sin(2 phi) + e2 cos(2 phi)

(phi = azimuth of k in the x-y plane, line of sight = z) and bins every
auto/cross spectrum:

    Phh,  PhE{0,2,4} PhB{0,2,4},
    PEE{0,2,4} PEB{0,2,4} PBB{0,2,4}     (Legendre multipoles in mu = k_z/k)

plus 2D P(k, mu) tables of the monopoles in <out_dir>/2d/.
1D rows: ``k  P(k)  sigma_P  N_modes``.

The analytic shape-noise floor (V/N_gal) <e1^2 + e2^2>/2, read from the
hinfo file of galaxy_ds_field.py, is subtracted from PEE0/PBB0 (and their
2D tables); without that file a high-k-shell estimate is used instead.
Phh is written without shot-noise subtraction (as in the TNG version); the
Poisson level V/N_gal is printed for reference.

Usage (configuration identical to the C version):
    python ia_power.py                          # ./params.cfg
    python ia_power.py myrun.cfg                # a specific config file
    python ia_power.py myrun.cfg sn=78 nfold=4
Precedence: built-in defaults < config file < key=value arguments.
"""

import os
import sys

import numpy as np

from binned import BinnedData, BinnedData2D
from common import c_atof, c_atoi, log

DEFAULTS = {
    "boxsize": "1906.8", "kmin": "0.003", "kmax": "20.0", "nbin": "30", "munbin": "20",
    "ng": "1024", "sn": "78", "nfold": "1",
    "base": "./results/",
    "gal_dir": "Galaxy_DS_Field/",
    "out_dir": "PS/",
    "hdens_stem": "galdens_fixng_ms_sub_red_sn",
    "e1_stem": "e1z_fixng_ms_sub_red_sn",
    "e2_stem": "e2z_fixng_ms_sub_red_sn",
    "shot_noise": "1",
}

# spectrum name -> (field a, field b, Legendre order) for the 1D outputs
SPECTRA_1D = {"Phh": ("h", "h", 0)}
for _a, _b in (("h", "E"), ("h", "B"), ("E", "E"), ("E", "B"), ("B", "B")):
    for _ell in (0, 2, 4):
        SPECTRA_1D[f"P{_a}{_b}{_ell}"] = (_a, _b, _ell)
# 2D (k, mu) tables of the monopoles
SPECTRA_2D = {f"P{a}{b}2D": (a, b) for a, b in
              (("h", "h"), ("h", "E"), ("h", "B"), ("E", "E"), ("E", "B"), ("B", "B"))}

SHOTNOISE_KFRAC = 0.7  # high-k-shell fallback uses [0.7, 1] * min(k_Nyquist, kmax)


# ---------------------------------------------------------------------------
# configuration (config.hpp)
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, argv):
        self.kv = dict(DEFAULTS)
        cfgfile = next((t for t in argv[1:] if "=" not in t), "params.cfg")
        if os.path.isfile(cfgfile):
            with open(cfgfile) as f:
                for line in f:
                    self._set_line(line)
            log(f"Config: loaded {cfgfile}")
        else:
            log(f"Config: no config file '{cfgfile}' found; using defaults + CLI overrides.")
        for tok in argv[1:]:
            if "=" in tok:
                self._set_line(tok)

    def _set_line(self, raw):
        line = raw.split("#", 1)[0].strip()
        if not line:
            return
        if "=" not in line:
            log(f"Config WARNING: ignoring line without '=': {raw.rstrip()}")
            return
        key, val = (s.strip() for s in line.split("=", 1))
        if key:
            self.kv[key] = val

    def get(self, key):
        if key not in self.kv:
            sys.exit(f"Config ERROR: missing required key '{key}'")
        return self.kv[key]

    def get_d(self, key):
        return c_atof(self.get(key))

    def get_i(self, key):
        return c_atoi(self.get(key))

    def dump(self):
        log("----- effective configuration -----")
        for k in sorted(self.kv):
            log(f"  {k} = {self.kv[k]}")
        log("-----------------------------------")


# ---------------------------------------------------------------------------
# input grids (fieldmap.hpp): read-only memory maps
# ---------------------------------------------------------------------------
def map_field(fname, ng):
    shape = (ng, ng, ng // 2 + 1)
    expected = 8 * shape[0] * shape[1] * shape[2]
    if not os.path.isfile(fname):
        sys.exit(f"FieldMap ERROR: cannot open {fname}")
    size = os.path.getsize(fname)
    if size != expected:
        sys.exit(f"FieldMap ERROR: size mismatch for {fname}\n  expected {expected} bytes "
                 f"for ng={ng}, found {size} (wrong ng, or truncated file)")
    return np.memmap(fname, dtype=np.complex64, mode="r", shape=shape)


def read_shot_noise(hinfo_path, boxsize):
    """Analytic floor (V/N) <e1^2+e2^2>/2 from the hinfo file, or None."""
    if not os.path.isfile(hinfo_path):
        log(f"WARNING: cannot open shape-stats file {hinfo_path}: shape-noise NOT "
            "subtracted analytically (falling back to high-k-shell estimate).")
        return None
    try:
        with open(hinfo_path) as f:
            f.readline()
            vals = [float(v) for v in f.readline().split()[:5]]
        _, _, ngal_used, e1rms, e2rms = vals
    except (ValueError, IndexError):
        vals = None
    if vals is None or not (ngal_used > 0 and e1rms > 0 and e2rms > 0):
        log(f"WARNING: could not parse {hinfo_path}: shape-noise NOT subtracted analytically.")
        return None
    vol = boxsize ** 3
    p_shot = vol / ngal_used * (e1rms ** 2 + e2rms ** 2) * 0.5
    log(f"analytic shot-noise from {hinfo_path}\n  N_gal={ngal_used:g}  e1_rms={e1rms:g}  "
        f"e2_rms={e2rms:g}  V={vol:g} (Mpc/h)^3  -> P_shot={p_shot:g}\n"
        f"  (Phh is written without subtraction; its Poisson level V/N_gal = {vol / ngal_used:g})")
    return p_shot


def check_boxsize(hinfo_path, boxsize):
    """Exit if the config boxsize differs from the one galaxy_ds_field.py recorded.

    A stale boxsize (e.g. the L2p8 default on an L1 run) would silently give
    wrong k and P. Old hinfo files without the column are accepted with a warning.
    """
    if not os.path.isfile(hinfo_path):
        return
    with open(hinfo_path) as f:
        names = f.readline().lstrip("#").split()
        vals = f.readline().split()
    if "boxsize" not in names or len(vals) != len(names):
        log(f"WARNING: {hinfo_path} has no boxsize column; cannot check boxsize={boxsize:g}.")
        return
    cat_box = float(vals[names.index("boxsize")])
    if abs(cat_box - boxsize) > 1e-4 * cat_box:
        sys.exit(f"ERROR: config boxsize = {boxsize:g} Mpc/h but the galaxy catalogue has "
                 f"BoxSize = {cat_box:g} Mpc/h ({hinfo_path}). Fix boxsize in the config.")
    log(f"boxsize check OK: {boxsize:g} Mpc/h matches the catalogue.")


# ---------------------------------------------------------------------------
# calc_power (power.hpp)
# ---------------------------------------------------------------------------
def signed_index(n, ng):
    """Grid index -> signed wavenumber index (i > ng/2 -> i - ng)."""
    n = np.asarray(n)
    return np.where(n > ng // 2, n - ng, n).astype(np.float64)


def calc_power(fields, boxlen, nfold, spectra_1d, spectra_2d, shot_analytic=None):
    """Accumulate all spectra over the half-complex grid, one x-plane at a time.

    ``fields`` = dict h/e1/e2 -> complex (ng, ng, ng//2+1) arrays (memmaps).
    Modes with k_z index 0 get weight 1/2 (they appear twice, as k and -k,
    in the stored half-space); the Nyquist plane k_z = ng/2 is not used.
    """
    ng = fields["h"].shape[0]
    kfund = 2.0 * np.pi / boxlen
    vol = (boxlen * nfold) ** 3
    kmax = spectra_1d["Phh"].maximum

    k_nyquist = kfund * (ng // 2)
    shot_kmax = min(k_nyquist, spectra_1d["PEE0"].maximum)
    shot_kmin = SHOTNOISE_KFRAC * shot_kmax
    shot_ee = shot_bb = shot_w = 0.0

    # Only indices whose component |n|*kfund <= kmax can land in a bin.
    jidx = np.arange(ng)
    jidx = jidx[np.abs(signed_index(jidx, ng)) * kfund <= kmax]
    kidx = np.arange(ng // 2)
    kidx = kidx[kidx * kfund <= kmax]
    jj = signed_index(jidx, ng)[:, None]
    kk = kidx.astype(np.float64)[None, :]
    weight = np.broadcast_to(np.where(kk == 0, 0.5, 1.0), (len(jidx), len(kidx))).ravel()
    pairs = ({SPECTRA_1D[n][:2] for n in spectra_1d} | {SPECTRA_2D[n] for n in spectra_2d}
             | {("E", "E"), ("B", "B")})

    log(f"looping over {ng} x-planes ...")
    for i in range(ng):
        ii = float(signed_index(i, ng))
        if abs(ii) * kfund > kmax:
            continue
        kperp2 = ii * ii + jj * jj                      # (nj, 1)
        ktot2 = (kperp2 + kk * kk).ravel()              # (nj*nk,)
        kperp2 = np.broadcast_to(kperp2, (len(jidx), len(kidx))).ravel()
        valid = ktot2 > 0                               # drop the k = 0 mode
        ktot2 = ktot2[valid]
        kperp2 = kperp2[valid]
        w = weight[valid]
        knorm = kfund * np.sqrt(ktot2)

        jjf = np.broadcast_to(jj, (len(jidx), len(kidx))).ravel()[valid]
        mu = np.broadcast_to(kk, (len(jidx), len(kidx))).ravel()[valid] / np.sqrt(ktot2)
        mu2 = mu * mu
        legendre = {0: 1.0, 2: 0.5 * (3.0 * mu2 - 1.0), 4: 0.125 * ((35.0 * mu2 - 30.0) * mu2 + 3.0)}

        # E/B projection; k_perp = 0 modes are assigned to E
        with np.errstate(invalid="ignore", divide="ignore"):
            f_e = np.where(kperp2 == 0, 1.0, (ii * ii - jjf * jjf) / kperp2)
            f_b = np.where(kperp2 == 0, 0.0, 2.0 * ii * jjf / kperp2)

        plane = {}
        for name in ("h", "e1", "e2"):
            plane[name] = np.asarray(fields[name][i][np.ix_(jidx, kidx)],
                                     dtype=np.complex128).ravel()[valid]
        plane["E"] = plane["e1"] * f_e + plane["e2"] * f_b
        plane["B"] = -plane["e1"] * f_b + plane["e2"] * f_e

        # vol * Re(a b*) for every field pair that any spectrum needs
        cross = {(a, b): vol * (plane[a].real * plane[b].real + plane[a].imag * plane[b].imag)
                 for a, b in pairs}

        bins_1d = spectra_1d["Phh"].get_bin(knorm)
        for name, bd in spectra_1d.items():
            a, b, ell = SPECTRA_1D[name]
            bd.push_data(knorm, cross[a, b] * legendre[ell], w, bins=bins_1d)
        if spectra_2d:
            first = next(iter(spectra_2d.values()))
            bins_2d = first.get_bin(knorm, mu)
            for name, bd in spectra_2d.items():
                a, b = SPECTRA_2D[name]
                bd.push_data(knorm, mu, cross[a, b], w, bins=bins_2d)

        shell = (knorm >= shot_kmin) & (knorm <= shot_kmax)
        shot_ee += float(np.sum(w[shell] * cross["E", "E"][shell]))
        shot_bb += float(np.sum(w[shell] * cross["B", "B"][shell]))
        shot_w += float(np.sum(w[shell]))
        if i % 64 == 0:
            log(f"  plane {i}/{ng}")

    # ---- shape-noise subtraction: E/B auto monopoles only ----
    auto_ee = [spectra_1d["PEE0"]] + ([spectra_2d["PEE2D"]] if "PEE2D" in spectra_2d else [])
    auto_bb = [spectra_1d["PBB0"]] + ([spectra_2d["PBB2D"]] if "PBB2D" in spectra_2d else [])
    if shot_analytic is not None:
        p_ee = p_bb = shot_analytic
        log(f"shot-noise floor (ANALYTIC Poisson): P_shot_EE={p_ee:g}  P_shot_BB={p_bb:g}")
    elif shot_w > 0:
        p_ee, p_bb = shot_ee / shot_w, shot_bb / shot_w
        log(f"shot-noise floor (high-k shell FALLBACK, kfrac={SHOTNOISE_KFRAC}): "
            f"P_shot_EE={p_ee:g}  P_shot_BB={p_bb:g}  [WARNING: under-subtracts for sparse "
            "samples; provide the hinfo file instead]")
    else:
        log("WARNING: no analytic floor and no modes in high-k shell; shot noise NOT subtracted.")
        return
    for bd in auto_ee:
        bd.subtract_const(p_ee)
    for bd in auto_bb:
        bd.subtract_const(p_bb)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv):
    cfg = Config(argv)
    boxsize = cfg.get_d("boxsize")      # Mpc/h
    kmin, kmax = cfg.get_d("kmin"), cfg.get_d("kmax")
    nbin, munbin, ng = cfg.get_i("nbin"), cfg.get_i("munbin"), cfg.get_i("ng")
    # Normalise to plain integers so names always match what the upstream
    # stages wrote (they format SnapNum/nfold as integers).
    sn = str(cfg.get_i("sn"))
    nfold = cfg.get_i("nfold")
    if nfold <= 0:
        sys.exit(f"ERROR: nfold must be a positive integer (got '{cfg.get('nfold')}')")
    # k = 2 pi / boxlen: boxsize in Mpc/h, the units of the soap_galaxies.py
    # catalogue (FLAMINGO L2p8: 2800 cMpc * h = 1906.8 Mpc/h).
    boxlen = boxsize / nfold
    cfg.dump()
    log(f"derived: boxlen={boxlen:g} (Mpc/h), nfold={nfold}")

    base = cfg.get("base")
    gal_dir = base + cfg.get("gal_dir")
    out_dir = base + cfg.get("out_dir")
    hinfo_dir = base + cfg.kv.get("hinfo_dir", cfg.get("gal_dir"))

    check_boxsize(os.path.join(hinfo_dir, f"hinfo_fixng_ms_sub_red_sn{sn}_nfold{nfold}.dat"), boxsize)

    shot = None
    if cfg.get_i("shot_noise") != 0:
        shot = read_shot_noise(
            os.path.join(hinfo_dir, f"hinfo_fixng_ms_sub_red_sn{sn}_nfold{nfold}.dat"), boxsize)
    else:
        log("shot_noise=0 in config: analytic subtraction disabled (using high-k-shell estimate).")

    tail = f"_nfold{nfold}.bin"
    files = {
        "h": gal_dir + cfg.get("hdens_stem") + sn + tail,
        "e1": gal_dir + cfg.get("e1_stem") + sn + tail,
        "e2": gal_dir + cfg.get("e2_stem") + sn + tail,
    }
    os.makedirs(os.path.join(out_dir, "2d"), exist_ok=True)

    log("mapping fields...")
    fields = {k: map_field(v, ng) for k, v in files.items()}
    log("fields mapped.")

    spectra_1d = {name: BinnedData(nbin, kmin, kmax, True) for name in SPECTRA_1D}
    spectra_2d = {name: BinnedData2D(nbin, kmin, kmax, True, munbin) for name in SPECTRA_2D}

    log("calculating power spectra...")
    calc_power(fields, boxlen, nfold, spectra_1d, spectra_2d, shot)

    phh = spectra_1d["Phh"]
    filled = phh.num > 0
    phh_sum = float(np.sum(phh.ysum[filled] / (phh.num[filled] + 1e-30)))
    log(f"Phh check: sum over bins = {phh_sum:g}")
    if phh_sum == 0.0:
        log(f"WARNING: Phh is identically zero; the galaxy field is probably empty. "
            f"Check {files['h']}")

    suffix = f"_sn{sn}_nfold{nfold}.dat"
    log(f"writing output to {out_dir}")
    for name, bd in spectra_1d.items():
        bd.dump(os.path.join(out_dir, name + suffix))
    for name, bd in spectra_2d.items():
        bd.dump(os.path.join(out_dir, "2d", name + suffix))
    log("Yatta!")


if __name__ == "__main__":
    main(sys.argv)
