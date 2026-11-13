"""Roman real-space 3x2pt scale-cut masks from the IR/UV split of
arXiv:2603.13031 (Saraivanov, Rubira, Miranda & Eifler 2026), for all
three probes.

The paper splits every projected spectrum at k_UV/D(z) along the Limber
integral, C = C_IR + C_UV, propagates the split to the real-space
two-point functions, and keeps an angular bin only if

    |X_UV| / (|X_UV| + |X_IR|) < alpha        (their Eq. 11)

with alpha = 0.05 (aggressive) or 0.01 (conservative). This script does
that without new C code: X_IR is the full data vector recomputed with
P_lin and P_NL set to (numerically) zero above k_UV/D(z), and
X_UV = X_full - X_IR. Every entry of the 2115-long vector gets its UV
fraction from one pair of data-vector evaluations, so xi_+-, gamma_t and
w(theta) are treated identically, with a separate k_UV per probe.

Defaults follow the IA paper's rules:
  * the split is computed on the NLA baseline (the wrapper's A1 = 0.6,
    eta = -1.5, A2 = bTA = 0), never on the TATT truth, so the mask does
    not adapt to the contamination it is meant to expose;
  * linear galaxy bias (the wrapper's B1), no point mass, no baryons;
  * monotonic cuts: one theta_min per row (every bin below the largest
    failing bin is cut), as in the paper.

Layout (rows of 15 theta bins):
  [0, 540) xi_+ | [540, 1080) xi_- | [1080, 1995) gamma_t | [1995, 2115) w

The covariance is never used. init_data_real still loads one, and the
example covariance is not positive definite with every entry kept, so the
script writes a throwaway identity covariance, an all-keep mask and a
matching .dataset (_uvcut_identity.cov, _uvcut_ones.mask, _uvcut_full.dataset)
into the data folder and loads those.

Run inside cocoa from projects/roman_real:

  python make_uv_scale_cuts.py --kuv-cs 0.3 --kuv-ggl 0.3 --kuv-gc 0.3 \\
      --alpha 0.05 --out 3x2_kuv0p3_a0p05.mask
  # a scan: one mask per k_UV (same k_UV for every probe), plus fractions
  python make_uv_scale_cuts.py --scan 0.15 0.2 0.25 0.3 0.4 0.5 \\
      --alpha 0.05 --out-prefix scan_a0p05 --save-fractions

Redshift-dependent cut. With one value per probe, the cut is k_UV/D(z) as
above. With several values and --z-nodes, each value is k_max [h/Mpc] at that
redshift (no growth factor), interpolated in z (linear in ln k by default,
--kz-interp linear for linear in k) and held constant outside the nodes:

  python make_uv_scale_cuts.py --z-nodes 0.0 0.5 1.0 1.5 2.0 \\
      --kuv-cs 0.30 0.40 0.52 0.70 0.94 --kuv-ggl 0.2 --kuv-gc 0.2 \\
      --alpha 0.05 --out cs_kz.mask
  # or from a file with columns  z  k_cs  [k_ggl  k_gc]
  python make_uv_scale_cuts.py --kz-file kmax_z.txt --alpha 0.05 --out kz.mask

Probes can mix the two forms (a list for xi_pm, a scalar for gamma_t and w).

Power-law form in the growth factor, k_max(z) = k0 D(z)^-n:
  * a scalar k with --kz-power n cuts at k/D(z)^n (n = 1 is k_UV/D(z),
    n = 0 a constant comoving k_max, as in cosmo2D_scuts.c);
  * a list of nodes with --kz-model dpow fits ln k_max = ln k0 - n ln D(z)
    to the nodes (least squares in ln k) and cuts on the fitted curve; n is
    fitted unless --kz-power fixes it. The fit is printed per probe.

  # fit k0 and n to the FLAMINGO TATT k_max per snapshot
  python make_uv_scale_cuts.py --kz-file kmax_z_tatt.txt --kz-model dpow \\
      --kuv-ggl 0.2 --kuv-gc 0.2 --alpha 0.05 --out cs_tatt_dpow.mask
  # constant comoving 0.25 h/Mpc for xi_pm
  python make_uv_scale_cuts.py --kuv-cs 0.25 --kz-power 0 --kuv-ggl 0.2 \\
      --kuv-gc 0.2 --alpha 0.05 --out cs_k0p25_flat.mask
(--kz-power applies to every scalar probe: give the lens probes as one-node
lists, or a list with --kz-model dpow, to keep them at a different power.)
"""
import argparse
import os
import numpy as np

import cosmolike_roman_real_interface as ci
import cosmolike_roman_real_notebook_wrappers as nw
cnu = nw.cnu

NTHETA = 15
SLICES = {"cosmic_shear": (0, 1080),
          "galaxy_galaxy_lensing": (1080, 1995),
          "galaxy_clustering": (1995, 2115)}
NDV = 2115
LN_FLOOR = -70.0  # P x e^-70 ~ 4e-31: zero for every integral, finite in ln


# ---------------------------------------------------------------------------
# an all-keep data set with a positive-definite (identity) covariance
# ---------------------------------------------------------------------------
def write_unmasked_dataset(data_dir, base="example1.dataset"):
    """Copy base .dataset with an identity covariance and an all-keep mask."""
    entries = {}
    with open(os.path.join(data_dir, base)) as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                key, val = line.split("=", 1)
                entries[key.strip()] = val.strip()
    idx = np.arange(NDV)
    np.savetxt(os.path.join(data_dir, "_uvcut_identity.cov"),
               np.column_stack((idx, idx, np.ones(NDV))), fmt="%d %d %.1f")
    np.savetxt(os.path.join(data_dir, "_uvcut_ones.mask"),
               np.column_stack((idx, np.ones(NDV))), fmt="%d %1.1f")
    entries["cov_file"] = "_uvcut_identity.cov"
    entries["mask_file"] = "_uvcut_ones.mask"
    name = "_uvcut_full.dataset"
    with open(os.path.join(data_dir, name), "w") as f:
        for key, val in entries.items():
            f.write(f"{key} = {val}\n")
    return name


# ---------------------------------------------------------------------------
# cosmology tables and the truncation
# ---------------------------------------------------------------------------
def camb_tables(halofit):
    return cnu.get_camb_cosmology(
        omegam=nw.omegam, omegab=nw.omegab, H0=nw.H0, ns=nw.ns,
        As_1e9=nw.As_1e9, w=nw.w, w0pwa=nw.w0pwa, mnu=nw.mnu,
        kmax=7.5, k_per_logint=10, CAMBAccuracyBoost=1.0,
        non_linear_emul=nw._CONFIG["non_linear_emul"],
        halofit_version=halofit)


def unflatten(lnP, nz, nk):
    """(n_z, n_k) table from the Fortran-flattened set_cosmology layout."""
    return np.asarray(lnP).reshape((nk, nz)).T


def flatten(tab):
    return np.ascontiguousarray(tab.T).ravel()


def growth_from_table(lnPL2d, z):
    """D(z)/D(z_min) from the low-k end of the linear table."""
    D = np.exp(0.5*(lnPL2d[:, 0] - lnPL2d[np.argmin(z), 0]))
    return D/D[np.argmin(z)]


def fit_dpow(zn, kn, z, D, power=None):
    """Fit ln k = ln k0 - n ln D(z) to the nodes (zn, kn); D(z) on the table
    redshifts z is interpolated to the nodes. Returns (k0, n, rms in ln k)."""
    o = np.argsort(z)
    Dn = np.interp(zn, np.asarray(z)[o], np.asarray(D)[o])
    x, y = -np.log(Dn), np.log(kn)
    if power is None:
        if np.ptp(x) == 0:
            raise ValueError("need nodes at two or more redshifts to fit n")
        n, lk0 = np.polyfit(x, y, 1)
    else:
        n = float(power)
        lk0 = float(np.mean(y - n*x))
    rms = float(np.sqrt(np.mean((y - (lk0 + n*x))**2)))
    return float(np.exp(lk0)), float(n), rms


def kcut_of_z(spec, z, D, z_nodes=None, interp="log", model="interp",
              power=None):
    """k_max(z) [h/Mpc] on the table redshifts.

    spec is one number k (cut at k/D(z)^n, n = power or 1: the paper's
    k_UV/D(z) by default) or a tuple of k_max values at z_nodes. A tuple
    is interpolated in z (model "interp"; interp "log": linear in ln k,
    "linear": linear in k; constant beyond the end nodes) or fitted with
    k0 D(z)^-n (model "dpow", n fitted unless power is given)."""
    D = np.asarray(D, float)
    if np.ndim(spec) == 0 or len(spec) == 1:
        n = 1.0 if power is None else float(power)
        return float(np.ravel(spec)[0])*D**(-n)
    zn, kn = np.asarray(z_nodes, float), np.asarray(spec, float)
    if zn.size != kn.size:
        raise ValueError(f"{kn.size} k values for {zn.size} z nodes")
    o = np.argsort(zn)
    zn, kn = zn[o], kn[o]
    if model == "dpow":
        k0, n, _ = fit_dpow(zn, kn, z, D, power)
        return k0*D**(-n)
    if interp == "log":
        return np.exp(np.interp(z, zn, np.log(kn)))
    return np.interp(z, zn, kn)


def truncate(lnP2d, log10k, kcut):
    """Zero P above k_max(z) (one value per table redshift): the IR part."""
    k = 10.0**log10k
    keep = k[None, :] <= np.asarray(kcut)[:, None]
    return np.where(keep, lnP2d, lnP2d + LN_FLOOR)


# ---------------------------------------------------------------------------
# one data-vector evaluation (mirrors nw.compute_probes)
# ---------------------------------------------------------------------------
def data_vector(tabs, lnPL, lnPNL, A1, A2, BTA):
    (log10k, z2d, _, _, G, zG, z1d, chi, omegan2, _) = tabs
    ci.init_ntable_lmax(int(nw._CONFIG["lmax"]))
    ci.init_photoz_conventions(int(nw._CONFIG["photoz_interpolation_type"]),
                               int(nw._CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(float(nw._CONFIG["internal_accuracyboost"]))
    ci.init_accuracy_boost(1.0, 0)
    ci.set_cosmology(omegam=nw.omegam, H0=nw.H0, log10k_2D=log10k, z_2D=z2d,
                     lnP_linear=lnPL, lnP_nonlinear=lnPNL, G=G, z_G=zG,
                     z_1D=z1d, chi=chi, omegan2=omegan2)
    ci.set_nuisance_shear_calib(M=nw.M_FID)
    ci.set_nuisance_shear_photoz(bias=nw.SHEAR_PHOTOZ_FID)
    ci.set_nuisance_clustering_photoz(bias=nw.LENS_PHOTOZ_FID)
    ci.set_nuisance_ia(A1=A1, A2=A2, B_TA=BTA)
    ci.set_nuisance_bias(B1=nw.B1_FID, B2=nw.ZEROS8, B_MAG=nw.ZEROS8,
                         B3nl=nw.ZEROS8, BK=nw.ZEROS8)
    ci.set_point_mass(PMV=nw.PM_FID)
    ci.reset_bary_struct()
    dv = np.array(ci.compute_data_vector_masked(), dtype=np.float64)
    if dv.size != NDV:
        raise RuntimeError(f"data vector has {dv.size} entries, expected {NDV}")
    return dv


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def uv_fraction(full, ir):
    uv = full - ir
    den = np.abs(uv) + np.abs(ir)
    return np.where(den > 0, np.abs(uv)/np.where(den > 0, den, 1.0), 0.0)


def contiguous_rows(keep):
    """One theta_min per row of NTHETA bins."""
    out = keep.copy()
    for r in range(0, keep.size, NTHETA):
        row = keep[r:r + NTHETA]
        fail = np.where(~row)[0]
        if fail.size:
            out[r:r + fail.max() + 1] = False
    return out


def mask_from_fractions(frac, alpha, monotonic=True):
    keep = frac < alpha
    return contiguous_rows(keep) if monotonic else keep


def select_probe(mask, probe):
    keep = {"3x2pt": list(SLICES),
            "2x2pt": ["galaxy_galaxy_lensing", "galaxy_clustering"],
            "cosmic_shear": ["cosmic_shear"],
            "galaxy_galaxy_lensing": ["galaxy_galaxy_lensing"],
            "galaxy_clustering": ["galaxy_clustering"]}[probe]
    m = mask.astype(float)
    for name, (a, b) in SLICES.items():
        if name not in keep:
            m[a:b] = 0.0
    return m


def write_mask(fname, mask):
    np.savetxt(fname, np.column_stack((np.arange(mask.size), mask)),
               fmt="%d %1.1f")


def summary(mask, label):
    parts = [f"{n.split('_')[0][:6]} {int(mask[a:b].sum())}/{b - a}"
             for n, (a, b) in SLICES.items()]
    print(f"{label:28s} " + "  ".join(parts))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuv-cs", type=float, nargs="+",
                    help="xi_+-: k_UV [h/Mpc] at z=0 (cut k_UV/D(z)), or one "
                         "k_max per --z-nodes entry (cut k_max(z))")
    ap.add_argument("--kuv-ggl", type=float, nargs="+", help="same, gamma_t")
    ap.add_argument("--kuv-gc", type=float, nargs="+", help="same, w(theta)")
    ap.add_argument("--z-nodes", type=float, nargs="+", default=None,
                    help="redshifts of the k_max lists")
    ap.add_argument("--kz-file", default=None,
                    help="columns z k_cs [k_ggl k_gc]: k_max(z) nodes (2 "
                         "columns: for every probe not given by --kuv-*)")
    ap.add_argument("--kz-interp", choices=["log", "linear"], default="log",
                    help="interpolation of k_max between nodes (log: in ln k)")
    ap.add_argument("--kz-model", choices=["interp", "dpow"], default="interp",
                    help="k lists: interpolate the nodes, or fit k0 D(z)^-n")
    ap.add_argument("--kz-power", type=float, default=None,
                    help="n of k0 D(z)^-n: fixes n in the dpow fit, and makes a "
                         "scalar k cut at k/D(z)^n (default 1, k_UV/D(z))")
    ap.add_argument("--scan", type=float, nargs="*",
                    help="k_UV list; one mask each, same k_UV for all probes")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05],
                    help="UV-fraction threshold(s); several write one mask each")
    ap.add_argument("--probe", default="3x2pt", choices=[
        "3x2pt", "2x2pt", "cosmic_shear", "galaxy_galaxy_lensing",
        "galaxy_clustering"])
    ap.add_argument("--no-monotonic", action="store_true")
    ap.add_argument("--halofit", default="takahashi",
                    help="must match the halofit_version of the chains")
    ap.add_argument("--emul", type=int, default=None,
                    help="non_linear_emul (1 EE2, 2 halofit); wrapper default")
    ap.add_argument("--path", default=None)
    ap.add_argument("--base-dataset", default="example1.dataset",
                    help=".dataset whose n(z) and binning are used (its "
                         "covariance and mask are replaced, see the docstring)")
    ap.add_argument("--out", default=None, help="mask file (single k_UV set)")
    ap.add_argument("--out-prefix", default="uvcut",
                    help="prefix of the scan masks")
    ap.add_argument("--save-fractions", action="store_true",
                    help="also write <prefix>_fractions.npz")
    a = ap.parse_args()

    z_nodes = a.z_nodes
    if a.kz_file:
        t = np.atleast_2d(np.loadtxt(a.kz_file))
        if t.shape[1] not in (2, 4):
            ap.error("--kz-file needs 2 or 4 columns: z k_cs [k_ggl k_gc]")
        z_nodes = t[:, 0].tolist()
        if t.shape[1] == 2:
            # one k_max(z) column: xi_pm always; gamma_t and w too, unless
            # they were given on the command line
            a.kuv_cs = t[:, 1].tolist()
            a.kuv_ggl = a.kuv_ggl or t[:, 1].tolist()
            a.kuv_gc = a.kuv_gc or t[:, 1].tolist()
        else:
            a.kuv_cs, a.kuv_ggl, a.kuv_gc = [t[:, c].tolist() for c in (1, 2, 3)]
    if a.scan:
        sets = [(k, k, k) for k in a.scan]
    else:
        if None in (a.kuv_cs, a.kuv_ggl, a.kuv_gc):
            ap.error("give --kuv-cs, --kuv-ggl and --kuv-gc, --kz-file, or --scan")
        for v in (a.kuv_cs, a.kuv_ggl, a.kuv_gc):
            if len(v) > 1 and (z_nodes is None or len(z_nodes) != len(v)):
                ap.error("a k list needs --z-nodes of the same length")
        sets = [tuple(v[0] if len(v) == 1 else tuple(v)
                      for v in (a.kuv_cs, a.kuv_ggl, a.kuv_gc))]


    data_dir = a.path or nw._CONFIG["path"]
    cfg = {"data_file": write_unmasked_dataset(data_dir, a.base_dataset),
           "path": data_dir}
    if a.emul is not None:
        cfg["non_linear_emul"] = a.emul
    nw.configure(**cfg)
    nw.init_cosmolike(CLprobe="3x2pt", with_data=True)

    tabs = camb_tables(a.halofit)
    log10k, z2d = np.asarray(tabs[0]), np.asarray(tabs[1])
    nz, nk = z2d.size, log10k.size
    lnPL2d, lnPNL2d = unflatten(tabs[2], nz, nk), unflatten(tabs[3], nz, nk)
    D = growth_from_table(lnPL2d, z2d)

    # NLA baseline IA (never the TATT truth), see the module docstring
    A1, A2, BTA = nw.A1_FID, nw.A2_FID, nw.BTA_FID
    full = data_vector(tabs, tabs[2], tabs[3], A1, A2, BTA)

    def ir_vector(spec):
        kc = np.broadcast_to(kcut(spec), z2d.shape)
        return data_vector(tabs,
                           flatten(truncate(lnPL2d, log10k, kc)),
                           flatten(truncate(lnPNL2d, log10k, kc)),
                           A1, A2, BTA)

    def kcut(spec):
        return kcut_of_z(spec, z2d, D, z_nodes, a.kz_interp, a.kz_model,
                         a.kz_power)

    if not a.scan:
        for name, spec in zip(("cs", "ggl", "gc"), sets[0]):
            if isinstance(spec, tuple) and a.kz_model == "dpow":
                k0, n, rms = fit_dpow(np.asarray(z_nodes, float),
                                      np.asarray(spec, float), z2d, D, a.kz_power)
                print(f"{name}: k_max(z) = {k0:.4f} D(z)^-{n:.3f} "
                      f"({'n fixed' if a.kz_power is not None else 'n fitted'}, "
                      f"rms {rms:.3f} in ln k over {len(spec)} nodes)")
            elif not isinstance(spec, tuple):
                n = 1.0 if a.kz_power is None else a.kz_power
                print(f"{name}: k_max(z) = {spec:g} D(z)^-{n:g}")

    if not a.scan:
        print("k_max(z) [h/Mpc] at the table redshifts (cs / ggl / gc):")
        for zz in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            j = int(np.argmin(np.abs(z2d - zz)))
            print(f"  z = {z2d[j]:5.2f}: " + " / ".join(
                f"{np.atleast_1d(kcut(s))[j]:.3f}"
                for s in sets[0]))

    cache = {}
    fractions = {}
    for kset in sets:
        frac = np.empty(NDV)
        for kuv, (s, e) in zip(kset, SLICES.values()):
            if kuv not in cache:
                cache[kuv] = uv_fraction(full, ir_vector(kuv))
            frac[s:e] = cache[kuv][s:e]
        fractions[kset] = frac
        for alpha in a.alpha:
            mask = select_probe(mask_from_fractions(frac, alpha,
                                                    not a.no_monotonic),
                                a.probe)
            if a.scan:
                fname = (f"{a.out_prefix}_kuv{kset[0]:.3g}_a{alpha:g}"
                         .replace(".", "p") + ".mask")
            else:
                tag = ["kz" if isinstance(k, tuple) else f"{k:g}" for k in kset]
                fname = a.out or (f"{a.out_prefix}_cs{tag[0]}_ggl{tag[1]}"
                                  f"_gc{tag[2]}_a{alpha:g}"
                                  .replace(".", "p") + ".mask")
            write_mask(fname, mask)
            summary(mask, f"kUV={kset} a={alpha:g}")
            print(f"  -> {fname}")

    if a.save_fractions:
        np.savez(f"{a.out_prefix}_fractions.npz", full=full,
                 kuv_sets=np.array([str(k) for k in fractions]),
                 fractions=np.array(list(fractions.values())))


if __name__ == "__main__":
    main()
