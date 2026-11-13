"""Why does the Cobaya-printed TATT data vector differ from gn_bias.py's?

Both paths call the same compiled cosmolike (cosmolike_roman_real_interface,
"ci"). This script records EVERY call into ci, with its arguments, from

  A  Cobaya: get_model(yaml) + one evaluation at the yaml's override point
  B  GN:     nw.configure / init_cosmolike / fisher_sigma.make_dv at TRUTH

then (1) prints the per-probe chi2 between the two data vectors, (2) diffs
the two call logs (which settings / inputs differ), and (3) swaps inputs:
B is recomputed with A's set_cosmology arguments (all of them, then one
group at a time), so the group that removes the chi2 is the cause.

Run from Cocoa/ with cocoa loaded (serial, about two data vectors per step).
The OMPI_MCA settings stop Open MPI from trying the InfiniBand (UCX) layer,
which fails outside mpirun on Puma:

  OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,tcp,self python compare_paths.py --yaml ./projects/roman_real/<your 3x2pt dv>.yaml

The yaml is the one that printed the data vector (combo_3x2pt, CAMB
mead2020, the TATT override). print_datavector is switched off here, so
nothing on disk is overwritten.
"""
import argparse
import copy
import functools
import types

import numpy as np
import yaml

import cosmolike_roman_real_interface as ci

# ---------------------------------------------------------------------------
# 1. record every call into ci
# ---------------------------------------------------------------------------
LOG = None
ORIG = {}


def _snap(x):
    if isinstance(x, np.ndarray) or (isinstance(x, (list, tuple)) and len(x) > 8):
        return np.array(x, dtype=float, copy=True)
    return copy.deepcopy(x)


def _wrap(name, f):
    @functools.wraps(f)
    def g(*a, **k):
        if LOG is not None:
            LOG.append((name, [_snap(v) for v in a],
                        {kk: _snap(v) for kk, v in k.items()}))
        return f(*a, **k)
    return g


for _n in dir(ci):
    _f = getattr(ci, _n)
    if not _n.startswith("_") and callable(_f):
        ORIG[_n] = _f
        setattr(ci, _n, _wrap(_n, _f))

import cosmolike_roman_real_notebook_wrappers as nw      # noqa: E402
from fisher_sigma import parameter_list, make_dv, match_likelihood_settings  # noqa: E402
from fisher_sigma import TRUTH, B1_TRUTH                   # noqa: E402


def truth_theta(names):
    """Truth of the TATT data vector (as in gn_bias.py, without its MPI)."""
    tr = {k: TRUTH[k] for k in ("As_1e9", "omegam", "H0", "ns", "omegab",
                                "A1", "eta1", "A2", "eta2", "bTA")}
    for i in range(8):
        tr[f"DZ{i+1}"] = 0.0
        tr[f"M{i+1}"] = 0.0
        tr[f"B1_{i+1}"] = B1_TRUTH[i]
    return np.array([tr[n] for n in names], dtype=float)

SECTIONS = [("xi+", 0, 540), ("xi-", 540, 1080),
            ("gammat", 1080, 1995), ("w", 1995, 2115)]
GROUPS = {  # set_cosmology inputs swapped together (they share grids)
    "power spectra (k,z grid, lnP_lin, lnP_lin_cb, lnP_nl)":
        ["log10k_2D", "z_2D", "lnP_linear", "lnP_linear_cb", "lnP_nonlinear"],
    "growth (G, z_G)": ["G", "z_G"],
    "distances (z_1D, chi)": ["z_1D", "chi"],
    "scalars (omegam, H0, omegab, omegan2)": ["omegam", "H0", "omegab",
                                              "omegan2"],
}


def chi2_report(tag, a, b, invcov):
    r = a - b
    out = [f"{tag}: total chi2 = {r @ invcov @ r:.4g}"]
    for n, s, e in SECTIONS:
        rs = np.zeros_like(r)
        rs[s:e] = r[s:e]
        if np.any(a[s:e] != 0):
            out.append(f"{n} {rs @ invcov @ rs:.3g}")
    print("   ".join(out))
    return float(r @ invcov @ r)


def describe(v):
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return "[] (empty)"
        return f"array n={v.size} [{v.min():.6g} .. {v.max():.6g}]"
    return repr(v)


def same(u, v):
    if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
        u, v = np.asarray(u, float), np.asarray(v, float)
        if u.shape != v.shape:
            return False, f"shape {u.shape} vs {v.shape}"
        if u.size == 0:
            return True, ""
        d = np.max(np.abs(u - v)/np.maximum(np.abs(v), 1e-300))
        return d < 1e-10, f"max rel diff {d:.2e}"
    return u == v, ""


def last_calls(log):
    """{name: (args, kwargs)} of the last call of each function."""
    out = {}
    for name, a, k in log:
        out[name] = (a, k)
    return out


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--probe", default="3x2pt")
    ap.add_argument("--match", type=int, default=1,
                    help="1 = give B the likelihood's Limber settings "
                         "(the fix); 0 = the old behaviour")
    a = ap.parse_args()

    # ---------------- path A: Cobaya -----------------------------------
    from cobaya.model import get_model
    info = yaml.safe_load(open(a.yaml))
    like_name = next(iter(info["likelihood"]))
    like = info["likelihood"][like_name]
    like["print_datavector"] = False
    point = dict(info["sampler"]["evaluate"]["override"])
    info.pop("sampler", None)
    info.pop("output", None)
    LOG = []
    model = get_model(info)
    model.logposterior(point)
    dA = np.array(ORIG["compute_data_vector_masked"](), dtype=float)
    logA, LOG = LOG, None
    print(f"A (Cobaya, {like_name}): {len(logA)} interface calls")

    # ---------------- path B: GN ---------------------------------------
    camb_extra = info["theory"]["camb"].get("extra_args", {})
    args = types.SimpleNamespace(
        halofit=camb_extra.get("halofit_version", "mead2020"),
        emul=int(like.get("non_linear_emul", 2)),
        lmax=int(like.get("lmax", 100000)),
        camb_kmax=float(camb_extra.get("kmax", 10.0)),
        camb_boost=float(camb_extra.get("AccuracyBoost", 1.1)),
        k_per_logint=int(camb_extra.get("k_per_logint", 15)))
    path = like.get("path", "./external_modules/data/roman_real")
    nw.cnu.get_camb_cosmology = functools.partial(
        nw.cnu.get_camb_cosmology, halofit_version=args.halofit)
    nw.configure(data_file=like["data_file"], path=path, lmax=args.lmax,
                 non_linear_emul=args.emul, IA_model=1, IA_code=0,
                 IA_redshift_evolution=3)
    params = parameter_list("tatt", a.probe)
    names = [p[0] for p in params]
    theta = truth_theta(names)
    dv = make_dv("tatt", a.probe, args)
    LOG = []
    nw.init_cosmolike(CLprobe=a.probe, with_data=True)
    if a.match:
        match_likelihood_settings(adopt_limber_gs=int(like.get('adopt_limber_gs', 0)))
    dB = dv(theta, names)
    logB, LOG = LOG, None
    invcov = np.asarray(ORIG["get_inv_cov_masked"](), dtype=float)
    print(f"B (GN wrappers): {len(logB)} interface calls\n")

    keep = (dA != 0) & (dB != 0)
    dA, dB = dA*keep, dB*keep
    chi2_report("A vs B", dB, dA, invcov)

    # ---------------- call-log diff ------------------------------------
    print("\n== interface calls that differ (last call of each) ==")
    LA, LB = last_calls(logA), last_calls(logB)
    for name in sorted(set(LA) | set(LB)):
        if name not in LA or name not in LB:
            print(f"  {name}: only in {'A (Cobaya)' if name in LA else 'B (GN)'}")
            continue
        (aA, kA), (aB, kB) = LA[name], LB[name]
        lines = []
        for i in range(max(len(aA), len(aB))):
            if i >= len(aA) or i >= len(aB):
                lines.append(f"    arg{i}: present in one only")
                continue
            ok, msg = same(aA[i], aB[i])
            if not ok:
                lines.append(f"    arg{i}: A {describe(aA[i])} | B "
                             f"{describe(aB[i])} {msg}")
        for kk in sorted(set(kA) | set(kB)):
            if kk not in kA or kk not in kB:
                lines.append(f"    {kk}: only in "
                             f"{'A' if kk in kA else 'B'} "
                             f"({describe(kA.get(kk, kB.get(kk)))})")
                continue
            ok, msg = same(kA[kk], kB[kk])
            if not ok:
                lines.append(f"    {kk}: A {describe(kA[kk])} | B "
                             f"{describe(kB[kk])} {msg}")
        if lines:
            print(f"  {name}:")
            print("\n".join(lines))
    seqA = [n for n, _, _ in logA]
    seqB = [n for n, _, _ in logB]
    print(f"\n  call counts: A {len(seqA)}, B {len(seqB)}; functions only "
          f"in A: {sorted(set(seqA)-set(seqB))}; only in B: "
          f"{sorted(set(seqB)-set(seqA))}")

    # ---------------- swap tests ---------------------------------------
    if "set_cosmology" not in LA or "set_cosmology" not in LB:
        print("\nset_cosmology missing from one log; no swap test")
        return
    _, kcA = LA["set_cosmology"]
    print("\n== B recomputed with A's set_cosmology inputs ==")

    def run_with(replace):
        def patched(*pa, **pk):
            pk = dict(pk)
            for key in replace:
                if key in kcA:
                    pk[key] = kcA[key]
                else:
                    pk.pop(key, None)
            return ORIG["set_cosmology"](*pa, **pk)
        ci.set_cosmology = patched
        try:
            return dv(theta, names)*keep
        finally:
            ci.set_cosmology = _wrap("set_cosmology", ORIG["set_cosmology"])

    chi2_report("  all inputs from A          ", run_with(list(kcA)), dA, invcov)
    for g, keys in GROUPS.items():
        chi2_report(f"  only {g:28s}", run_with(keys), dA, invcov)
    print("\nRead: the swap that drops chi2 to ~0 is the cause. If even 'all "
          "inputs from A' leaves chi2 ~6, the cause is in another call above "
          "(IA, bias, accuracy or FAST-PT settings).")


if __name__ == "__main__":
    main()
