"""Where does the pipeline truth differ from the Cobaya-printed data vector?

gn_bias.py prints one number, chi2(pipeline truth vs data_file). This
script breaks it down by probe and by (theta-bin block of) tomographic
pair, and prints the ratio pipeline/data_file, so the pattern points at
the cause:

  * constant ratio over a whole w(theta) pair   -> galaxy bias differs
  * gamma_t off mostly at small theta           -> point mass / small-scale
  * every probe off by the same small fraction  -> cosmology or CAMB setting

Run from Cocoa/ with cocoa loaded (serial, one data vector):

  python truth_diag.py --path ${ROOTDIR}/projects/roman_real/data \\
      --dataset lens_kuv0p5.dataset --probe 2x2pt
"""
import argparse
import functools

import numpy as np

import cosmolike_roman_real_interface as ci
import cosmolike_roman_real_notebook_wrappers as nw
from fisher_sigma import (parameter_list, make_dv, TRUTH, B1_TRUTH,
                          install_lowk_powerlaw, match_likelihood_settings)

SECTIONS = [("xi+", 0, 540), ("xi-", 540, 1080),
            ("gammat", 1080, 1995), ("w", 1995, 2115)]
NT = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--probe", default="2x2pt")
    ap.add_argument("--path", default=None)
    ap.add_argument("--halofit", default="mead2020")
    ap.add_argument("--limber-gs", type=int, default=0, choices=[0, 1],
                    help="gamma_t projection: 0 = non-Limber below l=150 "
                         "(the roman_real yaml default), 1 = Limber (the "
                         "old GN behaviour)")
    ap.add_argument("--emul", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=100000)
    ap.add_argument("--camb-kmax", type=float, default=10.0)
    ap.add_argument("--camb-boost", type=float, default=1.1)
    ap.add_argument("--k-per-logint", type=int, default=15)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--lowk-powerlaw", type=float, default=None,
                    help="replace P(k) below this k [h/Mpc] by a power law "
                         "(as Cobaya's extrap_kmin does), e.g. 3e-4")
    a = ap.parse_args()

    nw.cnu.get_camb_cosmology = functools.partial(
        nw.cnu.get_camb_cosmology, halofit_version=a.halofit)
    install_lowk_powerlaw(a.lowk_powerlaw, verbose=True)
    cfg = dict(data_file=a.dataset, lmax=a.lmax, non_linear_emul=a.emul,
               IA_model=1, IA_code=0, IA_redshift_evolution=3)
    if a.path:
        cfg["path"] = a.path
    nw.configure(**cfg)
    nw.init_cosmolike(CLprobe=a.probe, with_data=True)
    match_likelihood_settings(adopt_limber_gs=a.limber_gs)

    # settings the pipeline uses for lens probes (compare with the yaml)
    for name in ("mnu", "PM_FID", "ZEROS8"):
        if hasattr(nw, name):
            print(f"nw.{name} = {getattr(nw, name)}")
    print(f"B1 truth = {B1_TRUTH}")

    invcov = np.asarray(ci.get_inv_cov_masked(), dtype=np.float64)
    params = parameter_list("tatt", a.probe)
    names = [p[0] for p in params]
    theta = np.array([p[1] for p in params], dtype=float)
    d = make_dv("tatt", a.probe, a)(theta, names)
    d_file = np.asarray(ci.get_dv_masked(), dtype=np.float64) * (d != 0)
    r = d - d_file
    print(f"\ntotal chi2 = {r @ invcov @ r:.4g}  "
          f"({np.count_nonzero(d)} entries)")

    print("\nper probe (cross-probe terms left out):")
    for name, s, e in SECTIONS:
        rs = np.zeros_like(r)
        rs[s:e] = r[s:e]
        keep = d[s:e] != 0
        if not keep.any():
            continue
        ratio = d[s:e][keep] / np.where(d_file[s:e][keep] == 0, np.nan,
                                        d_file[s:e][keep])
        print(f"  {name:7s} chi2 = {rs @ invcov @ rs:10.4g}   "
              f"pipeline/file: median {np.nanmedian(ratio):.5f}, "
              f"range [{np.nanmin(ratio):.4f}, {np.nanmax(ratio):.4f}]")

    # which IA piece carries the mismatch? Fit the file-minus-pipeline
    # residual on the gamma_t entries as a combination of the pipeline's
    # own IA pieces: dA1 (NLA part), dA2 (tidal torquing), dBTA (density
    # weighting). A coefficient c means "the file has (1 + c) times the
    # pipeline's amount of that piece".
    def with_ia(A1, A2, bTA):
        t = dict(zip(names, theta))
        t.update(A1=A1, A2=A2, bTA=bTA)
        return make_dv("tatt", a.probe, a)(np.array(list(t.values())),
                                            list(t.keys()))
    d0 = with_ia(0.0, 0.0, 0.0)
    pieces = {"A1 (NLA part)": with_ia(TRUTH["A1"], 0.0, 0.0) - d0,
              "A2 (C2 mix)": with_ia(TRUTH["A1"], TRUTH["A2"], 0.0)
                             - with_ia(TRUTH["A1"], 0.0, 0.0),
              "bTA (dens. wt)": with_ia(TRUTH["A1"], 0.0, TRUTH["bTA"])
                                - with_ia(TRUTH["A1"], 0.0, 0.0)}
    g = np.zeros_like(r)
    g[1080:1995] = 1.0
    G = np.column_stack([v*g for v in pieces.values()])
    R = -r*g                          # file - pipeline
    A = G.T @ invcov @ G
    c = np.linalg.solve(A, G.T @ invcov @ R)
    res = R - G @ c
    print("\ngamma_t residual (file - pipeline) fitted with the pipeline's "
          "IA pieces:")
    for (n, _), ci_ in zip(pieces.items(), c):
        print(f"  {n:15s} coefficient {ci_:+.4f}")
    print(f"  chi2 before {R @ invcov @ R:.4g} -> after {res @ invcov @ res:.4g}")
    print("  (after << before: the file differs from the pipeline by those "
          "multiples of each IA piece)")

    # per 15-bin block (one tomographic pair), diagonal chi2
    diag = np.diag(invcov)
    rows = []
    for name, s, e in SECTIONS:
        for b in range(s, e, NT):
            idx = np.arange(b, b + NT)
            keep = idx[d[idx] != 0]
            if keep.size == 0:
                continue
            c2 = float(np.sum(r[keep]**2 * diag[keep]))
            ratio = d[keep] / np.where(d_file[keep] == 0, np.nan, d_file[keep])
            rows.append((c2, name, (b - s)//NT, keep[0] - b, ratio))
    rows.sort(key=lambda x: -x[0])
    print(f"\nlargest {a.top} pair contributions (diagonal chi2):")
    print("  probe   pair  first-kept-bin  chi2      ratio by theta bin (small -> large)")
    for c2, name, pair, first, ratio in rows[:a.top]:
        rt = " ".join(f"{x:.4f}" for x in ratio)
        print(f"  {name:7s} {pair:4d}  {first:6d}       {c2:9.3g}  {rt}")


if __name__ == "__main__":
    main()
