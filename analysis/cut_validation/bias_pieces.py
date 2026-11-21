#!/usr/bin/env python3
"""Where the bias of the every-parameter-free fits comes from: the IA model
error split by spectrum and range of k, each piece pushed through the full
Fisher matrix of fisher_minimize.py (every parameter free, the CMB prior,
the flat priors as Gaussians).

For one run of fisher_minimize.json (probe, model, K0, truth):
    ref      the run's theta_ref: D3A, Delta z = m = 0, b_i = B1_FID, the IA
             parameters fitted to the data at that fixed cosmology
    E        the model error at ref, E = d - v(ref) (data minus model, on the
             masked data vector), the vector the whole bias comes from
    pieces   E written on the response base of the truth (v9/ia_base*_<truth>
             .npz: R_X[s, j], the response of the data vector to
             P_X^truth -> P_X^truth (1 + tent_j(k)) at snapshot s, X = GI, II):
                 E = sum_{X, s, j} R_X[s, j] (1 - P_X^model/P_X^truth)(k_j, z_s),
             and split into the ranges of k of RANGES (and of BINS, finer),
             summed over snapshots. P_X^model is the native NLA or TATT
             (v7/model_errors.model_ratio, the conventions of cosmolike:
             c1rhocrit 0.01389, pivot 1.62) plus, for the halo-model variants,
             the alpha correction of hm_templates.py,
                 dR_GI = (1 - alpha_s) A_s C1_s (1 - I_1 P_lin/P_NL),
                 dR_II = -(1 - alpha_s) A_s^2 C1_s^2 (1 - P_lin/P_NL),
             and, for the freer redshift evolutions (NLAbk, NLAhz, TATTbk,
             TATThz), both amplitudes times g_s (fisher_minimize.zevo_g), so
             P_GI^model times g_s and P_II^model times g_s^2.
    shift    of each piece, linear: dtheta = -F^-1 J^T r, r the whitened
             piece, F = J^T J + priors + the flat priors as Gaussians (the
             Fflat of fisher_minimize.py, the sigma its FoB is quoted in);
             (Omega_m, S8) in units of that sigma and the 2D figure of bias
             of the piece. The pieces add up exactly to the linear shift of
             the base's E; the direct E (cosmolike) gives the same total up
             to the base's accuracy (stored: chi2 of the difference and the
             two linear figures of bias), and the minimizer's MAP figure of
             bias is copied from fisher_minimize.json for comparison.
    zshape   a second split: the model error at the largest scale (k = 0.01) of
             each snapshot, applied at every k, is one piece ("z|amplitude":
             the amplitude, i.e. redshift-evolution, error of GI and II
             together, which a split by k would cut into artificial pieces of
             one flat error); the scale dependence left over is split by
             spectrum and RANGES as above.
A cut that rests on no cancellation has every piece below the tolerance,
not only the total. The linear map from the data-minus-model vector to the
shift is saved per run in pieces_maps/, so other splits need no cosmolike.

    python bias_pieces.py <key> [<key> ...]      (keys of fisher_minimize.json)
Run from ~/cocoa/Cocoa after start_cocoa.sh. Appends to bias_pieces.json.
"""
import fcntl
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "roman_dchi2")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RD, "v7"))
sys.path.insert(0, os.path.join(HERE, "..", "tatt_fit"))
import fisher_minimize as FM                                      # noqa: E402
import model_errors as ME                                         # noqa: E402

OUT = os.path.join(HERE, "bias_pieces.json")
RANGES = [(0.0, 0.06), (0.06, 0.3), (0.3, np.inf)]
BINS = [0.0, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, np.inf]
C1RHO, OM = 0.01389, 0.306
_TM = {}


def tm():
    if "m" not in _TM:
        _TM["m"] = ME.F.TATTModel(halofit="takahashi")
    return _TM["m"]


def model_ratios(mdl, p, zs, fine, hmt):
    """(13, 2, nfine): P_X^model/P_NL of the model at its parameters p."""
    cfg = FM.IA[mdl]
    native = "TATT" if cfg["model"] == 1 else "NLA"
    best = dict(A1=p["roman_A1_1"], eta1=p["roman_A1_2"], A2=p.get("roman_A2_1", 0.0),
                eta2=p.get("roman_A2_2", 0.0), bTA=p.get("roman_BTA_1", 0.0))
    out = np.zeros((len(zs), 2, len(fine)))
    for s, z in enumerate(zs):
        out[s] = ME.model_ratio(tm(), float(z), fine, best, native)
    if cfg.get("hm"):
        if cfg["hm"] == "fixed":
            w = 1 - np.array(FM.ALPHA_FLAMINGO[cfg["alpha"]])
        else:
            w = np.full(len(zs), 1 - p["hm_alpha"])
        A = best["A1"]*((1 + zs)/FM.X0)**best["eta1"]
        C1 = np.array([C1RHO*OM/tm().D(float(z)) for z in zs])
        out[:, 0] += (w*A*C1)[:, None]*hmt["one_minus_I1Plin_over_PNL"]
        out[:, 1] -= (w*A**2*C1**2)[:, None]*hmt["one_minus_Plin_over_PNL"]
    if cfg.get("zevo"):      # a1 and a2 both times g(z) (fisher_minimize.zevo_g): GI x g, II x g^2
        g = FM.zevo_g(cfg["zevo"], zs, p)
        out[:, 0] *= g[:, None]
        out[:, 1] *= (g**2)[:, None]
    return out


def run(key):
    t0 = time.time()
    with open(FM.OUT + ".lock", "w") as lk:          # fisher_minimize.py may be writing it
        fcntl.flock(lk, fcntl.LOCK_EX)
        row = json.load(open(FM.OUT))["runs"][key]
    probe, mdl, K0, truth, gt = row["probe"], row["model"], row["K0"], row["truth"], row.get("gt", 0.3)
    S = FM.Setup(mdl, K0, truth, probe, row["cmb"], gt)
    assert S.names == row["names"]
    n = len(S.names)
    ref = np.array([row["ref"][q] for q in S.names])
    J, g = S.jac(ref)
    F = J.T @ J
    wflat = np.where(np.isfinite(S.lo), (S.hi - S.lo)**2/12, np.inf)
    if row["cmb"]:
        for q in ("ns", "H0", "omegab"):
            wflat[S.names.index(q)] = np.inf
    Finv = np.linalg.inv(F + np.diag(np.where(np.isfinite(wflat), 1/wflat, 0.0)))
    G = np.zeros((2, n))
    G[0, S.names.index("omegam")] = 1.0
    G[1] = g
    C2 = G @ Finv @ G.T
    sig = np.sqrt(np.diag(C2))
    npr = J.shape[0] - S.mask.sum()

    def shift(E):
        r = np.concatenate([S.Lw @ E[S.mask], np.zeros(npr)])
        dp = G @ (-Finv @ J.T @ r)
        return dp/sig, float(np.sqrt(dp @ np.linalg.solve(C2, dp)))

    v_ref, _ = S.vector(ref)
    E_direct = S.d - v_ref
    # the base and the truth's ratios on its grid
    bname = "ia_base" if probe == "cs" else "ia_base_3x2"
    b = np.load(os.path.join(RD, "v9", f"{bname}_{truth}.npz"))
    zs, fine = b["zs"], b["fine"]
    st = np.load(os.path.join(RD, "v3", "stitched_truths.npz"))
    Rt = np.array([[np.interp(np.log(fine), np.log(st["k"]), st[f"R_{truth}"][s, i]) for i in (0, 1)]
                   for s in range(len(zs))])
    hmt = np.load(os.path.join(HERE, f"hm_templates_{S.probe}_{truth}.npz"))
    p = dict(zip(S.names, ref))
    Rm = model_ratios(mdl, p, zs, fine, hmt)
    c = 1 - Rm/Rt                                     # (13, 2, nfine): data minus model per tent
    col = {"GI": b["R_GI"], "II": b["R_II"]}          # (13, nfine, n)

    def piece(X, lo, hi):
        sel = (fine >= lo) & (fine < hi)
        i = 0 if X == "GI" else 1
        return np.einsum("sj,sjn->n", c[:, i, sel], col[X][:, sel])
    E_base = piece("GI", 0, np.inf) + piece("II", 0, np.inf)
    dw = S.Lw @ (E_direct - E_base)[S.mask]
    ew = S.Lw @ E_direct[S.mask]
    mu_d, fob_d = shift(E_direct)
    mu_b, fob_b = shift(E_base)
    out = dict(key=key, probe=probe, model=mdl, K0=K0, truth=truth, gt=gt, sigma=sig.tolist(),
               fob_map=row["flatprior"]["fob_map"], mu_map=row["flatprior"]["mu_map"],
               fob_mean_linear=row["flatprior"]["fob_mean_linear"],
               direct=dict(mu=mu_d.tolist(), fob=fob_d), base=dict(mu=mu_b.tolist(), fob=fob_b),
               chi2_E=float(ew @ ew), chi2_base_error=float(dw @ dw), ranges={}, bins={})
    for X in ("GI", "II"):
        for name, edges in (("ranges", RANGES), ("bins", list(zip(BINS[:-1], BINS[1:])))):
            for lo, hi in edges:
                mu, fob = shift(piece(X, lo, hi))
                out[name][f"{X}|{lo:g}|{hi:g}"] = dict(mu=mu.tolist(), fob=fob)
    rng = np.array([v["mu"] for v in out["ranges"].values()])
    assert np.allclose(rng.sum(0), mu_b, atol=1e-8)
    # second split: the error at the largest scale (k = 0.01) of each snapshot,
    # applied at every k (the amplitude, i.e. redshift-evolution, error of both
    # spectra), and the scale dependence left over, by spectrum and range of k
    flat = np.repeat(c[:, :, :1], len(fine), axis=2)
    E_z = np.einsum("sj,sjn->n", flat[:, 0], col["GI"]) + np.einsum("sj,sjn->n", flat[:, 1], col["II"])
    mu, fob = shift(E_z)
    out["zshape"] = {"z|amplitude": dict(mu=mu.tolist(), fob=fob)}
    cshape = c - flat
    for X in ("GI", "II"):
        i = 0 if X == "GI" else 1
        for lo, hi in RANGES:
            sel = (fine >= lo) & (fine < hi)
            mu, fob = shift(np.einsum("sj,sjn->n", cshape[:, i, sel], col[X][:, sel]))
            out["zshape"][f"{X}|{lo:g}|{hi:g}"] = dict(mu=mu.tolist(), fob=fob)
    zsum = np.array([v["mu"] for v in out["zshape"].values()]).sum(0)
    assert np.allclose(zsum, mu_b, atol=1e-8)
    out["largest_zshape_fob"] = max(v["fob"] for v in out["zshape"].values())
    out["eps_flat"] = dict(GI=flat[:, 0, 0].tolist(), II=flat[:, 1, 0].tolist(), zs=zs.tolist())
    # the linear map from the (unwhitened, masked) data-minus-model vector to the
    # (Omega_m, S8) shift, so any other split needs no cosmolike
    M = -(G @ Finv @ J[:S.mask.sum()].T) @ S.Lw
    os.makedirs(os.path.join(HERE, "pieces_maps"), exist_ok=True)
    np.savez(os.path.join(HERE, "pieces_maps", key.replace("|", "_").replace("+", "p") + ".npz"),
             M=M, sig=sig, C2=C2, mask=S.mask, E_direct=E_direct, c=c, zs=zs, fine=fine)
    mu_r, fob_r = shift(E_direct - E_base)      # what the base misses (snapshot interpolation)
    out["remainder"] = dict(mu=mu_r.tolist(), fob=fob_r)
    out["largest_range_fob"] = max(v["fob"] for v in out["ranges"].values())
    out["rss"] = np.sqrt((rng**2).sum(0)).tolist()
    out["seconds"] = time.time() - t0
    with open(OUT + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        allr = json.load(open(OUT)) if os.path.exists(OUT) else dict(about=__doc__, runs={})
        allr["about"] = __doc__
        allr["runs"][key] = out
        json.dump(allr, open(OUT, "w"), indent=1)
    pcs = " ".join(f"{k}:{v['fob']:.2f}" for k, v in out["ranges"].items())
    print(f"{key}: FoB MAP {out['fob_map']:.3f}, linear direct {fob_d:.3f}, base {fob_b:.3f} "
          f"(chi2 E {out['chi2_E']:.2f}, base error {out['chi2_base_error']:.4f}); pieces {pcs}; "
          f"largest {out['largest_range_fob']:.2f} [{out['seconds']:.0f}s]", flush=True)


def main():
    for key in sys.argv[1:]:
        run(key)


if __name__ == "__main__":
    main()
