#!/usr/bin/env python3
"""Bias cancellation by B-spline nodes in ln k (step 3.2 of RERUN_PLAN_NBall.md).

For one run of bias_pieces.json (its linear map pieces_maps/<key>.npz):
    a[X, s, j]   the (Omega_m, S8) shift, in units of the run's sigma, of a
                 unit fractional error of P_X^truth on tent j at snapshot s:
                 a = M R_X[s, j] / sigma, M the linear map of bias_pieces.py
                 (every parameter free, CMB prior, flat priors as Gaussians),
                 R_X the response base (v9/ia_base*_<truth>.npz);
                 X = GI, II (E), IIB (B-mode II, truths with one).
    c[X, s, j]   the model error, 1 - P_X^model/P_X^truth at the run's
                 reference point (bias_pieces.py); the base total is
                 sum a c, the linear shift of the run up to the base's
                 snapshot interpolation.
    amplitude    first, the part of c along the amplitude directions (A1 and
                 eta1, the power law's amplitude and redshift slope, which
                 the data hold) is projected out: D = dc/dtheta (finite differences of the
                 model ratios at ref), delta = (D^T W D)^-1 D^T W c with W the
                 data weight of each tent (its chi2 per unit error, the
                 diagonal of R^T C^-1 R on the masked data), c_amp = D delta,
                 c_rest = c - c_amp. c_amp is kept as its own piece, split
                 by parameter; its shift is ~0 (the data hold A1 and eta1).
                 TATT's A2, eta2 and b_TA are not projected: in cs they rest
                 on their flat priors, an error along them is not absorbed,
                 and projecting them made an amplitude piece up to 0.46 that
                 the nodes cancelled (an artificial cancellation). Splitting
                 c_rest leaves no artificial +/- pairs from cutting a flat
                 amplitude error.
    nodes        cubic B-splines B_i in ln k, knots every 1/3 decade from
                 0.01 to 10 h/Mpc (12 functions, a partition of unity on the
                 tents): contribution[X, s, i] = sum_j B_i(k_j) a c_rest. They
                 add up to the total exactly.
    outputs      per (X, i) summed over snapshots: the contribution, the
                 response r_i = sum_s sum_j B_i a (shift of a unit error at
                 every snapshot) and the error p_i = the B_i-weighted mean of
                 c_rest, so a sign change of the error is told from one of
                 the response; per (X, s): summed over k; the running sum
                 over nodes in k; the cancellation ratio FoB(total)/rss, rss
                 = sqrt(sum FoB(piece)^2) over the (X, s, i) pieces (the FoB
                 if every piece had a random sign, the band's assumption),
                 and the same over (X, i); II E+B as one spectrum too.
    blocks       for every run, set 2 included: the direct error
                 E = d - v(ref) through M, split by data block (xi_+, xi_-,
                 gamma_t, w) and by bin pair.
Checks: pieces sum to the base total (1e-8); base against direct and against
the step-2 Fisher shift; c_amp shifts ~0; with the hard windows of
bias_pieces.RANGES the same a and c reproduce bias_pieces' "ranges".

    python spline_pieces.py <key> [<key> ...]     (run in the cocoa env)
Writes spline_pieces.json and spline_maps/<key>.npz.
"""
import fcntl
import json
import os
import sys

import numpy as np
from scipy.interpolate import BSpline

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "roman_dchi2")
sys.path.insert(0, HERE)
import bias_pieces as BP                                          # noqa: E402
import fisher_minimize as FM                                      # noqa: E402

OUT = os.path.join(HERE, "spline_pieces.json")
NPD, KLO, KHI = 3, 0.01, 10.0          # knots per decade, range [h/Mpc]
NXI, NGT = 1080, 915
IA_STEP = 0.01                         # finite-difference step of the IA parameters


def bspline_basis(k):
    """(nb, nk) cubic B-splines in ln k, clamped, a partition of unity on [KLO, KHI]."""
    br = np.linspace(np.log(KLO), np.log(KHI), int(round(NPD*np.log10(KHI/KLO))) + 1)
    t = np.concatenate([[br[0]]*3, br, [br[-1]]*3])
    nb = len(t) - 4
    x = np.clip(np.log(k), br[0], br[-1] - 1e-12)
    B = np.array([BSpline(t, np.eye(nb)[i], 3, extrapolate=False)(x) for i in range(nb)])
    B = np.nan_to_num(B)
    assert np.allclose(B.sum(0), 1, atol=1e-12)
    return B, np.exp(br)


def fob(mu, C2n):
    """2D figure of bias of a shift in sigma units; C2n the correlation matrix."""
    mu = np.atleast_2d(mu)
    return np.sqrt(np.einsum("...i,ij,...j->...", mu, np.linalg.inv(C2n), mu))


def blocks(key, row, m, E, mu_of):
    """Shift of the direct error by data block and bin pair."""
    mask = m["mask"]
    n = mask.size
    out = {}
    lab = np.empty(n, object)
    pair = np.empty(n, object)
    ij = [(i, j) for i in range(8) for j in range(i, 8)]
    for b, off in (("xi+", 0), ("xi-", 540)):
        lab[off:off + 540] = b
        pair[off:off + 540] = [f"{b} {ij[q//15][0] + 1}{ij[q//15][1] + 1}" for q in range(540)]
    if n > NXI:
        pairs = np.load(os.path.join(RD, "v9", "ia_base_3x2_common.npz"))["pairs"]
        lab[NXI:NXI + NGT] = "gt"
        pair[NXI:NXI + NGT] = [f"gt l{pairs[q//15][0] + 1}s{pairs[q//15][1] + 1}" for q in range(NGT)]
        lab[NXI + NGT:] = "w"
        pair[NXI + NGT:] = [f"w {q//15 + 1}" for q in range(n - NXI - NGT)]
    for name, labels in (("block", lab), ("pair", pair)):
        out[name] = {}
        for b in dict.fromkeys(labels[mask]):
            e = np.where(labels == b, E, 0.0)
            out[name][b] = mu_of(e).tolist()
    return out


def run(key):
    row = json.load(open(BP.OUT))["runs"][key]
    m = np.load(os.path.join(HERE, "pieces_maps", key.replace("|", "_").replace("+", "p") + ".npz"))
    M, sig, C2, mask = m["M"], m["sig"], m["C2"], m["mask"]
    C2n = C2/np.outer(sig, sig)

    def mu_of(E):
        return M @ E[mask]/sig
    mu_dir = mu_of(m["E_direct"])
    out = dict(key=key, model=row["model"], probe=row["probe"], K0=row["K0"], truth=row["truth"],
               gt=row["gt"], wc=row.get("wc", 0.2), gb=row.get("gb"),
               fob_step2=row["fob_mean_linear"], direct=dict(mu=mu_dir.tolist(), fob=float(fob(mu_dir, C2n)[0])))
    out["data_blocks"] = blocks(key, row, m, m["E_direct"], mu_of)
    bs = np.array(list(out["data_blocks"]["block"].values())).sum(0)
    out["checks"] = dict(blocks_sum=float(np.max(np.abs(bs - mu_dir))),
                         direct_vs_step2=abs(out["direct"]["fob"] - row["fob_mean_linear"]))
    if "c" not in m.files:                     # per-bin models, galaxy-bias runs: blocks only
        save(key, out)
        print(f"{key}: blocks only; FoB {out['direct']['fob']:.3f} "
              + " ".join(f"{b} {fob(np.array(v), C2n)[0]:.2f}" for b, v in out["data_blocks"]["block"].items()),
              flush=True)
        return

    # the base, its response a and the data weight W of every tent
    probe, truth = row["probe"], row["truth"]
    b = np.load(os.path.join(RD, "v9", f"{'ia_base' if probe == 'cs' else 'ia_base_3x2'}_{truth}.npz"))
    XS = [str(x) for x in m["spectra"]]
    colname = {"GI": "R_GI", "II": "R_II_E" if "IIB" in XS else "R_II", "IIB": "R_II_B"}
    c, zs, fine = m["c"].transpose(1, 0, 2), m["zs"], m["fine"]   # (X, s, j)
    if probe == "cs":
        cov = np.load(os.path.join(RD, "cocoa", "analysis", "cov_example1_xi.npy"))
    else:
        cov = np.load(os.path.join(RD, "v9", "ia_base_3x2_common.npz"))["cov"]
    Lw = np.linalg.inv(np.linalg.cholesky(cov[np.ix_(mask, mask)]))
    nX, nz, nf = len(XS), len(zs), len(fine)
    a = np.zeros((nX, nz, nf, 2))
    W = np.zeros((nX, nz, nf))
    for x, X in enumerate(XS):
        R = b[colname[X]][:, :, mask]                              # (nz, nf, nmask)
        a[x] = np.einsum("pn,sjn->sjp", M, R)/sig
        W[x] = np.sum(np.einsum("mn,sjn->sjm", Lw, R)**2, axis=-1)
    tot = np.einsum("xsjp,xsj->p", a, c)
    out["base"] = dict(mu=tot.tolist(), fob=float(fob(tot, C2n)[0]))

    # amplitude first: D = dc/dtheta_IA by central differences of the model ratios
    names = ["roman_A1_1", "roman_A1_2"]                       # A1, eta1 (evo 3)
    assert FM.IA[row["model"]].get("evo", 3) != 2
    ref = json.load(open(FM.OUT))["runs"][key]["ref"]
    hmt = None
    st = BP.BT.load_truths(os.path.join(RD, "v3", "stitched_truths.npz"))

    def cmodel(p):
        Rm = BP.model_ratios(row["model"], p, zs, fine, hmt)
        Rt = np.array([[np.interp(np.log(fine), np.log(st["k"]), st[f"R_{truth}"][s, i]) for i in (0, 1)]
                       for s in range(nz)])
        cc = [1 - Rm[:, 0]/Rt[:, 0], 1 - Rm[:, 1]/Rt[:, 1]]
        if "IIB" in XS:
            RtB = np.array([np.interp(np.log(fine), np.log(st["k"]), st[f"R_{truth}_B"][s, 1])
                            for s in range(nz)])
            cc.append(1 - BP.model_b_ratio(row["model"], p, zs, fine)/RtB)
        return np.array(cc)
    c0 = cmodel(dict(ref))
    assert np.allclose(c0, c, atol=1e-8), "the model ratios do not reproduce the stored error"
    D = np.zeros((nX, nz, nf, len(names)))
    for q, nm in enumerate(names):
        h = IA_STEP*max(1.0, abs(ref[nm]))
        pp, pm = dict(ref), dict(ref)
        pp[nm] += h
        pm[nm] -= h
        D[..., q] = (cmodel(pp) - cmodel(pm))/(2*h)
    Df, cf, Wf = D.reshape(-1, len(names)), c.ravel(), W.ravel()
    delta = np.linalg.lstsq(Df*np.sqrt(Wf)[:, None], cf*np.sqrt(Wf), rcond=None)[0]
    c_amp = (Df @ delta).reshape(c.shape)
    c_rest = c - c_amp
    mu_amp = np.einsum("xsjp,xsj->p", a, c_amp)
    mu_par = {nm: np.einsum("xsjp,xsj->p", a, D[..., q]*delta[q]).tolist() for q, nm in enumerate(names)}
    out["amplitude"] = dict(params=names, delta=delta.tolist(), mu=mu_amp.tolist(), mu_by_param=mu_par,
                            fob=float(fob(mu_amp, C2n)[0]),
                            frac_weighted_norm=float(np.sum(Wf*(Df @ delta)**2)/np.sum(Wf*cf**2)))

    # nodes
    Bk, knots = bspline_basis(fine)                                 # (nb, nf)
    nb = len(Bk)
    contrib = np.einsum("ij,xsjp,xsj->xsip", Bk, a, c_rest)         # (nX, nz, nb, 2)
    contrib_full = np.einsum("ij,xsjp,xsj->xsip", Bk, a, c)
    resp = np.einsum("ij,xsjp->xip", Bk, a)                         # (nX, nb, 2)
    perr = np.einsum("ij,xsj->xsi", Bk, c_rest)/Bk.sum(1)           # (nX, nz, nb)
    knode = np.exp(Bk @ np.log(fine)/Bk.sum(1))                     # centre of each node
    s_ = contrib.sum((0, 1, 2))
    out["checks"].update(nodes_sum=float(np.max(np.abs(s_ + mu_amp - tot))),
                         base_vs_direct=abs(out["base"]["fob"] - out["direct"]["fob"]),
                         amp_shift_fob=out["amplitude"]["fob"])
    # the hard windows of bias_pieces through the same a and c
    hard = {}
    for x, X in enumerate(XS):
        for lo, hi in BP.RANGES:
            sel = (fine >= lo) & (fine < hi)
            hard[f"{X}|{lo:g}|{hi:g}"] = np.einsum("sjp,sj->p", a[x][:, sel], c[x][:, sel])
    out["checks"]["hard_windows_vs_bias_pieces"] = float(max(
        np.max(np.abs(hard[k] - np.array(row["ranges"][k]["mu"]))) for k in hard))

    fobs_xsi = fob(contrib.reshape(-1, 2), C2n)
    by_xi = contrib.sum(1)                                          # (nX, nb, 2)
    ftot = float(fob(s_, C2n)[0])
    out["nodes"] = dict(
        spectra=XS, knots=knots.tolist(), k_node=knode.tolist(), zs=zs.tolist(),
        by_spectrum_node={f"{X}|{i}": by_xi[x, i].tolist() for x, X in enumerate(XS) for i in range(nb)},
        by_spectrum_snapshot={f"{X}|{s}": contrib[x, s].sum(0).tolist() for x, X in enumerate(XS)
                              for s in range(nz)},
        by_spectrum={X: contrib[x].sum((0, 1)).tolist() for x, X in enumerate(XS)},
        response={f"{X}|{i}": resp[x, i].tolist() for x, X in enumerate(XS) for i in range(nb)},
        running_S8=np.cumsum(contrib.sum((0, 1))[:, 1]).tolist(),
        running_Om=np.cumsum(contrib.sum((0, 1))[:, 0]).tolist(),
        fob_rest=ftot,
        cancellation=dict(
            xsi=ftot/float(np.sqrt(np.sum(fobs_xsi**2))),
            xi=ftot/float(np.sqrt(np.sum(fob(by_xi.reshape(-1, 2), C2n)**2))),
            xsi_no_amp_projection=float(fob(contrib_full.sum((0, 1, 2)), C2n)[0])
            / float(np.sqrt(np.sum(fob(contrib_full.reshape(-1, 2), C2n)**2)))))
    if "IIB" in XS:
        ii = by_xi[XS.index("II")] + by_xi[XS.index("IIB")]
        out["nodes"]["by_spectrum_node"].update({f"II E+B|{i}": ii[i].tolist() for i in range(nb)})
        out["nodes"]["by_spectrum"]["II E+B"] = ii.sum(0).tolist()
    os.makedirs(os.path.join(HERE, "spline_maps"), exist_ok=True)
    np.savez(os.path.join(HERE, "spline_maps", key.replace("|", "_").replace("+", "p") + ".npz"),
             a=a, W=W, c=c, c_amp=c_amp, contrib=contrib, contrib_full=contrib_full, resp=resp,
             perr=perr, B=Bk, fine=fine, zs=zs, spectra=np.array(XS), C2n=C2n)
    save(key, out)
    ch = out["checks"]
    print(f"{key}: FoB direct {out['direct']['fob']:.3f} base {out['base']['fob']:.3f} "
          f"(amp {out['amplitude']['fob']:.3f}, {out['amplitude']['frac_weighted_norm']:.0%} of the error; "
          + " ".join(f"{q.replace('roman_', '')} {fob(np.array(v), C2n)[0]:.2f}" for q, v in mu_par.items()) + "); "
          f"cancellation {out['nodes']['cancellation']['xsi']:.2f} (no projection "
          f"{out['nodes']['cancellation']['xsi_no_amp_projection']:.2f}); by spectrum "
          + " ".join(f"{X} {fob(np.array(v), C2n)[0]:.2f}" for X, v in out["nodes"]["by_spectrum"].items())
          + f"; checks nodes {ch['nodes_sum']:.1e} hard {ch['hard_windows_vs_bias_pieces']:.1e}", flush=True)


def save(key, out):
    with open(OUT + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        allr = json.load(open(OUT)) if os.path.exists(OUT) else dict(about=__doc__, runs={})
        allr["about"] = __doc__
        allr["runs"][key] = out
        json.dump(allr, open(OUT, "w"), indent=1)


def main():
    for key in sys.argv[1:]:
        run(key)


if __name__ == "__main__":
    main()
