#!/usr/bin/env python3
"""Linear scan of the 3x2pt bias from nonlinear galaxy bias over the gamma_t
and w cuts, with every parameter free (the setup of fisher_minimize.py).

One Jacobian of the whole 3x2pt vector (all 2115 rows) at the reference point
of a fisher_minimize.json run (IA fitted at D3A, b_i = B1_FID); for each
combination of cuts (xi_+- at K_XI, gamma_t at Kg with sources behind lenses
and the nested rows of gt_mask, w at Kw with the nested rows of
fisher_minimize.w_mask) the Fisher matrix
of those rows plus the priors, the CMB prior and the flat priors as
Gaussians (the "flatprior" sigma of fisher_minimize.py), and the linear
shift of (Omega_m, S8)
    dtheta = F^-1 J_m^T C_m^-1 E_m
for the data-minus-model vectors
    gb      the nonlinear-bias vector gbias_dv_flam.npz (gamma_t and w)
    gb_gt   its gamma_t rows only,  gb_w   its w rows only
    ia      the IA model error of the run at ref (d - v(ref))
    ia+gb   both.
Figures of bias in the 2D (Omega_m, S8) plane, in the flat-prior sigma of
each combination of cuts.

Per-bin mode (--perbin): lens bin 8 (z = 2.2 to 3.2, where w carries most
of the nonlinear-bias error at K0 = 0.2) gets its own gamma_t and w cuts
(Kg8, Kw8 in HI), bins 1 to 7 share theirs (Kg, Kw); stored under "perbin"
as "Kg|Kw|Kg8|Kw8". gbias_perbin.py sets every bin's cuts separately.

    python gbias_scan.py <fisher_minimize.json key> [--kxi 0.2] [--perbin]
Run from ~/cocoa/Cocoa after start_cocoa.sh. Writes gbias_scan.json (one
entry per key) and gbias_jac_<key>.npz (the Jacobian, reused if present).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fisher_minimize as FM                                         # noqa: E402

NXI, NGT, NDV = 1080, 915, 2115
KG = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]
KW = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]
HI = [0.1, 0.15, 0.2]
LAST = 7                     # lens bin 8 (0-based 7)


def gt_mask(c, k):
    """gamma_t rows kept at the cut k, nested: joined with those of every smaller cut."""
    ks = sorted(float(q[3:]) for q in c.files if q.startswith("gt|") and q != "gt|none")
    m = np.zeros(c["gt|none"].size, bool)
    for q in ks:
        if q <= k + 1e-9:
            m |= c[f"gt|{q:g}"]
    return m


def jacobian(key, row):
    path = os.path.join(HERE, "gbias_jac_" + key.replace("|", "_").replace("+", "p") + ".npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    S = FM.Setup(row["model"], row["K0"], row["truth"], row["probe"], row["cmb"], row.get("gt", 0.3))
    assert S.names == row["names"]
    S.mask = np.ones(NDV, bool)
    c = np.load(os.path.join(FM.RD, "v9", "ia_base_3x2_common.npz"))
    L = np.linalg.cholesky(c["cov"])
    S.Lw = np.linalg.inv(L)
    ref = np.array([row["ref"][q] for q in S.names])
    J, g = S.jac(ref)
    v_ref, _ = S.vector(ref)
    out = dict(D=-L @ J[:NDV], P=J[NDV:], g=g, names=np.array(S.names), lo=S.lo, hi=S.hi,
               E_ia=S.d - v_ref, ref=ref)
    np.savez(path, **out)
    return out


def main():
    a = sys.argv[1:]
    kxi = 0.2
    if "--kxi" in a:
        kxi = float(a.pop(a.index("--kxi") + 1))
        a.remove("--kxi")
    perbin = "--perbin" in a
    if perbin:
        a.remove("--perbin")
    key = a[0]
    row = json.load(open(FM.OUT))["runs"][key]
    jac = jacobian(key, row)
    D, P, g, names = jac["D"], jac["P"], jac["g"], list(jac["names"])
    n = len(names)
    c = np.load(os.path.join(FM.RD, "v9", "ia_base_3x2_common.npz"))
    cov = c["cov"]
    wflat = np.where(np.isfinite(jac["lo"]), (jac["hi"] - jac["lo"])**2/12, np.inf)
    if row["cmb"]:
        for q in ("ns", "H0", "omegab"):
            wflat[names.index(q)] = np.inf
    Fpr = P.T @ P + np.diag(np.where(np.isfinite(wflat), 1/wflat, 0.0))
    G = np.zeros((2, n))
    G[0, names.index("omegam")] = 1.0
    G[1] = g
    gb = {v: np.load(os.path.join(HERE, f"gbias_dv_{v}.npz"))["dD"] for v in ("flam", "flamgm", "flamgg")}
    E = {"gb": gb["flam"], "gb_gt": gb["flamgm"], "gb_w": gb["flamgg"], "ia": jac["E_ia"],
         "ia+gb": jac["E_ia"] + gb["flam"]}
    lens_gt = np.repeat(c["pairs"][:, 0], 15) == LAST
    lens_w = np.repeat(np.arange(8), 15) == LAST
    combos = [(kg, kw, kg, kw) for kg in KG for kw in KW]
    if perbin:
        combos = [(kg, kw, g8, w8) for kg in KG for kw in KW for g8 in HI for w8 in HI]
    res = {}
    for kg, kw, g8, w8 in combos:
            gtm = np.where(lens_gt, gt_mask(c, g8), gt_mask(c, kg)) & c["behind"]
            wm = np.where(lens_w, FM.w_mask(c, w8), FM.w_mask(c, kw))
            m = np.concatenate([c[f"xi|{kxi:g}"], gtm, wm])
            Ci = np.linalg.inv(cov[np.ix_(m, m)])
            Dm = D[m]
            Finv = np.linalg.inv(Dm.T @ Ci @ Dm + Fpr)
            C2 = G @ Finv @ G.T
            sig = np.sqrt(np.diag(C2))
            ent = dict(npts=int(m.sum()), sigma=sig.tolist())
            for name, e in E.items():
                dp = G @ (Finv @ Dm.T @ Ci @ e[m])
                ent[name] = dict(mu=(dp/sig).tolist(), fob=float(np.sqrt(dp @ np.linalg.solve(C2, dp))))
            res[f"{kg:g}|{kw:g}" + (f"|{g8:g}|{w8:g}" if perbin else "")] = ent
    out_path = os.path.join(HERE, "gbias_scan.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else dict(about=__doc__, runs={})
    out["about"] = __doc__
    tag = f"{key}|xi{kxi:g}" + ("|perbin" if perbin else "")
    out["runs"][tag] = dict(KG=KG, KW=KW, HI=HI if perbin else None, cuts=res)
    json.dump(out, open(out_path, "w"), indent=1)
    if perbin:
        for g8 in HI:
            for w8 in HI:
                for name in ("gb", "ia+gb"):
                    print(f"{name}, lens bin 8 at gamma_t {g8:g}, w {w8:g}: rows gamma_t cut of bins 1-7 {KG}, "
                          f"columns w cut {KW}")
                    for kg in KG:
                        print(f"  gt {kg:4g}: " + " ".join(f"{res[f'{kg:g}|{kw:g}|{g8:g}|{w8:g}'][name]['fob']:5.2f}"
                                                           for kw in KW))
        return
    for name in ("gb", "gb_gt", "gb_w", "ia", "ia+gb"):
        print(f"{name}: FoB, rows gamma_t cut {KG}, columns w cut {KW}")
        for kg in KG:
            print(f"  gt {kg:4g}: " + " ".join(f"{res[f'{kg:g}|{kw:g}'][name]['fob']:5.2f}" for kw in KW))
    print("sigma(S8):")
    for kg in KG:
        print(f"  gt {kg:4g}: " + " ".join(f"{res[f'{kg:g}|{kw:g}']['sigma'][1]:.4f}" for kw in KW))


if __name__ == "__main__":
    main()
