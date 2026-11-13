#!/usr/bin/env python3
"""Step 2: translate FLAMINGO-P(k) Delta chi2 into Roman Delta chi2.

For each model (NLA, TATT) and fit k_max (fits.npz, models fitted to the
smoothed FLAMINGO spectra), the SAME residual
    dP_X(k, z) = model_X - FLAMINGO_X,   X = hE, EE, BB (observed units)
enters both metrics:
    Delta chi2_FL(K)    = sum_snapshots sum_{0.05 <= k <= K} (dP / sigma_FL)^2
    Delta chi2_Roman(K) = dD^T C_Roman^-1 dD on the data vector cut at K
with dD the Limber transform of dP (dP_hE / b = dP_dE; white parts drop out),
Main ('matched'): dP kept only on 0.05 <= k <= K in BOTH metrics, white
offset common to P_EE, P_BB re-minimised (a nuisance in both). 'full': the
whole measured residual (0.05 <= k <= 17.7) on the mask at K: the extra
Delta chi2 from scales above K that leaks past the IR/UV cut.
For Roman real space (xi_+-, 1080 points, IR/UV mask at constant comoving
k_UV = K, alpha = 0.05) and Roman Fourier (E-mode bandpowers 30-4000,
540 points, same IR/UV rule per band). Gaussian covariances, 2415 deg^2.

Exchange rates:
  * models: every (model, k_max, K) gives a pair (Delta chi2_FL, Delta chi2_R);
  * amplitude: a pure IA amplitude change of the truth gives
    r_amp(K) = Delta chi2_R / Delta chi2_FL (total, and per snapshot).
The FLAMINGO threshold Delta chi2_FL* (K) maps to Delta chi2_R = 1.

    python roman_map.py --fits fits.npz --nz-real example1.nz --nz-fourier roman_example.nz
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import roman_limber as RL  # noqa: E402

FLAMINGO = dict(h=0.681, Om=0.306, Ob=0.0486, ns=0.967, sigma8=0.807, mnu=0.06)
COL = {"NLA": "#d1495b", "TATT": "#1f5fa8"}
X = ("dE", "EE", "BB")


def save(fig, path):
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def ratios(zs, k, R):
    return {x: RL.ratio_table(zs, k, R[:, i, :]) for i, x in enumerate(X)}


KMIN = 0.05


def dchi2_white(dP, err, k, K):
    """FLAMINGO Delta chi2 of a residual dP[3, nk] over KMIN <= k <= K, with the
    white shape-noise offset common to P_EE and P_BB re-minimised (a nuisance
    in both metrics: white power gives no xi_+- at theta > 0)."""
    m = (k >= KMIN) & (k <= K*(1 + 1e-9))
    w = np.concatenate([1/err[1][m]**2, 1/err[2][m]**2])
    r = np.concatenate([dP[1][m], dP[2][m]])
    c = np.sum(w*r)/np.sum(w)
    return float(np.sum((dP[0][m]/err[0][m])**2) + np.sum(w*(r - c)**2)), float(c)


def residual_tables(cosmo, zs, k, dP, norm, c, K=None, kext_lo=1e-4, kext_hi=2e3, n_ext=25):
    """Residual ratio tables for the Limber step, white offset c[s] removed
    from P_EE and P_BB.
    K given ('matched'): residual kept only on KMIN <= k <= K, zero elsewhere,
      i.e. exactly the k range of Delta chi2_FL(K).
    K None ('full'): residual on the whole measured range 0.05 <= k <= 17.7,
      zero outside (no extrapolation: beyond the data the residual is
      unknown, and extrapolating it dominated real space; leak_diag.py)."""
    lo = np.geomspace(kext_lo, KMIN, n_ext, endpoint=False)
    hi = np.geomspace(k[-1], kext_hi, n_ext + 1)[1:]
    m = k >= KMIN
    kk = np.concatenate([lo, k[m], hi])
    R = np.zeros((len(zs), 3, len(kk)))
    for s, z in enumerate(zs):
        pnl = cosmo.Pnl.P(z, kk)
        r_dE = dP[s, 0, m]/norm[s, 0, m]
        pe = [dP[s, x, m] - c[s] for x in (1, 2)]
        if K is None:
            z0 = np.zeros(len(lo)), np.zeros(len(hi))
            R[s, 0] = np.concatenate([z0[0], r_dE, z0[1]])
            for x, p in zip((1, 2), pe):
                R[s, x] = np.concatenate([z0[0], p, z0[1]])/pnl
        else:
            keep = (k[m] <= K*(1 + 1e-9)).astype(float)
            z0 = np.zeros(len(lo)), np.zeros(len(hi))
            R[s, 0] = np.concatenate([z0[0], r_dE*keep, z0[1]])
            for x, p in zip((1, 2), pe):
                R[s, x] = np.concatenate([z0[0], p*keep, z0[1]])/pnl
    return {x: RL.ratio_table(zs, kk, R[:, i, :]) for i, x in enumerate(X)}


def qform(v, cov, m):
    return float(v[m] @ np.linalg.solve(cov[np.ix_(m, m)], v[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="fits.npz")
    ap.add_argument("--nz-real", required=True)
    ap.add_argument("--nz-fourier", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="roman")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.fits)
    zs, kg = d["zs"], d["kgrid"]
    k = d["k"][0]
    assert np.allclose(d["k"], k)
    nsn, nK = len(zs), len(kg)
    norm = np.array([[d["b"][i]*d["PNL"][i], d["PNL"][i], d["PNL"][i]] for i in range(nsn)])

    print("CAMB ...", flush=True)
    cosmo = RL.Cosmology(**FLAMINGO, halofit="takahashi")
    RS = RL.RomanShear(cosmo, a.nz_real)
    FS = RL.RomanFourier(cosmo, a.nz_fourier)

    T = ratios(zs, k, d["Rtrue"])
    out = {"kgrid": kg, "zs": zs}
    sp = {}
    for name, S in (("real", RS), ("fourier", FS)):
        CEg, CBg = S.cl(None)
        CEi, CBi = S.cl(T, gg=False)
        if name == "real":
            tot = S.xi(CEg + CEi, CBg + CBi)
            cov = S.gaussian_cov(CEg + CEi, CBg + CBi)
            vec = lambda CE, CB, S=S: S.xi(CE, CB)
        else:
            tot = S.bandpowers(CEg + CEi)
            cov = S.gaussian_cov(CEg + CEi)
            vec = lambda CE, CB, S=S: S.bandpowers(CE)
        masks = []
        for K in kg:
            CEr, CBr = S.cl(T, kcut=lambda z, kk=K: np.full_like(z, kk))
            ir = vec(CEr, CBr)
            masks.append(RL.uv_mask(tot, ir, a.alpha, S.ntheta) if name == "real"
                         else RL.uv_mask_fourier(tot, ir, a.alpha, S.nband))
        masks = np.array(masks)
        ia = vec(CEi, CBi)
        sp[name] = dict(S=S, cov=cov, vec=vec, masks=masks)
        out[f"{name}_npts"] = masks.sum(1)
        out[f"{name}_snr_tot"] = np.sqrt([qform(tot, cov, m) for m in masks])
        out[f"{name}_snr_ia"] = np.sqrt([qform(ia, cov, m) for m in masks])
        print(f"{name}: points kept per K {masks.sum(1)}; IA S/N {np.round(out[name + '_snr_ia'], 1)}",
              flush=True)

    # amplitude exchange rate, total and per snapshot
    amp = np.array([d["Rtrue"][:, 0], 2*d["Rtrue"][:, 1], 2*d["Rtrue"][:, 2]]).transpose(1, 0, 2)
    ampP = amp*norm                                  # observed-unit amplitude residual
    for name in sp:
        S, cov, vec, masks = (sp[name][x] for x in ("S", "cov", "vec", "masks"))
        AR = np.zeros(nK)
        ARs = np.zeros((nsn, nK))
        for i, K in enumerate(kg):
            CE, CB = S.cl(residual_tables(cosmo, zs, k, ampP, norm, np.zeros(nsn), K=K), gg=False)
            AR[i] = qform(vec(CE, CB), cov, masks[i])
            for s_ in range(nsn):
                one = np.zeros_like(ampP)
                one[s_] = ampP[s_]
                CE, CB = S.cl(residual_tables(cosmo, zs, k, one, norm, np.zeros(nsn), K=K), gg=False)
                ARs[s_, i] = qform(vec(CE, CB), cov, masks[i])
        out[f"{name}_AR"] = AR
        out[f"{name}_AR_snap"] = ARs
        print(f"{name}: amplitude Delta chi2_R per eps^2 vs K: {np.round(AR, 1)}", flush=True)
    out["AFL"] = d["AFL"].sum(0)
    out["AFL_snap"] = d["AFL"]

    # models
    for mdl in ("NLA", "TATT"):
        dFL = d[f"{mdl}_dFL"]                         # [nsn, nkmax, nK]
        out[f"{mdl}_dFL"] = dFL.sum(0)
        out[f"{mdl}_lam"] = d[f"{mdl}_lam"].sum(0)
        out[f"{mdl}_chi2raw"] = d[f"{mdl}_chi2_raw"].sum(0)
        out[f"{mdl}_dofraw"] = d[f"{mdl}_dof"].sum(0)
        for name in sp:
            out[f"{mdl}_{name}"] = np.zeros((nK, nK))        # matched k range
            out[f"{mdl}_{name}_full"] = np.zeros((nK, nK))   # full residual (leakage)
        dfl = np.zeros((nK, nK))
        for j in range(nK):
            x = d[f"{mdl}_x_true"][:, j]
            Ns = x[:, -1]
            Mobs = d[f"{mdl}_R"][:, j]*norm
            Mobs[:, 1] += Ns[:, None]
            Mobs[:, 2] += Ns[:, None]
            dP = Mobs - d["T"]                          # [nsn, 3, nk], observed units
            for i, K in enumerate(kg):
                res = [dchi2_white(dP[s], d["err"][s], k, K) for s in range(nsn)]
                dfl[j, i] = sum(r[0] for r in res)
                c = [r[1] for r in res]
                for name in sp:
                    S, cov, vec, masks = (sp[name][x_] for x_ in ("S", "cov", "vec", "masks"))
                    CE, CB = S.cl(residual_tables(cosmo, zs, k, dP, norm, c, K=K), gg=False)
                    out[f"{mdl}_{name}"][j, i] = qform(vec(CE, CB), cov, masks[i])
                if i == j:
                    cj = c
            for name in sp:
                S, cov, vec, masks = (sp[name][x_] for x_ in ("S", "cov", "vec", "masks"))
                CE, CB = S.cl(residual_tables(cosmo, zs, k, dP, norm, cj), gg=False)
                v = vec(CE, CB)
                out[f"{mdl}_{name}_full"][j] = [qform(v, cov, m) for m in masks]
            print(f"{mdl} k_max={kg[j]:.3f}: dchi2_FL(K=k_max)={dfl[j, j]:9.2f}  Roman real "
                  f"{out[mdl + '_real'][j, j]:8.3f} (full {out[mdl + '_real_full'][j, j]:8.3f})  "
                  f"Fourier {out[mdl + '_fourier'][j, j]:8.3f} (full {out[mdl + '_fourier_full'][j, j]:8.3f})",
                  flush=True)
        out[f"{mdl}_dFL"] = dfl
    np.savez(os.path.join(a.out, "roman_map.npz"), **out)
    for name in sp:
        np.save(os.path.join(a.out, f"masks_{name}.npy"), sp[name]["masks"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
