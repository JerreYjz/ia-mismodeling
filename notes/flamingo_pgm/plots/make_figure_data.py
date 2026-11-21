#!/usr/bin/env python3
"""Figure data of the FLAMINGO P_gm note (flamingo_pgm.tex).

Inputs:
  analysis/cut_validation/galaxy_bias/plots/beta/data.json  (flamingo_beta.py:
      b1, beta_gm = b_gm/b1, beta_gg = (b_gg/b1)^2 and the Poisson-subtracted
      r, moved to total matter, stitched over the folds)
  flamingo/data/PS_matter/  (matter_cross.py: P_dm, P_gm, P_gg per fold)

Writes plots/bias_k/data.json and plots/cross_r/data.json, and prints the
numbers quoted in the note (its numbers block).

    python make_figure_data.py
"""
import json
import os

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BETA = os.path.join(REPO, "analysis", "cut_validation", "galaxy_bias", "plots", "beta", "data.json")
PS = os.path.join(REPO, "flamingo", "data", "PS_matter")
SNAPS = {"1.5": 48, "2": 38, "3": 18}
OM = 0.306                      # FLAMINGO D3A
CUTS = {"w": 0.2, "gt": 0.5}    # the K0 of w and gamma_t in the 3x2pt baseline: k_max = K0/D(z)
BOX, NG = 681.0, {1: 1024, 4: 1024, 16: 512}


def growth(z):
    """Linear growth of flat LCDM, D(0) = 1."""
    def d(a):
        e = lambda x: (OM/x**3 + 1 - OM)**-1.5/x**3
        return 2.5*OM*np.sqrt(OM/a**3 + 1 - OM)*quad(e, 0, a)[0]
    return d(1/(1 + z))/d(1.0)


def at(k, y, q):
    return float(np.interp(np.log(q), np.log(k), y))


def fold_spectra(sn, f):
    p = {x: np.loadtxt(os.path.join(PS, f"{x}_sn{sn}_nfold{f}.dat")) for x in ("Pmm_dm", "Pgm", "Pgg")}
    return p["Pgm"][:, 0], {x: v[:, 1] for x, v in p.items()}


def main():
    beta = json.load(open(BETA))["rows"]
    bias, cross, nums = {}, {}, {}
    for z, sn in SNAPS.items():
        row = beta[z]
        k = np.array(row["k"])
        sel = (k >= 0.03) & (k <= 6.0)
        bgm, bgg, r = np.array(row["beta_gm"]), np.sqrt(np.array(row["beta_gg"])), np.array(row["r"])
        meta = json.load(open(os.path.join(PS, f"matter_cross_sn{sn}.json")))
        kf, P = fold_spectra(sn, 1)
        fsel = (kf >= 0.03) & (kf <= 0.25*np.pi*NG[1]/BOX)
        rraw = P["Pgm"]/np.sqrt(P["Pgg"]*P["Pmm_dm"])
        D = growth(float(z))
        kmax = {c: K0/D for c, K0 in CUTS.items()}
        bias[z] = dict(k=k[sel].tolist(), b_gm=bgm[sel].tolist(), b_gg=bgg[sel].tolist(), b1=row["b1"],
                       b1_gg=row["b1_gg"], kmax=kmax)
        cross[z] = dict(k=k[sel].tolist(), r_poisson=r[sel].tolist(), k_raw=kf[fsel].tolist(),
                        r_raw=rraw[fsel].tolist())
        # fold agreement of b_gm = P_gm/P_dm on the same grid (no matter conversion)
        fold = {}
        for f in (1, 4):
            kk, Q = fold_spectra(sn, f)
            fold[f] = (kk, Q["Pgm"]/(Q["Pmm_dm"] - meta["folds"]["1"]["shot_dm"]))
        nums[z] = dict(D=D, b1=row["b1"], b1_gg=row["b1_gg"], N_gal=meta["folds"]["1"]["N_gal"],
                       shot_gal=meta["folds"]["1"]["shot_gal"],
                       **{f"kmax_{c}": v for c, v in kmax.items()},
                       **{f"bgm_{c}": at(k, bgm, v) for c, v in kmax.items()},
                       **{f"bgg_{c}": at(k, bgg, v) for c, v in kmax.items()},
                       **{f"r_{q}": at(k, r, q) for q in (0.3, 1.0, 5.0)},
                       **{f"rraw_{q}": at(kf, rraw, q) for q in (0.3, 1.0)},
                       # the whole Poisson term at the w cut, as a fraction of b_gg (fold 1)
                       shot_w=1 - np.sqrt(1 - meta["folds"]["1"]["shot_gal"]/at(kf, P["Pgg"], kmax["w"])),
                       fold_1v4=max(abs(at(fold[4][0], fold[4][1], q)/at(fold[1][0], fold[1][1], q) - 1)
                                    for q in (0.3, 0.5, 0.8, 1.0)))
    about = ("Producer: notes/flamingo_pgm/plots/make_figure_data.py, from flamingo_beta.py "
             "(analysis/cut_validation/galaxy_bias) and matter_cross.py spectra (flamingo/data/PS_matter).")
    json.dump(dict(about=about + " b_gm = P_gm/(b1 P_m), b_gg = sqrt(P_gg - V/N_g)/(b1 sqrt(P_m)), P_m the "
                   "total matter; kmax = K0/D(z) of the w (0.2) and gamma_t (0.5) cuts.", zs=list(SNAPS), rows=bias),
              open(os.path.join(HERE, "bias_k", "data.json"), "w"), indent=1)
    json.dump(dict(about=about + " r_poisson = P_gm/sqrt((P_gg - V/N_g)(P_dm - V/N_dm)), stitched; r_raw = "
                   "P_gm/sqrt(P_gg P_dm) without shot-noise subtraction, fold 1.", zs=list(SNAPS), rows=cross),
              open(os.path.join(HERE, "cross_r", "data.json"), "w"), indent=1)
    json.dump(nums, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
    for z, v in nums.items():
        print(z, {q: (round(x, 3) if isinstance(x, float) else x) for q, x in v.items()})


if __name__ == "__main__":
    main()
