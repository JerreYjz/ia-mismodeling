#!/usr/bin/env python3
"""Scale-dependent galaxy bias of the FLAMINGO L1_m8 galaxies, as tables for
the Roman 3x2pt contamination (gbias_contamination.py).

The analysis models gamma_t and w with linear bias, P_gm = b P_mm and
P_gg = b^2 P_mm. The measured spectra are written as
    P_gm(k, z) = b1 beta_gm(k, z) P_mm(k, z),   P_gg(k, z) = b1^2 beta_gg(k, z) P_mm(k, z),
with b1 the large-scale bias, so beta = 1 is linear bias and beta - 1 is the
nonlinear-bias error. P_mm is the total matter (SWIFT on-the-fly spectrum),
so the baryonic change of the matter spectrum stays out of beta.

Per snapshot:
  * with the galaxy-matter spectra of matter_cross.py (z = 1, 1.5, 2, 3):
        b_gm(k) = P_gm/P_dm,  b_gg^2(k) = (P_gg - V/N_g)/(P_dm - V/N_dm),
    all three on the same NGP grids and folds, so the window and the aliasing
    cancel in the ratios and the sample variance largely too; the dark-matter
    ratio is moved to total matter with the SWIFT spectra,
        x P_dm/P_tot = (Omega_m/Omega_c)^2 P_cdm^SWIFT/P_matter^SWIFT
    (SWIFT normalises P_cdm by the total mean density). b1 is the mode-weighted
    mean of b_gm over 0.01 < k < KFIT h/Mpc; r(k) = P_gm/sqrt(P_gg' P_dm') is
    the cross-correlation coefficient (shot noise removed).
  * without them (the other 9 snapshots): b_gg^2 = (P_hh - V/N_g)/(W P_tot)
    with P_hh the IA pipeline's galaxy spectrum and W(k) the window and
    aliasing of the folded NGP spectra, measured at the nearest snapshot that
    has the dark-matter grid (W = P_dm^NGP/P_dm^SWIFT); b1 from b_gg at
    0.01 < k < KFIT; beta_gm = r sqrt(beta_gg) with r of the nearest snapshot
    that has it (r = 1 when none exists yet).
The folded spectra are stitched as plot_stitch_flamingo.py: fold f is used for
0.25 k_Nyq(f/4) < k <= 0.25 k_Nyq(f). Below KLIN beta is set to 1 (the bins
there are the box's few modes; b1 is defined there).

Output: beta_tables.npz (z, k [h/Mpc], beta_gg, beta_gm, r, b1, source per
snapshot) and plots/beta/data.json.

    python flamingo_beta.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.expanduser("~/ia_power_flamingo/run_L1_m8_particles/results")
PMM = os.path.join(REPO, "flamingo", "data", "flamingo_pmm")
SNAPS = {78: 0.0, 76: 0.1, 74: 0.2, 72: 0.3, 70: 0.4, 68: 0.5, 66: 0.6, 63: 0.75,
         58: 1.0, 48: 1.5, 38: 2.0, 18: 3.0, 14: 4.0}
SWIFT = {0.0: "0122", 0.1: "0120", 0.2: "0118", 0.3: "0116", 0.4: "0114", 0.5: "0112", 0.6: "0110",
         0.75: "0107", 1.0: "0102", 1.5: "0092", 2.0: "0082", 3.0: "0062", 4.0: "0058"}
H, BOX = 0.681, 681.0
OM, OB, MNU = 0.306, 0.0486, 0.06
OC = OM - OB - MNU/(93.14*H**2)
KFIT, KLIN = 0.05, 0.03
KOUT = np.logspace(-4, 2, 241)
FOLDS_PHH = {1: 1024, 4: 1024, 16: 1024}
FOLDS_DM = {1: 1024, 4: 1024, 16: 512}


def knyq(f, ng):
    return np.pi*ng*f/BOX


def stitch(path_fmt, folds):
    """(k, P, N_modes) stitched over the folds (plot_stitch_flamingo.py rule)."""
    out = []
    for f, ng in folds.items():
        d = np.atleast_2d(np.loadtxt(path_fmt.format(f=f)))
        lo = 0.25*knyq(f/4, ng) if f > 1 else 0.0
        if f == 16 and ng == 512:
            lo = 0.25*knyq(4, 1024)
        sel = (d[:, 0] > lo) & (d[:, 0] <= 0.25*knyq(f, ng))
        out.append(d[sel])
    d = np.concatenate(out)
    return d[:, 0], d[:, 1], d[:, 3]


def swift(z, kind):
    """SWIFT spectrum (h/Mpc, (Mpc/h)^3), shot-noise subtracted; cdm in its own density."""
    d = np.loadtxt(os.path.join(PMM, f"power_{kind}_{SWIFT[z]}.txt"))
    d = d[d[:, 2] > 0]                        # shot-noise subtracted: can go negative at high k
    k, p = d[:, 1]/H, d[:, 2]*H**3
    if kind == "cdm":
        p = p*(OM/OC)**2
    return k, p


def interp_log(k, x, y):
    return np.exp(np.interp(np.log(k), np.log(x), np.log(y)))


def shot_gal(sn):
    v = open(os.path.join(RUN, "Galaxy_DS_Field", f"hinfo_fixng_ms_sub_red_sn{sn}_nfold1.dat")).read().split("\n")[1].split()
    return BOX**3/float(v[2])


def b1_fit(k, b, nm):
    sel = (k > 0.01) & (k < KFIT)
    return float(np.sum(b[sel]*nm[sel])/np.sum(nm[sel]))


def onto(kout, k, beta):
    """beta on KOUT: 1 below KLIN, measured (log-k interpolation) above, flat beyond."""
    sel = k >= KLIN
    kk, bb = np.concatenate([[KLIN*0.999], k[sel]]), np.concatenate([[1.0], beta[sel]])
    return np.where(kout < KLIN, 1.0, np.interp(np.log(kout), np.log(kk), bb))


def main():
    rows = {}
    PS = os.path.join(RUN, "PS_matter")
    have = [sn for sn in SNAPS if os.path.exists(os.path.join(PS, f"matter_cross_sn{sn}.json"))]
    W, R = {}, {}
    for sn in have:                           # the snapshots with the dark-matter grid
        z = SNAPS[sn]
        meta = json.load(open(os.path.join(PS, f"matter_cross_sn{sn}.json")))
        k, pmm, nm = stitch(os.path.join(PS, f"Pmm_dm_sn{sn}_nfold{{f}}.dat"), FOLDS_DM)
        _, pgm, _ = stitch(os.path.join(PS, f"Pgm_sn{sn}_nfold{{f}}.dat"), FOLDS_DM)
        _, pgg, _ = stitch(os.path.join(PS, f"Pgg_sn{sn}_nfold{{f}}.dat"), FOLDS_DM)
        sg, sd = meta["folds"]["1"]["shot_gal"], meta["folds"]["1"]["shot_dm"]
        kc, pc = swift(z, "cdm")
        km, pm = swift(z, "matter")
        bary = interp_log(k, kc, pc)/interp_log(k, km, pm)      # P_dm/P_tot
        bgm = pgm/(pmm - sd)
        bgg2 = (pgg - sg)/(pmm - sd)
        b1 = b1_fit(k, bgm, nm)
        r = pgm/np.sqrt((pgg - sg)*(pmm - sd))
        W[z] = (k, (pmm - sd)/interp_log(k, kc, pc))
        R[z] = (k, r)
        rows[z] = dict(sn=sn, source="P_gm", k=k, b1=b1, b1_gg=float(np.sqrt(b1_fit(k, bgg2, nm))),
                       beta_gm=bgm*bary/b1, beta_gg=bgg2*bary/b1**2, r=r, window=W[z][1], bary=bary)
    for sn, z in SNAPS.items():               # the others: P_hh only
        if z in rows:
            continue
        k, phh, nm = stitch(os.path.join(RUN, "PS", f"Phh_sn{sn}_nfold{{f}}.dat"), FOLDS_PHH)
        km, pm = swift(z, "matter")
        if W:
            zw = min(W, key=lambda q: abs(q - z))
            win = np.interp(np.log(k), np.log(W[zw][0]), W[zw][1], left=1.0, right=W[zw][1][-1])
        else:
            zw, win = None, np.ones_like(k)
        bgg2 = (phh - shot_gal(sn))/(win*interp_log(k, km, pm))
        b1 = float(np.sqrt(b1_fit(k, bgg2, nm)))
        bgg = bgg2/b1**2
        if R:
            zr = min(R, key=lambda q: abs(q - z))
            r = np.interp(np.log(k), np.log(R[zr][0]), R[zr][1], left=1.0, right=R[zr][1][-1])
        else:
            zr, r = None, np.ones_like(k)
        rows[z] = dict(sn=sn, source=f"P_hh (window from z={zw}, r from z={zr})", k=k, b1=b1, b1_gg=b1,
                       beta_gg=bgg, beta_gm=r*np.sqrt(np.clip(bgg, 0, None)), r=r, window=win)
    zs = sorted(rows)
    tab = dict(z=np.array(zs), k=KOUT,
               beta_gg=np.array([onto(KOUT, rows[z]["k"], rows[z]["beta_gg"]) for z in zs]),
               beta_gm=np.array([onto(KOUT, rows[z]["k"], rows[z]["beta_gm"]) for z in zs]),
               b1=np.array([rows[z]["b1"] for z in zs]), source=np.array([rows[z]["source"] for z in zs]))
    np.savez(os.path.join(HERE, "beta_tables.npz"), **tab)
    os.makedirs(os.path.join(HERE, "plots", "beta"), exist_ok=True)
    js = dict(about=f"Producer: flamingo_beta.py.\n{__doc__}", KFIT=KFIT, KLIN=KLIN,
              rows={f"{z:g}": {q: (v.tolist() if isinstance(v, np.ndarray) else v) for q, v in rows[z].items()}
                    for z in zs})
    json.dump(js, open(os.path.join(HERE, "plots", "beta", "data.json"), "w"), indent=1)
    for z in zs:
        d = rows[z]
        k = d["k"]
        at = {q: float(np.interp(np.log(q), np.log(k), d["beta_gg"])) for q in (0.1, 0.3, 1.0)}
        am = {q: float(np.interp(np.log(q), np.log(k), d["beta_gm"])) for q in (0.1, 0.3, 1.0)}
        print(f"z {z:4.2f} sn {d['sn']:2d} b1 {d['b1']:.3f} (gg {d['b1_gg']:.3f})  beta_gg(0.1,0.3,1) "
              + " ".join(f"{v:.3f}" for v in at.values()) + "  beta_gm " + " ".join(f"{v:.3f}" for v in am.values())
              + f"  [{d['source']}]")


if __name__ == "__main__":
    main()
