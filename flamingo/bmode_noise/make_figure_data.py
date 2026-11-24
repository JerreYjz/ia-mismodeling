#!/usr/bin/env python3
"""Figure data of the B-mode test: results/bmode_noise_sn*.json -> plots/bmode_pairs/data.json.

Each spectrum is stitched from the folds (nfold 1 below 0.5 h/Mpc, 4 from 0.5
to 2, 16 above), each fold kept below half its Nyquist. Written per
snapshot: P_BB of all galaxies, of shuffled orientations and of centrals
alone, (P_EE + P_BB)/2, and the close-pair prediction (V/N^2) sum c_ij j0(k r)
(pairs closer than 1 Mpc/h), in (Mpc/h)^3, with the Poisson floor that was
subtracted. Prints the numbers quoted.

    python make_figure_data.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = (78, 58, 38, 18)
FOLDS = ((1, 0.0, 0.5), (4, 0.5, 2.0), (16, 2.0, 20.0))
KRANGE = (0.1, 12.0)


def stitch(sp, name, q):
    k, p, e = [], [], []
    for nf, lo, hi in FOLDS:
        r = sp[f"{name}|{nf}"]
        for kk, pp, ee in zip(r[q]["k"], r[q]["P"], r[q]["err"]):
            if lo <= kk < hi and kk <= r["k_use_max"] and KRANGE[0] <= kk <= KRANGE[1]:
                k.append(kk), p.append(pp), e.append(ee)
    return np.array(k), np.array(p), np.array(e)


def main():
    panels = []
    for sn in SNAPS:
        d = json.load(open(os.path.join(HERE, "results", f"bmode_noise_sn{sn}.json")))
        sp = d["spectra"]
        k, bb, ebb = stitch(sp, "real", "PBB0")
        _, ee, _ = stitch(sp, "real", "PEE0")
        _, sbb, esbb = stitch(sp, "shuffled", "PBB0")
        _, cbb, ecbb = stitch(sp, "centrals", "PBB0")
        pk, pp = np.array(d["pairs"]["k"]), np.array(d["pairs"]["pred_EEBB_half"])
        pred = np.interp(np.log(k), np.log(pk), pp)
        hi = k > 4
        sh = d["pairs"]["shells"]
        tot = sum(s["sum_c"] for s in sh)
        cs = sum(s["cen-sat"] for s in sh)/tot
        r10 = sum(s["sum_c"] for s in sh if s["r_hi"] <= 0.01)/tot
        r30 = sum(s["sum_c"] for s in sh if s["r_hi"] <= 0.03)/tot
        chi = float(np.sum((sbb/esbb)**2))
        print(f"sn{sn} z={d['z']:.1f}: floor {sp['real|1']['floor']:.2f}; (EE+BB)/2 / pair prediction at k>4: "
              f"{np.round((0.5*(ee + bb)/pred)[hi], 2).tolist()}; P_BB/pred at k>0.2: "
              f"{np.round((bb/pred)[k > 0.2], 2).tolist()}; shuffled chi2 {chi:.1f}/{len(k)}; pair sum: "
              f"cen-sat {cs:.2f}, r<10 kpc/h {r10:.2f}, r<30 kpc/h {r30:.2f}; near-identical pairs "
              f"{d['pairs']['n_near_dup']} of {d['pairs']['n_near_10kpc']} closer than 10 kpc/h")
        panels.append(dict(sn=sn, z=d["z"], N=d["N"], floor=sp["real|1"]["floor"], k=k.tolist(),
                           PBB=bb.tolist(), PBB_err=ebb.tolist(), EEBB_half=(0.5*(ee + bb)).tolist(),
                           PBB_shuffled=sbb.tolist(), PBB_shuffled_err=esbb.tolist(),
                           PBB_centrals=cbb.tolist(), PBB_centrals_err=ecbb.tolist(), pair_pred=pred.tolist(),
                           frac_censat=cs, frac_r10kpc=r10, frac_r30kpc=r30))
    out = os.path.join(HERE, "plots", "bmode_pairs")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(about="Producer: flamingo/bmode_noise/make_figure_data.py from results/bmode_noise_sn*.json "
                   "(bmode_noise.py, ng = 1024, unreduced shapes of all bound stars, galaxies of the L1_m8 "
                   "catalogue). Spectra after the Poisson floor (V/N)<e1^2+e2^2>/2, in (Mpc/h)^3, stitched "
                   "from nfold 1, 4, 16. pair_pred: (V/N^2) sum_{i<j, r<1 Mpc/h} c_ij j0(k r_ij), "
                   "c_ij = e1_i e1_j + e2_i e2_j.", panels=panels),
              open(os.path.join(out, "data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
