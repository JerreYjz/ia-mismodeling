#!/usr/bin/env python3
"""Per-lens-bin gamma_t and w cuts against the FLAMINGO nonlinear galaxy bias.

The linear Fisher shift of gbias_scan.py (every parameter free, the CMB
prior, xi_+- at K_XI), with the cuts set per lens bin i = 1..8: gamma_t
keeps the nested rows of gt|Kg_i (sources behind lenses; a row kept at a
smaller cut stays kept, as fisher_minimize.w_mask does for w, because the
IR/UV fraction of gamma_t is not monotonic in K0 either), w the nested rows
of w|Kw_i. Data-minus-model vectors as in gbias_scan.py (gb, gb_gt, gb_w,
ia, ia+gb), and the pieces of gb: its rows of one probe and one lens bin
("gt3", "w8", ...), whose shifts add up to that of gb.

The bias model of the fit is linear (b1 per bin), or with --oneloop the
one-loop expansion of gbias_oneloop.py: b2 free per bin (flat prior
B2_RANGE), bs2 and b3 from b1, the model evaluated at b2_ref, so that
data minus model is gb - dv1. Three experiments:

  single   one bin raised from the floor (all others at the floor FLOOR):
           the figure of bias of each bin's rows alone, for gamma_t and w,
           and the chi2 of the nonlinear-bias change in those rows
           (dchi2 = E^T C^-1 E on the kept rows minus the floor's);
  greedy   from the floor, raise one (bin, probe) cut at a time by one step:
           the step that lowers sigma(S8) most among those after which the
           total (ia+gb), gb and every piece of gb stay below FOB_MAX;
           record the path and sigma(S8);
  uniform  the global gamma_t/w grid for comparison (as gbias_scan.py).

    python gbias_perbin.py <fisher_minimize.json key> [--kxi 0.2] [--oneloop] [--variant flam]
Run from ~/cocoa/Cocoa after start_cocoa.sh (the Jacobian is reused from
gbias_jac_<key>.npz). Writes gbias_perbin.json (one entry per key).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fisher_minimize as FM                                         # noqa: E402
from gbias_scan import gt_mask, jacobian                             # noqa: E402

NXI, NGT, NDV = 1080, 915, 2115
CUTS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]
B2_RANGE = (-4.0, 6.0)
FLOOR = 0.1
FOB_MAX = 0.3
NB = 8


class Scan:
    def __init__(self, key, kxi, gbias_dir=HERE, variant="flam", oneloop=False):
        row = json.load(open(FM.OUT))["runs"][key]
        jac = jacobian(key, row)
        self.D, names = jac["D"], list(jac["names"])
        n = len(names)
        c = np.load(os.path.join(FM.RD, "v9", "ia_base_3x2_common.npz"))
        self.c, self.cov = c, c["cov"]
        wflat = np.where(np.isfinite(jac["lo"]), (jac["hi"] - jac["lo"])**2/12, np.inf)
        if row["cmb"]:
            for q in ("ns", "H0", "omegab"):
                wflat[names.index(q)] = np.inf
        P = jac["P"]
        self.Fpr = P.T @ P + np.diag(np.where(np.isfinite(wflat), 1/wflat, 0.0))
        self.G = np.zeros((2, n))
        self.G[0, names.index("omegam")] = 1.0
        self.G[1] = jac["g"]
        sfx = "" if variant == "flam" else "_" + variant
        gb = {v: np.load(os.path.join(gbias_dir, f"gbias_dv_{v}{sfx}.npz"))["dD"]
              for v in ("flam", "flamgm", "flamgg")}
        self.E = {"gb": gb["flam"], "gb_gt": gb["flamgm"], "gb_w": gb["flamgg"], "ia": jac["E_ia"],
                  "ia+gb": jac["E_ia"] + gb["flam"]}
        self.oneloop = oneloop
        if oneloop:
            ol = np.load(os.path.join(gbias_dir, "gbias_oneloop.npz"))
            ib = [names.index(f"roman_B1_{i + 1}") for i in range(NB)]
            # check: the linear b1 column of the Jacobian against the driver's (w rows: no IA there)
            wr = slice(NXI + NGT, NDV)
            self.b1_check = float(np.abs(self.D[wr][:, ib] - ol["d_b1_lin"][wr]).max()/np.abs(ol["d_b1_lin"][wr]).max())
            D = self.D.copy()
            D[:, ib] += ol["d_b1"]
            self.D = np.hstack([D, ol["d_b2"]])
            w2 = (B2_RANGE[1] - B2_RANGE[0])**2/12
            F = np.zeros((n + NB, n + NB))
            F[:n, :n] = self.Fpr
            F[n:, n:] = np.eye(NB)/w2
            self.Fpr = F
            self.G = np.hstack([self.G, np.zeros((2, NB))])
            self.b2_ref = ol["b2_ref"]
            for q in ("gb", "gb_gt", "gb_w"):
                sel = np.zeros(NDV, bool)
                sel[NXI:] = True
                if q == "gb_gt":
                    sel[NXI + NGT:] = False
                if q == "gb_w":
                    sel[NXI:NXI + NGT] = False
                self.E[q] = np.where(sel, self.E[q] - ol["dv1"], 0.0)
            self.E["ia+gb"] = self.E["ia"] + self.E["gb"]
        self.xi = c[f"xi|{kxi:g}"]
        self.gt_lens = np.repeat(c["pairs"][:, 0], 15)
        self.w_lens = np.repeat(np.arange(NB), 15)
        self.gtm = {k: gt_mask(c, k) & c["behind"] for k in CUTS}
        # the pieces of gb: one probe, one lens bin
        self.pieces = {}
        for i in range(NB):
            e = np.zeros(NDV)
            r = np.zeros(NGT, bool)
            r[self.gt_lens == i] = True
            e[NXI:NXI + NGT] = np.where(r, self.E["gb"][NXI:NXI + NGT], 0.0)
            if np.any(e):
                self.pieces[f"gt{i + 1}"] = e
            e = np.zeros(NDV)
            r = self.w_lens == i
            e[NXI + NGT:] = np.where(r, self.E["gb"][NXI + NGT:], 0.0)
            self.pieces[f"w{i + 1}"] = e
        self.wm = {k: FM.w_mask(c, k) for k in CUTS}

    def mask(self, kg, kw):
        gt = np.zeros(NGT, bool)
        w = np.zeros(NDV - NXI - NGT, bool)
        for i in range(NB):
            gt |= self.gtm[kg[i]] & (self.gt_lens == i)
            w |= self.wm[kw[i]] & (self.w_lens == i)
        return np.concatenate([self.xi, gt, w])

    def evaluate(self, kg, kw):
        m = self.mask(kg, kw)
        Ci = np.linalg.inv(self.cov[np.ix_(m, m)])
        Dm = self.D[m]
        Finv = np.linalg.inv(Dm.T @ Ci @ Dm + self.Fpr)
        C2 = self.G @ Finv @ self.G.T
        sig = np.sqrt(np.diag(C2))
        out = dict(npts=int(m.sum()), sigma=sig.tolist())
        for name, e in self.E.items():
            dp = self.G @ (Finv @ Dm.T @ Ci @ e[m])
            out[name] = dict(mu=(dp/sig).tolist(), fob=float(np.sqrt(dp @ np.linalg.solve(C2, dp))),
                             chi2=float(e[m] @ Ci @ e[m]))
        pc = {}
        for name, e in self.pieces.items():
            dp = self.G @ (Finv @ Dm.T @ Ci @ e[m])
            pc[name] = float(np.sqrt(dp @ np.linalg.solve(C2, dp)))
        out["pieces"] = pc
        out["max_piece"] = max(pc.values())
        if self.oneloop:            # the fitted b2 (reference plus the shift by gb)
            sh = Finv @ Dm.T @ Ci @ self.E["gb"][m]
            out["b2_fit"] = (self.b2_ref + sh[-NB:]).tolist()
            out["b2_sigma"] = np.sqrt(np.diag(Finv)[-NB:]).tolist()
        return out


def passes(r):
    return r["ia+gb"]["fob"] < FOB_MAX and r["gb"]["fob"] < FOB_MAX and r["max_piece"] < FOB_MAX


def main():
    a = sys.argv[1:]
    kxi = 0.2
    if "--kxi" in a:
        kxi = float(a.pop(a.index("--kxi") + 1))
        a.remove("--kxi")
    variant = "flam"
    if "--variant" in a:
        variant = a.pop(a.index("--variant") + 1)
        a.remove("--variant")
    oneloop = "--oneloop" in a
    if oneloop:
        a.remove("--oneloop")
    key = a[0]
    S = Scan(key, kxi, variant=variant, oneloop=oneloop)
    if oneloop:
        print(f"check: linear b1 columns, Jacobian vs driver (w rows), max rel diff {S.b1_check:.1e}")
    floor = [FLOOR]*NB
    base = S.evaluate(floor, floor)
    single = {}
    for probe in ("gt", "w"):
        for i in range(NB):
            for k in CUTS:
                kg, kw = list(floor), list(floor)
                (kg if probe == "gt" else kw)[i] = k
                r = S.evaluate(kg, kw)
                single[f"{probe}|{i + 1}|{k:g}"] = dict(
                    fob={q: r[q]["fob"] for q in S.E}, sigma_S8=r["sigma"][1],
                    dchi2_gb=r["gb"]["chi2"] - base["gb"]["chi2"], npts=r["npts"], max_piece=r["max_piece"],
                    piece=r["pieces"].get(f"{probe}{i + 1}"), b2_fit=r.get("b2_fit"))
    # greedy path
    kg, kw = list(floor), list(floor)
    cur = S.evaluate(kg, kw)
    path = [dict(kg=list(kg), kw=list(kw), fob_total=cur["ia+gb"]["fob"], fob_gb=cur["gb"]["fob"],
                 max_piece=cur["max_piece"], sigma_S8=cur["sigma"][1], step=None, b2_fit=cur.get("b2_fit"))]
    while True:
        best = None
        for probe in ("gt", "w"):
            for i in range(NB):
                arr = kg if probe == "gt" else kw
                j = CUTS.index(arr[i])
                if j + 1 >= len(CUTS):
                    continue
                trial_g, trial_w = list(kg), list(kw)
                (trial_g if probe == "gt" else trial_w)[i] = CUTS[j + 1]
                r = S.evaluate(trial_g, trial_w)
                if not passes(r):
                    continue
                score = r["sigma"][1]       # the step that buys the most S8 precision among the passing ones
                if best is None or score < best[0]:
                    best = (score, trial_g, trial_w, r, f"{probe}{i + 1}->{CUTS[j + 1]:g}")
        if best is None:
            break
        _, kg, kw, cur, step = best
        path.append(dict(kg=list(kg), kw=list(kw), fob_total=cur["ia+gb"]["fob"], fob_gb=cur["gb"]["fob"],
                         max_piece=cur["max_piece"], sigma_S8=cur["sigma"][1], step=step, b2_fit=cur.get("b2_fit")))
    uniform = {}
    for g in CUTS:
        for w in CUTS:
            r = S.evaluate([g]*NB, [w]*NB)
            uniform[f"{g:g}|{w:g}"] = dict(fob={q: r[q]["fob"] for q in S.E}, sigma_S8=r["sigma"][1],
                                           max_piece=r["max_piece"], pieces=r["pieces"], b2_fit=r.get("b2_fit"))
    out_path = os.path.join(HERE, "gbias_perbin.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else dict(runs={})
    out["about"] = __doc__
    tag = f"{key}|xi{kxi:g}|{variant}" + ("|oneloop" if oneloop else "")
    out["runs"][tag] = dict(CUTS=CUTS, FLOOR=FLOOR, FOB_MAX=FOB_MAX, base=dict(
        fob={q: base[q]["fob"] for q in S.E}, sigma_S8=base["sigma"][1]), single=single, greedy=path,
        uniform=uniform)
    json.dump(out, open(out_path, "w"), indent=1)
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(f"{tag}: floor (all cuts {FLOOR}) FoB ia+gb {base['ia+gb']['fob']:.2f}, gb {base['gb']['fob']:.2f}, "
          f"sigma(S8) {base['sigma'][1]:.4f}")
    for probe in ("gt", "w"):
        print(f"single-bin {probe}: FoB of gb (rows: lens bin 1..8, columns cut {CUTS})")
        for i in range(NB):
            print(f"  bin {i + 1}: " + " ".join(f"{single[f'{probe}|{i + 1}|{k:g}']['fob']['gb']:5.2f}" for k in CUTS)
                  + "   dchi2_gb " + " ".join(f"{single[f'{probe}|{i + 1}|{k:g}']['dchi2_gb']:6.1f}" for k in CUTS))
    print("greedy path:")
    for p in path:
        print(f"  {str(p['step']):10s} total {p['fob_total']:.2f} gb {p['fob_gb']:.2f} piece {p['max_piece']:.2f} "
              f"sigma(S8) {p['sigma_S8']:.4f}"
              f"  gt {p['kg']}  w {p['kw']}")


if __name__ == "__main__":
    main()
