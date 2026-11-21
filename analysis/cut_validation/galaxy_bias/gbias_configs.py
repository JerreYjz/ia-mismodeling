#!/usr/bin/env python3
"""Named per-lens-bin cut configurations against the FLAMINGO nonlinear bias.

Evaluates each configuration of CONFIGS (gamma_t and w cuts K0 per lens
bin 1..8, xi_+- at 0.2) with gbias_perbin.Scan on every reference run of
REFS, with linear bias and with the one-loop model (b2 free): the figure of
bias of the total (ia+gb), of gb, of ia, the largest piece of gb (one
probe, one lens bin), sigma(S8), and for the one-loop model the fitted b2.
A configuration passes when the total, gb and every piece stay below 0.3
("passes"), or, judging galaxy bias as a systematic of its own, when gb and
every piece do ("passes_gb"). The one-loop columns are those of
gbias_oneloop.npz (b2_ref: the b2 fitted at the uniform 0.15 cuts);
b2_move, the largest move of b2 from b2_ref, checks the linearization.

    python gbias_configs.py      (cocoa Python, from ~/cocoa/Cocoa; writes gbias_configs.json)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gbias_perbin import FOB_MAX, Scan                               # noqa: E402

REFS = ["3x2+cmb|NLA|0.2|S1|gt0.5", "3x2+cmb|NLA|0.2|EB|gt0.5", "3x2+cmb|TATT|0.2|S1|gt0.5",
        "3x2+cmb|NLA|0.2|lin|gt0.5"]
CONFIGS = {
    "baseline":   ([0.5]*8, [0.2]*8),
    "w8_015":     ([0.5]*8, [0.2]*7 + [0.15]),
    "hiz_only":   ([0.5]*5 + [0.15, 0.15, 0.5], [0.2]*7 + [0.15]),
    "perbin":     ([0.8, 0.5, 0.5, 0.4, 0.35, 0.15, 0.15, 0.8], [0.5] + [0.3]*5 + [0.25, 0.15]),
    "perbin_w3":  ([0.5, 0.5, 0.5, 0.4, 0.35, 0.15, 0.15, 0.5], [0.3]*6 + [0.25, 0.15]),
    "perbin_w5":  ([0.5, 0.5, 0.5, 0.4, 0.35, 0.15, 0.15, 0.5], [0.5]*3 + [0.35]*3 + [0.25, 0.15]),
    "greedy_S1":  ([0.8, 0.35, 0.8, 0.4, 0.35, 0.15, 0.15, 0.8], [0.5, 0.25, 0.6, 0.35, 0.35, 0.5, 0.25, 0.15]),
    "w015_only":  ([0.5]*8, [0.15]*8),
    "uniform015": ([0.15]*8, [0.15]*8),
}


def main():
    out = dict(about=__doc__, configs={k: dict(gt=g, w=w) for k, (g, w) in CONFIGS.items()}, runs={})
    for key in REFS:
        for oneloop in (False, True):
            S = Scan(key, 0.2, oneloop=oneloop)
            tag = key + ("|oneloop" if oneloop else "")
            res = {}
            for name, (g, w) in CONFIGS.items():
                r = S.evaluate(g, w)
                ok = r["ia+gb"]["fob"] < FOB_MAX and r["gb"]["fob"] < FOB_MAX and r["max_piece"] < FOB_MAX
                res[name] = dict(total=r["ia+gb"]["fob"], gb=r["gb"]["fob"], ia=r["ia"]["fob"],
                                 gb_gt=r["gb_gt"]["fob"], gb_w=r["gb_w"]["fob"], max_piece=r["max_piece"],
                                 pieces=r["pieces"], sigma_S8=r["sigma"][1], sigma_Om=r["sigma"][0],
                                 mu=r["ia+gb"]["mu"], npts=r["npts"], passes=bool(ok), b2_fit=r.get("b2_fit"),
                                 passes_gb=bool(r["gb"]["fob"] < FOB_MAX and r["max_piece"] < FOB_MAX),
                                 b2_move=(float(max(abs(x - y) for x, y in zip(r["b2_fit"], S.b2_ref)))
                                          if oneloop else None))
                worst = max(r["pieces"], key=r["pieces"].get)
                print(f"{tag:38s} {name:11s} total {res[name]['total']:.2f} gb {res[name]['gb']:.2f} "
                      f"ia {res[name]['ia']:.2f} piece {r['max_piece']:.2f} ({worst}) sigma(S8) "
                      f"{res[name]['sigma_S8']:.4f} {'PASS' if ok else 'fail'} gb-only {'PASS' if res[name]['passes_gb'] else 'fail'}"
                      + (f" b2 move {res[name]['b2_move']:.2f}" if oneloop else ""), flush=True)
            out["runs"][tag] = res
    json.dump(out, open(os.path.join(HERE, "gbias_configs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
