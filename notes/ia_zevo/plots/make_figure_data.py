#!/usr/bin/env python3
"""data.json of the figures of the redshift-evolution note (ia_zevo.tex).

amp_error: the amplitude error of NLA per snapshot, eps(z) = 1 - P_model/P_truth
at k = 0.01 h/Mpc (bias_pieces.json, "eps_flat", the "z|amplitude" piece),
of the NLA fits at xi_+- cut 0.2 and gamma_t cut 0.5 (3x2pt, CMB prior) on the
stitched (S1) and linear-limit (lin) truths, GI and II; the figure of bias of
each snapshot's amplitude error alone (the "z|amplitude" piece split by
snapshot: the error at k = 0.01 of snapshot s applied at every k, pushed
through the linear map of pieces_maps/); and g(z) of the two freer
evolutions of fisher_minimize.zevo_g at example values.
zevo_3x2_nla, zevo_3x2_tatt: from analysis/cut_validation/zevo_summary.json, the
3x2pt runs (gamma_t 0.5, w 0.2, CMB prior) against the xi_+- cut K0, per truth:
the total MAP figure of bias, the largest piece of the bias split, sigma(S8),
whether a parameter sits at a prior bound, and the reliable cut, for NLA and
its two freer evolutions (NLAbk, NLAhz), and the same for TATT.

    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py      (from ~/cocoa/Cocoa)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.abspath(os.path.join(HERE, "..", "..", "..", "analysis", "cut_validation"))
sys.path.insert(0, CV)
import fisher_minimize as FM                                        # noqa: E402

P = json.load(open(os.path.join(CV, "bias_pieces.json")))["runs"]
out = dict(about=f"Producer: notes/ia_zevo/plots/make_figure_data.py.\n{__doc__}", z_break=FM.Z_BREAK)
for tr in ("S1", "lin"):
    key = f"3x2+cmb|NLA|0.2|{tr}|gt0.5"
    e = P[key]["eps_flat"]
    d = np.load(os.path.join(CV, "pieces_maps", key.replace("|", "_").replace("+", "p") + ".npz"))
    b = np.load(os.path.join(CV, "..", "roman_dchi2", "v9", f"ia_base_3x2_{tr}.npz"))
    fob_s = []
    for s in range(len(e["zs"])):
        E = (b["R_GI"][s].sum(0)*d["c"][s, 0, 0] + b["R_II"][s].sum(0)*d["c"][s, 1, 0])
        dp = d["M"] @ E[d["mask"]]
        fob_s.append(float(np.sqrt(dp @ np.linalg.solve(d["C2"], dp))))
    out[tr] = dict(key=key, zs=e["zs"], GI=e["GI"], II=e["II"], fob_snapshot=fob_s,
                   fob_amplitude=P[key]["zshape"]["z|amplitude"]["fob"])
zf = np.linspace(0, 4, 401)
out["z"] = zf.tolist()
out["g"] = {r"break, $\Delta\eta = +1$": FM.zevo_g("break", zf, {"zevo_deta": 1.0}).tolist(),
            r"break, $\Delta\eta = -1$": FM.zevo_g("break", zf, {"zevo_deta": -1.0}).tolist(),
            r"high $z$, $d_2 = 0.1$, $d_3 = 0.2$": FM.zevo_g("hz", zf, {"zevo_d2": 0.1, "zevo_d3": 0.2}).tolist()}
json.dump(out, open(os.path.join(HERE, "amp_error", "data.json"), "w"), indent=1)
print("wrote amp_error/data.json")
Z = json.load(open(os.path.join(CV, "zevo_summary.json")))["runs"]
for fam, models in (("nla", ["NLA", "NLAbk", "NLAhz"]), ("tatt", ["TATT", "TATTbk", "TATThz"])):
    d = dict(about=f"Producer: notes/ia_zevo/plots/make_figure_data.py.\n{__doc__}", threshold=0.3,
             truths=["S1", "EB", "lin"], models=models, K0=None, lines={})
    for m in models:
        for tr in d["truths"]:
            o = Z[f"3x2|{m}|{tr}"]
            d["K0"] = [float(k) for k in o["K0"]]
            rows = o["rows"]
            d["lines"][f"{m}|{tr}"] = dict(
                total=[r["fob"] if r else None for r in rows],
                largest=[r["largest"] if r else None for r in rows],
                sigS8=[r["sigS8"] if r else None for r in rows],
                unphysical=[bool(r["bound"]) if r else None for r in rows],
                reliable=float(o["reliable"]) if o["reliable"] else None)
    os.makedirs(os.path.join(HERE, f"zevo_3x2_{fam}"), exist_ok=True)
    json.dump(d, open(os.path.join(HERE, f"zevo_3x2_{fam}", "data.json"), "w"), indent=1)
    print(f"wrote zevo_3x2_{fam}/data.json")
