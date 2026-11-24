#!/usr/bin/env python3
"""data.json of the step-3 figures (RERUN_PLAN_NBall.md) from spline_pieces.json.

Truth CP, every parameter free, CMB prior, flat priors as Gaussians; every
shift in units of its sigma, every number a 2D figure of bias in
(Omega_m, S8) or a signed shift.

plots/cp_cancel_scan, plots/cp_cancel_signed (the layout of the note's
cancel_scan and cancel_signed): the total is the step-2 Fisher shift; the
pieces are those of spline_pieces.py. "z|amplitude" is the part of the model
error along the directions of A1 and eta1 (projected first); the rest is
split by B-spline nodes in ln k (3 per decade), each node assigned to the
range of bias_pieces.RANGES that holds its centre, and II is E + B. The
pieces add up to the tent base's total, which differs from the direct total
by the base's snapshot interpolation (stored as "remainder").

plots/cp_spline_nodes: at three cuts (cs at K0 0.2 and 0.4, 3x2pt gt0.5 at 0.2), per spectrum and node: the
response (S8 shift of a unit fractional error at every snapshot), the error
(mean of c_rest over the node) at three snapshots, and the running sum of the
S8 contributions over k.

plots/cp_data_blocks: the direct error through the linear map, split by data
block (xi_+, xi_-, gamma_t, w), signed S8 and Omega_m shifts against K0, for
the four models (set 2 has no tent split).

    python make_cp_pieces_figures.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K0 = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
RANGES = [(0.0, 0.06), (0.06, 0.3), (0.3, np.inf)]
NODE_CUTS = {"cs 0.2": ("cs+cmb", "", 0.2), "cs 0.4": ("cs+cmb", "", 0.4), "3x2 gt0.5 0.2": ("3x2+cmb", "|gt0.5", 0.2)}
Z_SHOW = (0.5, 1.0, 1.5)


def fob(mu, C2n):
    mu = np.asarray(mu)
    return float(np.sqrt(mu @ np.linalg.solve(C2n, mu)))


def main():
    SP = json.load(open(os.path.join(HERE, "spline_pieces.json")))["runs"]

    def key(pre, m, K, suf):
        return f"{pre}|{m}|{K:g}|CP{suf}|fisher"

    def c2n(k):
        mapf = os.path.join(HERE, "pieces_maps", k.replace("|", "_").replace("+", "p") + ".npz")
        m = np.load(mapf)
        return m["C2"]/np.outer(m["sig"], m["sig"])

    # cancel_scan / cancel_signed
    pieces = ["z|amplitude"] + [f"{X}|{lo:g}|{hi:g}" for X in ("GI", "II") for lo, hi in RANGES]
    lines = {}
    for probe, (pre, suf) in (("cs", ("cs+cmb", "")), ("3x2", ("3x2+cmb", ""))):
        for m in ("NLA", "TATT"):
            L = dict(total=[], mu_map=[], linear=[], remainder=[], cancellation=[],
                     pieces={p: dict(fob=[], mu=[]) for p in pieces})
            for K in K0:
                k = key(pre, m, K, suf)
                r, C2n = SP[k], c2n(k)
                L["total"].append(r["direct"]["fob"])
                L["mu_map"].append(r["direct"]["mu"])
                L["linear"].append(r["base"]["fob"])
                rem = np.array(r["direct"]["mu"]) - np.array(r["base"]["mu"])
                L["remainder"].append(fob(rem, C2n))
                L["cancellation"].append(r["nodes"]["cancellation"]["xsi"])
                amp = r["amplitude"]["mu"]
                L["pieces"]["z|amplitude"]["mu"].append(amp)
                L["pieces"]["z|amplitude"]["fob"].append(fob(amp, C2n))
                kn = np.array(r["nodes"]["k_node"])
                for X in ("GI", "II"):
                    src = "II E+B" if (X == "II" and "IIB" in r["nodes"]["spectra"]) else X
                    for lo, hi in RANGES:
                        sel = [i for i in range(len(kn)) if lo <= kn[i] < hi]
                        mu = np.sum([r["nodes"]["by_spectrum_node"][f"{src}|{i}"] for i in sel], axis=0)
                        L["pieces"][f"{X}|{lo:g}|{hi:g}"]["mu"].append(mu.tolist())
                        L["pieces"][f"{X}|{lo:g}|{hi:g}"]["fob"].append(fob(mu, C2n))
            lines[f"{probe}|{m}|CP"] = L
    about = __doc__
    for name in ("cp_cancel_scan", "cp_cancel_signed"):
        os.makedirs(os.path.join(HERE, "plots", name), exist_ok=True)
        json.dump(dict(about=about, K0=K0, threshold=0.3, pieces=pieces, probes=["cs", "3x2"],
                       models=["NLA", "TATT"], truths=["CP"], lines=lines),
                  open(os.path.join(HERE, "plots", name, "data.json"), "w"), indent=1)

    # nodes at one cut
    nodes = {}
    for su, (pre, suf, K) in NODE_CUTS.items():
        for m in ("NLA", "TATT"):
            k = key(pre, m, K, suf)
            r = SP[k]
            sm = np.load(os.path.join(HERE, "spline_maps", k.replace("|", "_").replace("+", "p") + ".npz"))
            zs = sm["zs"]
            isz = [int(np.argmin(abs(zs - z))) for z in Z_SHOW]
            XS = [str(x) for x in sm["spectra"]]
            contrib = sm["contrib"]                                  # (X, s, i, 2)
            run = np.cumsum(contrib.sum((0, 1))[:, 1])
            nodes[f"{su}|{m}"] = dict(
                key=k, k_node=r["nodes"]["k_node"], spectra=XS,
                response_S8={X: sm["resp"][x, :, 1].tolist() for x, X in enumerate(XS)},
                error={X: {f"{zs[s]:g}": sm["perr"][x, s].tolist() for s in isz} for x, X in enumerate(XS)},
                contrib_S8={X: contrib[x].sum(0)[:, 1].tolist() for x, X in enumerate(XS)},
                running_S8=run.tolist(), amplitude_S8=r["amplitude"]["mu"][1],
                total_S8=r["direct"]["mu"][1], base_S8=r["base"]["mu"][1],
                cancellation=r["nodes"]["cancellation"])
    os.makedirs(os.path.join(HERE, "plots", "cp_spline_nodes"), exist_ok=True)
    json.dump(dict(about=about, z_show=list(Z_SHOW), cuts={s: v[2] for s, v in NODE_CUTS.items()},
                   lines=nodes), open(os.path.join(HERE, "plots", "cp_spline_nodes", "data.json"), "w"), indent=1)

    # data blocks
    blocks = {}
    for su, (pre, suf) in (("cs", ("cs+cmb", "")), ("3x2 gt0.5", ("3x2+cmb", "|gt0.5"))):
        for m in ("NLA", "TATT", "NLAz", "TATTz"):
            L = dict(total=[], blocks={})
            for K in K0:
                r = SP[key(pre, m, K, suf)]
                L["total"].append(r["direct"]["mu"])
                for b, v in r["data_blocks"]["block"].items():
                    L["blocks"].setdefault(b, [None]*len(K0))
                for b in L["blocks"]:
                    L["blocks"][b][K0.index(K)] = r["data_blocks"]["block"].get(b, [0.0, 0.0])
            blocks[f"{su}|{m}"] = L
    os.makedirs(os.path.join(HERE, "plots", "cp_data_blocks"), exist_ok=True)
    json.dump(dict(about=about, K0=K0, threshold=0.3, setups=["cs", "3x2 gt0.5"],
                   models=["NLA", "TATT", "NLAz", "TATTz"], lines=blocks),
              open(os.path.join(HERE, "plots", "cp_data_blocks", "data.json"), "w"), indent=1)
    print("wrote cp_cancel_scan, cp_cancel_signed, cp_spline_nodes, cp_data_blocks")


if __name__ == "__main__":
    main()
