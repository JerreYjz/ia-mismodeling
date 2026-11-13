#!/usr/bin/env python3
"""The actual error of NLA and TATT against the stitched FLAMINGO truth, on the
nodes of the tolerance bands, to overlay on them.

For each mask, the model is taken at its best fit to the Roman xi_+- with
the truth injected (case A, example1 covariance): K0/D(z) masks from
tolerance_bands.json, and no mask fitted here (roman_test.fit with every
point). Its spectra, in the cosmolike convention of the truth tables, are

    a1(z) = A1 x^eta1,  a2(z) = A2 x^eta2,  x = (1 + z)/1.62 (evolution 3),
    P_dE, P_EE from tatt_fit/fit_tatt_flamingo.TATTModel.predict (NLA: A2 =
    b_TA = 0), CAMB Takahashi halofit at the D3A cosmology,

and the error is Delta_X(k, z) = P_X^model / P_X^truth - 1 at every node and
core snapshot (0.2 <= z <= 2), X = dE (GI) and EE (II); the truth is the S1
stitch of v3/stitched_truths.npz, interpolated in ln k as it is injected.
The cosmology of the best fit is not applied to the IA spectra: the error
is that of the IA model at the fiducial cosmology, which is what the bands
perturb.
Output: model_errors.json.

    python model_errors.py <cocoa so dir> <cocoa_roman_real/data>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tolerance_bands as TB                      # noqa: E402  (sets the import paths)
from tolerance_bands import B, RT, KW, extend_full  # noqa: E402
sys.path.insert(0, os.path.join(TB.ROOT, "..", "tatt_fit"))
import fit_tatt_flamingo as F                     # noqa: E402

X0 = 1.62
ZCORE = (0.2, 2.0)


def best_params(res, mask, model):
    r = [x for x in res if x["mask"] == mask and x["case"] == "A" and x["model"] == model][0]
    return r["baseline"]["best"]


def model_ratio(tm, z, k, best, model):
    """(R_dE, R_EE) = P_X / P_NL of the model at redshift z."""
    x = (1 + z)/X0
    a1 = best["A1"]*x**best["eta1"]
    if model == "TATT":
        a2, bTA = best["A2"]*x**best["eta2"], best["bTA"]
    else:
        a2, bTA = 0.0, 0.0
    P = tm.Pnl(z, k)
    out = tm.predict(z, k, a1, a2, bTA, 1.0, PNL=P)
    return out["dE"]/P, (out["EE"])/P          # Ns = 0, so EE is the IA term


def main():
    so_dir, data = sys.argv[1], sys.argv[2]
    t0 = time.time()
    os.chdir(TB.COCOA)
    tb = json.load(open(os.path.join(HERE, "tolerance_bands.json")))
    st = np.load(os.path.join(TB.ROOT, "v3", "stitched_truths.npz"))
    zs, kd, R = st["zs"], st["k"], st["R_S1"]
    nodes = np.array(tb["nodes_h_per_Mpc"])
    core = [s for s, z in enumerate(zs) if ZCORE[0] - 1e-9 <= z <= ZCORE[1] + 1e-9]

    # the no-mask best fits (the masked ones are in tolerance_bands.json)
    cl = B.CL(so_dir, data, threads=8)
    xi_gg = cl.model("NLA", B.FID, [0.0, 0.0])           # before any sim table
    h = cl.h
    PNLw = np.array([cl.PKNL.P(z, KW*h)*h**3 for z in zs])
    p = TB.write_set(TB.TABLES, "S1_err", st["sns"],
                     np.array([extend_full(kd, R[s, 0])*PNLw[s] for s in range(len(zs))]),
                     np.array([extend_full(kd, R[s, 1])*PNLw[s] for s in range(len(zs))]))
    cl.at(B.FID)
    cl.use_sim_tables(p, st["sns"], [float(z) for z in zs], 1)
    d = cl.xi_sim()
    cov = np.load("analysis/cov_example1_xi.npy")
    m = np.ones(len(d), bool)
    Ci = np.linalg.inv(cov)
    Lw = np.linalg.inv(np.linalg.cholesky(cov))
    fits = {}
    nla_sol = (3.5, 0.68)
    for model in ("NLA", "TATT"):
        r = RT.fit(cl, model, d, Ci, Lw, m, *RT.CASES["A"], print, nla_sol=nla_sol)
        if model == "NLA":
            nla_sol = (r["best"]["A1"], r["best"]["eta1"])
        fits[model] = r
        print(f"[{time.time() - t0:.0f}s] no mask {model}: chi2 {r['chi2_bestfit']:.1f}, FoB {r['FoB']:.2f}",
              flush=True)

    tm = F.TATTModel(halofit="takahashi")
    out = dict(about=__doc__, nodes_h_per_Mpc=nodes.tolist(), z_core=[float(zs[s]) for s in core],
               results=[])
    for mask in ["none"] + [f"K0/D {k}" for k in TB.MASK_K0]:
        for model in ("NLA", "TATT"):
            best = fits[model]["best"] if mask == "none" else best_params(tb["results"], mask, model)
            err = {"GI": [], "II": []}
            for s in core:
                z = float(zs[s])
                Rt = np.array([np.interp(np.log(nodes), np.log(kd), R[s, i]) for i in (0, 1)])
                Rm = model_ratio(tm, z, nodes, best, model)
                err["GI"].append((Rm[0]/Rt[0] - 1).tolist())
                err["II"].append((Rm[1]/Rt[1] - 1).tolist())
            out["results"].append(dict(mask=mask, model=model,
                                       best={k: float(v) for k, v in best.items()}, error=err))
            e = np.abs(np.array(err["GI"]))
            print(f"  {mask:9s} {model:4s} max |Delta_GI| over core z: " +
                  " ".join(f"{v:.3f}" for v in e.max(0)), flush=True)
            e = np.abs(np.array(err["II"]))
            print(f"  {mask:9s} {model:4s} max |Delta_II| over core z: " +
                  " ".join(f"{v:.3f}" for v in e.max(0)), flush=True)
    out["no_mask_fits"] = {k: dict(chi2=v["chi2_bestfit"], FoB=v["FoB"], best=v["best"])
                           for k, v in fits.items()}
    json.dump(out, open(os.path.join(HERE, "model_errors.json"), "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
