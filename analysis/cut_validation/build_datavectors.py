#!/usr/bin/env python3
"""Data vectors, mask and datasets of the cut-validation chains.

All vectors use the chain's own theory (emulmps P(k) + cosmolike, the
use_emulator 2 path) at the D3A fiducial cosmology with no nuisance shifts:
    flam   GG + xi_IA^FLAMINGO(prod), xi_IA from v9/ia_base_prod.npz (the
           Fisher base: FLAMINGO P_dE, P_EE tabulated into cosmolike)
    nla    NLA at its K0 = 0.2 best fit to flam (A1 3.486, eta1 0.700)
    tatt   TATT at its K0 = 0.2 best fit to flam
The mask keeps xi_+- at K0 = 0.2 (masks_n1 of cosmolike_xi_ia.npz) and
drops gamma_t and w. Checks, by running the chain likelihoods themselves:
    chi2(flam | NLA, TATT at their best fits) against the Fisher fits
    (scale_cut_scan.json), which used the CAMB/Takahashi path;
    chi2 = 0 for each control at its truth;
    chi2 between the emulator GG and the CAMB-path GG (emulator accuracy).
Output files go to ~/cocoa/Cocoa/projects/roman_real/data/cutval_k0p2_*.

Run from ~/cocoa/Cocoa after start_cocoa.sh:
    python ../ia_mismodeling/analysis/cut_validation/build_datavectors.py
"""
import copy
import json
import os

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "roman_dchi2")
DATA = "./projects/roman_real/data"
LIK = "roman_real.cosmic_shear"
NXI, NTOT = 1080, 2115
K0 = 0.2

FID = dict(As_1e9=2.099, ns=0.967, H0=68.1, omegab=0.0486, omegam=0.306)
NUIS = {**{f"roman_DZ_S{i}": 0.0 for i in range(1, 9)}, **{f"roman_M{i}": 0.0 for i in range(1, 9)}}
scan = json.load(open(os.path.join(RD, "v9", "scale_cut_scan.json")))["truths"]["prod"][f"{K0:g}"]
NAMES = dict(A1="roman_A1_1", eta1="roman_A1_2", A2="roman_A2_1", eta2="roman_A2_2", bTA="roman_BTA_1")
BEST = {m: {NAMES[k]: v for k, v in scan["models"][m]["best"].items()} for m in ("NLA", "TATT")}
ZERO_IA = {n: 0.0 for n in NAMES.values()}


def model_for(yaml, dataset):
    info = yaml_load_file(os.path.join(HERE, yaml))
    info.pop("sampler", None)
    info.pop("output", None)
    info["likelihood"][LIK]["data_file"] = dataset
    info["debug"] = False
    return get_model(info)


def point(model, ia):
    p = {**FID, **NUIS, **ia}
    return {k: v for k, v in p.items() if k in model.parameterization.sampled_params()}


def datavector(model, ia):
    p = point(model, ia)
    model.logposterior(p)
    lik = model.likelihood[LIK]
    return np.array(lik.get_datavector(**{**p, **{k: v for k, v in ia.items()}}))


def chi2(model, ia):
    return -2*float(model.loglikes(point(model, ia), as_dict=True)[0][LIK])


def write_vector(name, v):
    path = os.path.join(DATA, f"cutval_k0p2_{name}.modelvector")
    np.savetxt(path, np.column_stack([np.arange(NTOT), v]), fmt="%d %.10e")
    return os.path.basename(path)


def write_dataset(name, vec, mask):
    text = open(os.path.join(DATA, "example1.dataset")).read()
    text = text.replace("data_file = example1.modelvector", f"data_file = {vec}")
    text = text.replace("mask_file = example1.mask", f"mask_file = {mask}")
    assert vec in text and mask in text
    open(os.path.join(DATA, f"cutval_k0p2_{name}.dataset"), "w").write(text)


def main():
    m = np.load(os.path.join(RD, "cocoa", "run", "cosmolike_xi_ia.npz"))
    iK = int(np.argmin(np.abs(m["K_n1"] - K0)))
    mask = np.zeros(NTOT)
    mask[:NXI] = m["masks_n1"][iK]
    mname = "cutval_k0p2.mask"
    np.savetxt(os.path.join(DATA, mname), np.column_stack([np.arange(NTOT), mask]), fmt="%d %.1f")
    print(f"mask: K0 = {m['K_n1'][iK]}, {int(mask.sum())} xi points kept")

    base = np.load(os.path.join(RD, "v9", "ia_base_prod.npz"))
    nla = model_for("NLA_on_flam_k0p2.yaml", "ones.dataset")
    gg = datavector(nla, ZERO_IA)
    v_nla = datavector(nla, BEST["NLA"])
    tatt = model_for("TATT_on_flam_k0p2.yaml", "ones.dataset")
    v_tatt = datavector(tatt, {**ZERO_IA, **BEST["TATT"]})
    gg_t = datavector(tatt, ZERO_IA)
    print(f"GG from the NLA and TATT models agree to {np.max(np.abs(gg_t - gg)/np.max(np.abs(gg))):.1e}")
    v_flam = gg.copy()
    v_flam[:NXI] += base["xi_ia"]
    v_flam[NXI:] = 0.0
    v_nla[NXI:] = v_tatt[NXI:] = 0.0

    for name, v in (("flam", v_flam), ("nla", v_nla), ("tatt", v_tatt)):
        write_dataset(name, write_vector(name, v), mname)

    out = dict(about=__doc__, K0=K0, npts=int(mask.sum()), best=BEST, checks={})
    cov = np.load(os.path.join(RD, "cocoa", "analysis", "cov_example1_xi.npy"))
    k = mask[:NXI] > 0
    d = (gg[:NXI] - m["xi_gg"])[k]
    out["checks"]["chi2_GG_emul_vs_camb"] = float(d @ np.linalg.solve(cov[np.ix_(k, k)], d))
    del nla, tatt
    for yaml, ds, ia, ref in (
            ("NLA_on_flam_k0p2.yaml", "cutval_k0p2_flam.dataset", BEST["NLA"], scan["models"]["NLA"]["chi2"]),
            ("TATT_on_flam_k0p2.yaml", "cutval_k0p2_flam.dataset", BEST["TATT"], scan["models"]["TATT"]["chi2"]),
            ("NLA_on_flam_k0p2.yaml", "cutval_k0p2_nla.dataset", BEST["NLA"], 0.0),
            ("TATT_on_flam_k0p2.yaml", "cutval_k0p2_tatt.dataset", BEST["TATT"], 0.0)):
        mdl = model_for(yaml, ds)
        c = chi2(mdl, {**ZERO_IA, **ia})
        key = f"{yaml.split('_')[0]}|{ds}"
        out["checks"][key] = dict(chi2=c, fisher=ref)
        print(f"chi2 {key}: {c:.4f}  (Fisher pipeline {ref:.4f})")
    print(f"chi2 GG emulator vs CAMB path on the mask: {out['checks']['chi2_GG_emul_vs_camb']:.3f}")
    json.dump(out, open(os.path.join(HERE, "build_datavectors.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
