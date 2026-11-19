"""Write data.json of every figure of the note from the outputs in flamingo/resolution_tests/.

Run from anywhere: python make_figure_data.py. Needs only numpy (the analysis scripts that made the
JSON inputs need CAMB; see flamingo/resolution_tests/README.md).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "flamingo", "resolution_tests"))


def load(*p):
    with open(os.path.join(RT, *p)) as f:
        return json.load(f)


def dump(name, data):
    with open(os.path.join(HERE, name, "data.json"), "w") as f:
        json.dump(data, f, indent=1)


# particle_count: median N* against log M*
pc = load("particle_count.json")
dump("particle_count", {
    "about": "make_figure_data.py from flamingo/resolution_tests/particle_count.json (SOAP-HBT z = 0, "
             "first 2^22 rows of L1_m8 snap 78 and L2p8_m9 snap 78)",
    "runs": {r: {"logM": [x["logM"] for x in pc[r]["bins"]], "Nstar_med": [x["Nstar_med"] for x in pc[r]["bins"]],
                 "frac_ge100": [x["frac_ge100"] for x in pc[r]["bins"]], "mstar_particle": pc[r]["mstar_particle"]}
             for r in ["L1_m8", "L2p8_m9"]}})

# m9_m8_ratio: high-mass P_dE ratios to L1_m8, interpolated in log k to the L1_m8 bins
hm = load("restest_L1m9", "results.json")
ref = hm["L1_m8"]
k8, p8, s8 = (np.array(ref[x]) for x in ("k", "pde", "sigma"))
kny = {"L1_m9": 2.36, "L2p8_m9": 0.84}  # Nyquist of the 512^3 grid, nfold 1 [h/Mpc]
lines = {}
for box in ["L1_m9", "L2p8_m9"]:
    k, p, s = (np.array(hm[box][x]) for x in ("k", "pde", "sigma"))
    pi, si = np.interp(np.log(k8), np.log(k), p), np.interp(np.log(k8), np.log(k), s)
    m = (k8 >= 0.02) & (k8 <= 0.7 * min(kny[box], 2.0))
    r = pi / p8
    e = np.abs(r) * np.hypot(si / pi, s8 / p8)
    lines[box] = {"k": k8[m].tolist(), "ratio": r[m].tolist(), "err": e[m].tolist()}
dump("m9_m8_ratio", {
    "about": "make_figure_data.py from flamingo/resolution_tests/restest_L1m9/results.json: z = 0, "
             "M* > 4.05e11 Msun, SOAP StellarInertiaTensorNoniterative, ng 512, nfold 1",
    "anla": {b: [hm[b]["anla"], hm[b]["anla_err"]] for b in hm}, "erms": {b: hm[b]["erms"] for b in hm},
    "lines": lines})

# lowmass_ratio
lm = load("lowmass_L1m9", "results.json")
dump("lowmass_ratio", {
    "about": "make_figure_data.py from flamingo/resolution_tests/lowmass_L1m9/results.json: z = 0, "
             "10.0 < log M* < 10.41, unreduced tensor of all bound stars, ng 512, nfold 1 (k < 0.8) and 4",
    "ratios": lm["ratios"], "summary": lm["summary"]})

# lowk_sn
fc = load("lowk_forecast.json")
dump("lowk_sn", {
    "about": "make_figure_data.py from flamingo/resolution_tests/lowk_forecast.json: Gaussian S/N of P_hE "
             "per k bin, z = 0, pipeline binning", "cases": fc})
print("wrote data.json for", ", ".join(sorted(d for d in os.listdir(HERE) if os.path.isdir(os.path.join(HERE, d)))))
