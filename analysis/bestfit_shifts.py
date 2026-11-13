"""Best-fit (MAP) shifts of NLA relative to TATT, from Cobaya minimize runs.

Projection-free bias: compares the two minima, in units of the TATT chain's
marginal sigma, and the 2D (S8, Om) shift with the TATT chain covariance
(the gn_bias.py convention). Run from Cocoa/.
"""
import glob

import numpy as np
from getdist import loadMCSamples
from scipy.stats import chi2, norm

MIN = {  # minimize output prefixes (the MIN yamls' "output:" without ./)
    "TATT": "projects/roman_real/chains/MIN_TATT_cs_final_d3a_emul/MIN_TATT_cs_final_d3a_emul",
    "NLA":  "projects/roman_real/chains/MIN_NLA_cs_final_d3a_emul/MIN_NLA_cs_final_d3a_emul",
}
REF_CHAIN = "projects/roman_real/chains/MCMC_TATT_cs_final_d3a_emul"  # sigmas
TRUTH = {"S8": 0.81576, "omegam": 0.306, "sigma8": 0.80772, "As_1e9": 2.099,
         "H0": 68.1, "ns": 0.967, "omegab": 0.0486,
         "roman_A1_1": 3.80, "roman_A1_2": 0.57}
SHOW = ["S8", "omegam", "sigma8", "As_1e9", "H0", "ns", "omegab",
        "roman_A1_1", "roman_A1_2"]


def read_minimum(prefix):
    """Cobaya writes <prefix>.minimum.txt (header + one row); fall back to
    the key/value <prefix>.minimum (or .bestfit) file."""
    f = prefix + ".minimum.txt"
    if glob.glob(f):
        with open(f) as fh:
            head = fh.readline().lstrip("#").split()
        row = np.loadtxt(f, ndmin=2)[0]
        return dict(zip(head, row))
    for ext in (".minimum", ".bestfit"):
        if glob.glob(prefix + ext):
            out = {}
            for line in open(prefix + ext):
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        out[parts[2]] = float(parts[1])
                    except ValueError:
                        pass
                elif line.startswith("-log(Like)"):
                    out["minuslogpost"] = float(line.split("=")[1])
            return out
    raise SystemExit(f"no minimum file for {prefix}")


def two_d_sigma(d, C):
    q = float(d @ np.linalg.solve(C, d))
    return q, norm.ppf(0.5 + 0.5*chi2.cdf(q, 2))


best = {k: read_minimum(v) for k, v in MIN.items()}
ref = loadMCSamples(REF_CHAIN, settings={"ignore_rows": 0.3})

print(f"{'param':12s}{'truth':>10s}" + "".join(f"{k+' MAP':>11s}" for k in best)
      + f"{'NLA-TATT':>11s}{'in sigma':>9s}")
for p in SHOW:
    if not all(p in b for b in best.values()):
        continue
    t, bt, bn = TRUTH.get(p), best["TATT"][p], best["NLA"][p]
    sd = ref.std(p)
    row = f"{p:12s}" + (f"{t:10.4f}" if t is not None else f"{'':10s}")
    row += f"{bt:11.4f}{bn:11.4f}{bn - bt:+11.5f}{(bn - bt)/sd:+8.2f}s"
    print(row)

pair = ["S8", "omegam"]
C = ref.cov(pair)
for lab, d in [("NLA - TATT", [best["NLA"][p] - best["TATT"][p] for p in pair]),
               ("TATT - truth", [best["TATT"][p] - TRUTH[p] for p in pair]),
               ("NLA - truth", [best["NLA"][p] - TRUTH[p] for p in pair])]:
    q, n2 = two_d_sigma(np.array(d), C)
    print(f"2D (S8, Om) {lab:13s}: dchi2 = {q:.3f} -> {n2:.2f} sigma")
print("\nTATT - truth should be ~0 (it is the true model): a nonzero value "
      "means the TATT minimizer has not converged, or the truth values above "
      "do not match the data vector.")
