"""Print the LaTeX results table (MAP + marginal mean, NLA vs TATT) from the
minimize outputs and the MCMC chains. Run from Cocoa/.

  python make_results_table.py 3x2      # or: cs
"""
import sys

import numpy as np
from getdist import loadMCSamples
from scipy.stats import chi2, norm

from bestfit_shifts import read_minimum  # same minimum-file reader

TAG = sys.argv[1] if len(sys.argv) > 1 else "3x2"
C = "projects/roman_real/chains"
NAME = {m: f"{m}_{TAG}_final_d3a_emul" for m in ("TATT", "NLA")}
MIN = {m: f"{C}/MIN_{n}/MIN_{n}" for m, n in NAME.items()}
MCMC = {m: f"{C}/MCMC_{n}/MCMC_{n}" for m, n in NAME.items()}

# (name, latex, decimals)
ROWS_TOP = [("S8", r"$S_8$", 4), ("omegam", r"$\Omega_{\rm m}$", 4)]
ROWS_BOT = [("As_1e9", r"$10^9 A_{\rm s}$", 2), ("H0", r"$H_0$", 1),
            ("ns", r"$n_{\rm s}$", 3), ("omegab", r"$\Omega_{\rm b}$", 4)]


def two_d(d, cov):
    q = float(d @ np.linalg.solve(cov, d))
    return norm.ppf(0.5 + 0.5*chi2.cdf(q, 2))


best = {m: read_minimum(p) for m, p in MIN.items()}
s = {m: loadMCSamples(p, settings={"ignore_rows": 0.3}) for m, p in MCMC.items()}
ref = s["TATT"]


def row(p, lab, dec):
    sd = ref.std(p)
    cells = [lab]
    for m in ("TATT", "NLA"):
        cells += [f"{best[m][p]:.{dec}f}",
                  f"${s[m].mean(p):.{dec}f} \\pm {s[m].std(p):.{dec}f}$"]
    dmap = (best["NLA"][p] - best["TATT"][p])/sd
    dmean = (s["NLA"].mean(p) - s["TATT"].mean(p))/sd
    cells += [f"${dmap:+.2f}\\sigma$", f"${dmean:+.2f}\\sigma$"]
    return " & ".join(cells) + r" \\"


pair = ["S8", "omegam"]
cov = ref.cov(pair)
n_map = two_d(np.array([best["NLA"][p] - best["TATT"][p] for p in pair]), cov)
n_mean = two_d(np.array([s["NLA"].mean(p) - s["TATT"].mean(p) for p in pair]), cov)

print("\n".join(row(*r) for r in ROWS_TOP))
print(r"$(S_8,\Omega_{\rm m})$, 2D & --- & --- & --- & --- & "
      rf"$\mathbf{{{n_map:.2f}\sigma}}$ & ${n_mean:.2f}\sigma$ \\")
print(r"\hline")
print("\n".join(row(*r) for r in ROWS_BOT))
