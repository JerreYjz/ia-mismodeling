# ----------------------------------------------------------------------
# Triangle plot of one Cobaya MCMC chain, with the data-vector truth marked.
# Run from Cocoa/ (paths are relative to it), as a script or notebook cell.
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

# Chain root: everything before .1.txt, .2.txt, ...
# (the yaml's "output:" without the leading ./)
CHAIN_ROOT = "projects/roman_real/chains/MCMC_NLA_cs_final_d3a_emul"
LABEL = "NLA, cosmic shear"
BURN_IN = 0.3            # drop the first 30% of each chain
OUTPUT = "triangle_NLA_cs.pdf"

# What to plot. S8 and sigma8 come from sigma8_from_pk.
PARAMS = ["S8", "omegam", "sigma8", "As_1e9", "H0", "ns", "omegab",
          "roman_A1_1", "roman_A1_2"]

# Truth of the D3A TATT data vector. S8 and sigma8 are the emulator
# evaluate values (use those, not FLAMINGO's 0.807, so the emulator's own
# 0.07% offset cancels). Set the IA truth to what printed the data vector.
TRUTH = {"omegam": 0.306, "As_1e9": 2.099, "ns": 0.967, "H0": 68.1,
         "omegab": 0.0486, "sigma8": 0.80772, "S8": 0.81576,
         "roman_A1_1": 3.80, "roman_A1_2": 0.57,
         "roman_A2_1": -1.08, "roman_A2_2": -0.71, "roman_BTA_1": 0.18}

# ----------------------------------------------------------------------
s = loadMCSamples(CHAIN_ROOT, settings={"ignore_rows": BURN_IN})
have = [p.name for p in s.getParamNames().names]
params = [p for p in PARAMS if p in have]
missing = [p for p in PARAMS if p not in have]
if missing:
    print("not in this chain (skipped):", missing)

# quick health check: sample count and Gelman-Rubin R-1 across chains
nchains = len(s.chain_offsets) - 1 if getattr(s, "chain_offsets", None) is not None else "?"
print(f"{s.numrows} samples after {BURN_IN:.0%} burn-in, {nchains} chains")
try:
    print(f"R-1 (worst eigenvalue, getdist) = {s.getGelmanRubin():.3f}")
except Exception as e:
    print("R-1 not available:", e)

# marginal results: mean, 68% limits, shift from the truth in sigma
print(f"\n{'param':12s} {'mean':>9s} {'sigma':>8s} {'truth':>9s} {'shift':>7s}")
for p in params:
    m, sd = s.mean(p), s.std(p)
    t = TRUTH.get(p)
    shift = f"{(m - t)/sd:+.2f}s" if t is not None else ""
    tt = f"{t:9.4f}" if t is not None else " " * 9
    print(f"{p:12s} {m:9.4f} {sd:8.4f} {tt} {shift:>7s}")

# ----------------------------------------------------------------------
g = plots.get_subplot_plotter(width_inch=9)
g.settings.axes_fontsize = 10
g.settings.axes_labelsize = 13
g.settings.legend_fontsize = 13
g.settings.title_limit_fontsize = 10
g.settings.alpha_filled_add = 0.8
g.settings.num_plot_contours = 2          # 68% and 95%

g.triangle_plot(
    [s], params=params, filled=True,
    contour_colors=["#1f5fa8"],
    legend_labels=[LABEL], legend_loc="upper right",
    title_limit=1,                         # 68% limits above each 1D panel
    markers={p: TRUTH[p] for p in params if p in TRUTH},
    marker_args={"color": "k", "ls": "--", "lw": 1},
)
g.export(OUTPUT)
plt.show()
print("wrote", OUTPUT)
