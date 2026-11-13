# ----------------------------------------------------------------------
# NLA vs TATT on the same TATT data vector: triangle plot, (S8, Om) plot,
# and the shift table. Run from Cocoa/ (paths are relative to it).
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from getdist import loadMCSamples, plots

# Chain roots: the yaml's "output:" without the leading ./
CHAINS = [  # (label, root, colour, filled)
    ("TATT", "projects/roman_real/chains/MCMC_TATT_cs_final_d3a_emul", "#1f5fa8", True),
    ("NLA",  "projects/roman_real/chains/MCMC_NLA_cs_final_d3a_emul",  "#d1495b", False),
]
TITLE = r"Roman cosmic shear, $k_{\rm UV} = 0.35\,h\,{\rm Mpc}^{-1}$"
BURN_IN = 0.3
OUT_TRIANGLE = "triangle_cs_nla_vs_tatt.pdf"
OUT_2D = "S8_Om_cs_nla_vs_tatt.pdf"

PARAMS = ["S8", "omegam", "sigma8", "As_1e9", "H0", "ns", "omegab",
          "roman_A1_1", "roman_A1_2"]
IA_TATT = ["roman_A1_1", "roman_A1_2", "roman_A2_1", "roman_A2_2", "roman_BTA_1"]

# Truth of the D3A TATT data vector. S8 and sigma8 from the emulator
# evaluate run, so the emulator's own offset cancels. IA truth: set to the
# values that printed the data vector.
TRUTH = {"omegam": 0.306, "As_1e9": 2.099, "ns": 0.967, "H0": 68.1,
         "omegab": 0.0486, "sigma8": 0.80772, "S8": 0.81576,
         "roman_A1_1": 3.80, "roman_A1_2": 0.57,
         "roman_A2_1": -1.08, "roman_A2_2": -0.71, "roman_BTA_1": 0.18}


def two_d_sigma(d, C):
    """Shift d in a 2D plane with covariance C, as a 1D-equivalent sigma
    (the same convention as gn_bias.py)."""
    q = float(d @ np.linalg.solve(C, d))
    return q, norm.ppf(0.5 + 0.5*chi2.cdf(q, 2))


def health(label, s):
    n = len(s.chain_offsets) - 1 if getattr(s, "chain_offsets", None) is not None else "?"
    try:
        r = f"{s.getGelmanRubin():.3f}"
    except Exception:
        r = "n/a"
    print(f"{label:5s}: {s.numrows} samples, {n} chains, R-1 (getdist) = {r}")


# ----------------------------------------------------------------------
samples = {lab: loadMCSamples(root, settings={"ignore_rows": BURN_IN})
           for lab, root, _, _ in CHAINS}
for lab, s in samples.items():
    health(lab, s)
names = {lab: {p.name for p in s.getParamNames().names} for lab, s in samples.items()}
params = [p for p in PARAMS if all(p in n for n in names.values())]

# --- 1. shift of each chain from the truth -----------------------------
print(f"\n{'param':12s}" + "".join(f"{lab+' mean':>12s}{'sigma':>9s}{'shift':>8s}"
                                    for lab in samples) + f"{'truth':>10s}")
for p in params + [q for q in IA_TATT if q not in params]:
    row = f"{p:12s}"
    for lab, s in samples.items():
        if p in names[lab]:
            m, sd = s.mean(p), s.std(p)
            t = TRUTH.get(p)
            sh = f"{(m - t)/sd:+.2f}s" if t is not None else ""
            row += f"{m:12.4f}{sd:9.4f}{sh:>8s}"
        else:
            row += f"{'(fixed)':>12s}{'':9s}{'':8s}"
    t = TRUTH.get(p)
    row += f"{t:10.4f}" if t is not None else ""
    print(row)

# --- 2. NLA relative to TATT, in TATT's sigma (the bias of interest) ----
# TATT is the correct model here, so its own shift from the truth is the
# projection-effect baseline; NLA - TATT removes it.
ref, test = CHAINS[0][0], CHAINS[1][0]
sr, st = samples[ref], samples[test]
print(f"\n{test} relative to {ref} (in {ref}'s sigma):")
for p in ["S8", "omegam", "sigma8"]:
    if p in params:
        d = st.mean(p) - sr.mean(p)
        print(f"  d{p:7s} = {d:+.5f} = {d/sr.std(p):+.2f} sigma")
pair = ["S8", "omegam"]
C_ref = sr.cov(pair)
d2 = np.array([st.mean(p) - sr.mean(p) for p in pair])
q, n2 = two_d_sigma(d2, C_ref)
print(f"  2D (S8, Om): dchi2 = {q:.3f} -> {n2:.2f} sigma "
      f"({'below' if n2 < 0.3 else 'above'} the 0.3 sigma threshold)")
for lab, s in samples.items():
    dt = np.array([s.mean(p) - TRUTH[p] for p in pair])
    _, nt = two_d_sigma(dt, s.cov(pair))
    print(f"  {lab} vs truth, 2D: {nt:.2f} sigma")

# --- 3. triangle plot ---------------------------------------------------
def style(g):
    g.settings.axes_fontsize = 10
    g.settings.axes_labelsize = 13
    g.settings.legend_fontsize = 13
    g.settings.title_limit_fontsize = 9
    g.settings.alpha_filled_add = 0.75
    g.settings.num_plot_contours = 2      # 68% and 95%
    g.settings.linewidth_contour = 1.6
    g.settings.figure_legend_frame = False

order = [lab for lab, _, _, _ in CHAINS]
g = plots.get_subplot_plotter(width_inch=9)
style(g)
g.triangle_plot(
    [samples[l] for l in order], params=params,
    filled=[f for _, _, _, f in CHAINS],
    contour_colors=[c for _, _, c, _ in CHAINS],
    contour_ls=["-", "--"],
    legend_labels=order, legend_loc="upper right",
    title_limit=1,
    markers={p: TRUTH[p] for p in params if p in TRUTH},
    marker_args={"color": "k", "ls": ":", "lw": 1},
)
g.fig.suptitle(TITLE, x=0.62, y=0.98, fontsize=13)
g.export(OUT_TRIANGLE)
plt.show()

# --- 4. (S8, Om) panel: the paper figure -------------------------------
g2 = plots.get_single_plotter(width_inch=4.2)
style(g2)
g2.plot_2d([samples[l] for l in order], "omegam", "S8",
           filled=[f for _, _, _, f in CHAINS],
           colors=[c for _, _, c, _ in CHAINS], ls=["-", "--"])
g2.add_x_marker(TRUTH["omegam"], color="k", ls=":", lw=1)
g2.add_y_marker(TRUTH["S8"], color="k", ls=":", lw=1)
g2.add_legend(order, legend_loc="upper right")
g2.subplots[0, 0].set_title(TITLE + f"\nNLA - TATT: {n2:.2f}$\\sigma$ (2D)", fontsize=10)
g2.export(OUT_2D)
plt.show()
print("wrote", OUT_TRIANGLE, OUT_2D)
