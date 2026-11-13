# Roman cosmic-shear Delta chi2 of IA models vs FLAMINGO

Goal: judge the IA scale cut from the Delta chi2 it leaves in the Roman
xi_+- data vector, and compare with the P(k)-level chi2 against FLAMINGO's
error bars.

## Pipeline

1. `roman_limber.py`: Limber C_l (GG, GI, II E and B) and binned xi_+- for the
   Roman layout (8 bins of example1.nz, 15 log theta bins 2.5-250 arcmin,
   [xi_+ | xi_-], 36 pairs each), Gaussian xi_+- covariance
   (2415 deg^2, 41.3/8 gal/arcmin^2 per bin, sigma_e = 0.30 per component,
   exact discrete shape-noise term), IR/UV mask (alpha = 0.05, one theta_min
   per row). IA enters as R_X(k, z) = P_X / P_NL tables, linear in z
   between the 13 snapshots and in ln k.
2. `validate.py`: against cocoa_roman_real.
   * xi_+- (GG + NLA at the EXAMPLE_EVALUATE1 point) vs example1.modelvector:
     agrees to 1-2% (median 0.986; flat-sky Hankel vs cocoa's curved sky).
   * Covariance vs data/cov_roman_real (a 10-bin 3x2pt matrix, the only
     covariance in the repo that is not a Git LFS pointer): same structure,
     but that file was made for a different survey; the best match is
     A ~ 10850 deg^2 with sigma_e^2/n ~ 0.0113 arcmin^2 (diag ratio median 1.02,
     5-95% 0.84-1.05; correlation rms difference 0.024). The 15-bin
     example1_cov is an LFS object and could not be fetched.
3. `build_spectra.py`: FLAMINGO truth and per-snapshot model spectra.
   * truth: NLA below k_t, FLAMINGO above (notes' logistic stitch), folds
     1/4/16, P_dE = P_hE / b(k), P_EE and P_BB minus the white shape noise.
     Variants: `highk` (default; shape noise = high-k plateau of P_BB),
     `nla` (notes' Ns from the low-k NLA fit; over-subtracts and leaves a
     negative white C_l), `bconst` (P_hE / b0 with constant large-scale bias).
   * models: per-snapshot amplitudes only (NLA a1; TATT a1, a2, b_TA; + Ns),
     fitted to P_hE + P_EE over 0.05 <= k <= k_max with FLAMINGO's measured
     error bars, k_max on the data grid 0.12 ... 1.69.
4. `roman_dchi2.py`: Delta chi2 of model minus truth on
   all points, on IR/UV masks at k_UV = const (fit k_max x mask k_UV map),
   on k0/D(z) masks (fits at the largest k_max <= k0/D(z_snap)), and on the
   0.35/D(z) mask. FLAMINGO noise floor from 40 realizations of the truth
   perturbed by its error bars. P(k)-level chi2 sums for the mapping.
5. `roman_bestfit.py`: amplitudes chosen to minimise the Roman Delta chi2
   directly (fixed cosmology), free per snapshot or cosmolike power law:
   separates irreducible model failure from noisy P(k)-level amplitudes.

## Run

    P=.../psdata/PS; NZ=.../cocoa_roman_real/data/example1.nz
    python validate.py --repo .../cocoa_roman_real --out validation
    python build_spectra.py --ps-dir $P --out spectra.npz
    python roman_dchi2.py --spectra spectra.npz --nz $NZ --truth highk --out results_highk
    python roman_bestfit.py --spectra spectra.npz --res results_highk --nz $NZ

## Caveats

Gaussian covariance only (no SSC / connected NG): Delta chi2 values are upper
bounds on what the full covariance would give. Cosmology fixed at FLAMINGO's.
Truth above k ~ 17.7 h/Mpc extrapolated with a constant P_X / P_NL; P_dE above
k ~ 1 depends on the deterministic-bias convention (see `bconst`).

## Results (truth `highk`, Gaussian cov, alpha = 0.05)

Validation after reading example1.nz's z column as left bin edges (as cocoa
does): xi_+- vs example1.modelvector median ratio 0.997 (5-95%: 0.990-1.003).

Roman Delta chi2 on the IR/UV-masked xi_+- (`results_highk/*.csv`):

| cut | pts | NLA, P(k)-fit amps | TATT, P(k)-fit amps | NLA best fit (power law) | TATT best fit (power law) | noise floor |
|---|---|---|---|---|---|---|
| k_UV = 0.16 | 65 | 0.74 | 2.07 | 0.24 | 0.20 | 0.003 |
| k_UV = 0.22 | 113 | 1.14 | 3.69 | 0.68 | 0.64 | 0.010 |
| k_UV = 0.29 | 159 | 0.91 | 5.29 | 1.05 | 0.97 | 0.017 |
| k_UV = 0.39 | 192 | 1.11 | 8.18 | 1.43 | 1.27 | 0.025 |
| k_UV = 0.52 | 244 | 5.41 | 3.35 | 2.25 | 1.81 | 0.037 |
| k_UV = 0.70 | 297 | 22.1 | 4.20 | 3.11 | 2.44 | 0.061 |
| 0.35/D(z) | 208 | 1.11 (best k_max) | 2.73 (best k_max) | 1.55 | 1.49 | 0.040 |

* NLA with P(k)-fit amplitudes: Delta chi2 ~ 1 plateau for k_max = k_UV in
  0.16-0.39, then 5 at 0.52 and 22 at 0.70: cut ~0.4 h/Mpc (comoving), or
  k0 ~ 0.2 for k_UV = k0/D(z).
* TATT with P(k)-fit amplitudes: 2-8 at every cut above 0.12. The free a2,
  b_TA per snapshot fit P(k) to reduced chi2 ~ 1 up to k_max ~ 0.7 but are
  noisy and degenerate, and those directions show up in xi.
* Roman-level best fit at fixed cosmology: TATT gains only 5-25% in Delta chi2
  over NLA at the same mask; with the cosmolike power law both reach
  Delta chi2 = 1 at k_UV ~ 0.29 (comoving) or k0 ~ 0.29 (k0/D(z)); the
  0.35/D mask gives ~1.5 for both.
* P(k) chi2 does not predict the Roman Delta chi2 (pk_vs_roman_dchi2.png).
* FLAMINGO noise is negligible (<= 0.04 at the masks of interest); truth
  variants `nla`, `bconst` change the numbers by < ~2x (truth_variants.png).

## Stitched truth as the truth (`v3s/`, `report/stitched.pdf`)

The full v3 k_max analysis (GoF, anchored, 5%/10% deviation, Delta chi2_FL,
A(z), redshift-dependent cuts, Roman Delta chi2 and FoB) with the notes'
stitched spectra as the truth: NLA (notes' a1, Ns on 0.05-0.3) below k_t(z),
FLAMINGO spline above, logistic weight (width 0.3). `v3/v3_flamingo.py` now
takes a `truth` dict (default = production, reproduces the stored results
exactly); `v3s/run_stitched.py` builds S2 (notes' Ns, main), S1 (B-plateau
Ns), S2w (errors w sigma) and prod2 (production, P_hE + P_EE only) and runs
them into `v3s/<tag>/`; `v3/v3_roman.py --fits-dir --truth --out --stitched`
does the Roman side; `v3s/plot_stitched_cuts.py` the comparison figures.
Result: NLA GoF cut 0.22-0.39 (1.1-1.9 k_t), TATT 0.29-0.52; Roman FoB on the
stitched-shaped masks 0.08/0.06 (NLA shape) and 0.16/0.17 (TATT shape); TATT
on the stitched truth is nearly NLA (A2 -0.1 to -0.3, bTA 0). The change from
production is mostly the Ns convention (white II excess treated as noise),
shown by S2w; the anchored criterion is not meaningful for TATT on this truth.
The p-value uses a Satterthwaite effective dof (sum w^2) because the NLA side
is noise-free. The earlier Roman-test-only supplement is kept as
`report/stitched_romantest_v1.pdf` and included as Section 7.

## Stitched-truth control (`v3/stitched_test.py`, `v3/plot_stitched.py`, `report/stitched.pdf`)

The Roman test (all parameters free) on the notes' stitched truth (NLA below
k_t(z), simulation above, logistic weight) and three related variants (S1-S4,
see the PDF). Result: the stitch removes the white II excess below k_t and
rotates the shift from +0.3-0.6 sigma in S8 (production) to -0.3 sigma in
Omega_m, with chi2 nearly unchanged; the FoB = 0.3 crossing moves from
K0 ~ 0.30 to ~0.34 (NLA) / 0.39 (TATT). It assumes NLA in 0.1-0.2 h/Mpc where
the v3 GoF rejects it, so it is a control, not the production truth.
Outputs: v3/stitched_truths.npz, stitched_xi.npz, stitched_test.json/.csv,
fig_stitch_truths, fig_stitch_fob, fig_stitch_shift.

## Mask families and GGL / 3x2pt (`v3/mask_scan.py`, `v3/ggl3x2.py`, Secs. 11-12 of summary.pdf)

`mask_scan.py`: Roman FoB for k_UV = K0 D^-p with p = 1, 2/3, 0 (K0 0.15-0.5),
NLA/TATT cases A/B and with the white II template. The 0.3 sigma crossing is
at K0 ~ 0.3 in all three; at matched data volume K0/D is the most efficient;
with the white template a constant cut passes to k ~ 0.5 for both models.
`cocoa/driver3x2.py` (3x2pt cosmolike driver, sim_IA GI in gamma_t),
`cocoa/cov3x2.py` (Gaussian 3x2pt covariance, xi block = stored xi cov),
`ggl3x2.py` (parts all / 3x2_behind / ggl_behind): FLAMINGO IA injected into
xi_pm and gamma_t, Omega_m, A_s, 8 biases and IA free. With all lens-source
pairs the same-bin gamma_t is GI-dominated and separates the models (NLA fails
at K0_ggl ~ 0.18, TATT < 0.15 everywhere); with sources behind lenses only
both models show the cosmic-shear S8 shift and K0_ggl = 0.2 is safe.

## FLAMINGO vs best fits, and GoF as chi2/dof (report/compare.pdf)
- `v3/compare_fits.py`: model curves rebuilt from the stored best fits (`v3/v3.npz`) for fits to k <= 0.16, 0.29, 0.52;
  `fig_cmp_{GI,II,BB}_k{0p16,0p29,0p52}` (stack + percent difference), `fig_chi2dof`, `fig_chi2dof_comb`, `fig_chi2dof_cuts`,
  `chi2dof.csv`, `chi2dof_cuts.csv`. GoF rewritten as chi2_min/nu with the p = 0.05 equivalent chi2_0.95(nu)/nu and fixed thresholds 1.5, 2, 3.
- `report/compare.tex` -> `compare.pdf` (13 pages); table bodies `tab_chi2dof.tex`, `tab_chi2dof_cuts.tex` generated from the CSVs.

## Scale-cut strategies (report/strategies.pdf, 10 Nov)
- `v4/simfit.py` (fitter with covariances), `v4/sim_strategies.py` (GoF/stability rules, volume scaling, noise and calibration realisations -> sim_strategies.json),
  `v4/roman_strategies.py` (Fisher precision per mask, white-term response, tolerances, theory-error covariance fits -> roman_strategies.json, roman_theory.json, jacobians.npz),
  `v4/fisher_white.py` (cost of the white nuisance), `v4/split_pm.py` + `v4/verify_split.py` (separate xi+/xi- cuts), `v4/plot_strategies.py` (fig_strat_*, derived.json), `v4/make_tables.py` (report/tab_*.tex).
