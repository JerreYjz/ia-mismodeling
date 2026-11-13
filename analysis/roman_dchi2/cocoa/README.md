# FLAMINGO vs NLA/TATT IA in Roman xi_+- with cosmolike itself

Built and run here, without Cobaya:

* cosmolike core (cocoa-cosmolike-core b811ae0) with `core-sim-IA.patch`, and the
  cocoa_roman_real interface (0559522) with `roman_real-sim-IA.patch`, both
  applied unchanged. Compiled against the system GSL / FFTW / Armadillo /
  spdlog / HDF5, with carma v0.7.0, and with `-DSIM_KMIN_HMPC=1e-6
  -DSIM_KMAX_HMPC=0`, so tables are not trimmed. Library: `lib/`.
* `cosmolike_driver.py`: the roman_real likelihood's setup (initialize,
  set_cosmo_related, set_source_related) for xi_+-, CAMB at FLAMINGO D3A
  (takahashi, the EXAMPLE_EVALUATE2_d3a settings).
* `validate_cosmolike.py`: reproduces `example1.modelvector` (EVALUATE1 point)
  to 3e-5 median and 0.27% at the 99th percentile.

## Method (`cosmolike_xi_ia.py`, `analyze_cosmolike.py`)

FLAMINGO and every model fit enter cosmolike the same way, as sim_IA tables
(IA_code = 3). P_dE carries GI and P_EE carries II; both are in shear units
at mu = 0, per snapshot, on 1e-4 to 300 h/Mpc. Truth and models therefore
share the Limber integration, the C_l -> xi transform, the bin averaging and
the n(z).

* **FLAMINGO.** Smoothed measured spectra on 0.05 to 17.7 h/Mpc, with
  P_dE = P_hE / b(k) and P_EE minus the white shape noise (the high-k
  plateau). Outside the data, the ratio to P_NL is held at its edge value.
* **Models.** The v2 fits (`../v2/fits.npz`): per-snapshot amplitudes fitted
  to FLAMINGO P_hE + P_EE + P_BB over 0.05 <= k <= k_max.
* **Masks.** IR/UV, made exactly as `make_uv_scale_cuts.py` does: GG plus an
  NLA baseline (A1 0.6, eta -1.5), alpha 0.05, P truncated above k_UV(z).
  Families k_UV/D(z) (production) and constant k_UV.
* **Covariance.** Gaussian xi_+- (2415 deg^2, 41.3/8 gal/arcmin^2 per bin,
  sigma_e 0.30), at D3A with GG plus FLAMINGO IA.

## Checks (`analysis/checks.txt`)

* cosmolike native NLA vs the same NLA as tables: Delta chi2 0.16.
* cosmolike native TATT vs the same TATT as tables: Delta chi2 41. This is
  entirely TATT's IA B-mode term, which the sim path does not carry: the
  python B-mode xi gives 40.7, the residual is 0.17 and the correlation is
  0.996 (`check_tatt_bb.py`).
* Python Limber (`../roman_limber.py`) vs cosmolike, with the same masks and
  no B modes: within 10-50% on the masks, and 2% on all points.

## Results (E modes, as the sim path; `analysis/dchi2_masks.csv`)

Delta chi2 of xi_IA(model) - xi_IA(FLAMINGO), by fit k_max:

| mask | NLA 0.12 | NLA 0.16 | NLA 0.22 | NLA 0.29 | NLA 0.39 | TATT 0.16 | TATT 0.22 | TATT 0.29 | TATT 0.52 | TATT 0.70 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2/D | 0.53 | 0.83 | 1.38 | 2.41 | 4.49 | 0.74 | 0.74 | 1.24 | 1.07 | 1.48 |
| 0.25/D | 0.79 | 1.20 | 1.95 | 3.41 | 6.43 | 0.89 | 0.89 | 1.56 | 1.47 | 2.12 |
| 0.3/D | 1.05 | 1.49 | 2.32 | 4.02 | 7.72 | 1.03 | 1.05 | 1.72 | 1.63 | 2.36 |
| 0.35/D | 1.31 | 1.72 | 2.60 | 4.49 | 8.74 | 1.23 | 1.22 | 1.97 | 1.87 | 2.69 |
| 0.5/D | 2.13 | 2.39 | 3.26 | 5.49 | 10.9 | 1.57 | 1.48 | 2.28 | 2.15 | 3.10 |

With IA B modes added (python, `analysis/dchi2_masks_withBB.csv`), 0.35/D:
NLA rises by about 1.2, and TATT reaches 3.7-10. That depends on FLAMINGO's
P_BB, whose white level is uncertain.

## Pipeline issues found

1. **sim_IA trusted band.** With the default build (SIM_KMIN/KMAX = 0.5 / 7),
   every injected table is trimmed to that band and extrapolated flat below
   0.5 h/Mpc. That makes Delta chi2 about 1000 on the 0.35/D mask against
   the properly extended FLAMINGO (`check_simIA_band.py`). With all the
   measured rows (0.05-17.7) it is still 116, from the low-k power-law
   extrapolation. A FLAMINGO data vector needs tables that already cover
   1e-4 to ~300 h/Mpc, and a build that does not trim them.
2. **No IA B modes on the sim path.** TATT's model has them, so a TATT fit to
   a sim-injected data vector sees B modes the data cannot. For the Part 1a
   TATT fiducial that is Delta chi2 = 41 on all points. FLAMINGO's own P_BB
   is worth 1.75 on the 0.35/D mask.
3. **Noise in injected spectra.** Measured instead of smoothed FLAMINGO
   spectra add about 1.7 on the 0.35/D mask, mostly from 0.05-0.2 h/Mpc.
   Stitching NLA below k_t, as the notes do, removes it.

## FLAMINGO Delta chi2 -> Roman Delta chi2 (`translate.py`, `translate_matrix.py`)

Roman's Delta chi2 is a quadratic form in the FLAMINGO residual:

    Delta chi2_Roman = u^T A u,
    u = (model - FLAMINGO) / sigma_FLAMINGO   per snapshot, spectrum (P_dE, P_EE) and k bin
    A = S J^T C^-1 J S

* J is computed with cosmolike, one cell at a time (`analysis/J_cells.npy`).
* Delta chi2_FL = u^T u.
* The matrix reproduces the direct cosmolike numbers to within ~15%; it runs
  slightly low.

What A says (production masks k_UV = K0/D(z); `analysis/translate_matrix.csv`):

| | 0.25/D | 0.35/D | 0.5/D |
|---|---|---|---|
| amplitude-like residual: Delta chi2_FL giving Roman 1 | 718 | 717 | 649 |
| worst-case residual: Delta chi2_FL guaranteeing Roman < 1 | 11.9 | 10.4 | 8.2 |

* **Per bin** (`fig_sensitivity_map`), Roman Delta chi2 per unit FLAMINGO
  chi2 is:
  * 1e-2 to 5e-2 for k = 0.05-0.2 at z = 0.75-2;
  * below 1e-5 for k > 1, where FLAMINGO's errors are tiny and the mask
    removes the scales.
* **Per snapshot** (amplitude change), it is about 2e-3 at z = 1-1.5,
  4e-4 at z = 0.5, 1e-4 or less at z <= 0.3 and z = 3, and 5e-6 at z = 4.
* **The worst-case direction** (`fig_worst_direction`) is a large-scale
  (k ~ 0.06-0.15) GI/II residual at z = 1-2. FLAMINGO's box constrains it
  poorly, but Roman measures it well.
* **The in-fit FLAMINGO chi2 (k <= k_max) does not predict Roman's
  Delta chi2** (`fig_translate_infit_snap`). Roman also sees the residual
  above k_max, through the mask, and that residual sets a floor of about
  1-2 for TATT on 0.35/D.

## Verification and bias analysis (`verify_injection.py`, `bias_analysis.py`, `roman_summary.py`)

**Conversion and injection** (`analysis/verify_injection.txt`):

* **Formula.** Raw Pds files -> cosmolike convention:
  * P_dE = (3/2) M0^{hE} / (2R) / b(k)
  * P_EE = (M0 - 5/2 M2 + 27/8 M4)^{EE} / (2R)^2 / <W^2_NGP>
  * R = 1 - e_rms^2

  Recomputed term by term, it matches the analysis spectra exactly. The low-k
  aNLA reproduces the 2026-10-27 report to within 0.1-1.1% at all 13
  snapshots.
* **sim_IA.c's own conversion.** Applied to the raw files (apply_shear, hybrid
  LOS, NGP window, e_rms), it gives the same xi_IA as the pre-converted tables,
  to 2.5e-9.
* **Physical check.** The injected FLAMINGO xi_IA, divided by cosmolike's
  native NLA with the report's aNLA(z), has a median of 1.01-1.05 at
  theta >= 21'.

**Bias.**

* Data: GG + FLAMINGO IA.
* Model: cosmolike NLA / TATT with power-law z evolution.
* Fit: IA exact, cosmology through a refreshed linear response; the exact chi2
  agrees with the linearised one at the solution.
* Null test: TATT fitted to a TATT data vector gives zero bias.
* Table: `analysis/roman_dchi2_summary.csv`.

| mask | NLA chi2 (IA+cosmo) | NLA FoB | TATT chi2 | TATT FoB |
|---|---|---|---|---|
| 0.2/D | 0.44 | 0.24 | 0.37 | 0.23 |
| 0.25/D | 0.81 | 0.23 | 0.69 | 0.26 |
| 0.3/D | 1.18 | 0.31 | 0.96 | 0.32 |
| 0.35/D | 1.48 | 0.39 | 1.25 | 0.41 |
| 0.4/D | 1.90 | 0.61 | 1.49 | 0.66 |
| 0.5/D | 2.53 | 0.62 | 2.06 | 0.68 |

FoB is in (Omega_m, S8); case A frees Omega_m and A_s. Case B (n_s, H0 and
Omega_b also free) gives similar FoB.

## k_max summary (2026-11-06, `summary_kmax.py`, `bias_v2.py`, `bias_white.py`, `bias_wtemplate.py`)

Figures: `fig_kmax_gof`, `fig_kmax_dchi2`, `fig_kmax_weight`, `fig_bias_v2`,
`fig_white_ii`, `fig_kmax_fob` (png 220 dpi + pdf). Tables: `kmax_summary.csv`,
`kmax_weight.csv`, `bias_v2.csv`.

### 1. Goodness of fit in FLAMINGO (P_hE, P_EE, P_BB, 0.05 <= k <= k_max, measured errors)

Combined over the 13 snapshots (chi2/dof, p):

| k_max | NLA all z | NLA z=0.5-2 | TATT all z | TATT z=0.5-2 |
|---|---|---|---|---|
| 0.16 | 0.76, 0.98 | 0.93, 0.62 | 0.67, 1.00 | 0.78, 0.87 |
| 0.22 | 1.21, 0.03 | 1.17, 0.15 | 0.72, 1.00 | 0.86, 0.79 |
| 0.29 | 2.10, 2e-18 | 1.81, 2e-6 | 0.74, 1.00 | 0.84, 0.85 |
| 0.39 | 4.26, 0 | 2.93, 2e-23 | 0.89, 0.88 | 0.93, 0.68 |
| 0.52 | 10.6, 0 | 6.6, 0 | 1.19, 0.02 | 1.17, 0.10 |
| 0.70 | 21.8, 0 | 14.1, 0 | 1.73, 6e-14 | 1.51, 1e-4 |

NLA fails (p < 0.05) at k_max = 0.22-0.29 h/Mpc, TATT at 0.5-0.7 h/Mpc.

### 2. Delta chi2_FL and its translation to Roman

`kmax_summary.csv`: Delta chi2_FL (in-fit, vs the noise-free spectra) and
the Roman Delta chi2 of the same fits (cosmolike, E modes, IA amplitudes
from the FLAMINGO fit, production masks). Roman / FL ratio is 0.02-0.04 for
k_max <= 0.22, 0.01 at 0.29, 0.004-0.008 at 0.39-0.7, 1e-3 at k > 1.

Where Roman looks (`kmax_weight.csv`, `fig_kmax_weight`): Roman Delta chi2
per unit FLAMINGO chi2 placed at one k node (sum over 13 snapshots and both
spectra), mask 0.35/D:

| k | 0.067 | 0.09 | 0.12 | 0.16 | 0.22 | 0.29 | 0.39 | 0.52 | 0.70 | 0.94 | >2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| weight | 0.19 | 0.26 | 0.18 | 0.11 | 0.065 | 0.035 | 0.020 | 0.010 | 0.0035 | 0.0015 | <2e-4 |

90% of the weight is at k <= 0.22, 97% at k <= 0.39; per snapshot, z = 0.75-2
carry 80% of it. So Delta chi2_Roman = 1 corresponds to Delta chi2_FL of
roughly 20-100 if the misfit sits at k <= 0.15 and z = 0.75-2, 300-1000 at
k ~ 0.4-0.5, and > 1e4 at k > 1. The FLAMINGO GoF failures above sit where
Roman's weight is already small.

### 3. Figure of bias (Roman xi_+-, data = GG + FLAMINGO IA, case A)

Machinery checks: TATT(A2 = b_TA = 0) = NLA to machine precision; exact
non-linear fits (CAMB + cosmolike at every point) reproduce the linearised
chi2, dOmega_m, dS8 to all quoted digits; TATT's own best-fit B modes are
chi2 0.06-0.24, so the missing B modes on the sim path do not drive A2 to 0.
IA bounded to the chain priors (|A1|, |eta1|, |A2|, |eta2| <= 5, 0 <= b_TA
<= 2), TATT from 4 starting points; FoB uncertainty from FLAMINGO's error
bars propagated through the cell Jacobian.

| mask | NLA chi2 | NLA FoB | TATT chi2 | TATT FoB |
|---|---|---|---|---|
| 0.2/D | 0.44 | 0.24 +- 0.07 | 0.37 | 0.23 +- 0.06 |
| 0.25/D | 0.81 | 0.23 +- 0.07 | 0.68 | 0.26 +- 0.06 |
| 0.3/D | 1.18 | 0.31 +- 0.08 | 0.96 | 0.34 +- 0.05 |
| 0.35/D | 1.48 | 0.39 +- 0.09 | 1.25 | 0.42 +- 0.07 |
| 0.4/D | 1.90 | 0.61 +- 0.09 | 1.55 | 0.58 +- 0.06 |
| 0.5/D | 2.53 | 0.62 +- 0.11 | 2.06 | 0.63 +- 0.09 |
| 0.6/D | 3.38 | 0.80 +- 0.13 | 2.70 | 0.76 +- 0.10 |

NLA and TATT give the same FoB (within the FLAMINGO noise) although TATT's
chi2 is always lower. Adding FLAMINGO's P_BB to the data lowers both
(0.22 / 0.25 at 0.35/D); the raw spectra raise both (0.53 / 0.50).

### 4. Why: FLAMINGO's II has a white excess over its GI

X(k) = (P_EE - N_s) - P_dE^2 / P_NL is flat in k at every snapshot
(`fig_white_ii`), at 1-2 times the subtracted white level N_s (from P_BB):
E-mode white power 2-4 times the B-mode white power, i.e. small-scale
(one-halo-like) alignments, not shape noise. Any single-tidal-field model
(NLA exactly, TATT at leading order) has X = 0. A white P_EE gives a
constant C_l, hence xi_+ = 0 but xi_- = 2C/(pi theta^2), a theta^-2 tail
cosmolike reproduces exactly: for the (0,0) bin it is 2 sigma at 63' and
1 sigma at 116', so the k_UV masks remove only part of it.

Decomposition of the truth (`bias_white.py`, FoB NLA / TATT):

| truth | 0.25/D | 0.35/D | 0.5/D |
|---|---|---|---|
| FLAMINGO GI, II = GI^2/P_NL (single tidal field) | 0.08 / 0.09 | 0.10 / 0.09 | 0.12 / 0.07 |
| + white II excess, exactly white | 0.17 / 0.20 | 0.30 / 0.34 | 0.35 / 0.33 |
| + its measured k structure (= production) | 0.23 / 0.26 | 0.39 / 0.42 | 0.62 / 0.63 |

With one extra linear parameter, a free amplitude for the white II template
(`bias_wtemplate.py`, fitted a_w = 1.2-1.4): FoB 0.10 / 0.14 (0.25/D),
0.16 / 0.12 (0.35/D), 0.44 / 0.28 (0.5/D), and TATT is now the better
model where it should be. Holding the white term white below k = 0.05
instead of the ratio (`bias_phys.py`) changes the FoB by < 0.02.

### 5. Reading for the scale cut

* Perturbative (two-halo) mismodelling: Roman barely sees it. With a
  single-tidal-field truth both models have FoB ~ 0.1 out to 0.5/D; the
  residual Roman chi2 reaches 1 at K0 ~ 0.3/D (NLA) and ~ 0.37/D (TATT).
* The common bias is the white one-halo-like II term, which neither model
  has and which the k_UV masks do not remove: FoB = 0.3 at K0 ~ 0.3/D for
  both, 0.6 at 0.5/D.
* If that term is to be treated as part of the truth: K0 <= 0.25/D (FoB
  0.23-0.26 +- 0.06), and TATT does not buy anything over NLA. If it is
  modelled (one free white II amplitude, or a halo-model II), K0 = 0.35/D
  is safe for both and TATT is preferred at 0.5/D.
