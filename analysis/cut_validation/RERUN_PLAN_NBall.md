# Rerun plan: the stitched truth with the close-pair term kept (CP)

Written 2026-11-23. Do the steps in this order. Steps 0 and 1 are done
(2026-11-23); steps 2 to 4 have not been run.

**Scope:** only the stitched fiducial IA spectra injected into cosmolike. The
production, E minus B and linear-limit truths are not rerun.

**The truth:** CP (`v3/cp_truth.py` -> `v3/cp_truth.npz`), which replaces NBall
(step 0.1). It is S1 with the close-pair II term kept, in E and in B, with S1's
stitch weight w:
- E: R_S1,II + w N_s/P_NL + (1 - w) N_low/P_NL;
- B: w T_BB/P_NL + (1 - w) N_low/P_NL;
- GI: that of S1.

N_low, the term's low-k level, is the measured P_BB at 0.15 to 0.6 h/Mpc.
NBall (`v9/bmode_truth_3x2.py`) used the high-k plateau N_s instead.
In `fisher_minimize.py` the truth name is `CP`.

**Why:** `flamingo/bmode_noise/` showed that N_s is the shape correlation of close
(mostly central-satellite) pairs, equal in E and B, not noise. NB (N_s
switched off with the stitch weight below about 0.05 h/Mpc) removes a constant
term at low k only, which leaves a spurious degree-scale term in xi_+.

**Baseline already on disk** (`fisher_minimize.json`, `q_nb.log`, 2026-11-23):

| Setup | Truths | Models | K0 |
|---|---|---|---|
| cs+cmb | NB, NBall | NLA, TATT | 0.35 |
| 3x2+cmb gt0.5 | NB, NBall | NLA, NLAa, NLAf, TATT, TATTa, TATTf | 0.2 |

**Change of plan (2026-11-23): no minimizer.** Every figure of bias in this
rerun is the Fisher (linearized maximum-likelihood) shift; the minimizer runs
are cancelled. See step 2.

## 0. Settle the truth: done 2026-11-23

- **0.1 Low-k value of the term.** The pair sum (V/N^2) sum c_ij j0(k r_ij)
  is flat below k ~ 1 h/Mpc and falls to N_s only at 5 to 10 h/Mpc.
  - Its low-k level is 1.7 to 1.9 N_s at z = 1 to 3, and 3.3 N_s at z = 0.
  - The sum is not converged at its 1 Mpc/h limit: the 0.3 to 1 Mpc/h shell
    gives 28% of it at z = 1.
  - So N_low was taken from the measured P_BB at 0.15 to 0.6 h/Mpc, which the
    linear-limit IA cannot produce. It is 1.8 to 2.4 N_s at z = 0.2 to 2,
    measured at 22 to 30 sigma.
  - The pair sum agrees with it to 0.93 to 0.98 at z = 0, 1 and 2 (0.86 at z = 3).
  - CP equals NBall where w ~ 1. Its B modes stay flat at N_low below the
    stitch, instead of NBall's drop to N_s and the noisy smoothed T_BB at 0.05
    to 0.15.
  - Delta chi2 of xi_ia (example1 covariance):

    | cut | CP vs S1 | NBall vs S1 | CP vs NBall |
    |---|---|---|---|
    | none | 384 | 405 | 9.7 |
    | k_UV = 0.3/D | 0.11 | 0.55 | 0.48 |

  - Files: `v3/cp_truth.npz`, `v3/cp_truth.json`.
- **0.2 Decided: B modes under perturbation.** A knob of II moves E and B
  together. The base stores R_II = R_II_E + R_II_B, both parts kept, because
  the known model errors differ: NLA has no B modes, and native TATT has its
  own.
- **0.3 Decided: models.** Revised 2026-11-23: this rerun uses two sets of IA
  models.
  - **Set 1, power law in z:** NLA (A1, eta1) and TATT (A1, eta1, A2, eta2,
    b_TA).
  - **Set 2, one amplitude per source bin:**
    - NLAz: A1_1..8, cosmolike IA_REDSHIFT_BINNING, `IA_redshift_evolution 2`.
    - TATTz: A1_i and A2_i per bin.
  - The halo-model variants, including refitting HMc, and the other freer
    redshift evolutions (NLAbk, NLAhz, TATTbk, TATThz) are done together on
    another day.

## 1. Base and model fits for CP: done 2026-11-23

Code: `v9/bmode_tables.py` (the B part (xi_+, -xi_-) for the builders). B-mode
support was added to `v9/ia_base.py`, `v9/ia_base_3x2.py`, `v8/fid_fits.py` and
`v9/fits_3x2.py`. `fid_fits.py --truths` now adds to `fid_fits.json` instead of
rewriting it; S1 and prod were checked unchanged.

- **1.1 Cosmic-shear base.** `ia_base_CP.npz`: 2379 columns in 91 s.
  - Checks: stale 0, tents vs xi_ia 2e-9, vs dA 2e-8, vs deta 1.8e-3 (the
    usual finite-difference level).
  - R_GI equals S1's exactly. The B part matches NBall's xi_B (max xi_- 6.83e-5).
- **1.2 3x2pt base.** `ia_base_3x2_CP.npz`: 113 s. Every check passes, and the
  xi rows and J agree exactly with the cosmic-shear base.
- **1.3 Model errors at the fiducial cosmology.**
  - `fid_fits.json` has CP; drifts were added to `ia_base_CP.npz` by
    `v9/add_drifts.py`; `fits_3x2.json` has CP. That file never held S1, so
    its 3x2pt fits compare with prod.
  - With the cut k_UV = 0.3/D, A1 and eta1 are unchanged from S1 (3.39, 0.83).
  - The II error grows, as expected since the models have no close-pair term.
    At k = 0.1 to 1, mean over core z:

    | model | S1 | CP |
    |---|---|---|
    | NLA | -0.31 | -0.45 |
    | TATT | -0.27 | -0.39 |

  - GI is unchanged.
  - Without a cut the cosmic-shear chi2 drops (NLA 1517 -> 1012, TATT
    1192 -> 782). This is not yet understood; split it by xi_+ and xi_-
    before relying on no-cut numbers.
  - TATT parameters at a bound:
    - b_TA at 0 in 3x2|0.3 and 3x2|0.2;
    - eta2 at 5 in 3x2b|0.2, where A2 ~ 0, so eta2 is unconstrained.
    - Prod had eta2 at its bound in its 0.3 fits.

## 2. Reconfirm the figure of bias with the Fisher matrix only (no minimizer)

**Method** (Kable, Addison & Bennett 2019, arXiv:1908.01626, Section 2, Eqs. 1
to 6).
- With a Gaussian likelihood and a parameter-independent covariance, expand
  the model about a reference point theta_ref. The maximum-likelihood point
  is then
      theta_ML = theta_ref + F^-1 J^T C^-1 [d - mu(theta_ref)],
      F = J^T C^-1 J,  J = d mu/d theta at theta_ref.
- In our runs:
  - d = GG(D3A) + xi_IA of the truth (CP), noise free.
  - mu = the analysis model (NLA or TATT, every chain parameter).
  - theta_ref = D3A cosmology, Delta z = m = 0, b_i at truth, with the IA
    parameters fitted to d at that fixed cosmology.
  - F gets the Gaussian priors (Delta z, m, the CMB prior) and the flat
    priors as Gaussians of variance width^2/12.
- The shift of (Omega_m, S8) is G dtheta; G is the S8 row by finite
  differences.
- The figure of bias is FoB = sqrt(dp^T C2^-1 dp), with C2 = G F^-1 G^T the
  marginalized (Omega_m, S8) covariance from the same F.
- This is the `fob_mean_linear` that `fisher_minimize.py` already stores next
  to the MAP; the minimizer block is what is dropped.

**Why it can replace the minimizer here** (from the runs already on disk:
NLA and TATT, truths S1, NB, NBall).
- With the CMB prior (81 runs), Fisher minus MAP has median -0.004 and
  |diff| < 0.018 for 90% of runs; the worst case is 0.058 (13%,
  cs+cmb|TATT|0.6|S1).
- 80 of 81 runs land on the same side of 0.3; the exception is
  3x2+cmb|NLA|0.4|S1, at 0.302 MAP against 0.299 Fisher.
- Without the CMB prior (8 old runs) the Fisher value is up to 60% low, so
  the CMB prior is required. Every run below uses it.
- The flat priors as Gaussians are also required. Without them F is near
  singular in the poorly constrained directions: for cs+cmb|TATT|0.35|NBall
  the plain-F shift is (-0.09, +0.09) sigma against (-0.18, +0.16) MAP, while
  the flat-prior Fisher gives (-0.18, +0.16), FoB 0.279 against 0.284. That
  resolves the linear-against-MAP gap noted earlier.

**Code (to write).** Add `--fisher-only` to `fisher_minimize.py`.
- It keeps the IA-only reference fit, J, F and the linear shift; it skips the
  three full least-squares starts.
- It writes the same row without the MAP fields, under the same key, so
  bias_pieces.py (step 3) reads it. bias_pieces.py must accept rows without
  fob_map.

**Decided (2026-11-23): the reference fit.** For each model the IA parameters
are fitted to d at the fixed fiducial cosmology (IA-only least squares), and
the Fisher shift is taken from there, as Kable et al. expand about the best
fit. The fits:

| set | model | IA parameters fitted |
|---|---|---|
| 1 | NLA | 2 |
| 1 | TATT | 5 |
| 2 | NLAz | 8 |
| 2 | TATTz | 17 (A1_i, A2_i, one b_TA) |

**Set 2 per-bin models.**
- NLAz exists in `fisher_minimize.py`. Fisher against minimizer in the stored
  runs (cs+cmb and 3x2+cmb, K0 = 0.2 and 0.25): within 0.008 on S1 and 0.022
  on prod.
- TATTz is new: `model=1, evo=2`, with roman_A1_i and roman_A2_i for
  i = 1..8. cosmolike (IA.c, IA_REDSHIFT_BINNING) reads A1, A2 and b_TA per
  source bin, and the likelihood already passes roman_A2_<i> and
  roman_BTA_<i>.
- **Decided (2026-11-23): one shared b_TA** for TATTz, 17 IA parameters. One
  parameter is written into all 8 roman_BTA_i in Setup.vector (code to write),
  in [0, 2] as TATT.
- TATTz needs starts for the reference fit (A1_i, A2_i from the set-1 TATT fit
  at each bin's mean z), and the Fisher-against-minimizer agreement is
  untested for it. Run its four step-size checks first.

**Cost:** about 50 to 70 cosmolike evaluations per run for J, plus the IA
reference fit, against 1100 to 2600 for the minimizer: about 20 to 40 s per
run.

Runs, all with truth CP and the CMB prior, each for the four models (NLA,
TATT, NLAz, TATTz):
- **2.1 cs+cmb, K0 = 0.2 to 0.6** (7 cuts, 28 runs). This is the setup most
  sensitive to the close-pair term (FoB at K0 = 0.35: NB 0.54/0.42,
  NBall 0.22/0.27).
- **2.2 3x2+cmb gt0.5, K0 = 0.15 to 0.6** (8 cuts, 32 runs). Reconfirms the
  recommended cut (xi 0.2, gamma_t 0.5).
- **2.3 3x2+cmb at the other gamma_t cuts:** no gamma_t cut (7 xi cuts),
  gt0.4 (0.25 to 0.35) and gt0.6 (0.25, 0.3); about 48 runs.
- **2.4 2x2+cmb, K0 = 0.2 to 0.6** (28 runs). The separate-cut section;
  gamma_t has no II, so expect small changes. This is a check.
- **2.5 Other models: not in this rerun** (decision 0.3).
- **2.6 Galaxy-bias configurations** (gbflam, w0.15): NLA and NLAz, 6 runs.

All about 140 runs: about 25 to 40 min on 4 workers. TATTz is the slowest,
with 12 more derivative columns than TATT and the largest reference fit.

Checks:
- **Derivative steps.** Recompute J with every step halved and doubled for
  four runs (cs and 3x2, NLA and TATT, at the recommended cut). The FoB must
  move by less than 0.01; Kable et al. test 0.5% to 2% steps the same way.
- **Against S1.** Compare each run's Fisher FoB with the stored
  `fob_mean_linear` of the S1 run with the same key (same method, old truth),
  and with the NB and NBall points. NLAz on S1 exists at K0 = 0.2 and 0.25
  only; TATTz has no earlier runs.
- **Set 1 against set 2.** At each cut, the FoB of NLA against NLAz and TATT
  against TATTz: how much of the bias is the power-law redshift dependence.
- **Size.** Report every FoB above about 1.3 as outside the tested range; the
  largest one checked against a minimizer was 1.30 (3x2+cmb|NLA|0.2|S1|gt0.5|gbflam,
  Fisher 1.28).
- **Reference fit.** chi2 at the reference; IA parameters at a bound.

**Step 2 results (2026-11-23, done).** All 142 runs plus the step checks;
figure plots/cp_fisher_k0scan (make_cp_fisher_figure.py).
- Two code fixes on the way. (1) 16 runs (set 2 on 2x2, and at 3x2 gt0.5
  K0 0.15) had a singular plain F: A1_1 (and A2_1) see no data there. Only
  the flat-prior Fisher is defined; the row records no_info. (2) The cs set-2
  reference fits sat in the A1_8 ~ 0 basin of II (chi2 0.92 against 0.41 at
  A1_8 = 7.4 for NLAz 0.35). trend_restart gives a second start; it fired in
  all 14 cs set-2 runs and in no 3x2 run. The old S1 cs NLAz runs (0.2, 0.25,
  minimizer included) sat in the same basin.
- Step checks (set 2 redone after the restart): FoB moves at most 0.007.
- Set 1, CP against S1 (62 points): median -0.001, 87% within 0.03; largest
  cs TATT 0.6 (0.27 against 0.40), 3x2 TATT 0.6 (0.34 against 0.26). 2x2 is
  identical, as expected (no II in gamma_t or w).
- Set 2 against set 1: cs lower (NLAz/TATTz 0.05 to 0.20 against 0.07 to
  0.27); 2x2 near 0; 3x2 much higher (NLAz 0.78 to 1.0, TATTz 0.40 to 0.68)
  at every gamma_t cut, with a higher reference chi2 than set 1 (NLAz 4.8
  against NLA 1.7 at gt0.5 K0 0.2). That fit is a unique minimum (four
  starts). cs alone and 2x2 alone prefer different A1_i in the top bins
  (bin 7: 5.51 +- 0.25 against 5.00 +- 0.67), so a constant per bin cannot
  serve xi and gamma_t at once, while the power law in z inside each bin can.
- FoB above 1.3: NLAz gt0.5 gbflam (1.62) and gt0.5 w0.15 gbflam (1.93),
  outside the tested range.

## 3. Bias cancellation: the spline decomposition (numpy, after step 2)

- **3.1 Linear maps.** `bias_pieces.py` on the step-2 keys. It writes the
  linear map from the data-minus-model vector to the shift (`pieces_maps/`), so
  every later split is numpy only.
- **3.2 The decomposition** (new script). For each run, the contribution of
  each piece to the shift of Omega_m and S8:
  - **Basis:** positive B-splines in ln k, a partition of unity, 3 per decade
    over 0.01 to 10 h/Mpc. Contribution_i = sum_j a_j c_j B_i(k_j) over the 61
    tents, so the contributions add up to the linear total exactly.
  - **Split** by spectrum (GI, II in E, II in B) and by snapshot, or by B-splines
    in z.
  - **Amplitude first:** project the error onto the directions the IA
    parameters absorb (A, eta; also a2 and bTA for TATT) and decompose only the
    remainder. This removes the artificial +/- pairs of the hard k windows, and
    replaces the "error at k = 0.01 at every k" amplitude piece of
    `bias_pieces.py` "zshape".
  - **Outputs:**
    - the response a_i and the error p_i per node, plotted separately, to
      tell a sign change of the error from a sign change of the response
      (the Omega_m response to GI changes sign near 0.6 h/Mpc with A free);
    - the running sum of the contributions against k;
    - the cancellation ratio |actual shift| / rss (rss = the shift if each
      node had a random sign, the assumption of the band);
    - for 3x2pt, the split by data block (xi_+, xi_-, gamma_t).
- **3.3 Checks.**
  - The pieces sum to the linear total to 1e-8.
  - The linear total agrees with the base E (tents) and with the direct E
    (cosmolike), and with the step-2 Fisher shift of the same run.
  - The S1 decomposition through the new code reproduces `v9_pieces` and
    `cancel_signed` when the same hard windows are used.
- **3.4 Set-2 models.** Per-bin amplitudes are not a function of z, so their
  error cannot be written as P^model/P^truth - 1 per snapshot on the tent base
  (bias_pieces.py has no NLAz case). For NLAz and TATTz, split the direct
  error d - mu(ref) through the linear map by data block (bin pair, angle,
  xi_+/xi_-/gamma_t) instead. Alternatively, map each bin's amplitude to z
  through its n(z) and say so.
- **3.5 Figures.** Ask which existing figure each one replaces
  (`v9_pieces`, `cancel_signed`, `cancel_scan`) and keep its format.


**Step 3 results (2026-11-23, 3.1 to 3.4 done; 3.5 figures drafted).**
bias_pieces.py (CP: E and B parts of II, wc/gb passed, per-bin and gbias runs
store the linear map only) on all 142 keys; spline_pieces.py (new) on all;
figures plots/cp_cancel_scan, cp_cancel_signed, cp_spline_nodes,
cp_data_blocks (make_cp_pieces_figures.py).
- Checks: nodes sum to the base total 1.4e-15; data blocks to the direct
  1e-16; hard windows reproduce bias_pieces 1e-14; S1 rerun through the new
  bias_pieces equals the stored pieces (1e-14); direct = step-2 Fisher
  (2e-13); base against direct FoB median 0.009, worst 0.028 (snapshot
  interpolation, as before).
- Amplitude projection along A1 and eta1 only: its shift is at most 0.028.
  Projecting TATT's A2, eta2, b_TA as well gave an amplitude piece up to 0.46
  (they rest on their flat priors in cs, so an error along them is not
  absorbed) that the nodes cancelled, so they are not projected.
- Cosmic shear: GI and II (E+B) pieces each 0.15 to 1.2, total 0.07 to 0.27;
  cancellation ratio FoB/rss 0.24 to 0.44. The II error of NLA is one-signed
  and rises with k (the close-pair term: 0 at 0.1, 0.5 at 0.46, ~1 at high
  k; B error is 1, NLA has no B). The S8 response to II changes sign near
  0.3 h/Mpc (+1.6 at the 0.2 node, -0.65 at 0.46 for K0 0.4), so the
  cancellation is a sign change of the response, not of the error.
- 3x2pt gt0.5: GI carries the bias, little cancellation (ratio about 1.2).
- Set 2 by data block: in 3x2pt the per-bin models' Omega_m shift comes from
  gamma_t (NLAz -0.4 to -0.7 sigma, TATTz -0.25 to -0.5), against near zero
  from gamma_t for NLA and TATT; w is exactly zero (no IA).
- 3.5 done (2026-11-23, user: "Replace both", then "full rewrite"): the
  note's cancel_scan and cancel_signed now hold the CP figures; new figures
  cp_fisher_k0scan, cp_spline_nodes, cp_data_blocks. Section sec:cancel is
  rewritten for CP (Fisher-only total, the new split, reliable cut 0.25 /
  0.3 for 3x2pt NLA, per-bin amplitudes, the earlier stitched-truth split as
  one paragraph); Setup gains the close-pair truth; Summary, sepcuts,
  scale-cut, appendices updated. Numbers block Part 1g. Build: 0 errors.

## 4. Optional, if the note's linear products should follow the new truth

`scale_cut_scan.py`, `scale_cut_pieces.py`, `scale_cut_scan_3x2.py`,
`ia_budget*.py`, `spline_budget.py` and `band_kernel.py` on CP. All are numpy
on the step-1 bases, minutes each. `validate_direct.py` checks the base against
direct injections.

## Open, not in this rerun

- **Resolution test (m8 against m9) of the close-pair term.** 11 to 20% of it
  comes from pairs closer than 10 to 30 kpc/h; this sets its amplitude.
- **Cosmic-shear-alone cut rerun.** Previously deferred; covered by 2.1 if the
  cut is read off the K0 scan.
