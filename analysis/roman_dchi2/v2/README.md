# v2: FLAMINGO P(k) Delta chi2 and its translation into Roman Delta chi2

One residual, two metrics. Everything uses the same FLAMINGO spectra
(P_hE, P_EE, P_BB, folds 1/4/16 combined, FLAMINGO's measured error bars,
13 snapshots z = 0-4) and the same model fits.

## Step 1: FLAMINGO only (`fit_models.py`, `plot_step1.py`)

Per snapshot and k_max (0.12-3.0 h/Mpc), fit over 0.05 <= k <= k_max:
NLA (a1, Ns) and TATT (a1, a2, b_TA, Ns); Ns = white shape noise common to
P_EE, P_BB. Each fit twice:
* to the measured spectra: chi2_raw, dof, p-value -> goodness of fit;
* to a noise-free smoothed version of the spectra (weighted smoothing spline
  in ln k, chi2(data - spline) = N): Delta chi2_FL = misfit with FLAMINGO
  errors, used for the translation.

GOF cut (last k_max before two consecutive p < 0.05), `kmax_gof.csv`:
NLA 0.22-0.39 h/Mpc, TATT 0.39-0.70 h/Mpc depending on the snapshot.

## Step 2: Roman (`roman_map.py`, `plot_step2.py`)

The residual dP(k, z) = model - FLAMINGO of each fit, restricted to
0.05 <= k <= K, goes into
* Delta chi2_FL(K): sum over 13 snapshots with FLAMINGO errors;
* Delta chi2_Roman(K): Limber transform, Gaussian covariance (2415 deg^2,
  41.3/8 gal/arcmin^2 per bin, sigma_e = 0.30), IR/UV mask (alpha = 0.05) at
  constant comoving K; real space (xi_+-, 1080 points) and Fourier (EE
  bandpowers 30-4000, 540 points).
The white shape-noise offset is re-minimised in both (a nuisance: white
power gives no signal at theta > 0). "Full" adds the measured residual
above K (to 17.7 h/Mpc): the leakage past the cut.

Results (`roman/thresholds.csv`):
* Delta chi2_Roman = 1 corresponds to Delta chi2_FL ~ 1000 at K ~ 0.4 and
  ~ 500 at K ~ 0.5-0.7 (sum over the 13 snapshots; ~40-80 per snapshot),
  both for a pure amplitude change and for the actual NLA/TATT residuals
  (within a factor ~2).
* Model fails in Roman (Delta chi2_Roman = 1 with K = k_max):
  NLA k_max = 0.39 (real) / 0.42 (Fourier); TATT 0.66 / 0.69.
* Leakage from measured scales above K: <= 0.7 in real space for NLA
  (moves its real-space cut to 0.31), negligible in Fourier.

## Caveats

Gaussian covariance only; cosmology fixed (FLAMINGO); snapshots treated as
independent; residual set to zero beyond the measured range (extrapolating
it dominated small-theta xi_+-, see `leak_diag.py`).
