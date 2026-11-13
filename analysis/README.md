# NLA and TATT scale cuts for Roman (FLAMINGO-injected IA)

Code, inputs, results and reports for the NLA / TATT scale-cut work. The main
document is `roman_dchi2/report/strategies.pdf` ("Defining intrinsic-alignment
scale cuts for NLA and TATT", 10 Nov 2026). The companions are:

* `summary.pdf`: goodness of fit, Delta chi2, figure of bias and scale cuts.
* `compare.pdf`: FLAMINGO against the best fits.
* `fob_note.pdf`: how the figure of bias is computed.
* `stitched.pdf`: the stitched-truth control.

The halo-model work is in the separate HMc bundle.

Keep the folder layout as is: scripts find each other and their inputs through
relative paths (`../v2/fits.npz`, `../cocoa`, `../../tatt_fit`).

## Requirements

Python 3.10+, `numpy`, `scipy`, `matplotlib`, `camb` (tested 2.0.5),
`fast-pt` (tested 4.1.1). The Roman steps also need your CoCoA cosmolike build
(see "Roman steps" below).

## Pipeline

### 0. FLAMINGO spectra -> fits (`roman_dchi2/v2/`)
* `fit_models.py --ps-dir <raw FLAMINGO Pds files> --out fits.npz`: per-snapshot NLA and TATT fits against k_max.
  It also stores the FLAMINGO data already converted to the cosmolike convention:
  P_hE as 3/2 x monopole, P_dE = P_hE/b(k), errors, the smoothed truth, b(k), P_NL and N_s.
  The raw FLAMINGO files are not included (you have them); `fits.npz` is.
* `flamingo_truth.py`: the data, errors and noise-free truth used everywhere.
* `../../tatt_fit/fit_tatt_flamingo.py`: the TATT model (FAST-PT kernels, cosmolike conventions) and the loaders.

### 1. Simulation-level cuts (`roman_dchi2/v3/`, `v3s/`, `v4/`)
* `v3/v3_flamingo.py`: the production FLAMINGO-only cuts.
  * N_s is fixed to the B plateau, and TATT is bounded to the chain priors.
  * Four criteria per snapshot: GoF (last k_max before two consecutive p < 0.05), anchored, and 5% / 10% deviation.
  * Also gives A1(z), A2(z) and b_TA(z).
  * Outputs `v3.npz`, `cuts.csv` and the `fig_v3_*` figures.
* `v3/compare_fits.py`: chi2/dof against k_max, and data against best fits (`fig_chi2dof*`, `fig_cmp_*`). Runs in about 20 s.
* `v3/knl_scaling.py`, `v3/plot_kuv_z.py`: the redshift scaling of the cuts against k_UV(z) = K0/D(z).
* `v3s/run_stitched.py`, `v3s/plot_stitched_cuts.py`: the same analysis on stitched truths (control).
* `v4/sim_strategies.py`, `v4/simfit.py`: cut-strategy definitions, calibration and scatter over noise realisations.
* `v4/plot_strategies.py`, `v4/make_tables.py`: figures and tables for `strategies.tex`.

### 2. Roman steps (`roman_dchi2/cocoa/`, `v3/`, `v4/`, `v5/`)
These steps run through cosmolike itself (`cocoa/cosmolike_driver.py`), as
described in `cocoa/README.md`:
* cosmolike core and cocoa_roman_real, with `core-sim-IA.patch` and `roman_real-sim-IA.patch` (top level).
* Compiled with `-DSIM_KMIN_HMPC=1e-6 -DSIM_KMAX_HMPC=0`.
* The compiled library goes in `cocoa/lib/`. It is not included, because it is specific to the machine it was built on.

Each Roman script takes the cocoa_roman_real data directory as its argument.

* `cocoa/cosmolike_xi_ia.py`: injects FLAMINGO and the model fits as sim-IA tables, builds the IR/UV masks
  (as `make_uv_scale_cuts.py`), and writes `run/cosmolike_xi_ia.npz` (included).
* `cocoa/bias_analysis.py`, `cocoa/bias_v2.py`: the figure-of-bias machinery (the CL driver and fits).
* `v3/roman_test.py`: the Roman test "as the chain sees it".
  * Every parameter is free.
  * Case A: Omega_m and A_s free.
  * Case B: also n_s, H0 and Omega_b, with the chain priors.
  * FoB uncertainty from the FLAMINGO errors (`J_cells`).
  * Controls: TATT on TATT, NLA on TATT, and NLA on NLA through the injection path.
* `v3/mask_scan.py`: **production scan.** FoB against K0 for k_UV = K0/D(z) and constant K0, with
  NLA and TATT, cases A and B.
  * Results: `v3/mask_scan_ex1_{D1,flat}.json` and their logs.
  * Masks: `v3/masks_scan.npz`.
* `v3/ggl3x2.py`, `v3/plot_scan_3x2.py`: GGL and 3x2pt with FLAMINGO IA in gamma_t (lens-probe cuts).
* `v3/stitched_test.py`, `v3/plot_stitched.py`: the Roman test on stitched truths.
* `v4/split_pm.py`, `v4/verify_split.py`: separate cuts for xi_+ and xi_-.
* `v4/fisher_white.py`: the precision cost of a white II nuisance.
* `v5/linear_cov.py`, `v5/gof_masks.py`, `v5/plot_ex1.py`: the linear-response FoB and Fisher precision per mask,
  with the Gaussian and the production (example1) covariance compared.

### 3. Part 1 (NLA on a TATT fiducial, Cobaya; top level)
* `make_uv_scale_cuts.py`: the IR/UV scale-cut masks (method of arXiv:2603.13031).
* `gn_bias.py`, `fisher_sigma.py`, `make_ext_cov.py`: Gauss-Newton bias, Fisher sigma, and the Planck prior covariance.
* `*.yaml`, `*.sbatch`: the Cobaya MCMC and minimize setups.
* `make_minimize_yaml.py`, `bestfit_shifts.py`, `make_results_table.py`, `plot_*.py`: post-processing.
* `tex/`: tables and figures from that part.

## Inputs included

* `roman_dchi2/v2/fits.npz`: FLAMINGO spectra in the cosmolike convention, plus the NLA / TATT fits.
* **In the separate `nla_tatt_covariances.zip`** (58 MB; unzip it on top of this folder, it has the same layout):
  `roman_dchi2/cocoa/analysis/` covariances `cov_example1_xi.npy` (production), `cov_gauss_d3a.npy`
  (Gaussian) and `cov_example1_3x2.npy` (3x2pt), plus `J_cells.npy` (FLAMINGO-noise propagation).
  Only the Roman steps need them.
* `roman_dchi2/cocoa/run/cosmolike_xi_ia.npz`: GG and FLAMINGO xi_IA, masks.
* `roman_dchi2/v3/masks_scan.npz`, `masks_v3.npz`: the IR/UV masks.

Not included: the raw FLAMINGO power-spectrum files, and the compiled cosmolike
library. Also left out are the regenerable sim-IA table caches (`cocoa/run/tables_*`)
and the earliest v1 results (`results_*`, `validation/`). One script,
`cocoa/verify_injection.py`, has a hard-coded path to the raw spectra: edit
`PS` at its top.

## Quick check (no cosmolike needed)

```bash
cd roman_dchi2/v3
python3 compare_fits.py        # chi2/dof tables and figures, ~20 s
python3 v3_flamingo.py         # production FLAMINGO-only cuts
```

Figures are included as vector PDFs only, to keep the download small;
every figure script writes the 220 dpi PNG next to the PDF when rerun.
