# IA mismodeling for Roman: scale cuts, FLAMINGO and the halo model

Code, inputs, results and notes of the intrinsic-alignment (IA) mismodeling
paper: which scales of Roman cosmic shear (and 3x2pt) can be analyzed with the
NLA and TATT models when the true alignments are those measured in the
FLAMINGO simulation, and how a halo model with central alignment (HMc) does.

**Start here:**
* `notes/ia_error_budget/ia_error_budget.pdf`: the IA error budget of Roman
  cosmic shear (how wrong the FLAMINGO IA spectra may be, scale by scale,
  before Omega_m or S8 moves by 0.3 sigma), also for random errors
  correlated in k, and NLA / TATT against it.
  Scripts in `analysis/roman_dchi2/v8/`.
* `notes/scale_cuts_summary/scale_cuts_summary.pdf`: the scale-cut work
  (November 2026).

Every figure redraws from its own folder under the note's `plots/`.

## Layout

| folder | what it holds |
|---|---|
| `notes/ia_error_budget/` | the error-budget note and its figures; `plots/make_figure_data.py` exports the numbers from `analysis/roman_dchi2/v8/` |
| `notes/scale_cuts_summary/` | the summary note (`.tex`, `.pdf`) and its figure folders (`data.json`, `plot_<name>.py`, PDF, PNG); `plots/make_figure_data.py` exports the numbers from `analysis/` |
| `analysis/` | the scale-cut pipeline: Part 1 (NLA on a TATT fiducial, Cobaya yamls, Gauss-Newton bias, IR/UV masks) at the top level; `tatt_fit/` (the TATT model in cosmolike conventions, FLAMINGO loaders); `roman_dchi2/` (FLAMINGO fits `v2`, simulation cuts and Roman tests `v3`, strategies `v4`, production covariance `v5`, **halo model `v6`**, the cosmolike driver `cocoa/`, the work-log reports `report/`). See `analysis/README.md` and `analysis/roman_dchi2/v6/README.md` |
| `flamingo/extraction/` | the FLAMINGO IA power-spectrum extraction (SOAP catalogue to P_hh, P_hE, P_EE, P_BB) and the L1_m8 run scripts |
| `flamingo/data/` | its outputs used here: `PS/` (the 858 spectrum files), `hinfo/` (galaxy counts and e_rms), `flamingo_pmm/` (matter power), `stitched_all/` |
| `cosmolike_patches/` | the sim-IA patches for cocoa-cosmolike-core and cocoa_roman_real (IA_code = 3: tabulated simulation IA) |
| `.claude/skills/summary-notes/` | the note and figure style used for every note here (a copy of the personal skill) |

The scripts find each other through relative paths (`../v2/fits.npz`,
`../cocoa`, `../../tatt_fit`): keep the layout.

## Requirements

Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `camb` and `fast-pt`. On this
machine the cocoa interpreter has them: `~/cocoa/Cocoa/.local/bin/python`.
The Roman steps also need a cosmolike build with the two patches applied
(local branch `sim_ia` of `Cocoa/external_modules/code/cosmolike_core` and of
`Cocoa/projects/roman_real`), compiled with
`-DSIM_KMIN_HMPC=1e-6 -DSIM_KMAX_HMPC=0`, and the compiled library copied to
`analysis/roman_dchi2/cocoa/lib/` (not committed: it is machine specific).

## Reproducing

```bash
PY=~/cocoa/Cocoa/.local/bin/python
# FLAMINGO spectra -> fits in the cosmolike convention (minutes)
cd analysis/roman_dchi2/v2 && $PY fit_models.py --ps-dir ../../../flamingo/data/PS --out fits.npz
# simulation cuts and figures, no cosmolike needed (~20 s each)
cd ../v3 && $PY compare_fits.py && $PY v3_flamingo.py
# the note's figures from the shipped results
cd ../../../notes/scale_cuts_summary/plots && $PY make_figure_data.py
for d in */; do (cd $d && $PY plot_${d%/}.py); done
```

`fit_models.py` reproduces the shipped `fits.npz` data arrays exactly; the fits
themselves depend on the CAMB version (tested with 2.0.5; CAMB 1.6.7 moves
P_NL by 0.07% and the degenerate TATT directions by more), so the shipped
`fits.npz` is the reference.

## Provenance

The analysis and halo-model code were written in a Claude Code session
(1 to 11 November 2026) and delivered as `nla_tatt_bundle`,
`nla_tatt_covariances` and `hmc_bundle`; they are merged here unchanged except
for the raw-spectrum path of `analysis/roman_dchi2/cocoa/verify_injection.py`,
which now points to `flamingo/data/PS`. The scripts under
`flamingo/extraction/run_L1_m8_particles/` that drew the October update PDFs
still point to a local TNG stitch demo in `~/Downloads/ia_halofit_v2`.
