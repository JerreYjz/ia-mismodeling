# FLAMINGO spectra at z = 1: model fits and the halo-model refit

Work of 17 November 2026 on the FLAMINGO L1_m8 IA power spectra at z = 1
(snapshot 58): the best NLA, TATT and halo-model (HMc) fits drawn against the
measured spectra, and a refit of the halo model after its central 1-halo
amplitude turned out to be unphysical. The written record is the addendum of
`analysis/roman_dchi2/report/hmc_review.pdf` (its last section).

## Result

* The HMc + W fit needs a central 1-halo amplitude a_c of 13 to 190, against a
  physical size of about 0.5. The amplitude follows the fit range (at z = 1:
  102, 82, 38, 17, 2.7 for k_max = 0.39, 0.52, 0.70, 0.94, 3.04): it
  compensates for the nonlinear power that the P_NL to I_1 P_lin swap removes
  from the 2-halo term.
* With the A1 terms on I_1 P_lin + alpha (P_NL - I_1 P_lin) (GI) and
  P_lin + alpha (P_NL - P_lin) (II), alpha free and a_c <= 1, the summed
  chi2/nu over the nine snapshots at k_max = 0.52, 0.70, 0.94 is 0.97, 1.08,
  1.33 (no satellites), against 1.04, 1.29, 2.28 for HMc + W with a_c free and
  2.17, 4.49, 8.73 for HMc + W with a_c <= 1.
* Physical-sensibility checks (`fits/hmc_alpha_checks.py`, `fits/checks_z1.txt`):
  alpha (0.31 to 0.83) and A1 pass and are stable against the fit range; a_c
  is unconstrained at a physical amplitude; the satellite amplitude a_1h is
  unconstrained below k of about 1 and drifts with the fit range above it.
* Red and blue galaxies (18 November 2026, the second addendum of
  `hmc_review.pdf`): with red fractions from halo-quenching fits to observed
  galaxies (Zu & Mandelbaum 2016, Tinker et al. 2013) on the FLAMINGO HOD and
  one amplitude per colour at every redshift (Fortuna et al. 2021), the joint
  fit to the nine core snapshots gives chi2/nu = 12.5 (constant NLA amplitudes,
  F21 Eqs. 3-4) to 133 at k_max = 0.52, against 1.49 for one population with
  a power-law evolution. FLAMINGO's amplitude follows the galaxy bias
  (A1/b_g = 2.4 to 2.7 at every snapshot) while the red fraction falls; with
  each colour's amplitude scaled by its bias the fit sets A_blue = A_red.
  A physically consistent split (alignment from the host halo, scaled with
  its bias; blue scaled by a misalignment factor kappa_b; satellites aligned;
  nested on one population) gives kappa_b = 0.7 to 1.1 and no gain: the chi2
  profile in kappa_b has no minimum below 1, and every colour fit that beats
  one population needs kappa_b > 1. The fit improves instead with the
  2-halo nonlinear share alpha evolving with z (chi2/nu 1.49 to 1.18, floor
  1.15 to 0.95), partly by degeneracy with the shared b_TA and A2.

## Layout

| path | what it holds |
|---|---|
| `plots/make_figure_data.py` | writes every figure's `data.json` from `flamingo/data/` and `analysis/` |
| `plots/ia_fit_z1/` | best NLA and TATT fits (`analysis/roman_dchi2/v2/fits.npz`), with percent difference |
| `plots/ia_hmc_z1/` | best HMc + W fit (`analysis/roman_dchi2/v6/f21v_HMc_W.npz`), with its parts |
| `plots/ia_hmca_z1/` | the refit: alpha free, a_c <= 1, no satellites, fit to k <= 0.70 |
| `plots/ia_hmcas_z1/` | the refit with the satellites free, fit to k <= 1.69 |
| `fits/hmc_alpha_fit.py` | the refit of all variants, every snapshot and k_max; results in `fits/hmc_alpha_fits.json` |
| `fits/hmc_alpha_checks.py` | the physical-sensibility checks; output for z = 1 in `fits/checks_z1.txt` |
| `hmc_equations/` | the HMc equations on one slide-sized page (`.tex`, `.pdf`, `.png`) |
| `fits/hmc_rb_fit.py` | the joint red/blue and one-population fits over the core snapshots; results in `fits/hmc_rb_fits.json`, log `fits/hmc_rb_fit_log.txt` (tables from `analysis/roman_dchi2/v6/halo_redblue.py`) |
| `plots/ia_redblue/` | the effective amplitude against z, per snapshot and from the joint fits, with the red and blue fractions |
| `plots/ia_redblue_fit/` | the joint fits against the FLAMINGO spectra at z = 0.3, 1 and 2 |
| `fits/hmc_rb_kappa_profile.py` | the joint chi2 against a fixed blue misalignment factor kappa_b; `fits/hmc_rb_kappa_profile.json` |
| `plots/ia_rb_models/` | every joint model ranked by chi2/nu, with its physical-check status |
| `plots/ia_rb_profile/`, `plots/ia_redblue_kappa/` | the kappa_b profile, and kappa_b against the fit range |
| `plots/ia_rb_alpha/` | the 2-halo nonlinear share alpha against z, per snapshot and joint |

Each figure folder redraws from its `data.json` alone
(`python plot_<name>.py`). `make_figure_data.py` also knows a fifth figure,
`ia_ps_z1` (FLAMINGO against TNG300 at z = 1); its folder is not in the
repository, because the TNG300 spectra are a collaborator's measurements.

## Reproducing

```bash
PY=~/cocoa/Cocoa/.local/bin/python
cd notes/flamingo_vs_tng300/fits
$PY hmc_alpha_fit.py                 # all variants, about 20 minutes
$PY hmc_alpha_checks.py 1.0 > checks_z1.txt
(cd ../../../analysis/roman_dchi2/v6 && $PY halo_redblue.py > halo_redblue_log.txt)   # about 10 minutes
(cd ../../../analysis/roman_dchi2/v6 && $PY halo_redblue.py --light)   # central biases
$PY hmc_rb_fit.py > hmc_rb_fit_log.txt   # about 3 minutes
$PY hmc_rb_kappa_profile.py > hmc_rb_kappa_profile_log.txt   # about 1 minute
cd ../plots
$PY make_figure_data.py ia_fit_z1 ia_hmc_z1 ia_hmca_z1 ia_hmcas_z1 ia_redblue ia_redblue_fit ia_redblue_kappa ia_rb_models ia_rb_profile ia_rb_alpha
for d in ia_fit_z1 ia_hmc_z1 ia_hmca_z1 ia_hmcas_z1 ia_redblue ia_redblue_fit ia_redblue_kappa ia_rb_models ia_rb_profile ia_rb_alpha; do (cd $d && $PY plot_$d.py); done
```
