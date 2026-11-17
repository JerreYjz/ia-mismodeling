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
cd ../plots
$PY make_figure_data.py ia_fit_z1 ia_hmc_z1 ia_hmca_z1 ia_hmcas_z1
for d in ia_fit_z1 ia_hmc_z1 ia_hmca_z1 ia_hmcas_z1; do (cd $d && $PY plot_$d.py); done
```
