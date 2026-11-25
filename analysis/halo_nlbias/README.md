# Nonlinear halo bias in the IA halo model

Can the nonlinear halo bias beta_NL(k, M1, M2) of Mead & Verde 2021
(arXiv:2011.08858) replace the free nonlinear share alpha that the HMc refit
needs to fit the FLAMINGO IA spectra to k ~ 1 h/Mpc? The note
`halo_nlbias.pdf` lays out the model (equations, the beta_NL source, the
TATT terms, the 1-halo terms, exclusion and the low-k limit) and tests a
prototype on the nine core FLAMINGO snapshots (z = 0.2 to 2).

## Result

* **Fails beyond k_max ~ 0.5.** beta_NL from Dark Emulator reproduces most of
  the alpha correction at k = 0.3 to 0.5 (55 to 69 percent at z = 0.2 to 0.3,
  75 to 96 percent at z = 0.4 to 0.5, 110 to 130 percent at z = 0.6 to 1). It
  turns over above k ~ 0.7 (halo exclusion) while FLAMINGO keeps the shape of
  P_NL, i.e. 1-halo power. Summed chi2/nu at k_max = 0.52, 0.70, 0.94, all
  with a_c <= 1 and no free alpha:
  beta_NL 1.49, 4.08, 15.9; alpha refit 0.97, 1.08, 1.33; HMc 2.10, 4.79, 12.7.
  With b_TA = 0: 3.46, 17.6, 88.8. With the F21 satellites: 1.54, 3.70, 10.7,
  and a_1h is unconstrained and above its saturation value.
* **Drift.** b_TA rises from 0 to 0.38 (k_max <= 0.52) to 0.48 to 0.74 (0.94);
  A1 falls to 0.88 to 0.97 of its low-k NLA value at 0.94 (1.00 to 1.06 below
  0.70); a_c sits at its bound of 1.
* **Works.** Reduces to NLA/TATT at low k. On FLAMINGO's own galaxy-matter
  spectrum (z = 1.5, 2) the halo model with beta_NL is within 0.98 to 1.07 of
  the measurement at k = 0.3 to 0.9 (0.62 to 0.88 without). The alpha implied
  by beta_NL rises from 0.26 at z = 0.2 to 0.72 at z = 1, as the fitted alpha
  does.
* **Main limitation.** Dark Emulator is stable only above about
  10^12.6 M_sun/h (z <= 1) and stops at z = 1.48; most FLAMINGO centrals are
  at 10^11.5 to 10^12.5, so beta_NL is held below the emulator's range (with
  beta_NL = 0 there instead, the correction vanishes).

FLAMINGO's errors are underestimated by 3 to 9 times at high k: compare the
chi2 values between models, not with 1.

## What runs, and how long it takes

Plain Python (numpy, scipy, camb, fast-pt, dark_emulator, matplotlib), no
Cocoa or cosmolike. In a fresh environment:

```bash
pip install numpy scipy camb fast-pt dark_emulator matplotlib
```

| step | command (from this folder) | time | output |
|---|---|---|---|
| beta_NL from Dark Emulator | `python beta_tables.py` | 15 s | `beta_tables.npz` |
| halo-model integrals | `python halo_nlbias.py > halo_nlbias_log.txt` | 15 s | `nlbias_tables.npz` |
| the same at the 13 Roman snapshots (z = 0 to 4), for `cut_validation/hm_templates.py` | `python nlbias_roman_tables.py > nlbias_roman_tables_log.txt` | 15 s | `nlbias_roman_tables.npz` |
| fits, every variant | `python nlbias_fit.py > nlbias_fit_log.txt` | 4 min | `nlbias_fits.json` |
| quoted numbers and checks | `python nlbias_numbers.py > nlbias_numbers_log.txt` | 1 min | `nlbias_numbers.json` |
| figure data | `cd plots && python make_figure_data.py` | 10 s | `plots/*/data.json` |
| figures | `cd plots/<name> && python plot_<name>.py` | 5 s each | `<name>.pdf`, `<name>.png` |
| note | `bash ../../.claude/skills/summary-notes/assets/build_note.sh analysis/halo_nlbias` (from the repo root) | 10 s | `halo_nlbias.pdf` |

`nlbias_fit.py name ...` refits only the named variants and keeps the others.
The fits read `analysis/roman_dchi2/v2/fits.npz`, the v6 tables
(`halo_central_tables.npz`, `halo_strict_tables.npz`, `halo_tables.npz`) and
import `notes/flamingo_vs_tng300/fits/hmc_alpha_fit.py` (constants only) and
`analysis/tatt_fit/fit_tatt_flamingo.py` (TATT kernels); nothing outside this
folder is written. The P_gm check reads
`analysis/cut_validation/galaxy_bias/beta_tables.npz`.

## Fit variants (`nlbias_fit.py`)

| variant | spectra under A1 | A2 | b_TA | other |
|---|---|---|---|---|
| `lin_tatt` | HMc: I_1 P_lin, P_lin | free | free | |
| `alpha_tatt` | alpha refit (reproduces `alpha_cap_nosat`) | free | free | alpha free |
| `nlb_tatt` | P_lin (I_1 + I^beta_cm), P_lin (1 + I^beta_cc) | free | free | |
| `nlb_tt` | as nlb_tatt | free | 0 | the layout's fiducial |
| `nlb_nla` | as nlb_tatt | 0 | 0 | |
| `nlbT_tt` | shape bias independent of mass | free | 0 | |
| `nlblo_tt` | beta_NL = 0 below the emulator's masses | free | 0 | |
| `nlbsame_tt` | z = 1.5, 2 only: z = 1.48 table on the same masses | free | 0 | |
| `alpha_nla`, `nla` | alpha refit, alpha free or 1 | 0 | 0 | |
| `*_sat` | as the base variant | | | F21 satellites, a_1h free |
| `tatt` | P_NL, P_NL (TATT reference) | free | free | |
| `lin_nla` | HMc in the NLA form | 0 | 0 | |

All variants: a_c in [0, 1], free white term W, N_s fixed, P_hE = b(k) P_dE.

On the Mac (25 November 2026) Dark Emulator lives in its own venv,
`~/cocoa/venvs/darkemu`, and CAMB comes from the cocoa env: run every script as
`PYTHONPATH=~/cocoa/Cocoa/.local/lib/python3.11/site-packages ~/cocoa/venvs/darkemu/bin/python <script>`.
The local rerun reproduces the cloud tables to 1e-3 (P_lin, CAMB 1.6.7) and
the summed chi2/nu to 1e-3. The GOF, parameter checks and the Roman figure of
bias of every halo model are written up in `notes/halo_ia_models`.

## Layout

| path | what it holds |
|---|---|
| `halo_nlbias.tex`, `.pdf` | the note |
| `beta_tables.py` | beta_NL per mass bin from Dark Emulator |
| `halo_nlbias.py` | the 2-halo integrals with beta_NL, and the P_gm, P_gg, P_mm checks |
| `nlbias_fit.py` | the fits |
| `nlbias_numbers.py` | every number of the note, the sign checks |
| `plots/make_figure_data.py` | writes each figure's `data.json` |
| `plots/gi_kernel/` | the GI 2-halo spectrum per unit A1 C1 P_lin, models and FLAMINGO |
| `plots/bias_checks/` | the halo model against FLAMINGO's P_gm and P_gg |
| `plots/fit_gof/` | summed chi2/nu against k_max |
| `plots/fit_params/` | A1 and a_1h against k_max |
