# Halo-model IA with central 1-halo alignment (HMc)

Code, inputs and results behind `roman_dchi2/report/hmc_review.pdf`.
Keep the folder layout as is: the scripts find each other through relative
paths (`../v4`, `../v2`, `../v3`, `../cocoa`, `../../tatt_fit`).

## Requirements

Python 3.10+, `numpy`, `scipy`, `matplotlib`, `camb` (tested 2.0.5),
`fast-pt` (tested 4.1.1). The Roman part also needs your CoCoA cosmolike
build (see below).

## What runs where

| step | script (in `roman_dchi2/v6/`) | needs | time |
|---|---|---|---|
| central 1-halo tables P_c(k) (matter and galaxy versions) | `halo_central.py` | CAMB | ~2 min |
| radial-profile / mass-dependence grids | `halo_central_grid.py` | CAMB | ~1 min |
| coherent-satellite tables | `halo_satcoh.py` | CAMB | ~2 min |
| halo-model galaxy bias I_g(k) (no measured b(k)) | `halo_gbias.py` | CAMB | ~1 min |
| FLAMINGO fits of every variant | `halo_central_fit.py <variants>` | the tables above | seconds to minutes per variant |
| FLAMINGO figures | `make_hmc.py`, `make_hmcw.py` | `f21v_*.npz` | seconds |
| review tables and term figure | `make_hmc_review.py` | `f21v_*.npz`, `hmc_scan_ex1_*.json` | seconds |
| Roman K0 scan (NLA, TATT, HMc) | `hmc_scan.py <cocoa data dir>` | cosmolike | ~3 h per cut family |
| Roman FoB figure | `make_hmc_roman.py` | `hmc_scan_ex1_*.json` | seconds |

All tables and results are already included, so every figure script runs
immediately. Example:

```bash
cd roman_dchi2/v6
python3 halo_central_fit.py HMc_W,TATT_W,HMc_nob   # refit; writes f21v_<variant>.npz and a log
python3 make_hmcw.py                               # fig_hmcw_gof, fig_hmcw_spectra_z0p3 / z1
```

### FLAMINGO variants (`halo_central_fit.py`)

| name | model |
|---|---|
| `HMc_lin` | **HMc**: TATT with the A1 terms on I_1 P_lin (GI) and P_lin (II), central 1-halo term, F21 satellites |
| `HMc_W` | HMc + free white E = B term W (main FLAMINGO result) |
| `HMc_nob` | HMc + W with no measured b(k): P_hE from the halo model (I_g, galaxy-position central term) |
| `TATT_W`, `TATT_nob` | TATT references with W (measured b(k) / constant b_g) |
| `HMc_add`, `HMc`, `HMc_g`, `HMc_flat` | diagnostics (P_NL kept; II on P_NL; galaxy-position central term; mass-independent halo shapes) |
| `HMc_lin_p`, `HMc_lin_b`, `HMc_lin_pb` | radial ellipticity profile / mass-dependent alignment grids |
| `HMcs`, `HMcs_noGI`, `HMcs_norad` | coherent satellites (indistinguishable from W) |

The references `TATT` and `hyb` (= HM) come from `f21_variants.py TATT,hyb`.

### Roman scan (`hmc_scan.py`)

`hmc_scan.py` uses the same production machinery as `v3/mask_scan.py` and
`v3/roman_test.py`, through `cocoa/cosmolike_driver.py`. That driver loads the
compiled cosmolike Python library from `cocoa/lib/`, which is not included
(it is specific to the machine it was built on): point it at your CoCoA
build. The run takes the cocoa data directory (with the Roman n(z), masks
etc.) as its argument:

```bash
FAMS=D1 OUTTAG=_ex1_D1 python3 hmc_scan.py /path/to/cocoa_roman_real/data
FAMS=flat OUTTAG=_ex1_flat python3 hmc_scan.py /path/to/cocoa_roman_real/data
```

Environment variables: `COV` (default `analysis/cov_example1_xi.npy`), `K0S`,
`MODELS` (default `NLA,TATT,HMc`), `AC_HI` (upper prior bound of a_c, default 100).
On first run it builds `templates2_hmc.npz` (included) and prints the
validation numbers quoted in the review.

## Key files

* `halo_ia.py`: halo-model core (CAMB linear power, Tinker mass function and bias,
  Bhattacharya concentrations, NFW, HOD, F21 satellite kernel, I_1).
* `halo_central.py`: elliptical-NFW quadrupole kernel `u_E`, Allgood halo shapes
  `sigma_E2`, central 1-halo spectra.
* `halo_central_fit.py`: all FLAMINGO variants (`m_hmc`, `m_hmcw`, `m_nob`, `m_hmcs`).
* `hmc_scan.py`: Roman templates (`hmc_templates`) and the HMc model wiring
  (`model_hm`).
* `f21_variants.py`: shared fitting machinery and the earlier F21/HM variants.
  The extended satellite grids it can use (`halo_ext_tables_*.npz`, 250 MB) are
  not included; only the `F21_ext`, `F21_all` and `hyb_ext` variants need them.
* `../v4/simfit.py`, `../../tatt_fit/fit_tatt_flamingo.py`: FLAMINGO data
  (`../v2/fits.npz`, already in the cosmolike convention), TATT kernels (FAST-PT).

## Results included

* `f21v_*.npz`: FLAMINGO fits (chi2, nu, parameters, cuts per snapshot and k_max).
* `hmc_scan_ex1_{D1,flat}.json`, `hm_scan_ex1_{D1,flat}.json`: Roman FoB scans.
* `fig_hmc*`, `fig_hmcw*`: figures (220 dpi PNG and PDF).
* `../report/hmc_review.{tex,pdf}` and `tab_hmcr_*.tex`.
