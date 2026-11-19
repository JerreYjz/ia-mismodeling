# FLAMINGO m9 vs m8 resolution tests for the IA spectra

Can the 2.8 Gpc FLAMINGO box (L2p8_m9, 8 times heavier particles than L1_m8) be used for the IA
spectra of the paper's sample (log M* ~ 10)? The tests here answer it; the write-up is
`notes/flamingo_m9_resolution/flamingo_m9_resolution.pdf`. Run on 2026-11-19 from the extraction
code in `../extraction/` (unchanged).

| item | what it does | output |
|---|---|---|
| `particle_count.py` | star and DM particle counts against M* in the z = 0 SOAP catalogues of L1_m8, L1_m9 and L2p8_m9 (first 2^22 rows of each, streamed with hdfstream) | `particle_count.json` |
| `discreteness_mc.py` | toy Monte Carlo: how well an N-particle unreduced inertia tensor recovers a galaxy's projected shape (alignment response and extra noise against N) | `discreteness_mc.json` |
| `restest_L1m9/` | high-mass test, M* > 4.05e11 Msun, SOAP 1 R_half tensor: L1_m9 rerun (`run.sh`) next to the L1_m8 and L2p8_m9 spectra copied from `ia_power_flamingo/restest_soap_m9_vs_m8`; `compare3.py` (a_nla, P_dE), `match.py` (same galaxies across L1_m8 and L1_m9) | `results.json`, `match.json`, quick-look PNG |
| `lowmass_L1m9/` | low-mass test, 10.0 < log M* < 10.41, unreduced tensor of all bound stars from the particles: L1_m9 with MinStars 10 against the L1_m8 z = 0 catalogue, by mass cut and as the same matched galaxies (`make_matched.py`); `run.sh` then `compare.py` | `results.json` |
| `lowk_forecast.py` | Gaussian S/N of P_hE per k bin at low k for L1 and L2p8, with exact mode counts | `lowk_forecast.json` |

All spectra are `ia_power.py` outputs at Ngrid 512 (nfold 1 and, for `lowmass_L1m9`, 4), with the
hinfo files (N_gal, e_rms). The run logs are kept locally in each `logs/` (`*.log` is git-ignored).

**Not in the repository** (GB-size galaxy catalogues; scripts take their paths from the
environment or an absolute path):

* the L1_m8 z = 0 catalogue, `ia_power_flamingo/run_L1_m8_particles/results/Galaxies/galaxies_78.hdf5`
  (`lowmass_L1m9/make_matched.py`, `lowmass_L1m9/params/gds_L1_m8.dat`);
* the L1_m8 high-mass SOAP catalogue, `ia_power_flamingo/restest_soap_m9_vs_m8/L1_m8/Galaxies/` (`restest_L1m9/match.py`);
* the L1_m9 catalogues written by the two `run.sh` (`L1_m9/Galaxies/`, `L1_m9_match/`, `L1_m8_match/`):
  `match.py` reads `RESTEST_L1M9_CAT`, `lowmass_L1m9/compare.py` reads `LOWMASS_CATDIR`, and without
  them keeps the matched-shape numbers already in `results.json`.

`compare3.py`, `compare.py` and `lowk_forecast.py` need `ia_halofit_v2` (`cosmology.py`,
`stitch.py`) for the CAMB/halofit P_NL, as `notes/flamingo_vs_tng300` does. Run everything with
`~/cocoa/Cocoa/.local/bin/python`; the `run.sh` scripts run at `nice -n 19` with one thread.
