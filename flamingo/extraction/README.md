# ia-power for FLAMINGO (SOAP catalogue only)

Intrinsic-alignment power spectra for FLAMINGO, measured straight from the
SOAP-HBT halo catalogue: no particle data, no matter field (the one exception,
`matter_cross.py`, is described below). Adapted from the
TNG300 Python pipeline (`ia_power_py`).

| stage | script | input | output |
|---|---|---|---|
| 1 | `soap_galaxies.py` | `SOAP-HBT/halo_properties_XXXX.hdf5` | `galaxies_<sn>.hdf5` |
| 2 | `galaxy_ds_field.py` | stage 1 | `galdens_/e1z_/e2z_…_sn<sn>_nfold<n>.bin`, `hinfo_…dat` |
| 3 | `ia_power.py` | stage 2 | `Phh, PhE{0,2,4}, PhB{0,2,4}, PEE/PEB/PBB{0,2,4}`, `2d/…` |

`binned.py` and the FFT and grid code in `common.py` are unchanged from the TNG
version. The grid and output file formats are unchanged too.

## Running

```sh
pip install numpy scipy h5py hdfstream     # hdfstream only for remote reading
python soap_galaxies.py   params/soap_galaxies_078.dat
python galaxy_ds_field.py params/galaxy_ds_078.dat  <nfold>
python ia_power.py        params/ia_power.cfg sn=78 nfold=<nfold>
```

or `./run_pipeline.sh` (edit `SNAPSHOTS`, `FOLDS`, `MAX_JOBS`).

`SoapFile` can be a local path, or `cosma:/FLAMINGO/...` to stream the file
from the COSMA data server. Stage 1 reads the catalogue in blocks of 2^24
rows. It reads the mass and particle-number columns for every row, then fetches
the tensors and positions only for selected rows. Remotely this takes about
10 s per 2M subhalos, so roughly 30 min for L2p8_m9 z=0 (393M subhalos).
On COSMA, point `SoapFile` at the local file instead.

## What changed relative to TNG

* **No particles, no matter field.** SOAP already stores stellar inertia
  tensors, so `shape_tensor.py` and `density_field.py` are gone. All
  spectra involving `d` (Pdd, Pdh, PdE, PdB) were removed from `ia_power.py`.
* **Units.** FLAMINGO uses comoving Mpc and 10^10 Msun, both *without* h.
  Stage 1 converts to Mpc/h and 10^10 Msun/h, so mass cuts, `boxsize` and k
  keep the TNG conventions. L2p8: `boxsize = 2800 * 0.681 = 1906.8` Mpc/h.
  `ia_power.py` does not read the box size, so set it in the config.
* **Shapes.** `ShapeField` selects any SOAP tensor:
  * `BoundSubhalo/StellarInertiaTensorNoniterative` (default): Σm x x / Σm over
    bound stars within 1 R_half,* (checked against the L1_m8 particles).
    Stored as (xx, yy, zz, xy, xz, yz). Note: L1_m8 SOAP stores stellar
    tensors only for M* > 4e11 Msun; use `particle_shapes.py` below that.
  * `BoundSubhalo/StellarInertiaTensor[Reduced][Noniterative]`: iterative and/or
    reduced variants. 3D tensors exist only in `BoundSubhalo`.
  * `ProjectedAperture/{10,30,50,100}kpc/projz/ProjectedStellarInertiaTensor…`:
    2D tensor in an aperture. Only `projz` is accepted (LOS = z). Its x/y
    ellipticities are NaN.
  Ellipticities use the TNG definitions (e1 = (Iaa−Ibb)/(Iaa+Ibb), e2 = 2Iab/(…)).
* **Selection.** SOAP zeroes stellar tensors for subhalos with
  `NumberOfStarParticles < 100` (the dataset's `Mask Threshold` attribute).
  Stage 1 reads that attribute and applies the cut automatically.
  `Select all|centrals|satellites` uses `InputHalos/IsCentral`. Stage 2 can
  apply a tighter mass or central cut without rerunning stage 1.
* **Positions** are `InputHalos/HaloCentre` (the most-bound particle, like TNG's
  `SubhaloPos`).
* Config keys `den_sn` and `ellip_sn` are now a single `sn`. `dens_dir` and
  `dens_stem` are gone.

## Galaxy-matter cross spectrum (`matter_cross.py`)

No FLAMINGO product holds P_gm, and the reduced snapshots keep particles by
halo membership, so `matter_cross.py stream <sn>` reads every dark-matter
particle of a full snapshot from the server (46.7 billion for L1_m8, about
2e6 rows/s, 5.5 to 7 h per snapshot) onto the folded grids of the galaxy
field, with a checkpoint every 30 min. `matter_cross.py power <sn>` then
writes `Pmm_dm`, `Pgm` and `Pgg` per fold (no shot-noise subtraction, V/N in
`matter_cross_sn<sn>.json`). `run_matter_cross.sh` runs the queue (z = 3, 2,
1.5), restarting a failed stream from its checkpoint. The spectra are in
`flamingo/data/PS_matter/`; the measurement is written up in
`notes/flamingo_pgm/`.

## Things to know

* **Mass floor.** At m9 a star particle weighs about 10^9 Msun. The 100-particle
  mask therefore means M* ≳ 4×10^10 Msun/h (10^10.64 in a test on the real
  z=0 catalogue). TNG-style cuts at 10^9 Msun/h have no effect. For lower
  masses use the m8 boxes (L1_m8), or a shape measured with fewer particles,
  which SOAP does not provide.
* **Sign.** In this code's E convention, *radial* alignment toward
  overdensities gives **P_hE < 0**. This is checked in the mock test.
* **Noise.** The shape-noise floor (V/N)⟨e1²+e2²⟩/2 is subtracted from
  PEE0/PBB0 as before. Phh is written *without* Poisson subtraction (as in
  TNG). The log prints V/N_gal for reference.
* No window deconvolution (NGP fields), as before. Use folding (`nfold`) for
  high k.

## Testing

`bash tests/run_mock_test.sh` builds mock SOAP files with the real group,
dataset and mask-attribute layout and runs all stages. It checks masking,
unit conversion, the e1/e2 formula, 2D vs 3D tensor consistency, and that
a radially aligned sample gives a strong P_hE with P_hB ≈ 0 while a random
sample gives P_hE ≈ 0.
