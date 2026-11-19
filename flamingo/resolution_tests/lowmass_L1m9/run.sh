#!/bin/bash
# Low-mass resolution test, z = 0, 10.0 < log M* < 10.41, ng 512, nfold 1 and 4:
# L1_m9 (particle shapes, MinStars 10) vs L1_m8, by mass cut and as the same matched galaxies.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="nice -n 19 $HOME/cocoa/Cocoa/.local/bin/python"; CODE=../../extraction
ts() { date "+%F %T"; }
echo "$(ts) shapes L1_m9"
[ -f L1_m9/Galaxies/galaxies_78.hdf5 ] || $PY $CODE/particle_shapes.py params/shapes_L1_m9.dat > logs/shapes.log 2>&1
echo "$(ts) $(grep -E 'galaxies with shapes' logs/shapes.log | tail -1)"
$PY make_matched.py
for b in L1_m9 L1_m8 L1_m9_match L1_m8_match; do
  for nf in 1 4; do
    [ -f $b/PS/Phh_sn78_nfold$nf.dat ] && continue
    $PY $CODE/galaxy_ds_field.py params/gds_$b.dat $nf > logs/gds_${b}_$nf.log 2>&1
    $PY $CODE/ia_power.py boxsize=681.0 kmin=0.003 kmax=20 nbin=30 munbin=20 ng=512 sn=78 nfold=$nf \
        base=$b/ gal_dir=Galaxy_DS_Field/ hinfo_dir=Galaxy_DS_Field/ out_dir=PS/ > logs/ps_${b}_$nf.log 2>&1
    rm -f "${b:?}"/Galaxy_DS_Field/*.bin
    echo "$(ts) $b nfold $nf done"
  done
done
echo "$(ts) all done"
