#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=~/cocoa/Cocoa/.local/bin/python; CODE=../../extraction
$PY $CODE/soap_galaxies.py params/soap_L1_m9.dat > logs/soap.log 2>&1
grep 'galaxies selected' logs/soap.log
$PY $CODE/galaxy_ds_field.py params/gds_L1_m9.dat 1 > logs/gds.log 2>&1
$PY $CODE/ia_power.py boxsize=681.0 kmin=0.003 kmax=3 nbin=25 munbin=10 ng=512 sn=78 nfold=1 \
    base=L1_m9/ gal_dir=Galaxy_DS_Field/ hinfo_dir=Galaxy_DS_Field/ out_dir=PS/ > logs/ps.log 2>&1
rm -f L1_m9/Galaxy_DS_Field/*.bin
echo done
