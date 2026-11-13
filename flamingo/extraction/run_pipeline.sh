#!/bin/bash
# ---------------------------------------------------------------------------
# FLAMINGO IA power spectra from the SOAP catalogue (no particle data).
#
#   1. soap_galaxies.py    params/soap_galaxies_<sn>.dat      (once per snapshot)
#   2. galaxy_ds_field.py  params/galaxy_ds_<sn>.dat nfold
#   3. ia_power.py         params/ia_power.cfg sn=<sn> nfold=<nfold>
#
# Each stage-2 job holds one ng^3 float32 grid (~4.3 GB at ng=1024); set
# MAX_JOBS to fit memory. Logs go to logs/.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

SNAPSHOTS=(078)                 # zero-padded, as in the parameter-file names
FOLDS=(1 2 4 8 16 32)
MAX_JOBS=2
LOGDIR=logs
mkdir -p "$LOGDIR"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done; }

for sn in "${SNAPSHOTS[@]}"; do
  snum=$((10#$sn))
  echo "== snapshot $sn: galaxy catalogue from SOAP"
  python3 soap_galaxies.py "params/soap_galaxies_${sn}.dat" > "$LOGDIR/soap_${sn}.log" 2>&1

  for nf in "${FOLDS[@]}"; do
    throttle
    (
      python3 galaxy_ds_field.py "params/galaxy_ds_${sn}.dat" "$nf" \
        > "$LOGDIR/gds_${sn}_nfold${nf}.log" 2>&1
      python3 ia_power.py params/ia_power.cfg sn="$snum" nfold="$nf" \
        > "$LOGDIR/ps_${sn}_nfold${nf}.log" 2>&1
      echo "   done sn=$sn nfold=$nf"
    ) &
  done
done
wait
echo "all runs finished"
