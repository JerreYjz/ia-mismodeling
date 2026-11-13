#!/bin/bash
# FLAMINGO L1_m8 IA spectra with the TNG shape definition (unreduced inertia tensor of all bound
# stars, Galaxy_Shape_Tensor/eigen_iteration.c), z = 0-3 (78-74 done 2026-10-22, 72-66 on 2026-10-25, 63 and 58 on 2026-10-26; next: 48, 38, 18, 14 [z=4, z=3 mass cut]).
# Per snapshot: stream star particles -> catalogue (resumable), grids + spectra per nfold (grids
# deleted after use), then stitch plot, summary plots and cosmolike export.
set -euo pipefail
cd "$(dirname "$0")"
CODE=..
SNAPSHOTS=(048 038 018 014)
FOLDS=(1 4 16)
GDS=results/Galaxy_DS_Field
ts() { date "+%F %T"; }

for sn in "${SNAPSHOTS[@]}"; do
  snum=$((10#$sn))
  echo "$(ts) == snapshot $sn"
  if [ ! -f "results/Galaxies/galaxies_${snum}.hdf5" ]; then
    python "$CODE/particle_shapes.py" "params/particle_shapes_${sn}.dat" >> "logs/shapes_${sn}.log" 2>&1
  fi
  echo "$(ts)    catalogue: $(grep -E 'galaxies with shapes' logs/shapes_${sn}.log | tail -1); $(grep -E 'primary shapes' logs/shapes_${sn}.log | tail -1)"
  for nf in "${FOLDS[@]}"; do
    if [ -f "results/PS/Phh_sn${snum}_nfold${nf}.dat" ]; then echo "$(ts)    nfold=$nf already done"; continue; fi
    python "$CODE/galaxy_ds_field.py" "params/galaxy_ds_${sn}.dat" "$nf" > "logs/gds_${sn}_nfold${nf}.log" 2>&1
    python "$CODE/ia_power.py" params/ia_power.cfg sn="$snum" nfold="$nf" > "logs/ps_${sn}_nfold${nf}.log" 2>&1
    rm -f "$GDS"/galdens_*_sn${snum}_nfold${nf}.bin "$GDS"/e1z_*_sn${snum}_nfold${nf}.bin "$GDS"/e2z_*_sn${snum}_nfold${nf}.bin
    echo "$(ts)    nfold=$nf spectra done, grids deleted"
  done
  # plots + export after every snapshot (failures here must not stop the run)
  echo "$(ts)    $(python plot_stitch_flamingo.py "$snum" all 2>&1 | grep -E '^PLOT|Error' | tail -1 || echo 'PLOT failed')"
  python plot_summary.py > "logs/summary_after_${sn}.log" 2>&1 && echo "$(ts)    SUMMARY plots updated: $(grep -c a_nla logs/summary_after_${sn}.log) snapshot(s)" || echo "$(ts)    SUMMARY plot failed (see logs/summary_after_${sn}.log)"
  python export_cosmolike.py all > "logs/export_after_${sn}.log" 2>&1 && echo "$(ts)    cosmolike export updated" || echo "$(ts)    cosmolike export failed (see logs/export_after_${sn}.log)"
  echo "$(ts) == snapshot $sn finished"
done
echo "$(ts) all snapshots finished"
