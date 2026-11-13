#!/bin/bash
# ---------------------------------------------------------------------------
# Make one IR/UV scale-cut mask and a .dataset that uses it.
#
#   bash make_mask_dataset.sh NAME PROBE KCS KGGL KGC [BASE]
#
#   NAME   name of the mask and dataset (writes NAME.mask, NAME.dataset)
#   PROBE  cosmic_shear | galaxy_galaxy_lensing | galaxy_clustering | 2x2pt | 3x2pt
#   KCS, KGGL, KGC  cuts for xi_pm, gamma_t, w (unused probes: any value).
#          One number: k_UV [h/Mpc] at z = 0, cut at k_UV/D(z).
#          A quoted list ("0.3 0.4 0.5"): k_max [h/Mpc] at the redshifts in
#          ZNODES, interpolated in z (needs ZNODES with the same length).
#   BASE   .dataset to copy (data vector, covariance, n(z)); default
#          tatt_kuv0p35.dataset. Its data vector must come from the 3x2pt yaml
#          whenever PROBE includes gamma_t or w.
#
# Environment: ZNODES="0 0.5 1 2"  redshifts of the k lists
#              KZFILE=kmax_z.txt   columns z k_cs [k_ggl k_gc]; overrides KCS..
#              KZMODEL=dpow        fit k0 D(z)^-n to the k lists (default: interpolate)
#              KZPOWER=n           fix n (with a scalar k: cut at k/D(z)^n)
#              HALOFIT=mead2020    (default; match the truth vector)
#              DATA=...            output folder (default roman_real/data)
#
#   ZNODES="0 0.5 1 1.5 2" bash make_mask_dataset.sh cs_kz cosmic_shear \
#       "0.29 0.39 0.52 0.70 0.94" 0.2 0.2
#
# Run from Cocoa/ with the cocoa env loaded (needs make_uv_scale_cuts.py there).
# alpha = 0.05.
# ---------------------------------------------------------------------------
set -e
NAME=$1; PROBE=$2; KCS=$3; KGGL=$4; KGC=$5
BASE=${6:-tatt_kuv0p35.dataset}
if [ -z "$KGC" ] && [ -z "$KZFILE" ]; then
  sed -n '2,29p' "$0"; exit 1
fi
DATA=${DATA:-${ROOTDIR:?load cocoa first}/projects/roman_real/data}
HALOFIT=${HALOFIT:-mead2020}

if [ -n "$KZFILE" ]; then
  KARGS="--kz-file ${KZFILE}"
else
  KARGS="--kuv-cs ${KCS} --kuv-ggl ${KGGL} --kuv-gc ${KGC}"
  [ -n "$ZNODES" ] && KARGS="${KARGS} --z-nodes ${ZNODES}"
fi

[ -n "$KZMODEL" ] && KARGS="${KARGS} --kz-model ${KZMODEL}"
[ -n "$KZPOWER" ] && KARGS="${KARGS} --kz-power ${KZPOWER}"

python make_uv_scale_cuts.py --path ${DATA} --probe ${PROBE} --halofit ${HALOFIT} \
    ${KARGS} --alpha 0.05 --out ${NAME}.mask
cp ${NAME}.mask ${DATA}/
sed "s/^\(mask_file *= *\).*/\1${NAME}.mask/" ${DATA}/${BASE} > ${DATA}/${NAME}.dataset
echo "-> ${DATA}/${NAME}.dataset (mask ${NAME}.mask, from ${BASE})"
grep -E '^(data_file|mask_file)' ${DATA}/${NAME}.dataset
