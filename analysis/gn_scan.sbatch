#!/bin/bash
# ---------------------------------------------------------------------------
# Gauss-Newton IA bias for one .dataset: TATT (sanity check) then NLA.
#
# Settings come from the submit line (environment variables):
#   NAME   .dataset name without extension (required), e.g. cs_kuv0p4
#   PROBE  xi | 2x2pt | 3x2pt (gn_bias --probe)              default xi
#   TAG    label for the output JSON                         default fixed
#   EXTRA  extra gn_bias options, e.g. the external priors   default none
#   MODELS models to run                                     default "tatt nla"
#   DATA   data folder  default ${ROOTDIR}/projects/roman_real/data
#
#   NAME=cs_kuv0p4 sbatch gn_scan.sbatch
#   NAME=ggl_kuv0p3 PROBE=2x2pt sbatch gn_scan.sbatch
#   NAME=3x2_final PROBE=3x2pt TAG=priors MODELS=nla \
#     EXTRA="--free-cosmo --prior-ns 0.0042 --prior-omegab 0.0012 --prior-H0 5" \
#     sbatch gn_scan.sbatch
#
# Output: gn_<model>_<probe>_<NAME>_<TAG>.json in the submit folder (Cocoa/).
# ---------------------------------------------------------------------------
#SBATCH --job-name=IA_gn
#SBATCH --output=IA_gn-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

PROBE=${PROBE:-xi}
TAG=${TAG:-fixed}
MODELS=${MODELS:-"tatt nla"}

echo Running on host `hostname`
echo Time is `date`
echo Slurm job ID is $SLURM_JOBID, tasks $SLURM_NTASKS, cpus/task $SLURM_CPUS_PER_TASK
echo "NAME=${NAME} PROBE=${PROBE} TAG=${TAG} MODELS=${MODELS} EXTRA=${EXTRA}"

cd $SLURM_SUBMIT_DIR

# Load cocoa in the shell you submit from (conda activate cocoa; source
# start_cocoa.sh): sbatch passes that environment to the job. Do NOT run
# stop_cocoa.sh / start_cocoa.sh here: several jobs starting together race
# on cocoa's symlinks ("SYMLINK CREATION FAILED") and break each other.
if [ -z "${ROOTDIR}" ]; then
  echo "ROOTDIR is empty: load cocoa (conda activate cocoa; source start_cocoa.sh)"
  echo "in the shell you submit from, then resubmit"; exit 1
fi
echo "ROOTDIR: ${ROOTDIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$(command -v python)
GN=${SLURM_SUBMIT_DIR}/gn_bias.py
DATA=${DATA:-${ROOTDIR}/projects/roman_real/data}

echo "python:  ${PY:-NOT FOUND}"
echo "script:  ${GN}"
echo "dataset: ${DATA}/${NAME}.dataset"
[ -z "${NAME}" ] && { echo "set NAME=<dataset name without .dataset>"; exit 1; }
[ -z "${PY}" ] && { echo "python not on PATH"; exit 1; }
[ -f "${GN}" ] || { echo "gn_bias.py not in ${SLURM_SUBMIT_DIR}"; exit 1; }
[ -f "${DATA}/${NAME}.dataset" ] || { echo "missing ${DATA}/${NAME}.dataset"; exit 1; }

for MODEL in ${MODELS}; do
  echo "=== ${MODEL}, ${PROBE}, ${NAME}, ${TAG}, `date`"
  mpirun -n ${SLURM_NTASKS} --oversubscribe \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x ROOTDIR -x OMP_NUM_THREADS \
    --mca pml ob1 --mca btl vader,tcp,self --map-by slot --bind-to none \
    ${PY} ${GN} --path ${DATA} --dataset ${NAME}.dataset \
      --model ${MODEL} --probe ${PROBE} --halofit mead2020 ${EXTRA} \
      --out ${SLURM_SUBMIT_DIR}/gn_${MODEL}_${PROBE}_${NAME}_${TAG}.json
done

echo Finished at `date`
