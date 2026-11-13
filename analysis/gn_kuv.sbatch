#!/bin/bash
# ---------------------------------------------------------------------------
# Gauss-Newton bias at one k_UV mask: TATT (sanity check) then NLA,
# fixed ns, H0, Omega_b, cosmic shear.
#
# Needs ${DATA}/tatt_kuv${KUV}.dataset pointing at the k_UV mask.
# Submit from Cocoa/ (where gn_bias.py and fisher_sigma.py live):
#   sbatch gn_kuv.sbatch                    # KUV=0p4 (default)
#   KUV=0p45 sbatch gn_kuv.sbatch
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

KUV=${KUV:-0p4}
PROBE=${PROBE:-xi}

echo Running on host `hostname`
echo Time is `date`
echo k_UV = $KUV, probe = $PROBE

cd $SLURM_SUBMIT_DIR

# conda and (.local) envs should be already loaded prior to submitting
if [ -z "${ROOTDIR}" ]; then
  source ~/.bashrc
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
DATA=${ROOTDIR}/external_modules/data/roman_real
DATASET=tatt_kuv${KUV}.dataset
PY=$(command -v python)

if [ ! -f "${DATA}/${DATASET}" ]; then
  echo "missing ${DATA}/${DATASET}"; exit 1
fi
if [ -z "${PY}" ]; then
  echo "python not found: is the cocoa env loaded?"; exit 1
fi

for MODEL in tatt nla; do
  echo "=== ${MODEL} on TATT, ${DATASET}, started `date`"
  mpirun -n ${SLURM_NTASKS} --oversubscribe \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x ROOTDIR -x OMP_NUM_THREADS \
    --mca pml ob1 --mca btl vader,tcp,self --map-by slot --bind-to none \
    ${PY} gn_bias.py --path ${DATA} --dataset ${DATASET} \
      --model ${MODEL} --probe ${PROBE} --halofit mead2020 \
      --out gn_${MODEL}_${PROBE}_kuv${KUV}_fixed.json
done

echo Finished at `date`
