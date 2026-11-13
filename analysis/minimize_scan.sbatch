#!/bin/bash
# ---------------------------------------------------------------------------
# Cobaya minimize scan over k_UV scale cuts (IA mismodeling paper).
#
# One array task = one yaml = one MAP fit. The yamls are expected at
#   projects/roman_real/ia_paper/${MODEL}_on_TATT_${PROBE}_kuv${KUV}.yaml
# for the k_UV values in KUVS below (array index -> KUVS[index]).
#
# Submit from Cocoa/ (conda and .local envs loaded), e.g.
#   sbatch --array=0-6 minimize_scan.sbatch                 # NLA, cosmic shear
#   MODEL=TATT sbatch --array=0-6 minimize_scan.sbatch      # matched-model reference
#   PROBE=3x2 sbatch --array=0-6 minimize_scan.sbatch
#   sbatch --array=4 minimize_scan.sbatch                   # only kuv0p5
# ---------------------------------------------------------------------------
#SBATCH --job-name=IA_min
#SBATCH --output=IA_min-%A-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # = best_of: one MPI rank per starting point
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8          # OpenMP threads per rank for cosmolike/CAMB
#SBATCH --time=12:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

MODEL=${MODEL:-NLA}                # NLA (analysis) or TATT (matched reference)
PROBE=${PROBE:-cs}                 # cs, 2x2, 3x2: must match the yaml names
KUVS=(0p1 0p15 0p2 0p25 0p3 0p4 0p5)

TASK=${SLURM_ARRAY_TASK_ID:-0}
KUV=${KUVS[$TASK]}
YAML=./projects/roman_real/ia_paper/${MODEL}_on_TATT_${PROBE}_kuv${KUV}.yaml

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID, array task $TASK
echo Number of tasks is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK
echo yaml is $YAML

cd $SLURM_SUBMIT_DIR

if [ -z "$KUV" ]; then
  echo "array index $TASK is outside KUVS (${#KUVS[@]} entries)"; exit 1
fi
if [ ! -f "$YAML" ]; then
  echo "missing $YAML"; exit 1
fi

# conda and (.local) envs should be already loaded prior to submitting a slurm script
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 10 + TASK*20 )) # stagger start_cocoa.sh across array tasks: it
                            # calls stop_cocoa.sh, which deletes the links
                            # another task may be reading
  source ~/.bashrc
  conda activate cocoa
  source start_cocoa.sh
fi

# cosmolike and CAMB are OpenMP codes: give each MPI rank its cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -n ${SLURM_NTASKS} --oversubscribe \
  --mca pml ob1 --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --rank-by slot \
  --map-by slot:pe=${OMP_NUM_THREADS} --mca mpi_yield_when_idle 1 \
  cobaya-run ${YAML} -f

echo Finished at `date`
