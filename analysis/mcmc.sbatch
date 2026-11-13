#!/bin/bash
# ---------------------------------------------------------------------------
# Cobaya MCMC on one node: 8 MPI chains x 10 OpenMP threads = 80 cores.
#
#   YAML=./projects/roman_real/MCMC_NLA_cs_final_d3a_emul.yaml sbatch mcmc.sbatch
#
# The job reloads cocoa itself (stop_cocoa / conda activate / start_cocoa),
# so it does not depend on the submit shell. Jobs that start at the same
# moment can race on cocoa's symlinks ("SYMLINK CREATION FAILED"): stagger
# submissions, e.g. sbatch --begin=now+60 for the second job.
# cobaya-run -r resumes an existing chain, so resubmitting after the time
# limit continues where it stopped.
# ---------------------------------------------------------------------------
#SBATCH --job-name=IA_mcmc
#SBATCH --output=IA_mcmc-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
## emulbaosn runs with device: "cuda". On a GPU node uncomment the next line
## (and use your GPU partition); on a CPU node set device: "cpu" in the yaml.
##SBATCH --gres=gpu:1

echo Running on host `hostname`, job $SLURM_JOBID, `date`
echo "YAML=${YAML}  tasks ${SLURM_NTASKS} x cpus ${SLURM_CPUS_PER_TASK}"
cd $SLURM_SUBMIT_DIR

source ~/.bashrc
source stop_cocoa.sh
conda deactivate
conda activate cocoa
source start_cocoa.sh

if [ -z "${ROOTDIR}" ]; then
  echo "ROOTDIR is empty after start_cocoa.sh: check the cocoa setup"; exit 1
fi
echo "ROOTDIR: ${ROOTDIR}"
[ -f "${YAML}" ] || { echo "missing yaml: ${YAML}"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# PyTorch (and Keras on its torch backend) size their CPU thread pools from
# these; without them every rank would start one thread per core of the node
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# full path to the runner: prterun does not search PATH the way the shell
# does, so a bare "cobaya-run" can fail with "unable to find the executable".
# Fall back to "python -m cobaya run" when the entry point is missing.
CR=$(command -v cobaya-run)
if [ -n "${CR}" ]; then
  RUN="${CR}"
else
  PY=$(command -v python)
  [ -z "${PY}" ] && { echo "neither cobaya-run nor python on PATH"; exit 1; }
  RUN="${PY} -m cobaya run"
fi
echo "runner: ${RUN}"

mpirun -n ${SLURM_NTASKS} --oversubscribe \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x ROOTDIR -x OMP_NUM_THREADS \
  -x OMP_PROC_BIND -x OMP_PLACES -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS \
  --mca pml ob1 --mca btl vader,tcp,self \
  --map-by slot:PE=${SLURM_CPUS_PER_TASK} --bind-to core \
  ${RUN} "${YAML}" -r

echo Finished at `date`
