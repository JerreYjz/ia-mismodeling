#!/bin/bash
# Cut validation queue on the local Mac (8 cores): one chain at a time with
# 4 MPI processes x 2 OpenMP threads, each followed by the minimizer.
#
# The minimizer (cobaya-run --minimize on the chain's own yaml) starts from the
# chain's maximum a posteriori and rescales the parameters with the chain's
# covariance; 4 MPI processes = 4 starts, the best kept. It maximizes the
# posterior (the Gaussian Delta z and m priors included, flat cosmology), so
# its shift from the truth carries no projection (volume) effect.
# Never pass -f to --minimize: it would delete the chain it starts from.
#
#   bash run_queue.sh [--wait PID] chain1 chain2 ...     (from anywhere)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = "--wait" ]; then
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
  shift 2
fi
source ~/miniconda/etc/profile.d/conda.sh
conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2
REL=../ia_mismodeling/analysis/cut_validation
CH=./projects/roman_real/chains/cutval
for name in "$@"; do
  if [ ! -s "$CH/$name.progress" ] || ! grep -q "converged" "$HERE/$name.log" 2>/dev/null; then
    echo "$(date) chain $name"
    mpirun -n 4 --oversubscribe cobaya-run -f "$REL/$name.yaml" > "$HERE/$name.log" 2>&1
  fi
  echo "$(date) minimize $name"
  mpirun -n 4 --oversubscribe cobaya-run --minimize "$REL/$name.yaml" > "$HERE/${name}_min.log" 2>&1
  echo "$(date) done $name: $(ls $CH/$name.minimum.txt 2>/dev/null)"
done
