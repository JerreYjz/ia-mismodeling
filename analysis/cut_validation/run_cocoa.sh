#!/bin/bash
# Run one python command in the cocoa env (as run_fisher_jobs.sh):  bash run_cocoa.sh script.py args...
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/miniconda/etc/profile.d/conda.sh
conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
s="$1"; shift
python "$HERE/$s" "$@"
