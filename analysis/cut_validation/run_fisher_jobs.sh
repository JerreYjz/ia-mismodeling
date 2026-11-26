#!/bin/bash
# fisher_minimize.py on every line of a job file, N at a time, in the cocoa env (2 OpenMP threads each).
#   bash run_fisher_jobs.sh jobs.txt [N] > log
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/miniconda/etc/profile.d/conda.sh
conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2
N=${2:-4}
grep -v '^\s*$' "$1" | xargs -S 4096 -P "$N" -I{} sh -c "python '$HERE/fisher_minimize.py' {} 2>&1 | grep -E 'pts, chi2 ref|RESTART|Error|Traceback' | sed 's|^|[{}] |'"
echo "ALL DONE"
