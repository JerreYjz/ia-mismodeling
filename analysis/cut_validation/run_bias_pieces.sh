#!/bin/bash
# bias_pieces.py on the given keys, in the cocoa environment (2 OpenMP threads).
#   bash run_bias_pieces.sh <key> [<key> ...]
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/miniconda/etc/profile.d/conda.sh
conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
python "$HERE/bias_pieces.py" "$@" 2>&1 | grep -E "^\S+\+cmb\||Error|Traceback|line [0-9]+" 
