#!/bin/bash
# split every fit of a jobs file (bias_pieces.py) as it appears, until all are split
#   bash follow_list.sh <jobs file>
source ~/miniconda/etc/profile.d/conda.sh; conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2
CV=~/cocoa/ia_mismodeling/analysis/cut_validation
export JOBS="$1"
while true; do
  todo=$(python3 - <<'PY'
import json, os
CV = os.path.expanduser("~/cocoa/ia_mismodeling/analysis/cut_validation")
R = json.load(open(f"{CV}/fisher_minimize.json"))["runs"]
P = json.load(open(f"{CV}/bias_pieces.json"))["runs"]
keys = []
for line in open(os.environ["JOBS"]):
    a = line.split()
    if not a: continue
    gt = a[a.index("--gt") + 1] if "--gt" in a else "0.3"
    pr = a[a.index("--probe") + 1]
    m, k0, tr = [x for i, x in enumerate(a) if not x.startswith("--") and a[i-1] not in ("--gt", "--probe")][-3:]
    keys.append(f"{pr}+cmb|{m}|{k0}|{tr}" + (f"|gt{gt}" if gt != "0.3" and pr != "2x2" else ""))
todo = [k for k in keys if k in R and k not in P]
print("DONE" if all(k in P for k in keys) else " ".join(todo))
PY
)
  [ "$todo" = "DONE" ] && { echo "split all $(basename $JOBS)"; break; }
  for k in $todo; do python $CV/bias_pieces.py "$k" 2>&1 | tail -1 | cut -c1-200; done
  sleep 30
done
