#!/bin/bash
# Halo-model scale-cut scan on the local Mac: every job one fisher_minimize.py
# run (model, K0, truth), 4 at a time with 2 OpenMP threads each (NPAR=n
# for n at a time; NICE=19 runs them at the lowest priority, behind another job).
#
# Battery guard: on a weak charger the battery can drain under full load. A
# background loop pauses (SIGSTOP) every running fit when the battery falls
# below LOW percent and resumes them (SIGCONT) above HIGH; nothing is lost,
# the fits only wait. Its log is hm_guard.log.
#
#   bash run_hm_scan.sh <jobs file>      (one "--probe P --cmb MODEL K0 TRUTH" per line)
#   NPAR=2 NICE=19 bash run_hm_scan.sh <jobs file>
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
LOW=15
HIGH=25
source ~/miniconda/etc/profile.d/conda.sh
conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2
set -u

guard() {
  local paused=0
  while true; do
    local pct
    pct=$(pmset -g batt | grep -o '[0-9]*%' | tr -d '%')
    if [ "$paused" = 0 ] && [ "$pct" -lt "$LOW" ]; then
      pkill -STOP -f fisher_minimize.py && paused=1
      echo "$(date) battery $pct%: fits paused" >> "$HERE/hm_guard.log"
    elif [ "$paused" = 1 ] && [ "$pct" -ge "$HIGH" ]; then
      pkill -CONT -f fisher_minimize.py && paused=0
      echo "$(date) battery $pct%: fits resumed" >> "$HERE/hm_guard.log"
    fi
    sleep 60
  done
}
guard &
GPID=$!
trap 'kill $GPID 2>/dev/null; pkill -CONT -f fisher_minimize.py' EXIT

mkdir -p "$HERE/logs_hm"
export HERE
grep -v '^#' "$JOBS" | grep -v '^$' | xargs -P "${NPAR:-4}" -L 1 bash -c '
  name=$(echo "$*" | tr " " "_" | tr -d "-")
  nice -n "${NICE:-0}" python ../ia_mismodeling/analysis/cut_validation/${SCRIPT:-fisher_minimize.py} "$@" > "$HERE/logs_hm/$name.log" 2>&1
  echo "$(date) done $*: $(tail -1 "$HERE/logs_hm/$name.log" | cut -c1-200)"' worker
echo "$(date) all done"
