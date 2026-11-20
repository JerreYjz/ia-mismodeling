#!/bin/bash
# P_gm for FLAMINGO L1_m8: stream every dark-matter particle of each snapshot
# onto count grids (about 7.5 h per snapshot), then P_mm, P_gm, P_gg
# (matter_cross.py). High redshift first: the lens bins at z >= 1.5 are where
# linear bias fails. A run resumes from its 30-min checkpoint if restarted.
#
# Battery guard: below LOW percent the streamer is paused (SIGSTOP) and above
# HIGH it resumes; a read that times out while paused is retried. caffeinate
# keeps the Mac awake for the whole queue.
#
#   nohup bash run_matter_cross.sh > logs/matter_cross.log 2>&1 &
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/miniconda/etc/profile.d/conda.sh; conda activate cocoa
cd ~/cocoa/Cocoa && source start_cocoa.sh >/dev/null 2>&1
cd "$HERE"
export OMP_NUM_THREADS=4
SNAPSHOTS=(18 38 48)           # z = 3, 2, 1.5 (z = 1 dropped: the bias is linear to 4% there)
LOW=15
HIGH=25
mkdir -p logs
ts() { date "+%F %T"; }

guard() {
  local paused=0
  while true; do
    local pct
    pct=$(pmset -g batt | grep -o '[0-9]*%' | tr -d '%')
    if [ "$paused" = 0 ] && [ "$pct" -lt "$LOW" ]; then
      pkill -STOP -f "matter_cross.py" && paused=1 && echo "$(ts) battery $pct%: paused"
    elif [ "$paused" = 1 ] && [ "$pct" -ge "$HIGH" ]; then
      pkill -CONT -f "matter_cross.py" && paused=0 && echo "$(ts) battery $pct%: resumed"
    fi
    sleep 60
  done
}
guard &
GPID=$!
trap 'kill $GPID 2>/dev/null; pkill -CONT -f matter_cross.py' EXIT

for sn in "${SNAPSHOTS[@]}"; do
  if [ -f "$HOME/ia_power_flamingo/run_L1_m8_particles/results/PS_matter/matter_cross_sn${sn}.json" ]; then
    echo "$(ts) sn $sn already done"; continue
  fi
  echo "$(ts) == sn $sn: streaming"
  # a failed stream is restarted from its checkpoint (every 30 min), never skipped
  until caffeinate -i python matter_cross.py stream "$sn" >> "logs/matter_stream_${sn}.log" 2>&1; do
    echo "$(ts) sn $sn: stream FAILED, restarting from the checkpoint in 10 min"; sleep 600
  done
  echo "$(ts) sn $sn: $(tail -1 logs/matter_stream_${sn}.log)"
  caffeinate -i python matter_cross.py power "$sn" >> "logs/matter_power_${sn}.log" 2>&1 \
    && echo "$(ts) sn $sn: spectra done" || echo "$(ts) sn $sn: power FAILED"
done
echo "$(ts) all snapshots finished"
