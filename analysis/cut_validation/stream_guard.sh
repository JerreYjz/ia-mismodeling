#!/bin/bash
# Keep the FLAMINGO dark-matter stream (flamingo/extraction/matter_cross.py) at
# its speed while low-priority fits run: every 3 min the stream's rate is
# measured from its log; two slow readings in a row (below LOW rows/s) pause
# the fits and bias splits (SIGSTOP), two fast ones (above HIGH) resume them.
# Exits when the stream queue has finished.
LOW=1600000; HIGH=1900000
LOGS=~/cocoa/ia_mismodeling/flamingo/extraction/logs
rows() { grep -h 'rows ' $(ls -t $LOGS/matter_stream_*.log | head -1) | tail -1 | sed 's/.*rows \([0-9,]*\)\/.*/\1/' | tr -d ,; }
slow=0; fast=0; paused=0; prev=$(rows); tp=$(date +%s)
while pgrep -f run_matter_cross.sh > /dev/null; do
  sleep 180
  now=$(rows); t=$(date +%s)
  [ -z "$now" ] || [ -z "$prev" ] || [ "$now" -lt "$prev" ] && { prev=$now; tp=$t; continue; }
  rate=$(( (now - prev)/(t - tp) )); prev=$now; tp=$t
  if [ "$rate" -lt "$LOW" ]; then slow=$((slow+1)); fast=0; else slow=0; fi
  if [ "$rate" -gt "$HIGH" ]; then fast=$((fast+1)); else fast=0; fi
  if [ "$paused" = 0 ] && [ "$slow" -ge 2 ]; then
    pkill -STOP -f fisher_minimize.py; pkill -STOP -f bias_pieces.py; paused=1
    echo "$(date) stream $rate rows/s: fits paused"
  elif [ "$paused" = 1 ] && [ "$fast" -ge 2 ]; then
    pkill -CONT -f fisher_minimize.py; pkill -CONT -f bias_pieces.py; paused=0
    echo "$(date) stream $rate rows/s: fits resumed"
  fi
done
pkill -CONT -f fisher_minimize.py; pkill -CONT -f bias_pieces.py
echo "$(date) stream queue finished; guard exits"
