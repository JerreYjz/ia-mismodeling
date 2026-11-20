#!/bin/bash
# After the first redshift-evolution queue: the 3x2pt cuts 0.5 and 0.6 for the
# freer models and the missing NLA/TATT baselines, then their bias splits.
cd "$(dirname "$0")"
until grep -q "all done" q_zevo.log; do sleep 60; done
NPAR=2 NICE=19 bash run_hm_scan.sh jobs_zevo_ext.txt > q_zevo_ext.log 2>&1
