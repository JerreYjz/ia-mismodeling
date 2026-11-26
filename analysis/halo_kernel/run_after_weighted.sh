#!/bin/bash
# After hk_weighted.py has written every snapshot: model GOF, Roman templates, Fisher runs.
cd "$(dirname "$0")"
PY=~/cocoa/Cocoa/.local/bin/python
until [ $(ls kernels/weighted_*.npz 2>/dev/null | wc -l) -ge 13 ] && ! pgrep -f hk_weighted.py >/dev/null; do sleep 20; done
echo "weighted fields done $(date)"
$PY hk_model.py 2>&1 | grep -v Warn > model/hk_model_run.log
echo "hk_model done $(date)"
$PY hk_templates.py w-ls hmk0 2>&1 | grep -v Warn
$PY hk_templates.py w-ls-grad@1.26 hmk 2>&1 | grep -v Warn
echo "templates done $(date)"
bash ../cut_validation/run_fisher_jobs.sh $PWD/../cut_validation/jobs_hmk.txt 4 > ../cut_validation/fisher_hmk.log 2>&1
echo "fisher done $(date)"
