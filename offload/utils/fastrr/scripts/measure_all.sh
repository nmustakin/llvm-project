#!/bin/bash
# Quiet-machine measurements (host side, one lane at a time, GPU 1): space and load cost of a
# tuner replay per kernel for both stages, then the batching cost (paper section 5.1).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
J=$JOB
date
for st in stage1 stage2; do
  $J space-$st 1 $st space_measure.py --stage $st --out $X/measure/space-$st.csv > $X/measure/space-$st.log 2>&1
  $J cost-$st 1 $st batch_cost.py --stage $st --kernels lulesh:649,lulesh:844,minife:290 --out $X/measure/batch-cost-$st.csv > $X/measure/batch-cost-$st.log 2>&1
done
date; echo 0 > $X/exit/measure-all.txt
