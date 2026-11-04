#!/bin/bash
# References on the final system (stage 2): FastRR and Batch TPE, three seeds, on all 43 kept
# kernels (FastRR on 20 of them was run by orchestrate_stage2.sh and is reused). Validated
# together in validation-ref.csv. GPUs 1-4.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
st=stage2; M=final-wave,tpe-batch-nocls
J=$JOB
run() { $J ref-$1 $2 $st run_tuner.py --stage $st --app $3 --lines $4 --pool $5 --methods $M \
          --out $X/$st/runs --validation-name validation-ref.csv > $X/reruns/logs/ref-$1.log 2>&1; }
date
( run lulesh 1 lulesh 649,844,1201,333,584,1029,1080,1721,1913,2103 loko ) &
( run minife 2 minife 290,521,157,169,205,210,254 loko; run xsbench 2 xsbench 180 loao
  run amgmk 2 amgmk 258 loao; run haccmk 2 haccmk 35 loao ) &
( run s3dA 3 s3d 159,165,171,183,177,189,195 loao; run minisweep 3 minisweep 282,315,348,383 loao ) &
( run s3dB 4 s3d 220,226,200,205,210,215,232 loao; run miniweather 4 miniWeather 760,253,308 loao
  run simplemoc 4 simplemoc 379 loao; run su3 4 su3 41 loao ) &
wait; date; echo 0 > $X/exit/refs.txt
