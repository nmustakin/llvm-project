#!/bin/bash
# Stage 2 (space-saving changes, replays with --relocate): FastRR on the same kernels, seeds,
# budget, and lane layout as stage 1 (orchestrate.sh with rerun_stage1.sh), validated in 5 rounds.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
st=stage2; M=${1:-final-wave}
J=$JOB
mkdir -p $X/$st/logs
run() { $J $st-$1 $2 $st run_tuner.py --stage $st --app $3 --lines $4 --pool $5 --methods $M \
          --out $X/$st/runs > $X/$st/logs/$1.log 2>&1; }
date
( run lulesh 1 lulesh 649,844,1201 loko ) &
( run xsbench 2 xsbench 180 loao; run amgmk 2 amgmk 258 loao ) &
( run s3dA 3 s3d 159,165,171 loao ) &
( run s3dB 4 s3d 183,220,226 loao ) &
( run minisweep 5 minisweep 282,315,348,383 loao ) &
( run simplemoc 6 simplemoc 379 loao; run su3 6 su3 41 loao ) &
( run minife 7 minife 290,521 loko; run miniweather 7 miniWeather 760 loao ) &
wait; date; echo 0 > $X/exit/orchestrate-$st.txt
