#!/bin/bash
# usage: orchestrate.sh <stage1|stage2> <methods>   (on the host, inside a tmux session)
# The paper's online T4 experiments on one stage's LLVM build, one lane per GPU (GPU 0 is
# avoided; it overheated in earlier runs):
#   5 LULESH/miniFE kernels, rankers trained without the target kernel (LOKO), and
#   15 HeCBench kernels with headroom, rankers trained without the target application (LOAO);
# three seeds, 300 s cap, no cache, every shipped configuration validated in 5 rounds.
st=$1; M=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
J=$JOB
mkdir -p $X/$st/logs
run() {  # tag GPU app lines pool
  $J $st-$1 $2 $st run_tuner.py --stage $st --app $3 --lines $4 --pool $5 --methods $M \
     --out $X/$st/runs > $X/$st/logs/$1.log 2>&1
}
date
( run lulesh 1 lulesh 649,844,1201 loko ) &
( run minife 2 minife 290,521 loko; run xsbench 2 xsbench 180 loao; run amgmk 2 amgmk 258 loao ) &
( run s3dA 3 s3d 159,165,171 loao ) &
( run s3dB 4 s3d 183,220,226 loao ) &
( run minisweep 5 minisweep 282,315,348,383 loao ) &
( run miniweather 6 miniWeather 760 loao; run simplemoc 6 simplemoc 379 loao; run su3 6 su3 41 loao ) &
wait; date; echo 0 > $X/exit/orchestrate-$st.txt
