#!/bin/bash
# References on the MI50 (FASTRR_TARGET=amdgpu, stage 2 = this branch): FastRR and Batch TPE,
# three seeds, on the 46 kept kernels (the 43 of the T4 and sw4ck l23, l24, l25), with the
# ranking model trained on the MI50 pool (make_pool.py from the MI50 grids), leaving out the
# kernel (LULESH, miniFE) or the application (HeCBench). Validated in validation-ref.csv.
# Waits for the sw4ck grids, rebuilds the pool with them, then runs one lane per GPU (0-7).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
st=stage2; M=final-wave,tpe-batch-nocls
J=$JOB
L=$X/reruns/logs; mkdir -p $L
wait_for() { for f in "$@"; do until [ -f $X/exit/$f.txt ]; do sleep 60; done; done; }
run() { $J ref-$1 $2 $st run_tuner.py --stage $st --app $3 --lines $4 --pool $5 --methods $M \
          --out $X/$st/runs --validation-name validation-ref.csv > $L/ref-$1.log 2>&1; }
date
wait_for grid-sw4ck-23 grid-sw4ck-24 grid-sw4ck-25
PYTHONPATH=$FASTRR_PYLIB $FASTRR_PYTHON $FASTRR/scripts/make_pool.py --grids $X/grids \
  --out $FASTRR_POOL > $L/make_pool.log 2>&1 || { echo 1 > $X/exit/refs.txt; exit 1; }
tail -n 1 $L/make_pool.log
( run luleshA 0 lulesh 649,844,1201,333,584 loko ) &
( run luleshB 1 lulesh 1029,1080,1721,1913,2103 loko ) &
( run minife 2 minife 290,521,157,169,205,210,254 loko ) &
( run s3dA 3 s3d 159,165,171,183,177,189,195 loao ) &
( run s3dB 4 s3d 220,226,200,205,210,215,232 loao ) &
( run minisweep 5 minisweep 282,315,348,383 loao; run xsbench 5 xsbench 180 loao
  run amgmk 5 amgmk 258 loao ) &
( run miniweather 6 miniWeather 760,253,308 loao; run simplemoc 6 simplemoc 379 loao
  run su3 6 su3 41 loao ) &
( run sw4ck 7 sw4ck 23,24,25 loao; run haccmk 7 haccmk 35 loao ) &
wait; date; echo 0 > $X/exit/refs.txt
