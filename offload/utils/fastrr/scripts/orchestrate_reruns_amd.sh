#!/bin/bash
# The MI50 re-runs after the references (orchestrate_refs_amd.sh), on GPUs 0-7:
#   phase A  GPUs 0-4: the conventional strategies (BO, TPE, random, 60 full evaluations, shared
#            cache) on the 10 LULESH kernels;
#            GPUs 5-7: the input study (retune at the second input with FastRR, then validate the
#            first input's configurations unchanged and rescaled at the second input)
#            (miniFE at 150^3 finished on 2026-11-05 before GPU 6 hung, and is not rerun)
#   phase B  FastRR on 1, 2, 4, and 8 GPUs on the 5 LULESH/miniFE kernels of the batch table,
#            all validated together in validation-mgpu.csv
# There is no MOTPE: upstream replays measure no energy.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
J=$JOB
L=$X/reruns/logs; mkdir -p $L
wait_for() { for f in "$@"; do until [ -f $X/exit/$f.txt ]; do sleep 60; done; done; }
tune() {  # tag GPUs out app lines pool methods [extra args]
  local tag=$1 g=$2 out=$3 app=$4 lines=$5 pool=$6 m=$7; shift 7
  $J $tag $g stage2 run_tuner.py --stage stage2 --app $app --lines $lines --pool $pool --methods $m \
     --out $X/reruns/$out "$@" > $L/$tag.log 2>&1
}
inputs() {  # GPU a:b:lines ...
  local g=$1; shift
  for ab in "$@"; do
    IFS=: read a b lines <<< "$ab"
    tune inputs-retune-$b $g inputs/runs $b $lines loao final-wave --validate-rounds 0
    $J inputs-validate-$b $g stage2 validate_inputs.py --app-a $a --app-b $b --lines $lines \
       > $L/inputs-validate-$b.log 2>&1
  done
}
CONV=bo-60,tpe-60,random-60
date
wait_for refs
g=0
for pair in 649,844 1201,333 584,1029 1080,1721 1913,2103; do
  ( tune search-$g $g search lulesh $pair loko $CONV --cache ) &
  g=$((g + 1))
done
( inputs 5 s3d:s3d-s3:159,165,171,183,220,226 ) &
( inputs 6 minisweep:minisweep-c884:282,315,348,383 su3:su3-l24:41 ) &
( inputs 7 simplemoc:simplemoc-s1500k:379 miniWeather:miniWeather-1600:760 ) &
wait; date
# Phase B: the whole machine. Runs of the same application share a work directory (compiled
# binaries), so two runs of one application never run at the same time.
tune mgpu1-lulesh 0 mgpu lulesh 649,844,1201 loko final-wave --validate-rounds 0 &
tune mgpu2-minife 2,3 mgpu minife 290,521 loko final-wave-2gpu --devices 2 --validate-rounds 0 &
wait
tune mgpu1-minife 0 mgpu minife 290,521 loko final-wave --validate-rounds 0 &
tune mgpu2-lulesh 2,3 mgpu lulesh 649,844,1201 loko final-wave-2gpu --devices 2 --validate-rounds 0 &
wait
tune mgpu4-lulesh 0,1,2,3 mgpu lulesh 649,844,1201 loko final-wave-4gpu --devices 4 --validate-rounds 0 &
tune mgpu4-minife 4,5,6,7 mgpu minife 290,521 loko final-wave-4gpu --devices 4 --validate-rounds 0 &
wait
tune mgpu8-lulesh 0,1,2,3,4,5,6,7 mgpu lulesh 649,844,1201 loko final-wave-8gpu --devices 8 --validate-rounds 0
tune mgpu8-minife 0,1,2,3,4,5,6,7 mgpu minife 290,521 loko final-wave-8gpu --devices 8 --validate-rounds 0
M=final-wave,final-wave-2gpu,final-wave-4gpu,final-wave-8gpu
tune mgpu-validate-lulesh 0 mgpu lulesh 649,844,1201 loko $M --validation-name validation-mgpu.csv &
tune mgpu-validate-minife 1 mgpu minife 290,521 loko $M --validation-name validation-mgpu.csv &
wait; date; echo 0 > $X/exit/reruns-amd.txt
