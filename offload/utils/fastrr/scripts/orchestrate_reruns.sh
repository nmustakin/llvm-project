#!/bin/bash
# The re-runs for the upstream version of the paper, on the final system (stage 2), one lane per
# GPU, each waiting for its prerequisites (exit files of the earlier lanes):
#   GPUs 1-3  after the references: the online incremental store (3 random application orders),
#             then the FastRR ablations on the 15 HeCBench kernels (without wave-aligned team
#             counts, without refinement)
#   GPU 4     after the references: the input study (record the second inputs, retune, validate)
#   GPUs 5-6  after the grids: the conventional strategies (BO, TPE, random, 60 full
#             evaluations) on the 10 LULESH kernels
#   GPU 7     after the cold-load measurement: MF screening and TPE with 60 full evaluations on
#             the 5 LULESH/miniFE kernels (the batch table)
#   finally   FastRR on four GPUs (1-4) and with promotions instead of batch confirmation (5)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
A="$FASTRR_RUN bash"
J=$JOB
L=$X/reruns/logs
wait_for() { for f in "$@"; do until [ -f $X/exit/$f.txt ]; do sleep 60; done; done; }
tune() {  # tag GPU out app lines pool methods [extra args]
  local tag=$1 g=$2 out=$3 app=$4 lines=$5 pool=$6 m=$7; shift 7
  $J $tag $g stage2 run_tuner.py --stage stage2 --app $app --lines $lines --pool $pool --methods $m \
     --out $X/reruns/$out "$@" > $L/$tag.log 2>&1
}
APPS="lulesh minife s3d minisweep miniWeather amgmk haccmk simplemoc su3 xsbench"
ABL=final,final-wave-norefine
date
for lane in 0 1 2; do
  order=$(python3 -c "import random; a='$APPS'.split(); random.Random($lane).shuffle(a); print(','.join(a))")
  echo "incremental lane $lane: $order" >> $X/reruns/orders.txt
  g=$((lane + 1))
  ( wait_for refs
    $J inc-lane$lane $g stage2 inc_run.py --lane $lane --order $order > $L/inc-lane$lane.log 2>&1
    case $lane in
      0) tune abl-s3d $g ablation s3d 159,165,171,183,220,226 loao $ABL ;;
      1) tune abl-minisweep $g ablation minisweep 282,315,348,383 loao $ABL
         tune abl-miniweather $g ablation miniWeather 760 loao $ABL ;;
      2) for a in amgmk:258 simplemoc:379 su3:41 xsbench:180; do
           tune abl-${a%%:*} $g ablation ${a%%:*} ${a##*:} loao $ABL; done ;;
    esac ) &
done
( wait_for refs
  for b in minife-150 s3d-s3 su3-l24 simplemoc-s1500k minisweep-c884 miniWeather-1600; do
    $A $FASTRR/scripts/record_all_inputs.sh $b 4 >> $L/inputs-record.log 2>&1; done
  for ab in minife:minife-150:290,521 s3d:s3d-s3:159,165,171,183,220,226 su3:su3-l24:41 \
            simplemoc:simplemoc-s1500k:379 minisweep:minisweep-c884:282,315,348,383 \
            miniWeather:miniWeather-1600:760; do
    IFS=: read a b lines <<< "$ab"
    tune inputs-retune-$b 4 inputs/runs $b $lines loao final-wave --validate-rounds 0
    $J inputs-validate-$b 4 stage2 validate_inputs.py --app-a $a --app-b $b --lines $lines \
       > $L/inputs-validate-$b.log 2>&1
  done ) &
( wait_for gridA
  tune search-A 5 search lulesh 649,844,1201,333,584 loko bo-60,tpe-60,random-60 --cache ) &
( wait_for gridB
  tune search-B 6 search lulesh 1029,1080,1721,1913,2103 loko bo-60,tpe-60,random-60 --cache ) &
( wait_for cold-stage2
  tune batch-lulesh 7 batch lulesh 649,844,1201 loko tpe-mf,tpe-60
  tune batch-minife 7 batch minife 290,521 loko tpe-mf,tpe-60 ) &
wait
# Four GPUs and promotions on the 5 LULESH/miniFE kernels, after everything else.
( tune mgpu-lulesh 1,2,3,4 mgpu lulesh 649,844,1201 loko final-wave-4gpu --devices 4
  tune mgpu-minife 1,2,3,4 mgpu minife 290,521 loko final-wave-4gpu --devices 4 ) &
( tune promote-lulesh 5 promote lulesh 649,844,1201 loko final-wave-promote
  tune promote-minife 5 promote minife 290,521 loko final-wave-promote ) &
wait; date; echo 0 > $X/exit/reruns.txt
