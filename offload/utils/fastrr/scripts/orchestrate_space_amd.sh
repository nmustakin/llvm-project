#!/bin/bash
# The space table on the MI50 (S8): recordings, recording time, device and host memory of one
# tuner replay, and cold and warm load, for stage 1 (batching and the timing protocol only,
# rr-taco-batch-amd, built by `build-amd.sh stage1`) against stage 2 (this branch), on the
# primary inputs of the 11 applications. Host-side times need a quiet machine, so it waits for
# the stage 1 build and uses one GPU (FASTRR_SPACE_GPU, default 1). Stage 1 recordings are
# large (151 GiB on the T4), so the applications are recorded smallest first, and the script
# stops before an application when less than FASTRR_SPACE_MIN_FREE_GB (80) is free.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
J=$JOB
G=${FASTRR_SPACE_GPU:-1}
M=$X/measure; mkdir -p $M
APPS="xsbench rsbench simplemoc su3 haccmk amgmk minisweep miniWeather sw4ck s3d minife lulesh"
declare -A KEEP=(
  [lulesh]=333,584,649,844,1029,1080,1201,1721,1913,2103
  [minife]=157,169,205,210,254,290,521
  [s3d]=159,165,171,177,183,189,195,200,205,210,215,220,226,232
  [minisweep]=282,315,348,383 [miniWeather]=253,308,760 [amgmk]=258 [haccmk]=35
  [rsbench]=37 [simplemoc]=379 [su3]=41 [xsbench]=180 [sw4ck]=23,24,25)
declare -A MEM=([lulesh]=8 [minife]=8)
until [ -f $X/build/stage1.exit ]; do sleep 60; done
[ "$(cat $X/build/stage1.exit)" = 0 ] || { echo "stage 1 build failed"; echo 1 > $X/exit/space-amd.txt; exit 1; }
# The omp-kernel-replay lit tests of stage 1 (batching, warm-up, restore from device) must pass.
bash $FASTRR/scripts/build-test.sh stage1 $G > $X/build/stage1-test.log 2>&1
[ "$(cat $X/build/stage1-test.exit)" = 0 ] || { echo "stage 1 lit tests failed"; echo 1 > $X/exit/space-amd.txt; exit 1; }
date
# Stage 2: the recordings exist (sizes.csv and sizes-kept.csv from record_all.sh).
LIST=$(echo $APPS | tr ' ' ,)
$J space-stage2 $G stage2 space_measure.py --stage stage2 --apps $LIST --out $M/space-stage2.csv > $M/space-stage2.log 2>&1
$J cold-stage2 $G stage2 cold_load.py --stage stage2 --apps $LIST --out $M/cold-stage2.csv > $M/cold-stage2.log 2>&1
# Stage 1, one application at a time.
export PYTHONPATH=$FASTRR_PYLIB${PYTHONPATH:+:$PYTHONPATH}
for app in $APPS; do
  free=$(df -BG --output=avail $FASTRR_REC | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "${FASTRR_SPACE_MIN_FREE_GB:-80}" ]; then
    echo "$(date) stopping before $app: ${free} GB free"; echo 2 > $X/exit/space-amd.txt; exit 2
  fi
  echo "$(date) stage1 $app (${free} GB free)"
  env $FASTRR_VISIBLE=$G $FASTRR_PYTHON -u $FASTRR/scripts/record_app.py stage1 $app --keep ${KEEP[$app]} \
    --memsize ${MEM[$app]:-4} >> $M/record-stage1.log 2>&1 || { echo "$app: recording failed"; continue; }
  $J space-stage1-$app $G stage1 space_measure.py --stage stage1 --apps $app --out $M/space-stage1.csv >> $M/space-stage1.log 2>&1
  $J cold-stage1-$app $G stage1 cold_load.py --stage stage1 --apps $app --out $M/cold-stage1.csv >> $M/cold-stage1.log 2>&1
done
df -h $FASTRR_REC; date; echo 0 > $X/exit/space-amd.txt
