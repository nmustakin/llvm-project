#!/bin/bash
# The bulk HeCBench study on upstream LLVM (the further kernels of the transfer study, Table
# mi50more of the paper): one worker per GPU takes applications from a shared queue, builds and
# records each with stage 2 (4 GiB region, as the LLVM 16 study), keeps the kernels with a trip
# count, at most 1 GiB of recorded memory, and a default time of 0.1 to 100 ms (bulk_filter.py),
# and measures their grid datasets (grid.py). Set FASTRR_EXP and FASTRR_REC to separate
# directories (e.g. .../exp-bulk and .../rec-bulk) before running.
# usage: bulk_amd.sh <GPU> ...   (one worker per GPU, in the background; the queue is filled once)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP; J=$JOB
Q=$X/bulk-queue.txt; L=$X/bulk-queue.lock; S=$X/bulk-status.txt
mkdir -p $X/logs/bulk
if [ ! -f $Q ]; then
  $FASTRR_PYTHON -c "import json; print('\n'.join(json.load(open('$FASTRR/data/bulk_hecbench.json'))))" > $Q
fi
worker() {
  local g=$1 app
  while true; do
    free=$(df -BG --output=avail $FASTRR_REC | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 20 ]; then echo "$(date) gpu$g: stopping, $free GB free" >> $S; return; fi
    app=$(flock $L bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$app" ] && return
    $J bulk-rec-$app $g stage2 record_app.py stage2 $app --keep all --memsize 4 \
      > $X/logs/bulk/record-$app.log 2>&1
    if [ "$(cat $X/exit/bulk-rec-$app.txt)" != 0 ]; then
      echo "$(date) $app gpu$g record failed" >> $S; continue
    fi
    $J bulk-filter-$app $g stage2 bulk_filter.py --stage stage2 --app $app \
      > $X/logs/bulk/filter-$app.log 2>&1
    if [ "$(cat $X/exit/bulk-filter-$app.txt)" != 0 ]; then
      echo "$(date) $app gpu$g filter failed" >> $S; continue
    fi
    kernels=$($FASTRR_PYTHON -c "import json; print(','.join('$app:'+l for l in json.load(open('$X/stage2/kernels/$app.json'))))")
    if [ -n "$kernels" ]; then
      $J bulk-grid-$app $g stage2 grid.py --stage stage2 --kernels $kernels \
        > $X/logs/bulk/grid-$app.log 2>&1
    fi
    echo "$(date) $app gpu$g kept=${kernels:-none} grid=$(cat $X/exit/bulk-grid-$app.txt 2>/dev/null)" >> $S
  done
}
for g in "$@"; do worker $g & done
wait
echo "$(date) all workers done" >> $S
