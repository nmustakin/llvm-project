#!/bin/bash
# The energy study of the conference workflow on upstream LLVM (the paper's "Upstream LLVM on
# the T4" energy paragraph): BO with 30 initial and 200 suggestions, one seed, no measurement
# cache, on the 24 LULESH and miniFE kernels of the conference version, once per objective
# (speedup, energy, EDP, ED2P: 96 runs). Every evaluation measures the kernel energy per launch
# over a search window of RR_ENERGY_MIN_MS (default 1000 ms), and the shipped configurations are
# validated against the default in 10 randomized rounds with 3000 ms windows: on the T4, energy
# over 300 ms varies between repetitions by a median of 14.3%, at 1000 ms by 5.8% and at
# 3000 ms by 3.5%.
#
# usage: orchestrate_energy24.sh [record] [search] [validate] [summary]   (default: all four)
#   Run it on the host, in tmux: $JOB starts every job in the container (on NVIDIA). One lane
#   per GPU of FASTRR_ENERGY_GPUS.
#
# Environment (besides env.sh):
#   FASTRR_ENERGY_GPUS     GPUs of the lanes (default "0 1 2 3 4 5 6 7")
#   FASTRR_ENERGY_BUILD    LLVM build of this branch (replay tool with --energy-ms)
#   RR_ENERGY_MIN_MS       search window in ms (default 1000)
#   FASTRR_ENERGY_VAL_MS   validation windows in ms, comma-separated (default 3000)
#   FASTRR_ENERGY_ROUNDS   validation rounds (default 10)
#   RR_UPSTREAM_METRICS    1: CUPTI metrics in every evaluation, as the conference workflow
#                          (NVIDIA; about 4x the search time with energy). Default 0.
#   FASTRR_ENERGY_OUT      output directory (default $FASTRR_EXP/energy24)
#
# Output: $FASTRR_ENERGY_OUT/runs/<method>/<app>-l<line>/ (energy_search.py),
#   validation/<app>-l<line>.csv (energy_validate.py), summary.csv and summary.log
#   (energy_summary.py), logs/, and exit codes in $FASTRR_EXP/exit/energy24-*.txt.
# Recordings: $FASTRR_REC/energy/<app>, registries $FASTRR_EXP/energy/kernels/<app>.json.
# Every phase is resumable: finished runs and validation rounds are skipped.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP; J=$JOB
O=${FASTRR_ENERGY_OUT:-$X/energy24}; mkdir -p $O/logs $X/exit
read -r -a G <<< "${FASTRR_ENERGY_GPUS:-0 1 2 3 4 5 6 7}"
export RR_ENERGY_MIN_MS=${RR_ENERGY_MIN_MS:-1000} RR_UPSTREAM_METRICS=${RR_UPSTREAM_METRICS:-0}
VAL_MS=${FASTRR_ENERGY_VAL_MS:-3000}; ROUNDS=${FASTRR_ENERGY_ROUNDS:-10}
METHODS="bo-speedup bo-energy bo-edp bo-ed2p"
declare -A LINES=(
  [lulesh]=333,584,649,844,1029,1080,1201,1235,1279,1312,1340,1377,1721,1841,1913,2103,2372
  [minife]=157,169,205,210,254,290,521)
PHASES=${@:-record search validate summary}
echo "start $(date -Is) phases: $PHASES; GPUs ${G[*]}; search window ${RR_ENERGY_MIN_MS} ms," \
     "validation ${VAL_MS} ms x $ROUNDS rounds, metrics $RR_UPSTREAM_METRICS"

for phase in $PHASES; do
  case $phase in
  record)
    # The 24 kernels, recorded with this branch's runtime (8 GiB pool, as for the space study).
    fails=0
    for app in lulesh minife; do
      $J energy24-record-$app ${G[0]} energy record_app.py energy $app --keep ${LINES[$app]} \
        --memsize 8 > $O/logs/record-$app.log 2>&1
      rc=$(cat $X/exit/energy24-record-$app.txt); [ "$rc" = 0 ] || fails=$((fails+1))
      tail -1 $O/logs/record-$app.log
    done
    [ $fails = 0 ] || { echo "recording failed ($fails apps)"; exit 1; } ;;
  search)
    # One item per kernel and objective; each lane pops items under a lock.
    Q=$O/queue.txt
    if [ ! -s $Q ]; then
      : > $Q
      for app in lulesh minife; do
        for line in ${LINES[$app]//,/ }; do
          for m in $METHODS; do
            [ -n "$(ls $O/runs/$m/$app-l$line/$m-s0/trace_*.csv 2>/dev/null)" ] || echo "$app|$line|$m" >> $Q
          done
        done
      done
    fi
    echo "search: $(wc -l < $Q) items"
    for g in "${G[@]}"; do
      ( fails=0
        while true; do
          item=$(flock $O/queue.lock bash -c "head -1 $Q; sed -i 1d $Q")
          [ -n "$item" ] || break
          IFS='|' read app line m <<< "$item"
          $J energy24-item-g$g $g energy energy_search.py --app $app --lines $line --methods $m \
            --seeds 0 --budget 230 --init 30 --no-cache --out $O/runs \
            >> $O/logs/search-$app-l$line-$m.log 2>&1
          rc=$(cat $X/exit/energy24-item-g$g.txt)
          [ "$rc" = 0 ] || { fails=$((fails+1)); echo "$item failed rc=$rc"; }
        done
        echo $fails > $X/exit/energy24-search-g$g.txt ) > $O/logs/search-lane-g$g.log 2>&1 &
    done
    wait
    fails=$(cat $X/exit/energy24-search-g*.txt | awk '{s += $1} END {print s}')
    echo "search done $(date -Is), failed items: $fails" ;;
  validate)
    $J energy24-vqueue ${G[0]} energy energy_validate.py --runs $O/runs --out $O/validation \
      --make-queue > $O/logs/validate-queue.log 2>&1
    cat $O/logs/validate-queue.log | tail -1
    for g in "${G[@]}"; do
      $J energy24-validate-g$g $g energy energy_validate.py --runs $O/runs --out $O/validation \
        --lane --rounds $ROUNDS --windows $VAL_MS > $O/logs/validate-lane-g$g.log 2>&1 &
    done
    wait
    echo "validation done $(date -Is)" ;;
  summary)
    $FASTRR_RUN $FASTRR_PYTHON $FASTRR/scripts/energy_summary.py --runs $O/runs --validation $O/validation \
      --window ${VAL_MS##*,} --out $O/summary.csv | tee $O/summary.log ;;
  *) echo "unknown phase $phase"; exit 2 ;;
  esac
done
echo 0 > $X/exit/energy24.txt; echo "end $(date -Is)"
