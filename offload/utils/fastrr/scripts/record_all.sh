#!/bin/bash
# usage: record_all.sh <stage> <GPU> [apps...]  (inside rr_current.sif on NVIDIA): record the paper's
# applications with a stage's runtime, keeping the kernels of the transfer pool.
st=$1; g=$2; shift 2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
export PYTHONPATH=$FASTRR_PYLIB${PYTHONPATH:+:$PYTHONPATH}
export $FASTRR_VISIBLE=$g
declare -A KEEP=(
  [lulesh]=333,584,649,844,1029,1080,1201,1721,1913,2103
  [minife]=157,169,205,210,254,290,521
  [s3d]=159,165,171,177,183,189,195,200,205,210,215,220,226,232
  [minisweep]=282,315,348,383 [miniWeather]=253,308,760 [amgmk]=258 [haccmk]=35
  [rsbench]=37 [simplemoc]=379 [su3]=41 [xsbench]=180 [sw4ck]=none)
declare -A MEM=([lulesh]=8 [minife]=8)
APPS=${@:-"xsbench rsbench simplemoc su3 haccmk amgmk minisweep miniWeather sw4ck s3d minife lulesh"}
echo "start $(date -Is) $st GPU $g"; fails=0
for app in $APPS; do
  $FASTRR_PYTHON -u $FASTRR/scripts/record_app.py $st $app --keep ${KEEP[$app]} --memsize ${MEM[$app]:-4} || fails=$((fails+1))
done
echo "fails=$fails end $(date -Is)"; mkdir -p $X/exit; echo $fails > $X/exit/record-$st-gpu$g.txt
