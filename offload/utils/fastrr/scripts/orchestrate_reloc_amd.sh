#!/bin/bash
# Stage 2 space measurement on the MI50 after moving the AMD record/replay base address to
# 0x153540000000 (amdgpu/src/rtl.cpp). With the former address, the index arrays of LULESH,
# miniFE, and XSBench held 8-byte values that looked like pointers, so their replays could not
# relocate. Records the 12 applications again into a separate directory (FASTRR_EXP and
# FASTRR_REC are set by the caller, e.g. .../exp-reloc and .../rec-reloc; the earlier stage 2
# recordings and results stay untouched) and repeats space_measure.py and cold_load.py.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP; J=$JOB; G=${FASTRR_SPACE_GPU:-1}
M=$X/measure; mkdir -p $M $X/exit $X/build
APPS="xsbench rsbench simplemoc su3 haccmk amgmk minisweep miniWeather sw4ck s3d minife lulesh"
declare -A KEEP=(
  [lulesh]=333,584,649,844,1029,1080,1201,1721,1913,2103
  [minife]=157,169,205,210,254,290,521
  [s3d]=159,165,171,177,183,189,195,200,205,210,215,220,226,232
  [minisweep]=282,315,348,383 [miniWeather]=253,308,760 [amgmk]=258 [haccmk]=35
  [rsbench]=37 [simplemoc]=379 [su3]=41 [xsbench]=180 [sw4ck]=23,24,25)
declare -A MEM=([lulesh]=8 [minife]=8)
bash $FASTRR/scripts/build-test.sh stage2 $G > $X/build/stage2-test.log 2>&1
[ "$(cat $X/build/stage2-test.exit)" = 0 ] || { echo "stage 2 lit tests failed"; echo 1 > $X/exit/reloc-amd.txt; exit 1; }
export PYTHONPATH=$FASTRR_PYLIB${PYTHONPATH:+:$PYTHONPATH}
for app in $APPS; do
  echo "$(date) stage2 $app"
  env $FASTRR_VISIBLE=$G $FASTRR_PYTHON -u $FASTRR/scripts/record_app.py stage2 $app --keep ${KEEP[$app]} \
    --memsize ${MEM[$app]:-4} >> $M/record-stage2.log 2>&1 || echo "$app: recording failed"
done
LIST=$(echo $APPS | tr ' ' ,)
$J space-stage2 $G stage2 space_measure.py --stage stage2 --apps $LIST --out $M/space-stage2.csv > $M/space-stage2.log 2>&1
$J cold-stage2 $G stage2 cold_load.py --stage stage2 --apps $LIST --out $M/cold-stage2.csv > $M/cold-stage2.log 2>&1
date; echo 0 > $X/exit/reloc-amd.txt
