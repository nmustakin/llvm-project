#!/bin/bash
# usage: record_all_inputs.sh <app> <GPU>  (inside rr_current.sif on NVIDIA): record a second input of the
# input study with stage 2, keeping the kernels of the study.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
export PYTHONPATH=$FASTRR_PYLIB${PYTHONPATH:+:$PYTHONPATH}
export $FASTRR_VISIBLE=$2
declare -A KEEP=([minife-150]=290,521 [s3d-s3]=159,165,171,183,220,226 [su3-l24]=41
                 [simplemoc-s1500k]=379 [minisweep-c884]=282,315,348,383 [miniWeather-1600]=760)
declare -A MEM=([minife-150]=8)
$FASTRR_PYTHON -u $FASTRR/scripts/record_app.py stage2 $1 --keep ${KEEP[$1]} --memsize ${MEM[$1]:-4}
echo $? > $X/exit/record-input-$1.txt
