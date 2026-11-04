#!/bin/bash
# usage: build-test.sh <stage1|stage2> <GPU>: build the stage's LLVM (incremental) and run the
# omp-kernel-replay lit tests on one GPU (inside rr_current.sif).
st=$1; g=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP/build; mkdir -p $X
case $st in
  stage1) B=$FASTRR_STAGE1_BUILD ;;
  stage2) B=$FASTRR_STAGE2_BUILD ;;
  *) echo "unknown stage $st"; exit 2 ;;
esac
echo "start $(date -Is) $st"
CMAKE_PREFIX_PATH=$FASTRR_ZSTD nice -n 19 make -C $B -j32 > $X/$st-make.log 2>&1
rc=$?; echo "build rc=$rc"
if [ $rc -ne 0 ]; then echo $rc > $X/$st-test.exit; exit $rc; fi
export CUDA_VISIBLE_DEVICES=$g
fails=0
for cfg in nvptx64-nvidia-cuda nvptx64-nvidia-cuda-LTO; do
  $B/bin/llvm-lit -v --filter omp-kernel-replay $B/runtimes/runtimes-bins/offload/test/$cfg > $X/$st-lit-$cfg.log 2>&1 || fails=$((fails+1))
  grep -E '^(PASS|FAIL|UNSUPPORTED|XFAIL)' $X/$st-lit-$cfg.log
done
echo "lit fails=$fails"; echo $fails > $X/$st-test.exit; echo "end $(date -Is)"
