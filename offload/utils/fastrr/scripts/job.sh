#!/bin/bash
# usage: job.sh <tag> <GPU> <stage1|stage2> <script.py> <args...>   (inside rr_current.sif)
# Runs a Python script of this experiment on one GPU with the tuner's environment and the
# stage's LLVM build, and leaves its exit code in exit/<tag>.txt. Stage 2 replays with
# --relocate (the space-saving changes).
tag=$1; g=$2; st=$3; script=$4; shift 4
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP
# GCC 12 as gcc/g++ (as update-gcc.sh of openmp-rr does).
mkdir -p /tmp/gcc12-bin
ln -sf "$(command -v gcc-12)" /tmp/gcc12-bin/gcc; ln -sf "$(command -v g++-12)" /tmp/gcc12-bin/g++
export PATH="/tmp/gcc12-bin:$PATH"
export PYTHONPATH=$FASTRR_PYLIB${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g RR_TIMING_TRIES=2 RR_RANK_CACHE=$X/rank-cache
case $st in
  stage1) export RR_UPSTREAM_BUILD=$FASTRR_STAGE1_BUILD ;;
  stage2) export RR_UPSTREAM_BUILD=$FASTRR_STAGE2_BUILD
          export RR_UPSTREAM_REPLAY_FLAGS=--relocate ;;
  *) echo "unknown stage $st"; exit 2 ;;
esac
mkdir -p $X/exit
date; rc=0
python3 -u $FASTRR/scripts/$script "$@" || rc=$?
echo $rc > $X/exit/$tag.txt; date
