#!/bin/bash
# Search strategies with time and energy on the MI50 (the paper's energy section, upstream LLVM):
# BO, TPE, and random search on speedup, BO on energy, and MOTPE on both, with 60 full
# evaluations, three seeds, and a measurement cache shared by the runs of a kernel, on the ten
# LULESH kernels. Every full evaluation measures the energy per launch over a 300 ms window
# (RR_ENERGY_MIN_MS; Upstream.UpstreamKernel.execute). One lane per GPU, the kernels split
# between the lanes; then the joint validation of the shipped configurations.
# usage: orchestrate_energy_amd.sh   (GPUs 4, 5, 6; FASTRR_ENERGY_GPUS overrides)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP; J=$JOB
O=$X/energy; mkdir -p $O/logs
M=bo-60,tpe-60,random-60,bo-energy-60,motpe-60
read -r -a G <<< "${FASTRR_ENERGY_GPUS:-4 5 6}"
LANES=("333,584,649,844" "1029,1080,1201" "1721,1913,2103")
for i in 0 1 2; do
  ( $J energy-$i ${G[$i]} stage2 run_tuner.py --stage stage2 --app lulesh --lines ${LANES[$i]} \
      --methods $M --seeds 0,1,2 --cache --pool loko --out $O > $O/logs/lane-$i.log 2>&1 ) &
done
wait
date; echo 0 > $X/exit/energy-amd.txt
