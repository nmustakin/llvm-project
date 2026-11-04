# Paths and container of the FastRR replication (README.md). Source this file in bash; set any
# variable beforehand to override the defaults, which are those of the original machine.
FASTRR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLVM_SRC=$(cd "$FASTRR/../../.." && pwd)
# Settings of this machine, kept out of version control (variables as below). Jobs started
# by tmux do not inherit the variables of the shell that starts them, so set them here.
[ -f "$FASTRR/scripts/env.local.sh" ] && source "$FASTRR/scripts/env.local.sh"

# Outputs (registries, runs, logs, exit codes) and recordings.
: "${FASTRR_EXP:=/data2/nmn/rr-metrics/experiment-archives/taco-replication-2026-11-03}"
: "${FASTRR_REC:=/data3/nmn/rr-metrics-recordings/taco-replication-2026-11-03}"
# LLVM builds. Stage 2 is this branch (rr-taco-space); stage 1 is branch rr-taco-batch.
: "${FASTRR_STAGE2_BUILD:=$LLVM_SRC/build-rr-2026-10-13}"
: "${FASTRR_STAGE1_SRC:=/data2/nmn/llvm/llvm-project-taco}"
: "${FASTRR_STAGE1_BUILD:=$FASTRR_STAGE1_SRC/build}"
# zstd installation the stage 2 build links (CMAKE_PREFIX_PATH).
: "${FASTRR_ZSTD:=/data2/nmn/zstd-1.5.6/install}"
# ROCm installation for AMD GPUs (build-amd.sh, Upstream.py); needs hsa_amd_vmem_* (ROCm 6.0+).
: "${FASTRR_ROCM:=/opt/rocm}"
# HeCBench sources (github.com/zjin-lcf/HeCBench, commit f0c817a), src/ directory.
: "${FASTRR_HECBENCH:=/data2/nmn/rr-metrics/hecbench/src}"
# Training pool of the ranking models (LLVM 16 grid datasets).
: "${FASTRR_POOL:=$FASTRR/data/pool_landscapes.csv}"
# Python packages missing from the container (xgboost 2.1.4, optuna); prepended to PYTHONPATH.
: "${FASTRR_PYLIB:=/data2/nmn/rr-metrics/experiment-archives/learned-transfer-2026-10-13/pylib:/data2/nmn/rr-metrics/experiment-archives/search-study-2026-10-09/pylib}"
# Figures only: the LLVM 16 incremental simulation and the paper's figure directory.
: "${FASTRR_SIM:=/data2/nmn/rr-metrics/experiment-archives/incremental-ranker-2026-10-28/incremental.csv}"
: "${FASTRR_FIGURES:=/data2/nmn/rr-metrics/RR-GPGPU2025/figures-ext/upstream}"
# Container (GCC 12, CUDA 12.3, Python with pandas and scikit-learn) and its bind mounts.
: "${FASTRR_SIF:=/data2/nmn/rr_current.sif}"
: "${FASTRR_BINDS:=--bind /data2/nmn:/data2/nmn --bind /data3/nmn:/data3/nmn}"

# Python 3.10 or later with the packages of requirements.txt.
: "${FASTRR_PYTHON:=python3}"
# Offload target: nvptx (NVIDIA, the paper's T4s) or amdgpu (AMD; native, no container).
: "${FASTRR_TARGET:=nvptx}"
case $FASTRR_TARGET in
  amdgpu) FASTRR_VISIBLE=ROCR_VISIBLE_DEVICES ;;
  *) FASTRR_VISIBLE=CUDA_VISIBLE_DEVICES ;;
esac

export FASTRR LLVM_SRC FASTRR_EXP FASTRR_REC FASTRR_STAGE2_BUILD FASTRR_STAGE1_SRC \
  FASTRR_STAGE1_BUILD FASTRR_ZSTD FASTRR_HECBENCH FASTRR_POOL FASTRR_PYLIB FASTRR_SIM \
  FASTRR_FIGURES FASTRR_SIF FASTRR_BINDS FASTRR_ROCM \
  FASTRR_TARGET FASTRR_VISIBLE FASTRR_AMD_ARCH FASTRR_PYTHON FASTRR_ROCM_DEVICE_LIBS

# Prefix of a command in the container, and of a Python job of this directory (job.sh).
# On AMD the jobs run natively (the shell must be in the render group).
if [ "$FASTRR_TARGET" = amdgpu ]; then
  FASTRR_RUN=""
else
  FASTRR_RUN="apptainer exec --nv $FASTRR_BINDS $FASTRR_SIF"
fi
JOB="$FASTRR_RUN bash $FASTRR/scripts/job.sh"
