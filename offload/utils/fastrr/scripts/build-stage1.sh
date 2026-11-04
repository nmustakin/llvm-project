#!/bin/bash
# Build stage 1 (rr-taco-batch worktree): clang/lld + openmp/offload with the NVPTX device RTL,
# configured like build-rr-2026-10-13 (GCC 12, Release + assertions).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP/build; mkdir -p $X
SRC=$FASTRR_STAGE1_SRC
B=$FASTRR_STAGE1_BUILD
echo "start $(date -Is) src=$(git -C $SRC rev-parse HEAD) (+ uncommitted)"
set -o pipefail
if [ ! -f $B/CMakeCache.txt ]; then
  CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake -S $SRC/llvm -B $B \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_ENABLE_RUNTIMES="openmp;offload" \
    -DLLVM_RUNTIME_TARGETS="default;nvptx64-nvidia-cuda" \
    -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_APPEND_VC_REV=OFF \
    -DLIBOMPTARGET_ENABLE_DEBUG=ON || { echo 1 > $X/stage1.exit; exit 1; }
fi
nice -n 19 make -C $B -j40
rc=$?
echo "$rc" > $X/stage1.exit
echo "end $(date -Is) rc=$rc"
exit $rc
