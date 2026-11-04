#!/bin/bash
# Build stage 2 (this branch, rr-taco-space): clang/lld + openmp/offload with the NVPTX device
# RTL, GCC 12, Release + assertions, and zstd (FASTRR_ZSTD) for the compressed recordings.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP/build; mkdir -p $X
SRC=$LLVM_SRC
B=$FASTRR_STAGE2_BUILD
echo "start $(date -Is) src=$(git -C $SRC rev-parse HEAD)"
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
    -DLLVM_ENABLE_ZSTD=FORCE_ON -Dzstd_DIR=$FASTRR_ZSTD/lib/cmake/zstd \
    -DLIBOMPTARGET_ENABLE_DEBUG=ON || { echo 1 > $X/stage2.exit; exit 1; }
fi
CMAKE_PREFIX_PATH=$FASTRR_ZSTD nice -n 19 make -C $B -j40
rc=$?
echo "$rc" > $X/stage2.exit
echo "end $(date -Is) rc=$rc"
exit $rc
