#!/bin/bash
# Build stage 2 (this branch) for AMD GPUs: clang/lld + openmp/offload with the AMDGPU device
# RTL, GCC 12, Release + assertions, and zstd (FASTRR_ZSTD) for the compressed recordings.
# The AMDGPU plugin dlopens libhsa-runtime64.so (the first on LD_LIBRARY_PATH, which
# Upstream.environment and build-test.sh set to FASTRR_ROCM/lib); it needs hsa_amd_vmem_*
# (ROCm 6.0 or later). Run it natively (not in the NVIDIA container), as a member of the
# render group, so that the offload configuration finds the GPU and enables its lit tests.
#
# clang of LLVM 24 emits code object version 6 by default, which needs ROCm 6.3; the build
# writes clang.cfg and clang++.cfg next to clang (read by every compile, including the lit
# tests and the device link of the linker wrapper) with -mcode-object-version=5 and the
# ROCm device libraries FASTRR_ROCM_DEVICE_LIBS (default $FASTRR_ROCM/amdgcn/bitcode).
# clang of LLVM 24 no longer links the oclc_daz_opt and oclc_correctly_rounded_sqrt control
# libraries, which the ocml of older ROCm (6.0) still references (single-precision pow, exp2,
# log10, ...): the JIT link of such kernels fails on an undefined __oclc_daz_opt. The build
# therefore points clang at $B/amdgcn-device-libs: links to the ROCm libraries, except
# oclc_finite_only_{off,on}.bc (which clang always links), which also define the two missing
# constants with clang's former defaults (no denormal flushing, correctly rounded sqrt).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
X=$FASTRR_EXP/build; mkdir -p $X
SRC=$LLVM_SRC
B=$FASTRR_STAGE2_BUILD
: "${FASTRR_AMD_ARCH:=gfx906}"
echo "start $(date -Is) src=$(git -C $SRC rev-parse HEAD) rocm=$FASTRR_ROCM"
set -o pipefail
export CMAKE_PREFIX_PATH="$FASTRR_ZSTD:$FASTRR_ROCM"
if [ ! -f $B/CMakeCache.txt ]; then
  CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake -S $SRC/llvm -B $B -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_ENABLE_RUNTIMES="openmp;offload" \
    -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa" \
    -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_APPEND_VC_REV=OFF \
    -DLLVM_ENABLE_ZSTD=FORCE_ON -Dzstd_DIR=$FASTRR_ZSTD/lib/cmake/zstd \
    -DLIBOMPTARGET_PLUGINS_TO_BUILD="amdgpu;host" \
    -DLIBOMPTARGET_FORCE_AMDGPU_TESTS=ON \
    -DLIBOMPTARGET_ENABLE_DEBUG=ON || { echo 1 > $X/stage2.exit; exit 1; }
fi
: "${FASTRR_ROCM_DEVICE_LIBS:=$FASTRR_ROCM/amdgcn/bitcode}"
mkdir -p $B/bin
for cfg in clang.cfg clang++.cfg; do
  printf -- '-mcode-object-version=5\n--rocm-device-lib-path=%s\n' "$B/amdgcn-device-libs" > $B/bin/$cfg
done
nice -n 19 ninja -C $B -j${FASTRR_JOBS:-48}
rc=$?
if [ $rc -eq 0 ]; then
  L=$B/amdgcn-device-libs; rm -rf $L; mkdir -p $L
  for f in $FASTRR_ROCM_DEVICE_LIBS/*.bc; do ln -s $f $L/; done
  extra=""
  for f in oclc_daz_opt_off.bc oclc_correctly_rounded_sqrt_on.bc; do
    [ -f $FASTRR_ROCM_DEVICE_LIBS/$f ] &&
      extra+=$($B/bin/llvm-dis $FASTRR_ROCM_DEVICE_LIBS/$f -o - | grep '^@__oclc_')$'\n'
  done
  for f in oclc_finite_only_off oclc_finite_only_on; do
    rm $L/$f.bc
    { $B/bin/llvm-dis $FASTRR_ROCM_DEVICE_LIBS/$f.bc -o -; printf '%s' "$extra"; } |
      $B/bin/llvm-as -o $L/$f.bc || rc=1
  done
fi
echo "$rc" > $X/stage2.exit
echo "end $(date -Is) rc=$rc"
exit $rc
