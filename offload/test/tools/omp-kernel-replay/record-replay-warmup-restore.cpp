// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: printf '0 0\n0 64\n' > %t.testdir/batch.txt
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify --repetitions=3 --warmup-ms=50 --rewarm-ms=10 --restore-from-device --batch %t.testdir/batch.txt {} | FileCheck --check-prefix=BATCH %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify --repetitions=3 --restore-from-device {} | FileCheck --check-prefix=SINGLE %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify --repetitions=3 --warmup-ms=20 {} | FileCheck --check-prefix=HOST %s
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify --relocate --repetitions=3 --warmup-ms=20 --restore-from-device {} | FileCheck --check-prefix=RELOCATED %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// Every configuration of a batch is warmed up, with a longer warm-up for the
// first one, and every replay restores the recorded memory from its device
// copy. The kernel is not idempotent, so verification fails unless every
// replay starts from the recorded memory.

// BATCH:      ||ReplayConfig|| index: 0
// BATCH-NEXT: Warm-up replays: {{[1-9][0-9]*}}
// BATCH-NEXT: Replay time (1): {{[0-9]+}} ns
// BATCH-NEXT: Replay device time (1): {{[0-9]+}} ns
// BATCH:      ||ReplayTiming|| duration_seconds: {{[0-9.e+-]+}}
// BATCH-NEXT: Replay done, device memory verified
// BATCH-NEXT: ||ReplayConfig|| index: 1 num_teams: {{[0-9]+}} num_threads: 64
// BATCH-NEXT: Warm-up replays: {{[1-9][0-9]*}}
// BATCH:      ||ReplayTiming||
// BATCH-NEXT: Replay done, device memory verified

// SINGLE-NOT: Warm-up replays
// SINGLE-COUNT-3: Replay device time
// SINGLE:     Replay done, device memory verified

// HOST: Warm-up replays: {{[1-9][0-9]*}}
// HOST: Replay done, device memory verified

// RELOCATED: Relocated the replay memory
// RELOCATED: Warm-up replays: {{[1-9][0-9]*}}
// RELOCATED-COUNT-3: Replay device time
// RELOCATED: Replay done, device memory verified

#include <cstdint>
#include <cstdio>

int main() {
  size_t Size = 1000;
  uint64_t *Data = new uint64_t[Size];

  for (size_t I = 0; I < Size; ++I)
    Data[I] = I;

#pragma omp target teams distribute parallel for map(tofrom : Data[0 : Size])
  for (size_t I = 0; I < Size; ++I)
    Data[I] += 10;

  uint64_t Sum = 0;
  for (size_t I = 0; I < Size; ++I)
    Sum += Data[I];

  // CHECK: PASS
  if (Sum == 509500)
    printf("PASS\n");

  delete[] Data;
}
