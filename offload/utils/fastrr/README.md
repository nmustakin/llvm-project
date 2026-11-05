# FastRR on upstream LLVM record/replay

This directory holds the FastRR tuner and the scripts that replicate the upstream LLVM results
of the TACO paper (`RR-GPGPU2025/taco/extension-TACO-upstream.tex`) on an NVIDIA T4. The tuner
tunes the launch bounds of an OpenMP offload kernel (its `nvvm.maxntid` and `nvvm.minctasm`
attributes, set in the recorded IR) and its team and thread counts by replaying the kernel's
recording with `llvm-omp-kernel-replay`, so the application never reruns.

## Layout

| Path | Contents |
|---|---|
| `RRUtil/` | The tuner. `Upstream.py` connects it to upstream record/replay; `Search.py` holds the search strategies and the ranking model; `Benchmark.py` holds `Kernel.Search`. Copied from branch `fastrr-upstream` of openmp-rr (`record-replay/RR/RRUtil`). |
| `benchmarks/` | LULESH (`lulesh-omp`), miniFE (`miniFE-omp`), and the second inputs of the input study (`inputs`), from openmp-rr. |
| `data/pool_landscapes.csv` | Training pool of the ranking model: grid datasets of 48 kernels measured with the LLVM 16 fork. |
| `scripts/env.sh` | Every path and the container. Set variables before sourcing it to override the defaults. |
| `scripts/` | Build, recording, tuning, measurement, and analysis scripts (below). |
| `requirements.txt` | Python packages. |

The record/replay changes the tuner needs are in this branch (`rr-taco-space`): batch replay,
the timing protocol (`--warmup-ms`, `--rewarm-ms`, `--restore-from-device`, device-event kernel
time), and the space-efficient recordings (compressed block store, `--relocate`, in-place
verification). This is "stage 2" in the scripts and the paper. "Stage 1" is the same tuner on
upstream with batching and the timing protocol only, in branch `rr-taco-batch`; it is needed
only for the stage 1 against stage 2 comparisons (`space_measure.py`, `cold_load.py`,
`analyze.py`).

## Requirements

- An NVIDIA GPU (the paper used T4s, `sm_75`, which `Upstream.ARCH` sets) and CUDA 12.3.
- Apptainer and the container `rr_current.sif` (Ubuntu 22.04, GCC 12.3, CUDA 12.3, CMake 3.22,
  Python 3.10 with the packages of `requirements.txt` except xgboost and optuna).
  Install those two into a directory and point `FASTRR_PYLIB` at it:
  `pip install --target DIR xgboost-cpu==2.1.4 optuna==3.6.1`.
- zstd 1.5.6 installed with its CMake package (`FASTRR_ZSTD`).
- HeCBench sources (github.com/zjin-lcf/HeCBench, commit f0c817a), the `src/` directory
  (`FASTRR_HECBENCH`).
- Disk: the stage 2 recordings of all applications take about 1.3 GiB; stage 1 recordings
  take about 100 GiB.

## Setup on a new machine

Write the machine's paths to `scripts/env.local.sh`, which `env.sh` sources and Git ignores:

```bash
FASTRR_EXP=/path/to/experiment          # outputs
FASTRR_REC=/path/to/recordings
FASTRR_STAGE2_BUILD=/path/to/build       # default: <llvm-project>/build-rr-2026-10-13
FASTRR_ZSTD=/path/to/zstd/install
FASTRR_HECBENCH=/path/to/HeCBench/src
FASTRR_PYLIB=/path/to/pylib
FASTRR_SIF=/path/to/rr_current.sif
FASTRR_BINDS="--bind /path:/path"        # every path above must be visible in the container
```

Then `source offload/utils/fastrr/scripts/env.sh` in the shell. Every job script sources
`env.sh` itself. Use `env.local.sh` rather than exported variables: a job started by
`tmux new-session` gets the environment of the tmux server, not of the calling shell. Run long
jobs in a named tmux session, as in the commands below.

## Steps

1. **Build LLVM** (inside the container):
   `$FASTRR_RUN bash $FASTRR/scripts/build-stage2.sh`, then
   `$FASTRR_RUN bash $FASTRR/scripts/build-test.sh stage2 <GPU>` to run the
   `omp-kernel-replay` lit tests (15 pass and 1 unsupported, 16 with LTO).
2. **Record the applications**: `$FASTRR_RUN bash $FASTRR/scripts/record_all.sh stage2 <GPU>`.
   `record_app.py` copies each application's sources to `$FASTRR_EXP/apps`, builds it with
   the stage's clang, records it into `$FASTRR_REC/<stage>/<app>`, keeps the kernels of the
   paper, and writes the kernel registry `$FASTRR_EXP/<stage>/kernels/<app>.json`.
3. **Grid datasets** (the optimum of every kernel, for true speedups against the optimum):
   `$JOB <tag> <GPU> stage2 grid.py --stage stage2 --kernels lulesh:649,...`.
4. **Tuning runs.** `run_tuner.py` tunes kernels with the listed methods and seeds, then
   validates every shipped configuration against the default in randomized rounds (true
   speedup). Methods: `final-wave` (FastRR), `tpe-batch-nocls` (Batch TPE), the ablations
   `final` and `final-wave-norefine`, `final-wave-promote`, `final-wave-4gpu` (with
   `--devices 4`), `tpe-mf`, and `bo-60`, `tpe-60`, `random-60`. The orchestration scripts
   run the paper's experiments, one lane per GPU:
   - `orchestrate_stage2.sh` and `orchestrate_refs.sh`: FastRR and Batch TPE on the 43 kernels
     (tables batch, summary, crossgpu; the HeCBench figure);
   - `orchestrate_reruns.sh`: the incremental store (`inc_run.py`), the ablations, the input
     study (`record_all_inputs.sh`, `validate_inputs.py`), the search strategies, the batch
     table, four GPUs, and promotions;
   - `measure_all.sh` (`space_measure.py`, `batch_cost.py`) and `cold_load.py`: recording
     sizes, device memory, and load times of both stages (the space table).
   `orchestrate.sh <stage> <methods>` runs the 20 kernels of the first replication on one
   stage. The scripts' lanes use GPUs 1 to 7; edit them to fit the machine.
5. **Analysis**: `analyze.py` (stage 1 against stage 2), then `analyze_paper.py` (every number
   of the paper, `paper-analysis.log` and `paper/*.csv`) and `make_figures.py`
   (`$FASTRR_FIGURES`). Run them in the container after sourcing `env.sh`, with
   `FASTRR_PYLIB` on `PYTHONPATH`.

Each job leaves its exit code in `$FASTRR_EXP/exit/<tag>.txt`; the orchestration scripts wait
for these files to order their lanes.

## AMD GPUs

`FASTRR_TARGET=amdgpu` (in `env.local.sh`) switches the scripts and the tuner to AMD GPUs; it was
brought up on an AMD Instinct MI50 (gfx906).

- ROCm 6.0 or later (`FASTRR_ROCM`): the AMDGPU plugin maps the recording at its recorded
  address with `hsa_amd_vmem_*`, which ROCm 5.x lacks (the plugin then does not load at all).
- Build natively, without the container, in a shell of the render group:
  `bash $FASTRR/scripts/build-amd.sh` (X86 and AMDGPU, runtime targets `default` and
  `amdgcn-amd-amdhsa`, the AMDGPU plugin linked against `$FASTRR_ROCM`'s hsa-runtime64), then
  `bash $FASTRR/scripts/build-test.sh stage2 <GPU>` for the lit tests on the
  `amdgcn-amd-amdhsa` configurations.
- clang of LLVM 24 emits code object version 6, which needs ROCm 6.3. `build-amd.sh` writes
  `clang.cfg` and `clang++.cfg` next to clang with `-mcode-object-version=5` and the device
  libraries `FASTRR_ROCM_DEVICE_LIBS` (default `$FASTRR_ROCM/amdgcn/bitcode`, from the
  `rocm-device-libs` package). Every compile reads them.
- rocprof of ROCm 6.0 (`rocprofiler`, `hsa-amd-aqlprofile`) profiles upstream replays with the
  counter passes of `Device.AMDGPU.MetricPasses` and `RRUtil/amd/rr_metrics.xml`; the tuner
  does not call it yet (`SkipMetrics`).
- The plugin dlopens `libhsa-runtime64.so`; `Upstream.environment` and `build-test.sh` put
  `$FASTRR_ROCM/lib` first on `LD_LIBRARY_PATH`. With ROCm 6.0 the plugin also needs the
  `checkIfAPU` fix of this branch (an unchecked `llvm::Error` aborted device setup).
- Upstream generates no LTO lit configuration for AMDGPU: the 16 tests run once.
- `FASTRR_AMD_ARCH` (default gfx906) is the `-march` of the applications and the llc target.
- `FASTRR_VISIBLE` is `ROCR_VISIBLE_DEVICES`; each replay process sees exactly its own GPU.
- `FASTRR_PYTHON` is the Python 3.10 interpreter if `python3` is older.
- `Upstream.UpstreamAMDGPU` sets MaxThreads as `amdgpu-flat-work-group-size` (1,MaxThreads)
  and MinTeams as `amdgpu-waves-per-eu` (`Device.AMDGPU.getWavesPerEU`) on the recorded IR,
  compiles it with llc (resource usage from the `kernel-resource-usage` remarks) and links
  it with ld.lld into `<stem>.image`.
- As on NVIDIA, upstream replays collect no metrics and no energy.
- `scripts/measure_noise.py` times every kept kernel in randomized rounds, one replay process
  per measurement, and reports the coefficient of variation (the measurement-noise check);
  `$JOB noise 0,...,7 stage2 measure_noise.py --out $FASTRR_EXP/noise --gpus 0,...,7`.

## Environment variables of the tuner

`job.sh` sets these for every job: `RR_UPSTREAM_BUILD` (the stage's build),
`RR_UPSTREAM_REPLAY_FLAGS=--relocate` (stage 2), `RR_TIMING_TRIES=2`, `RR_RANK_CACHE`
(trained ranking models, in `$FASTRR_EXP/rank-cache`), `CUDA_VISIBLE_DEVICES`, and
`CUDA_DEVICE_ORDER=PCI_BUS_ID`.
