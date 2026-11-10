# HeCBench build files of the T4 experiments

HeCBench applications are built from a checkout of github.com/zjin-lcf/HeCBench at commit
`f0c817a` (`$FASTRR_HECBENCH` is its `src/` directory). The checkout needs the sibling
directories that some applications read (`xsbench-cuda`, `miniWeather-cuda`, `sw4ck-cuda`,
`minibude-sycl`). Two applications also need data from HeCBench's public DVC store
(`s3://hecbench-geekdude`, anonymous access):

- `testSNAP-omp/refdata.tar.gz` (md5 a1658855...);
- `minibude-sycl/data.tar.gz` (md5 36c4cec4...).

`record_app.py` builds each application with `make -f Makefile.rr` in a copy of its directory.
These files are not part of HeCBench. Install them into the checkout with:

```bash
bash $FASTRR/benchmarks/hecbench/install.sh        # or: install.sh <HeCBench src directory>
```

This does three things, and running it again changes nothing:

1. **Makefiles.** It copies `makefiles/<app>/Makefile.rr` into `<src>/<app>/`. There are 225
   applications: the 12 of the paper and those of the bulk transfer study.
2. **Second miniWeather input.** It creates `miniWeather-omp-1600` as a copy of
   `miniWeather-omp`.
3. **RSBench fix.** It applies `rsbench-map-data.patch`.

## Makefile.rr

Each `Makefile.rr` is the application's `Makefile.aomp` with these changes, which `make_rr.sh`
reproduces:

- The AMD offload target is replaced by NVIDIA: `-fopenmp-targets=nvptx64
  -Xopenmp-target=nvptx64 -march=sm_75`.
- It adds the flags used for LULESH and miniFE, `-fopenmp-offload-mandatory -foffload-lto`.
- It adds `LDFLAGS += -Wl,--embed-bitcode`. This embeds the device bitcode in the binary, which
  upstream Record-Replay records and the tuner recompiles with new launch bounds.

`make_rr.sh <app dir>...` regenerates the file from `Makefile.aomp`. It reproduces 223 of the
files here exactly; the other two are miniWeather's, which have these changes:

- **`miniWeather-omp`:**
  - It builds with `clang++` instead of `mpicxx`, against `mpi_stub/mpi.h`. That is a
    single-rank stub of the 11 MPI calls that miniWeather uses: the halo exchange sends to
    itself. The binary runs without `mpiexec`. Set `MPI_STUB` to the stub's directory if
    `$FASTRR` is not set.
  - `SIM_TIME` is 10 instead of 600 (60 steps), because each kernel is recorded at its first
    launch.
- **`miniWeather-omp-1600`,** the second input of the input study: `NX=1600`, `NZ=800`
  (16 times the cells), and `SIM_TIME=2.5` (about 60 steps).

The makefiles target the T4 (`sm_75`). For `FASTRR_TARGET=amdgpu`, the MI50 experiments
generate their own `Makefile.rr` (`make_rr_amd.sh`, which `bulk_registry.py` refers to).

## RSBench

With upstream LLVM, recording RSBench fails. Its event kernel dereferences pointer members of
the `SimulationData` structure `data`, but its `target data` region maps only the arrays and
not the structure itself; LLVM 16 tolerated this. `rsbench-map-data.patch` adds
`map(to:data)` to `simulation.c`. With it, RSBench records and replays on the T4. The paper
reports RSBench separately from the 43 kernels.

## Run arguments

The run arguments of the recordings are in `scripts/record_app.py` (`HECBENCH`), not in the
makefiles' `run` targets.
