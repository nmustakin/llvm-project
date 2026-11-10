#!/bin/bash
# usage: make_rr.sh <HeCBench app directory>...   e.g. make_rr.sh $FASTRR_HECBENCH/rsbench-omp
# Writes Makefile.rr for NVIDIA (sm_75) from the app's Makefile.aomp, as the T4 experiments did:
# the AMD offload target replaced by nvptx64, the flags of LULESH and miniFE
# (-fopenmp-offload-mandatory -foffload-lto), and the device bitcode embedded in the binary
# (-Wl,--embed-bitcode), which upstream Record-Replay records and the tuner recompiles.
# makefiles/ holds the files the T4 used; this script reproduces all of them except the two
# miniWeather ones (single-rank MPI stub, shorter run; see README.md).
for d in "$@"; do
  [ -f "$d/Makefile.aomp" ] || { echo "$d: no Makefile.aomp"; continue; }
  sed -e 's/amdgcn-amd-amdhsa/nvptx64/g' -e 's/^ARCH *= *gfx[0-9a-f]*/ARCH      = sm_75/' \
      -e 's/-march=\$(ARCH)/-march=$(ARCH) -fopenmp-offload-mandatory -foffload-lto/' \
      "$d/Makefile.aomp" > "$d/Makefile.rr"
  printf '\n# record-replay: keep device bitcode in the binary for kernel extraction\nLDFLAGS += -Wl,--embed-bitcode\n' >> "$d/Makefile.rr"
done
