# TODO

- **NVIDIA path: run `opt -O3` before `llc`/`ptxas` in `UpstreamNVIDIA.deviceCompile`
  (RRUtil/Upstream.py).** The bitcode embedded with `-Wl,--embed-bitcode` is the linked
  device module *before* the LTO optimization pipeline (OpenMPOpt, O3). Compiling it with
  `llc` alone gives much slower code: on the MI50 the tuner's default configuration ran up
  to 22x slower than the recorded image (miniFE l169 4.9x) until `UpstreamAMDGPU` ran
  `opt -O3` first. `UpstreamNVIDIA` has the same llc-only pipeline, so the T4 upstream
  (LLVM 24) grids and online results may be affected. Check: compare the replay time of the
  recorded image with the tuner's default build for a few kernels on a T4; if they differ,
  fix the compile and rerun the T4 upstream grids and online runs.
