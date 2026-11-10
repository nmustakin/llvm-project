#!/bin/bash
# usage: install.sh [HeCBench src directory]   (default $FASTRR_HECBENCH from scripts/env.sh)
# Prepares a HeCBench checkout (github.com/zjin-lcf/HeCBench, commit f0c817a, src/) for the T4
# experiments, as on the original machine:
#  1. copies makefiles/<app>/Makefile.rr into <src>/<app>/ (225 applications);
#  2. creates miniWeather-omp-1600, the second miniWeather input, as a copy of miniWeather-omp;
#  3. applies rsbench-map-data.patch, which maps the SimulationData structure in RSBench's
#     event kernel (upstream LLVM fails to record it otherwise).
# Idempotent. record_app.py then builds each application with `make -f Makefile.rr`.
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
S=${1:-$FASTRR_HECBENCH}; H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -d "$S" ] || { echo "no HeCBench src directory $S"; exit 1; }
if [ ! -d "$S/miniWeather-omp-1600" ] && [ -d "$S/miniWeather-omp" ]; then
  cp -r "$S/miniWeather-omp" "$S/miniWeather-omp-1600"
  rm -f "$S/miniWeather-omp-1600"/*.o "$S/miniWeather-omp-1600/main"
fi
n=0; missing=0
for m in "$H"/makefiles/*/Makefile.rr; do
  app=$(basename "$(dirname "$m")")
  if [ -d "$S/$app" ]; then cp "$m" "$S/$app/Makefile.rr"; n=$((n+1)); else missing=$((missing+1)); fi
done
echo "Makefile.rr installed for $n applications ($missing not in $S)"
if (cd "$S" && patch -p1 -N --dry-run -s < "$H/rsbench-map-data.patch" >/dev/null 2>&1); then
  (cd "$S" && patch -p1 -N -s < "$H/rsbench-map-data.patch") && echo "rsbench: patch applied"
else
  echo "rsbench: patch already applied (or the source differs)"
fi
