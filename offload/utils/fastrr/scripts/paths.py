"""Paths of the replication, from the environment env.sh exports (job.sh sources it; for a
script run directly, source env.sh first)."""
import os
from pathlib import Path

# The tuner (RRUtil/), the benchmark sources (benchmarks/) and the pool (data/).
FASTRR = str(Path(__file__).resolve().parents[1])


def _env(name):
    if name not in os.environ:
        raise SystemExit(f'{name} is not set: source {FASTRR}/scripts/env.sh first')
    return os.environ[name]


X = _env('FASTRR_EXP')
REC = _env('FASTRR_REC')
BUILDS = {'stage1': _env('FASTRR_STAGE1_BUILD'), 'stage2': _env('FASTRR_STAGE2_BUILD'),
          'energy': _env('FASTRR_ENERGY_BUILD')}
HEC = _env('FASTRR_HECBENCH')
POOL = _env('FASTRR_POOL')
SIM = _env('FASTRR_SIM')
FIGURES = _env('FASTRR_FIGURES')
