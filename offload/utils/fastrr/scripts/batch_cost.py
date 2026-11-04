#!/usr/bin/env python3
"""Cost of evaluating a candidate configuration (paper section 5.1), inside rr_current.sif.

usage: batch_cost.py --stage stage1 --kernels lulesh:649,lulesh:844,minife:290 --out FILE

For each kernel, 16 launch configurations of the default binary are timed
(warm-up and median of 10 launches, as in the tuner) three ways, twice each:
  single   one replay process per configuration (1 s warm-up each)
  batch    one replay process for all 16, restoring the recorded memory from a
           device copy (1 s warm-up, then 100 ms before each later configuration)
  host     the same batch, restoring the recorded memory from the host
The wall time per configuration and every configuration's median kernel time go
to the CSV, so that batch and single-process timings can be compared.
"""
import argparse, json, os, sys, time
from pathlib import Path

import pandas as pd

from paths import FASTRR, X  # noqa: E402
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=2)
    args = ap.parse_args()
    device = Upstream.get_device(0)
    rows = []
    for item in args.kernels.split(','):
        app, line = item.split(':')
        registry = json.loads(Path(X, args.stage, 'kernels', f'{app}.json').read_text())
        work = Path(args.out).parent / 'batch-cost-work' / f'{app}-l{line}'
        stem = Upstream.stage(registry[line], work)
        os.chdir(work)
        kernel = Upstream.UpstreamKernel(device, stem, str(work))
        trip = kernel.getTripCount()
        device.deviceCompile(kernel.original, kernel.Name, stem, MaxThreads=1024)
        configs = [dict(NumThreads=nt, NumTeams=max(1, int(trip / nt * frac)))
                   for nt in (64, 128, 256, 512) for frac in (1 / 16, 1 / 4, 1 / 2, 1)]
        for rnd in range(args.rounds):
            for mode in ('single', 'batch', 'host'):
                t0 = time.perf_counter()
                if mode == 'single':
                    times = {}
                    for i, c in enumerate(configs):
                        t, _ = kernel._replay_batch([c], kernel.TimingWarmupMs)
                        if 0 in t:
                            times[i] = t[0]
                else:
                    times, _ = kernel._replay_batch(configs, kernel.TimingWarmupMs,
                                                    restore_from_device=mode == 'batch')
                wall = time.perf_counter() - t0
                for i, c in enumerate(configs):
                    rows.append(dict(app=app, line=line, round=rnd, mode=mode, index=i,
                                     threads=c['NumThreads'], teams=c['NumTeams'],
                                     kernel_s=times.get(i), wall_per_config_s=wall / len(configs)))
                print(f'{app} l{line} round {rnd} {mode}: {wall / len(configs):.3f} s per '
                      f'configuration, {len(times)}/{len(configs)} timed', flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
