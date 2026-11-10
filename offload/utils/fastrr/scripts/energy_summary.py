#!/usr/bin/env python3
"""Summary of the energy study: per run, the validated speedup and energy improvement of its
shipped configuration, and per method their geometric means over the kernels (the paper's
"four objectives give 1.15x in energy and 1.15x to 1.17x in speed" on the T4).

usage: energy_summary.py --runs DIR --validation DIR [--window 3000] [--out FILE.csv]

- Validated ratio of a shipped configuration: the default's median duration (energy) over the
  configuration's median, over the validation rounds at --window (energy_validate.py).
- Also the in-search search time per run (from the run's trace), the number of runs whose
  validated energy improvement is below 0.97, and the median launch-mode time ratio of the
  shipped configurations (far from 1: the kernel changes on its own outputs in the window).
"""
import argparse, glob, json, math, os
from pathlib import Path

import numpy as np
import pandas as pd


def gmean(a):
    a = [x for x in a if x and x == x and x > 0]
    return math.exp(sum(map(math.log, a)) / len(a)) if a else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--validation', required=True)
    ap.add_argument('--window', type=int, default=3000)
    ap.add_argument('--out')
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f'{a.validation}/*.csv')):
        v = pd.read_csv(f)
        v = v[(v.window_ms == a.window) & v.valid]
        if v.empty:
            continue
        kd = v.kernel.iloc[0]
        base = v[v.roles.str.startswith('default')]
        bt, be = base.duration.median(), base.energy.median()
        shipped = {}
        for sf in sorted(Path(a.runs).glob(f'*/{kd}/validation-shipped.json')):
            for key, tags in json.loads(sf.read_text()).items():
                shipped.setdefault(key, []).extend(tags)
        for key, tags in shipped.items():
            s = v[v.key == key]
            for tag in tags:
                run, obj = tag.split(':')
                method, seed = run.rsplit('-s', 1)
                trace = glob.glob(f'{a.runs}/{method}/{kd}/{run}/trace_*.csv')
                # The trace's wall column is the run's cumulative wall time.
                wall = float(pd.read_csv(trace[0]).wall.max()) if trace else None
                rows.append(dict(kernel=kd, method=method, seed=int(seed), objective=obj, key=key,
                                 rounds=len(s), speedup=bt / s.duration.median(),
                                 energy_gain=be / s.energy.median(),
                                 time_ratio=s.time_ratio.median(), search_s=wall))
    df = pd.DataFrame(rows)
    if df.empty:
        print('no validated runs'); return
    if a.out:
        df.to_csv(a.out, index=False)
    print(f'validation window {a.window} ms, {df.kernel.nunique()} kernels, {len(df)} runs')
    for m, g in df.groupby('method'):
        print(f'  {m:12s} runs {len(g):3d}  speedup gmean {gmean(g.speedup):.3f}  '
              f'energy gain gmean {gmean(g.energy_gain):.3f}  energy gain < 0.97: '
              f'{int((g.energy_gain < 0.97).sum())}  time_ratio median {g.time_ratio.median():.2f}'
              + (f'  search median {g.search_s.median():.0f} s' if g.search_s.notna().any() else ''))
    print(f'  all          energy gain < 0.97: {int((df.energy_gain < 0.97).sum())} of {len(df)}')


if __name__ == '__main__':
    main()
