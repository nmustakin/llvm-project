#!/usr/bin/env python3
"""Validation of the energy study: the default and every configuration that a run of
energy_search.py shipped, re-measured in randomized rounds with long energy windows. Ported
from the T4 study's e_repeat.py (C6-2), which found that kernel energy over a 300 ms window
varies between repetitions by a median of 14.3% on the T4, against 5.8% at 1,000 ms and 3.5%
at 3,000 ms, so short windows cannot validate energy.

usage: energy_validate.py --runs DIR --out DIR --make-queue
       energy_validate.py --runs DIR --out DIR --lane [--rounds 10] [--windows 3000]

- --runs: the output directory of energy_search.py
  (<method>/<app>-l<line>/validation-shipped.json); the shipped configurations of all methods
  of a kernel are validated together, in the same rounds.
- --make-queue lists the kernels of --runs in OUT/queue.txt; each --lane process pops kernels
  from it under a lock, so one lane per GPU splits the kernels.
- Per kernel and round, every key (the default and the shipped configurations of all methods
  and seeds) is measured once at every window of --windows (ms), in random order, each in its
  own timing process (Tries=1), without metrics.
Output: OUT/<app>-l<line>.csv, one row per measurement (resumable: finished rounds are
skipped). energy_summary.py computes the validated ratios.
"""
import argparse, fcntl, json, os, random, sys
from pathlib import Path

import pandas as pd

from paths import FASTRR, X
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil.Benchmark import Kernel  # noqa: E402


def parse_key(key):
    cfg = {}
    for part in key.split('-'):
        k, v = part.split('=')
        if k != 'DeviceID' and v != 'None':
            cfg[k] = int(v)
    return cfg


def pop(out):
    with open(out / 'queue.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = out / 'queue.txt'
        lines = queue.read_text().splitlines()
        if not lines:
            return None
        queue.write_text('\n'.join(lines[1:]) + ('\n' if lines[1:] else ''))
        return lines[0]


def run_kernel(runs, out, kd, rounds, windows, registry_dir):
    app, line = kd.rsplit('-l', 1)
    roles = {}
    for f in sorted(Path(runs).glob(f'*/{kd}/validation-shipped.json')):
        for key, tags in json.loads(f.read_text()).items():
            roles.setdefault(key, []).extend(tags)
    registry = json.loads(Path(registry_dir, f'{app}.json').read_text())
    work = out / kd / 'work'
    stem = Upstream.stage(registry[line], work)
    os.chdir(work)
    kernel = Upstream.UpstreamKernel(Upstream.get_device(0), stem, str(out / kd))
    default = kernel.getDefaultKey()
    keys = [default] + [k for k in roles if k != default]
    roles.setdefault(default, []).insert(0, 'default')
    csv = out / f'{kd}.csv'
    rows = pd.read_csv(csv).to_dict('records') if csv.exists() else []
    done = {r['round'] for r in rows}
    rng = random.Random(int(''.join(c for c in line if c.isdigit())))
    for rnd in range(rounds):
        order = [(k, w) for k in keys for w in windows]
        rng.shuffle(order)
        if rnd in done:
            continue
        for key, w in order:
            Kernel.EnergyMinMs = w
            valid, _, _, dur, en, _, _ = kernel.sample(Tries=1, **parse_key(key))
            lm = getattr(kernel, 'LastMetrics', {}) or {}
            rows.append(dict(kernel=kd, round=rnd, window_ms=w, key=key,
                             roles='+'.join(roles[key]), valid=valid,
                             duration=dur[0] if valid else None,
                             energy=en[0] if valid else None,
                             time_ratio=lm.get('energy_time_ratio'),
                             busy_fraction=lm.get('energy_busy_fraction'),
                             sm_clock_mhz=lm.get('sm_clock_mhz')))
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f'{kd}: round {rnd} done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--registry-dir', default=str(Path(X, 'energy', 'kernels')))
    ap.add_argument('--make-queue', action='store_true')
    ap.add_argument('--lane', action='store_true')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--windows', default='3000')
    a = ap.parse_args()
    runs, out = Path(a.runs).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.make_queue:
        kernels = sorted({p.parent.name for p in runs.glob('*/*/validation-shipped.json')})
        (out / 'queue.txt').write_text('\n'.join(kernels) + '\n')
        print(len(kernels), 'kernels')
        return
    if not a.lane:
        ap.error('--make-queue or --lane')
    # Energy only: no metrics process.
    Kernel.SkipMetrics = True
    Upstream.UpstreamKernel.SkipMetrics = True
    windows = [int(w) for w in a.windows.split(',')]
    while True:
        kd = pop(out)
        if not kd:
            break
        run_kernel(runs, out, kd, a.rounds, windows, a.registry_dir)
    print('done', flush=True)


if __name__ == '__main__':
    main()
