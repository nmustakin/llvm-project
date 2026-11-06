#!/usr/bin/env python3
"""Classification study on AMD GPUs with upstream LLVM (the MI50 rows of the paper's
classification results; after ~/rr-amd-work/calib/landscape.py of the LLVM 16 study).

usage: classify_study.py --stage stage2 --kernels lulesh:649,s3d:159,... [--out DIR]

For every kernel, with its grid dataset (grids/<app>-<line>.csv, grid.py) as the landscape:
- three baseline profiles of the default configuration with rocprof (UpstreamKernel.
  profile_metrics, with about 100 ms of launches), timing each, which is the cost of the
  tuner's baseline profile;
- one profile of the best configuration of the grid;
- a validated speedup of that configuration: five rounds of one timing process of the default
  and one of the best configuration, alternating, and the median of the paired ratios.
Writes <out>/<app>-<line>.summary.json in the layout of the LLVM 16 study, which
~/rr-amd-work/calib/analyze.py and compare_rules.py read (labels and rule agreement).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from paths import FASTRR, X
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402


def median_metrics(samples):
    keys = set().union(*samples) if samples else set()
    return {k: float(np.median([s[k] for s in samples if k in s])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage2')
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--rounds', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out or Path(X, 'classify'))
    out.mkdir(parents=True, exist_ok=True)
    device = Upstream.get_device(0)
    for item in args.kernels.split(','):
        app, line = item.split(':')
        summary = out / f'{app}-{line}.summary.json'
        if summary.exists():
            print(f'{app} l{line}: done, skipping', flush=True)
            continue
        registry = json.loads(Path(X, args.stage, 'kernels', f'{app}.json').read_text())
        kdir = out / 'work' / f'{app}-l{line}'
        stem = Upstream.stage(registry[line], kdir / 'work')
        os.chdir(kdir / 'work')
        kernel = Upstream.UpstreamKernel(device, stem, str(kdir))
        grid = pd.read_csv(Path(X, 'grids', f'{app}-{line}.csv'))
        grid = grid[grid.valid.astype(bool) & (grid.speedup > 0)]
        best = grid.loc[grid.speedup.idxmax()]
        best_cfg = {k: int(best[k]) for k in ('MaxThreads', 'MinTeams', 'NumThreads',
                                              'NumTeams')}
        # Launches per profile: about the 100 ms warm-up of a timing replay (3 to 2000).
        meta = json.loads(Path(X, 'grids', f'{app}-{line}.meta.json').read_text())
        launches = int(min(2000, max(3, np.ceil(0.1 / meta['baseline_batch_s']))))

        def compile_(cfg=None):
            lb = {k: cfg[k] for k in ('MaxThreads', 'MinTeams')} if cfg else {}
            stats = device.deviceCompile(kernel.original, kernel.Name, kernel.HashName, **lb)
            s = stats[kernel.Name]
            return int(s[0]), int(s[3] + s[4])

        t0 = time.time()
        default_bin = compile_()
        base, seconds = [], []
        for _ in range(3):
            m, sec = kernel.profile_metrics(launches=launches)
            if m:
                m['compile_spill_bytes'] = float(default_bin[1])
                base.append(m)
            seconds.append(sec)
        best_bin = compile_(best_cfg)
        best_m, _ = kernel.profile_metrics(best_cfg['NumThreads'], best_cfg['NumTeams'],
                                           launches=launches)
        if best_m:
            best_m['compile_spill_bytes'] = float(best_bin[1])
        ratios = []
        for _ in range(args.rounds):
            compile_()
            td, _ = kernel._replay_batch([{}], kernel.TimingWarmupMs)
            compile_(best_cfg)
            tb, _ = kernel._replay_batch([best_cfg], kernel.TimingWarmupMs)
            if 0 in td and 0 in tb:
                ratios.append(td[0] / tb[0])
        validated = float(np.median(ratios)) if ratios else float('nan')
        s = dict(app=app, line=int(line), kernel=kernel.Name,
                 default_binary=list(default_bin),
                 baseline_metrics=median_metrics(base),
                 baseline_full=dict(metrics=median_metrics(base)),
                 best_full=dict(metrics=best_m or {}),
                 best_screen=dict(best_cfg, regs=int(best.regs), spill=int(best.spill),
                                  speedup=float(best.speedup)),
                 validated_speedup=validated, validation_ratios=ratios,
                 profile_seconds=seconds, profile_launches=launches,
                 seconds=time.time() - t0)
        summary.write_text(json.dumps(s, indent=1))
        print(f'{app} l{line}: default {default_bin}, best {best_bin} {best_cfg}, grid '
              f'{best.speedup:.3f}x, validated {validated:.3f}x, profile '
              f'{np.mean(seconds):.1f} s, metrics {len(base)}/3', flush=True)


if __name__ == '__main__':
    main()
