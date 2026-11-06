#!/usr/bin/env python3
"""Training pool of the ranking model from grid datasets (grid.py output).

Reads <grids>/<app>-<line>.csv and .meta.json, keeps the valid configurations with a
positive speedup, and writes them in the layout of data/pool_landscapes.csv: kernel id
(<APP>-<line>, the application names of run_tuner.POOL_APP), the launch configuration,
the loop trip count, the measured speedup, and the free features of
Search.compile_features (registers, spills, theoretical occupancy, waves, tail) for the
current device (FASTRR_TARGET; on AMD the MI50 limits).

Without --classes, every kernel gets class 'none' (the final tuner blanks the pool's
classes anyway, no_classify). --classes takes a CSV with columns app,line,klass.

usage: make_pool.py --grids $FASTRR_EXP/grids --out POOL.csv [--classes CSV]
"""
import argparse, glob, json, sys
from pathlib import Path

import pandas as pd

from paths import FASTRR
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil import Search  # noqa: E402
from run_tuner import POOL_APP  # noqa: E402

COLS = ['kid', 'app', 'bench', 'line', 'klass', 'MaxThreads', 'MinTeams', 'NumThreads',
        'NumTeams', 'trip', 'teams_frac', 'speedup', 'regs', 'spill', 'occ', 'waves', 'tail']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes')
    args = ap.parse_args()
    Search.set_device(Upstream.get_device(0))
    classes = {}
    if args.classes:
        c = pd.read_csv(args.classes)
        classes = {(a, int(l)): k for a, l, k in zip(c.app, c.line, c.klass)}

    frames = []
    for path in sorted(glob.glob(f'{args.grids}/*.csv')):
        meta = json.loads(Path(path[:-len('.csv')] + '.meta.json').read_text())
        app, line, trip = meta['app'], int(meta['line']), meta['trip']
        d = pd.read_csv(path)
        d = d[d.valid.astype(bool) & (d.speedup > 0)].copy()
        name = POOL_APP.get(app, app.upper())
        feats = []
        for r in d.itertuples():
            cfg = dict(MaxThreads=r.MaxThreads, MinTeams=r.MinTeams, NumThreads=r.NumThreads,
                       NumTeams=r.NumTeams)
            f, _ = Search.compile_features(cfg, {(r.MaxThreads, r.MinTeams): (r.regs, r.spill)},
                                           trip)
            feats.append((f[6], 2 ** f[7], f[8]))
        d['occ'], d['waves'], d['tail'] = zip(*feats) if feats else ([], [], [])
        d['kid'], d['app'], d['bench'] = f'{name}-{line}', name, name
        d['trip'], d['klass'] = float(trip), classes.get((app, line), 'none')
        frames.append(d[COLS])
        print(f'{name}-{line}: {len(d)} configurations, best {d.speedup.max():.3f}')
    pool = pd.concat(frames, ignore_index=True)
    pool.to_csv(args.out, index=False)
    print(f'{args.out}: {len(pool)} rows, {pool.kid.nunique()} kernels, '
          f'{pool.app.nunique()} applications')


if __name__ == '__main__':
    main()
