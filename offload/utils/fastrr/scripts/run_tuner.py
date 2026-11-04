#!/usr/bin/env python3
"""Online tuning of upstream recordings with the paper's final tuner and Batch TPE
(inside rr_current.sif; adapted from tuner-fixes-2026-10-30/fix_run.py).

usage: run_tuner.py --stage stage1 --app s3d --lines 159,165 --out DIR [--pool loao|loko] ...

Methods (Kernel.Search through RRUtil.Upstream):
  final-wave       FastRR as published: ranking model trained on the pool, no baseline
                   profile, patience 2, refinement, batch confirmation, wave-aligned
                   team counts
  tpe-batch-nocls  Batch TPE: timing-only baseline, constant-liar TPE batches of 16,
                   the 5 best screens promoted to full measurements
The other methods (ablations, MF screening, and conventional strategies with 60 full
evaluations) are listed in METHODS. Every batch or MF run has a 300 s wall-clock cap and no
measurement cache (--cache shares one between the runs of a kernel). After the runs, every
shipped configuration and the default are re-measured in randomized rounds
(validation.csv); the true speedup is the median paired ratio.
Resumable: finished runs (trace present) and validation rounds are skipped.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import pandas as pd

from paths import FASTRR, X, POOL  # noqa: E402
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil.DataBase import DB  # noqa: E402

# Application names of the pool's kernel ids.
POOL_APP = {'lulesh': 'LULESH', 'minife': 'MINIFE', 's3d': 'S3D', 'minisweep': 'MINISWEEP',
            'miniWeather': 'MINIWEATHER', 'amgmk': 'AMGMK', 'simplemoc': 'SIMPLEMOC',
            'su3': 'SU3', 'xsbench': 'XSBENCH', 'rsbench': 'RSBENCH', 'haccmk': 'HACCMK',
            # second inputs of the input study
            's3d-s3': 'S3D', 'su3-l24': 'SU3', 'simplemoc-s1500k': 'SIMPLEMOC',
            'minisweep-c884': 'MINISWEEP', 'miniWeather-1600': 'MINIWEATHER', 'minife-150': 'MINIFE'}
FINAL = dict(no_classify=True, patience=2, confirm='batch', refine=True)
METHODS = {  # name -> (algorithm, options)
    'final-wave': ('rank-batch', dict(FINAL, wave_teams=True)),
    'tpe-batch-nocls': ('tpe-batch', dict(no_classify=True)),
    # Ablations of FastRR: without wave-aligned team counts, without refinement, with two
    # promotions to full measurements instead of batch confirmation, on several GPUs.
    'final': ('rank-batch', dict(FINAL)),
    'final-wave-norefine': ('rank-batch', dict(FINAL, wave_teams=True, refine=False)),
    'final-wave-promote': ('rank-batch', dict(no_classify=True, patience=2, refine=True,
                                              wave_teams=True)),
    'final-wave-4gpu': ('rank-batch', dict(FINAL, wave_teams=True)),
    # Multi-fidelity screening: one timing process per proposal, 5 promotions.
    'tpe-mf': ('tpe-mf', {}),
}
# Conventional strategies: 60 full evaluations (two timing processes each), 15 initial samples
# for the model-based ones, no wall-clock cap.
CONVENTIONAL = {'bo-60': 'bo', 'tpe-60': 'tpe', 'random-60': 'random'}
METHODS.update({m: (a, {}) for m, a in CONVENTIONAL.items()})
SCENARIO = dict(MaxThreads=True, MinTeams=True, NumTeams=True, NumThreads=True)


def parse_key(key):
    cfg = {}
    for part in key.split('-'):
        k, v = part.split('=')
        if k != 'DeviceID' and v != 'None':
            cfg[k] = int(v)
    return cfg


def best_keys(db):
    if db.MetaData.get('shipped'):
        return [db.MetaData['shipped']]
    rows = [r for r in db.GetExplorationStats() if r['Valid'] and not r['Baseline']]
    return [max(rows, key=lambda r: r['Speedup'])['Configuration']] if rows else []


def target_kid(pool, app, line):
    kids = pool[(pool.app == POOL_APP.get(app, app.upper())) & (pool.line == line)].kid.unique()
    return kids[0] if len(kids) else f'{POOL_APP.get(app, app.upper())}-{line}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--app', required=True)
    ap.add_argument('--lines', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', default='final-wave,tpe-batch-nocls')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--wall-budget', type=float, default=300)
    ap.add_argument('--pool', choices=('loao', 'loko'), default='loao',
                    help='leave the target application (loao) or kernel (loko) out')
    ap.add_argument('--validate-rounds', type=int, default=5)
    ap.add_argument('--validation-name', default='validation.csv')
    ap.add_argument('--devices', type=int, default=1,
                    help='split each batch over this many visible GPUs (final-wave-4gpu)')
    ap.add_argument('--cache', action='store_true',
                    help='share a measurement cache between the runs of a kernel')
    args = ap.parse_args()

    device = Upstream.get_device(0)
    registry = json.loads(Path(X, args.stage, 'kernels', f'{args.app}.json').read_text())
    pool_all = pd.read_csv(POOL)
    pool = pool_all[pool_all.app != POOL_APP.get(args.app, args.app.upper())] \
        if args.pool == 'loao' else pool_all

    for line in [int(x) for x in args.lines.split(',')]:
        kdir = Path(args.out) / f'{args.app}-l{line}'
        work = kdir / 'work'
        stem = Upstream.stage(registry[str(line)], work)
        os.chdir(work)
        kernel = Upstream.UpstreamKernel(device, stem, str(kdir))
        kid = target_kid(pool_all, args.app, line)
        compile_table = str(kdir / 'compile_table.json')
        ref_table = Path(X, args.stage, 'runs', f'{args.app}-l{line}', 'compile_table.json')
        if not Path(compile_table).exists() and ref_table.exists():
            Path(compile_table).write_text(ref_table.read_text())
        if not Path(compile_table).exists():          # not charged to the searches
            from RRUtil.Search import build_compile_table
            t0 = time.time()
            build_compile_table(kernel, compile_table)
            print(f'{args.app} l{line}: compile table in {time.time() - t0:.0f}s', flush=True)
        for seed in [int(s) for s in args.seeds.split(',')]:
            for method in args.methods.split(','):
                algorithm, opts = METHODS[method]
                kernel.dbDir = str(kdir / f'{method}-s{seed}')
                os.makedirs(kernel.dbDir, exist_ok=True)
                if Path(kernel.dbDir, f'trace_{kernel.HashName}.csv').exists():
                    print(f'{args.app} l{line} {method} s{seed}: done, skipping', flush=True)
                    continue
                options = dict(opts)
                if method not in CONVENTIONAL:
                    options['wall_budget'] = args.wall_budget
                if algorithm == 'rank-batch':
                    options.update(pool=pool, target_kid=kid)
                if args.devices > 1:
                    options['devices'] = list(range(args.devices))
                cache = str(kdir / 'cache.json') if args.cache else None
                t0 = time.time()
                kernel.Search(algorithm, ['speedup'], 60, 15, seed, cacheDB=cache, device=0,
                              options=options, compileTable=compile_table, **SCENARIO)
                print(f'{args.app} l{line} {method} s{seed}: {time.time() - t0:.0f}s', flush=True)

        # Ground truth: every shipped configuration and the default, in randomized rounds.
        shipped = {}
        for seed in [int(s) for s in args.seeds.split(',')]:
            for method in args.methods.split(','):
                kernel.dbDir = str(kdir / f'{method}-s{seed}')
                db = DB(kernel.getDBName(**SCENARIO), kernel.getDefaultKey())
                for key in best_keys(db):
                    shipped.setdefault(key, []).append(f'{method}-s{seed}')
        vfile = kdir / args.validation_name
        Path(str(vfile).replace('.csv', '-shipped.json')).write_text(json.dumps(shipped, indent=2))
        rows = pd.read_csv(vfile).to_dict('records') if vfile.exists() else []
        done_rounds = {r['round'] for r in rows}
        keys = [kernel.getDefaultKey()] + list(shipped)
        rng = random.Random(line)
        for rnd in range(args.validate_rounds):
            order = keys[:]
            rng.shuffle(order)
            if rnd in done_rounds:
                continue
            for key in order:
                valid, _, _, dur, _, _, _ = kernel.sample(Tries=1, **parse_key(key))
                rows.append(dict(line=line, round=rnd, key=key,
                                 baseline=key == kernel.getDefaultKey(), valid=valid,
                                 duration=dur[0] if valid else None))
            pd.DataFrame(rows).to_csv(vfile, index=False)
            print(f'{args.app} l{line}: validation round {rnd} done', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
