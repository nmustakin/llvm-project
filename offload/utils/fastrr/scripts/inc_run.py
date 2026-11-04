#!/usr/bin/env python3
"""Online incremental ranker on upstream LLVM (inside rr_current.sif; adapted from
incremental-ranker-2026-10-28/inc_run.py).

usage: inc_run.py --stage stage2 --lane L --order app1,app2,... [--seed S]

A lane tunes the applications in the given order with its own store (reruns/incremental/
lane<L>/store.csv, the pool layout), which holds only the configurations its earlier
searches screened. While the store is empty, an application is tuned with Batch TPE
('cold'; upstream has no profile, so there is no class budget); afterwards with FastRR
(final-wave) trained on the store ('model'). 300 s cap, no cache; a kernel whose confirmed
speedup is below 1 ships the default. Each kernel's final configuration is validated with the
default in 5 randomized rounds. After each application, its screens join the store.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from paths import FASTRR, X  # noqa: E402
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil.DataBase import DB  # noqa: E402
from RRUtil.Search import theoretical_blocks_per_sm, SM_COUNT  # noqa: E402

SCENARIO = dict(MaxThreads=True, MinTeams=True, NumTeams=True, NumThreads=True)
COLS = ['kid', 'app', 'line', 'klass', 'MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams', 'trip',
        'teams_frac', 'speedup', 'regs', 'spill', 'occ', 'waves', 'tail']


def parse_key(key):
    return {k: int(v) for k, v in (p.split('=') for p in key.split('-'))
            if k != 'DeviceID' and v != 'None'}


def store_rows(trace, table, app, line, trip):
    """Screens of one run in the pool layout (no class)."""
    t = pd.read_csv(trace)
    t = t[t.source.isin(['batch', 'cache']) & (t.speedup > 0)].drop_duplicates('key')
    rows = []
    for key, sp in zip(t.key, t.speedup):
        c = parse_key(key)
        mt, mn, nt, teams = c['MaxThreads'], c['MinTeams'], c['NumThreads'], c['NumTeams']
        regs, spill = table[f'{mt},{mn}']
        blocks = theoretical_blocks_per_sm(regs, nt)
        waves = teams / max(1, SM_COUNT * blocks)
        rows.append(dict(kid=f'{app}-{line}', app=app, line=line, klass='none', MaxThreads=mt,
                         MinTeams=mn, NumThreads=nt, NumTeams=teams, trip=trip,
                         teams_frac=teams * nt / trip, speedup=sp, regs=regs, spill=spill,
                         occ=blocks * np.ceil(nt / 32) * 32 / 1024, waves=waves,
                         tail=waves / np.ceil(waves) if waves > 0 else 0.0))
    return rows


def run_app(args, lane, pos, app, store, device):
    registry = json.loads(Path(X, args.stage, 'kernels', f'{app}.json').read_text())
    n_store = store.kid.nunique() if len(store) else 0
    mode = 'model' if n_store >= 1 else 'cold'
    new_rows = []
    for line in sorted(int(x) for x in registry):
        kdir = lane / f'{pos:02d}-{app}' / f'l{line}'
        stem = Upstream.stage(registry[str(line)], kdir / 'work')
        os.chdir(kdir / 'work')
        kernel = Upstream.UpstreamKernel(device, stem, str(kdir))
        ct = kdir / 'compile_table.json'
        if not ct.exists():
            ct.write_text(Path(X, args.stage, 'runs', f'{app}-l{line}', 'compile_table.json').read_text())
        table = json.loads(ct.read_text())
        trip = float(kernel.getTripCount())
        kernel.dbDir = str(kdir / f'{mode}-s{args.seed}')
        os.makedirs(kernel.dbDir, exist_ok=True)
        trace = Path(kernel.dbDir, f'trace_{kernel.HashName}.csv')
        info_f = kdir / 'run.json'
        if not trace.exists():
            t0 = time.time()
            options = dict(wall_budget=300, no_classify=True)
            if mode == 'cold':
                algorithm = 'tpe-batch'
            else:
                algorithm = 'rank-batch'
                options.update(patience=2, confirm='batch', refine=True, wave_teams=True,
                               pool=store.copy(), target_kid=f'{app}-{line}')
            kernel.Search(algorithm, ['speedup'], 60, 15, args.seed, cacheDB=None, device=0,
                          options=options, compileTable=str(ct), **SCENARIO)
            info_f.write_text(json.dumps(dict(lane=args.lane, pos=pos, app=app, line=line,
                                              mode=mode, seed=args.seed, store_kernels=n_store,
                                              store_rows=len(store), end_to_end=time.time() - t0)))
        info = json.loads(info_f.read_text())
        print(f'{app} l{line} {mode} s{args.seed}: {info["end_to_end"]:.0f}s', flush=True)
        db = DB(kernel.getDBName(**SCENARIO), kernel.getDefaultKey())
        shipped = db.MetaData.get('shipped')
        if not shipped:                       # Batch TPE ships the best promotion
            rows = [r for r in db.GetExplorationStats() if r['Valid'] and not r['Baseline']]
            shipped = max(rows, key=lambda r: r['Speedup'])['Configuration'] if rows else None
        t = pd.read_csv(trace)
        conf = t[(t.key == shipped) & t.source.isin(['confirm', 'promote'])].speedup
        conf = float(conf.iloc[-1]) if len(conf) else float('nan')
        final = shipped if shipped and conf >= 1 else kernel.getDefaultKey()
        info.update(shipped=shipped, confirmed=conf, final=final,
                    stop=db.MetaData.get('stop_reason'), search_s=db.MetaData.get('search_wall_s'),
                    screens=int(t.source.isin(['batch', 'cache']).sum()))
        info_f.write_text(json.dumps(info))
        new_rows += store_rows(trace, table, app, line, trip)

        vfile = kdir / 'validation.csv'
        rows = pd.read_csv(vfile).to_dict('records') if vfile.exists() else []
        done_rounds = {r['round'] for r in rows}
        keys = list(dict.fromkeys([kernel.getDefaultKey(), final]))
        rng = random.Random(line * 100 + args.lane)
        for rnd in range(5):
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
    return new_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage2')
    ap.add_argument('--lane', type=int, required=True)
    ap.add_argument('--order', required=True)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    args.seed = args.lane % 3 if args.seed is None else args.seed
    device = Upstream.get_device(0)
    lane = Path(X, 'reruns', 'incremental', f'lane{args.lane}')
    lane.mkdir(parents=True, exist_ok=True)
    store_f = lane / 'store.csv'
    for pos, app in enumerate(args.order.split(',')):
        store = pd.read_csv(store_f) if store_f.exists() else pd.DataFrame(columns=COLS)
        if len(store) and app in set(store.app):
            print(f'{app}: already in the store, skipping', flush=True)
            continue
        new_rows = run_app(args, lane, pos, app, store, device)
        # One write per application, so an interrupted application is redone completely.
        store = pd.concat([store, pd.DataFrame(new_rows, columns=COLS)], ignore_index=True)
        store.to_csv(lane / f'store-after-{pos:02d}-{app}.csv', index=False)
        store.to_csv(lane / 'store.csv.tmp', index=False)
        os.replace(lane / 'store.csv.tmp', store_f)
        print(f'{app}: store now {store.kid.nunique()} kernels, {len(store)} rows', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
