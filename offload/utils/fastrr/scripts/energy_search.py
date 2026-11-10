#!/usr/bin/env python3
"""The conference version's tuner with time and energy objectives on upstream LLVM: BO (or
another strategy) on one or more kernels, every evaluation a full measurement with the kernel
energy per launch over a window of Kernel.EnergyMinMs (RR_ENERGY_MIN_MS; job.sh energy) and,
with RR_UPSTREAM_METRICS=1 on NVIDIA, the CUPTI metrics. Ported from the T4 study's
search24.py (C6-1 and C6-8).

usage: energy_search.py --app lulesh --lines 649,844 --out DIR
                        [--registry FILE.json] [--methods bo-speedup,bo-energy,bo-edp,bo-ed2p]
                        [--seeds 0] [--budget 230] [--init 30] [--no-cache]
                        [--validate-rounds 0]

- The registry (line -> recording JSON) defaults to $FASTRR_EXP/energy/kernels/<app>.json,
  which `record_app.py energy <app> --keep all` writes.
- Output per method and kernel: DIR/<method>/<app>-l<line>/<method>-s<seed>/ (the run's
  database and trace) and validation-shipped.json ({configuration key:
  ['<method>-s<seed>:<objective>']}), the configurations the runs ship: the best measured
  value of each objective of the run.
- --validate-rounds N re-measures the shipped configurations with the default in N randomized
  rounds at the search window (validation.csv). The study validates instead with
  energy_validate.py, which pools the runs of all methods and uses a longer window.
Resumable: finished runs (trace present) and validation rounds are skipped.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import pandas as pd

from paths import FASTRR, X
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil.DataBase import DB  # noqa: E402

METHODS = {  # name -> (algorithm, objectives)
    # The conference tuner's four objectives.
    'bo-speedup': ('bo', ['speedup']),
    'bo-energy': ('bo', ['energy']),
    'bo-edp': ('bo', ['edp']),
    'bo-ed2p': ('bo', ['ed2p']),
    # Other strategies (the search-strategy comparison with energy).
    'tpe-speedup': ('tpe', ['speedup']),
    'random': ('random', ['speedup']),
    'motpe': ('motpe', ['speedup', 'energy']),
}
VALUE = {'speedup': lambda s, e: s, 'energy': lambda s, e: e,
         'edp': lambda s, e: s * e, 'ed2p': lambda s, e: s * s * e}
SCENARIO = dict(MaxThreads=True, MinTeams=True, NumTeams=True, NumThreads=True)


def parse_key(key):
    cfg = {}
    for part in key.split('-'):
        k, v = part.split('=')
        if k != 'DeviceID' and v != 'None':
            cfg[k] = int(v)
    return cfg


def best_keys(db, objectives):
    """Configurations a run ships: the best measured value of each objective."""
    if db.MetaData.get('shipped'):
        return {objectives[0]: db.MetaData['shipped']}
    rows = [r for r in db.GetExplorationStats() if r['Valid'] and not r['Baseline']]
    keys = {}
    for obj in objectives:
        vals = [(VALUE[obj](r.get('Speedup'), r.get('EnergyImprovement')), r) for r in rows]
        good = [(v, r) for v, r in vals if v is not None and v == v]   # drop NaN
        if good:
            keys[obj] = max(good, key=lambda t: t[0])[1]['Configuration']
    return keys


def validate(kernel, kdir, line, shipped, rounds):
    """The shipped configurations and the default in `rounds` randomized rounds at the search
    window (validation.csv)."""
    vfile = kdir / 'validation.csv'
    rows = pd.read_csv(vfile).to_dict('records') if vfile.exists() else []
    done_rounds = {r['round'] for r in rows}
    default = kernel.getDefaultKey()
    keys = [default] + [k for k in shipped if k != default]
    rng = random.Random(int(''.join(c for c in line if c.isdigit())))
    for rnd in range(rounds):
        order = keys[:]
        rng.shuffle(order)
        if rnd in done_rounds:
            continue
        for key in order:
            valid, _, _, dur, en, _, _ = kernel.sample(Tries=1, **parse_key(key))
            rows.append(dict(line=line, round=rnd, key=key, baseline=key == default, valid=valid,
                             duration=dur[0] if valid else None, energy=en[0] if valid else None,
                             time_ratio=getattr(kernel, 'LastMetrics', {}).get('energy_time_ratio')))
        pd.DataFrame(rows).to_csv(vfile, index=False)
        print(f'l{line}: validation round {rnd} done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry')
    ap.add_argument('--app', required=True)
    ap.add_argument('--lines', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', default='bo-speedup,bo-energy,bo-edp,bo-ed2p')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--budget', type=int, default=230)
    ap.add_argument('--init', type=int, default=30)
    ap.add_argument('--validate-rounds', type=int, default=0)
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    registry_file = args.registry or Path(X, 'energy', 'kernels', f'{args.app}.json')
    registry = json.loads(Path(registry_file).read_text())
    device = Upstream.get_device(0)
    seeds = [int(s) for s in args.seeds.split(',')]
    for line in args.lines.split(','):
        for method in args.methods.split(','):
            # One tree per method, so that lanes can search one kernel with several methods
            # at once (each work directory compiles its own binaries).
            kdir = Path(args.out).resolve() / method / f'{args.app}-l{line}'
            stem = Upstream.stage(registry[line], kdir / 'work')
            os.chdir(kdir / 'work')
            kernel = Upstream.UpstreamKernel(device, stem, str(kdir))
            cache = None if args.no_cache else str(kdir / 'cache.json')
            algorithm, objectives = METHODS[method]
            shipped = {}
            for seed in seeds:
                kernel.dbDir = str(kdir / f'{method}-s{seed}')
                os.makedirs(kernel.dbDir, exist_ok=True)
                if Path(kernel.dbDir, f'trace_{kernel.HashName}.csv').exists():
                    print(f'{args.app} l{line} {method} s{seed}: done, skipping', flush=True)
                else:
                    t0 = time.time()
                    kernel.Search(algorithm, objectives, args.budget, args.init, seed,
                                  cacheDB=cache, device=0, options={},
                                  compileTable=str(kdir / 'compile_table.json'), **SCENARIO)
                    print(f'{args.app} l{line} {method} s{seed}: {time.time() - t0:.0f}s',
                          flush=True)
                db = DB(kernel.getDBName(**SCENARIO), kernel.getDefaultKey())
                for obj, key in best_keys(db, objectives).items():
                    shipped.setdefault(key, []).append(f'{method}-s{seed}:{obj}')
            (kdir / 'validation-shipped.json').write_text(json.dumps(shipped, indent=2))
            if args.validate_rounds:
                validate(kernel, kdir, line, shipped, args.validate_rounds)
    print('done', flush=True)


if __name__ == '__main__':
    main()
