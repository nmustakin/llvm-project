#!/usr/bin/env python3
"""Grid datasets on upstream LLVM (inside rr_current.sif; after hecbench-2026-10-13/landscape_hec.py).

usage: grid.py --stage stage2 --kernels lulesh:649,s3d:159,...

The paper's grid: one launch bound per distinct binary (the largest MaxThreads among the pairs
with the same registers and spills), every multiple of the warp size (32 threads; 64 on AMD
GPUs) up to it that fits the register file, and 16 teams
fractions spaced logarithmically in [1/256, 1], timed in batch replay (chunks of 160) with the
tuner's protocol. The default is timed in batch mode at the start and the end, and speedups are
relative to their mean. Output: grids/<app>-<line>.csv and .meta.json.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from paths import FASTRR, X  # noqa: E402
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil.Search import build_compile_table  # noqa: E402

FRACS = np.geomspace(1 / 256, 1.0, 16)
CHUNK = 160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--kernels', required=True)
    args = ap.parse_args()
    device = Upstream.get_device(0)
    os.makedirs(f'{X}/grids', exist_ok=True)
    for item in args.kernels.split(','):
        app, line = item.split(':')
        out = f'{X}/grids/{app}-{line}.csv'
        if os.path.exists(out):
            print(f'{app} l{line}: exists, skipping', flush=True)
            continue
        registry = json.loads(Path(X, args.stage, 'kernels', f'{app}.json').read_text())
        kdir = Path(X, 'grids', 'work', f'{app}-l{line}')
        stem = Upstream.stage(registry[line], kdir / 'work')
        os.chdir(kdir / 'work')
        kernel = Upstream.UpstreamKernel(device, stem, str(kdir))
        ct = kdir / 'compile_table.json'
        ref = Path(X, args.stage, 'runs', f'{app}-l{line}', 'compile_table.json')
        if not ct.exists() and ref.exists():
            ct.write_text(ref.read_text())
        t0 = time.time()
        table = build_compile_table(kernel, str(ct))
        trip = kernel.getTripCount()
        reps = {}
        for (mt, mn), binary in table.items():
            if binary not in reps or mt > reps[binary][0]:
                reps[binary] = (mt, mn)
        base = [kernel.sample_batch([dict(NumThreads=None, NumTeams=None)])[0]]
        rows = []
        for (regs, spill), (mt, mn) in sorted(reps.items()):
            cfgs, seen = [], set()
            step = kernel.Profiler.WarpSize
            for nt in range(step, mt + 1, step):
                if not kernel.Profiler.blockFits(regs, nt):
                    continue
                for fr in FRACS:
                    teams = max(1, int(math.ceil(trip * fr / nt)))
                    if (nt, teams) not in seen:
                        seen.add((nt, teams))
                        cfgs.append(dict(MaxThreads=mt, MinTeams=mn, NumThreads=nt, NumTeams=teams))
            for i in range(0, len(cfgs), CHUNK):
                chunk = cfgs[i:i + CHUNK]
                for c, (valid, dur) in zip(chunk, kernel.sample_batch(chunk)):
                    rows.append(dict(app=app, line=int(line), regs=regs, spill=spill, **c,
                                     teams_frac=c['NumTeams'] * c['NumThreads'] / trip,
                                     valid=valid, duration=dur))
            print(f'{app} l{line}: binary regs={regs} spill={spill}: {len(cfgs)} configs, '
                  f'{time.time() - t0:.0f}s so far', flush=True)
        base.append(kernel.sample_batch([dict(NumThreads=None, NumTeams=None)])[0])
        bdur = float(np.mean([d for v, d in base if v]))
        df = pd.DataFrame(rows)
        df['baseline_duration'] = bdur
        df['speedup'] = np.where(df.valid, bdur / df.duration, np.nan)
        df.to_csv(out, index=False)
        json.dump(dict(app=app, line=int(line), kernel=kernel.Name, trip=trip,
                       baseline_batch_s=bdur, binaries=len(reps), configs=len(df),
                       baseline_drift=base[0][1] / base[1][1] if base[0][0] and base[1][0] else None,
                       seconds=time.time() - t0),
                  open(out.replace('.csv', '.meta.json'), 'w'), indent=1)
        print(f'{app} l{line}: done, {len(df)} configs ({df.valid.mean():.1%} valid), best '
              f'{df.speedup.max():.3f}x, {time.time() - t0:.0f}s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
