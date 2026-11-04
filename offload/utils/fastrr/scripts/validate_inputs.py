#!/usr/bin/env python3
"""Input sensitivity of tuned configurations (inside rr_current.sif; after
inputs-2026-10-14/input_study.py).

usage: validate_inputs.py --app-a s3d --app-b s3d-s3 --lines 159,165

At the second input B, in the same 5 randomized rounds as B's default, validates:
  transfer  the configurations shipped at input A (FastRR and Batch TPE, three seeds each,
            stage2/runs/<A>-l<line>/validation-ref-shipped.json), unchanged
  scaled    the same with the number of teams rescaled to keep the teams fraction
            (teams * threads / trip count) at B's trip count
  retuned   the configurations FastRR shipped when retuned at B
            (reruns/inputs/runs/<B>-l<line>, run_tuner.py)
Output: reruns/inputs/<B>-l<line>/validation.csv and configs.json.
"""
import argparse, json, os, random, sys
from pathlib import Path

import pandas as pd

from paths import FASTRR, X  # noqa: E402
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402


def parse_key(key):
    return {k: int(v) for k, v in (p.split('=') for p in key.split('-'))
            if k != 'DeviceID' and v != 'None'}


def key_of(c):
    return '-'.join(f'{k}={c[k]}' for k in ('MaxThreads', 'MinTeams', 'NumThreads', 'NumTeams')) + '-DeviceID=0'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage2')
    ap.add_argument('--app-a', required=True)
    ap.add_argument('--app-b', required=True)
    ap.add_argument('--lines', required=True)
    args = ap.parse_args()
    device = Upstream.get_device(0)
    reg_a = json.loads(Path(X, args.stage, 'kernels', f'{args.app_a}.json').read_text())
    reg_b = json.loads(Path(X, args.stage, 'kernels', f'{args.app_b}.json').read_text())
    for line in args.lines.split(','):
        trip_a = json.loads(Path(reg_a[line]).read_text())['LoopTripCount']
        kdir = Path(X, 'reruns', 'inputs', f'{args.app_b}-l{line}')
        stem = Upstream.stage(reg_b[line], kdir / 'work')
        os.chdir(kdir / 'work')
        kernel = Upstream.UpstreamKernel(device, stem, str(kdir))
        trip_b = kernel.getTripCount()
        who = {}
        shipped_a = json.loads(Path(X, args.stage, 'runs', f'{args.app_a}-l{line}',
                                    'validation-ref-shipped.json').read_text())
        for key, runs in shipped_a.items():
            c = parse_key(key)
            for r in runs:
                who.setdefault(key, []).append(f'transfer:{r}')
                if 'NumTeams' in c:
                    s = dict(c, NumTeams=max(1, round(c['NumTeams'] * trip_b / trip_a)))
                    who.setdefault(key_of(s), []).append(f'scaled:{r}')
        retuned = json.loads(Path(X, 'reruns', 'inputs', 'runs', f'{args.app_b}-l{line}',
                                  'validation-shipped.json').read_text())
        for key, runs in retuned.items():
            for r in runs:
                who.setdefault(key, []).append(f'retuned:{r}')
        (kdir / 'configs.json').write_text(json.dumps(dict(trip_a=trip_a, trip_b=trip_b, who=who),
                                                      indent=1))
        vfile = kdir / 'validation.csv'
        rows = pd.read_csv(vfile).to_dict('records') if vfile.exists() else []
        done = {r['round'] for r in rows}
        keys = [kernel.getDefaultKey()] + [k for k in who if k != kernel.getDefaultKey()]
        rng = random.Random(int(line))
        for rnd in range(5):
            order = keys[:]
            rng.shuffle(order)
            if rnd in done:
                continue
            for key in order:
                valid, _, _, dur, _, _, _ = kernel.sample(Tries=1, **parse_key(key))
                rows.append(dict(line=int(line), round=rnd, key=key,
                                 baseline=key == kernel.getDefaultKey(), valid=valid,
                                 duration=dur[0] if valid else None))
            pd.DataFrame(rows).to_csv(vfile, index=False)
        print(f'{args.app_b} l{line}: validated {len(keys)} configurations', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
