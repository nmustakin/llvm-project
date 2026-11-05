#!/usr/bin/env python3
"""Measurement noise of the replay timing protocol (the paper's "median CV").

For every kept kernel of a stage (<stage>/kernels/*.json), three launch configurations are
timed in --rounds randomized rounds, each measurement in its own replay process with the
tuner's protocol: a warm-up of the device's TimingWarmupMs (Kernel.TimingWarmupMs on NVIDIA),
the median of Kernel.TimingRepeats launches timed with device events, memory restored from a
device copy before each launch, and RR_UPSTREAM_REPLAY_FLAGS (--relocate for stage 2).

The configurations are the recorded default, 64-thread teams and 1024-thread teams (teams
covering the loop trip count). A configuration the binary cannot launch is skipped. Kernels
are spread over the visible GPUs (--gpus), one replay at a time per GPU.

Writes <out>/noise.csv (one row per measurement) and prints the coefficient of variation
over rounds per kernel and configuration, and its median.

usage: measure_noise.py --stage stage2 --out DIR [--rounds 6] [--gpus 0,1,...]
"""
import argparse, csv, glob, json, math, os, random, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paths import FASTRR, X, BUILDS
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
from RRUtil.Benchmark import Kernel  # noqa: E402


def configs(info):
    trip = info.get('LoopTripCount') or 0
    out = [('default', 0, 0)]
    for threads in (64, 1024):
        if trip > 0:
            out.append((f'threads{threads}', math.ceil(trip / threads), threads))
    return out


def measure(build, js, teams, threads, warmup_ms, gpu, visible):
    with tempfile.NamedTemporaryFile('w', suffix='.batch', delete=False) as fd:
        fd.write(f'{teams} {threads}\n')
    env = Upstream.environment(build=build)
    env[visible] = str(gpu)
    cmd = [f'{build}/bin/llvm-omp-kernel-replay', '--batch', fd.name,
           '--repetitions', str(Kernel.TimingRepeats), '--warmup-ms', str(warmup_ms),
           '--rewarm-ms', str(Kernel.BatchRewarmMs), '--restore-from-device', '--ignore-limits',
           *Upstream.REPLAY_FLAGS, js]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        m = re.search(r'\|\|ReplayTiming\|\| duration_seconds: (\S+)', p.stdout)
        return float(m.group(1)) if m else None
    finally:
        os.unlink(fd.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--gpus', default=os.environ.get('FASTRR_GPUS', '0'))
    args = ap.parse_args()
    build = BUILDS[args.stage]
    visible = os.environ.get('FASTRR_VISIBLE', 'CUDA_VISIBLE_DEVICES')
    device = Upstream.get_device(0)
    warmup_ms = getattr(device, 'TimingWarmupMs', Kernel.TimingWarmupMs)
    gpus = [int(g) for g in args.gpus.split(',')]

    kernels = []
    for reg in sorted(glob.glob(f'{X}/{args.stage}/kernels/*.json')):
        for line, js in json.loads(Path(reg).read_text()).items():
            kernels.append((Path(reg).stem, line, js))
    rng = random.Random(0)

    def lane(work):
        gpu, items = work
        rows = []
        for app, line, js in items:
            cfgs = configs(json.loads(Path(js).read_text()))
            for r in range(args.rounds):
                order = cfgs[:]
                rng_k = random.Random(hash((app, line, r)) & 0xffffffff)
                rng_k.shuffle(order)
                for name, teams, threads in order:
                    s = measure(build, js, teams, threads, warmup_ms, gpu, visible)
                    rows.append(dict(app=app, line=line, config=name, round=r, seconds=s))
            print(f'{app} l{line} done on GPU {gpu}', flush=True)
        return rows

    lanes = [(g, kernels[i::len(gpus)]) for i, g in enumerate(gpus)]
    with ThreadPoolExecutor(len(gpus)) as ex:
        rows = [row for rs in ex.map(lane, lanes) for row in rs]
    os.makedirs(args.out, exist_ok=True)
    with open(Path(args.out, 'noise.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['app', 'line', 'config', 'round', 'seconds'])
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    d = pd.DataFrame(rows).dropna()
    g = d.groupby(['app', 'line', 'config']).seconds
    cv = (100 * g.std(ddof=1) / g.mean()).rename('cv_pct')
    n = g.size().rename('rounds')
    t = pd.concat([g.median().rename('median_s'), cv, n], axis=1)
    t = t[t.rounds >= 2]
    print(t.round(4).to_string())
    print(f'\n{len(t)} kernel configurations ({d.app.nunique()} apps); warm-up {warmup_ms} ms, '
          f'{Kernel.TimingRepeats} timed launches, flags {Upstream.REPLAY_FLAGS}')
    print(f'median CV {t.cv_pct.median():.2f}%, 90th percentile {t.cv_pct.quantile(0.9):.2f}%, '
          f'max {t.cv_pct.max():.2f}%')
    skipped = pd.DataFrame(rows)
    skipped = skipped[skipped.seconds.isna()].groupby(['app', 'line', 'config']).size()
    if len(skipped):
        print(f'configurations without a timing (not launchable): {len(skipped)}')


if __name__ == '__main__':
    main()
