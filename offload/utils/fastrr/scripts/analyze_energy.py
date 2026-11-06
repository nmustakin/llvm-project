#!/usr/bin/env python3
"""Time and energy of the search strategies with upstream LLVM (the paper's energy section;
after ~/rr-amd-work/phase5/analyze_strategies.py of the LLVM 16 study).

usage: analyze_energy.py <run_tuner.py --out dir> [--exclude l1201]

Reads <dir>/<app>-l<line>/<method>-s<seed>/trace_*.csv (every evaluation's speedup and energy
gain) and <dir>/<app>-l<line>/validation.csv (true speedups of the shipped configurations), and
prints:
- the true speedup of each strategy, with its ratio to bo-60 and a bootstrap CI over kernels;
- per kernel, the fastest and the most energy-efficient configuration measured by any strategy,
  the energy gain that shipping the fastest leaves, and the speed that shipping the most
  efficient costs;
- the hypervolume of each strategy's measurements relative to all measurements of the kernel.
Kernels in --exclude (default l1201, whose energy window runs on modified inputs) are left out
of the energy results, and kernels whose implied power exceeds the board limit are reported.
"""
import argparse, glob, math, os, re
import numpy as np
import pandas as pd

BOARD_W = 300.0


def hypervolume(pts):
    """Area dominated by points (speedup, energy gain) above the reference (1, 1)."""
    p = sorted([(a, b) for a, b in pts if a > 1 and b > 1], key=lambda t: -t[0])
    area, top = 0.0, 1.0
    for a, b in p:
        if b > top:
            area += (a - 1) * (b - top)
            top = b
    return area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--exclude', default='l1201')
    ap.add_argument('--ref', default='bo-60')
    args = ap.parse_args()
    exclude = set(filter(None, args.exclude.split(',')))
    rng = np.random.default_rng(0)
    runs = []
    for f in glob.glob(f'{args.root}/*-l*/*-s*/trace_*.csv'):
        kdir, mdir = f.split('/')[-3], f.split('/')[-2]
        method, seed = re.match(r'(.*)-s(\d+)$', mdir).groups()
        t = pd.read_csv(f)
        runs.append(dict(kernel=kdir, line='l' + kdir.split('-l')[-1], method=method,
                         seed=int(seed), trace=t))
    if not runs:
        raise SystemExit('no traces')

    # True speedups of the shipped configurations: the median over the validation rounds of
    # the round's default time divided by the configuration's time (as analyze_paper.py).
    import json
    rows = []
    for vf in glob.glob(f'{args.root}/*-l*/validation.csv'):
        v = pd.read_csv(vf)
        v = v[v.valid]
        base = v[v.baseline].set_index('round').duration
        for key, names in json.load(open(vf.replace('.csv', '-shipped.json'))).items():
            g = v[v.key == key].set_index('round').duration
            c = g.index.intersection(base.index)
            sp = float(np.median(base.loc[c] / g.loc[c])) if len(c) else np.nan
            for n in names:
                m, seed = n.rsplit('-s', 1)
                rows.append(dict(kernel=vf.split('/')[-2], method=m, seed=int(seed), true=sp))
    if rows:
        v = pd.DataFrame(rows)
        k = v.groupby(['kernel', 'method']).true.apply(lambda x: np.exp(np.log(x).mean())).unstack()
        print('True speedup of the shipped configurations (geometric mean over kernels and '
              'seeds; every method ships its fastest measured configuration):')
        for m in k.columns:
            line = f'  {m:14s} {np.exp(np.log(k[m]).mean()):.3f}'
            if args.ref in k and m != args.ref:
                r = np.log(k[m] / k[args.ref]).dropna()
                bt = [np.exp(rng.choice(r.values, len(r)).mean()) for _ in range(4000)]
                line += (f'  ratio to {args.ref} {np.exp(r.mean()):.3f} '
                         f'[{np.percentile(bt, 2.5):.3f}, {np.percentile(bt, 97.5):.3f}]')
            print(line)

    # Every configuration measured for a kernel, over all strategies and seeds.
    meas = {}
    for r in runs:
        t = r['trace']
        t = t[(t.speedup > 0.02) & t.energy_gain.notna() & (t.energy_gain > 0)]
        for key, s, e in zip(t.key, t.speedup, t.energy_gain):
            meas.setdefault(r['line'], {})[key] = (s, e)
    print('\nTime-optimal against energy-optimal configuration per kernel:')
    left, cost, differ = [], [], 0
    for l, dd in sorted(meas.items()):
        pts = np.array(list(dd.values()))
        f, e = pts[np.argmax(pts[:, 0])], pts[np.argmax(pts[:, 1])]
        flag = ' (excluded)' if l in exclude else ''
        print(f'  {l:6s} fastest {f[0]:.3f}x speed, {f[1]:.3f}x energy gain | most efficient '
              f'{e[0]:.3f}x speed, {e[1]:.3f}x energy gain{flag}')
        if l in exclude:
            continue
        differ += int(not np.allclose(f, e))
        left.append(math.log(e[1]) - math.log(f[1]))
        cost.append(math.log(f[0]) - math.log(e[0]))
    if left:
        print(f'  the optima differ on {differ} of {len(left)} kernels; shipping the fastest '
              f'leaves {100 * (1 - math.exp(-np.mean(left))):.1f}% of the energy gain; shipping '
              f'the most efficient costs {100 * (1 - math.exp(-np.mean(cost))):.1f}% in speed')

    hv = []
    for r in runs:
        if r['line'] in exclude or r['line'] not in meas:
            continue
        t = r['trace']
        t = t[(t.speedup > 0.02) & t.energy_gain.notna() & (t.energy_gain > 0)]
        full = hypervolume(list(meas[r['line']].values()))
        if full > 0:
            hv.append(dict(method=r['method'], hv=hypervolume(zip(t.speedup, t.energy_gain))
                           / full))
    if hv:
        print('\nHypervolume relative to all configurations found (mean over runs):')
        print(pd.DataFrame(hv).groupby('method').hv.mean().sort_values(ascending=False)
              .round(3).to_string())


if __name__ == '__main__':
    main()
