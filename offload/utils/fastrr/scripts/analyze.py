#!/usr/bin/env python3
"""Analysis of the online experiments of taco-replication-2026-11-03 (inside rr_current.sif).

usage: analyze.py > analysis.log

Per kernel and run (as tuner-fixes-2026-10-30/analyze.py): the true speedup of the
shipped configuration, i.e. the median over the validation rounds of the same round's
default time divided by the configuration's time, with the default fallback (the
default is shipped when the last confirmed or promoted speedup is below 1); the
end-to-end time of Kernel.Search (driver log), split into setup (before the first
screen), search (DB search_wall_s), and confirmation (the rest); and the number of
screens. Batch TPE and FastRR are compared on the same kernels and seeds, and
FastRR on stage 2 (space-saving changes) with FastRR on stage 1, with kernel-clustered
bootstrap 95% CIs. The share of the optimum uses the optima of the paper's LLVM 16
grid datasets, which a different compiler may exceed.
"""
import glob, json, os, re
import numpy as np, pandas as pd

pd.set_option('display.width', 200)
from paths import X, POOL
POOL_APP = {'lulesh': 'LULESH', 'minife': 'MINIFE', 's3d': 'S3D', 'minisweep': 'MINISWEEP',
            'miniWeather': 'MINIWEATHER', 'amgmk': 'AMGMK', 'simplemoc': 'SIMPLEMOC',
            'su3': 'SU3', 'xsbench': 'XSBENCH'}
NAMES = {'final-wave': 'FastRR', 'tpe-batch-nocls': 'Batch TPE'}
FIVE = {'lulesh-649', 'lulesh-844', 'lulesh-1201', 'minife-290', 'minife-521'}
rng = np.random.default_rng(0)
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))


def boot(d, n=4000):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def load(stage):
    pool = pd.read_csv(POOL)
    optimum = pool.groupby(['app', 'line']).speedup.max()
    times = {}
    for f in glob.glob(f'{X}/{stage}/logs/*.log'):
        for m in re.finditer(r'^(\S+) l(\d+) (\S+) s(\d+): (\d+)s$', open(f).read(), re.M):
            times[(m[1], int(m[2]), m[3], int(m[4]))] = float(m[5])
    rows = []
    for kd in sorted(glob.glob(f'{X}/{stage}/runs/*-l*')):
        app, line = os.path.basename(kd).rsplit('-l', 1)
        line = int(line)
        vf = f'{kd}/validation.csv'
        if not os.path.exists(vf):
            continue
        v = pd.read_csv(vf)
        v = v[v.valid]
        if v['round'].nunique() < 5:
            continue
        base = v[v.baseline].set_index('round').duration
        for key, runs in json.load(open(f'{kd}/validation-shipped.json')).items():
            g = v[v.key == key].set_index('round').duration
            c = g.index.intersection(base.index)
            sp = float(np.median(base.loc[c] / g.loc[c]))
            for r in runs:
                meth, seed = r.rsplit('-s', 1)
                rd = f'{kd}/{meth}-s{seed}'
                tr = pd.read_csv(glob.glob(f'{rd}/trace_*.csv')[0])
                meta = json.load(open(glob.glob(f'{rd}/analysis*.json')[0]))['MetaData']
                conf = tr[(tr.key == key) & tr.source.isin(['confirm', 'promote'])].speedup
                fb = len(conf) > 0 and float(conf.iloc[-1]) < 1
                e2e = times.get((app, line, meth, int(seed)), np.nan)
                last = float(tr.wall.max()) if len(tr) else 0.0
                search = float(meta.get('search_wall_s', np.nan))
                rows.append(dict(stage=stage, kernel=f'{app}-{line}', method=meth, seed=int(seed),
                                 key=key, raw=sp, shipped=1.0 if fb else sp, fallback=fb,
                                 opt=optimum.get((POOL_APP.get(app, app.upper()), line), np.nan),
                                 screens=int(tr.source.isin(['batch', 'cache']).sum()),
                                 stop=meta.get('stop_reason'), end_to_end=e2e,
                                 setup=e2e - last, search=search, confirm=last - search))
    return pd.DataFrame(rows)


def summary(r, title):
    print(f'\n=== {title}: {r.kernel.nunique()} kernels, {len(r)} runs')
    h = r[r.opt > 1.05].copy()
    h['share'] = np.clip(np.log(h.shipped), 0, None) / np.log(h.opt)
    t = pd.DataFrame({'true speedup': r.groupby('method').shipped.apply(gm),
                      'screens': r.groupby('method').screens.mean(),
                      'setup s': r.groupby('method').setup.mean(),
                      'search s': r.groupby('method').search.mean(),
                      'confirm s': r.groupby('method').confirm.mean(),
                      'end to end s': r.groupby('method').end_to_end.mean(),
                      'share of LLVM16 optimum': h.groupby('method').share.mean(),
                      'fallbacks': r.groupby('method').fallback.sum()})
    print(t.rename(index=NAMES).round(3).to_string())
    return t


def compare(a, b, label):
    """Ratio of shipped speedups and difference of end-to-end time, b against a, paired
    by kernel and seed, with kernel-clustered CIs."""
    p = a.merge(b, on=['kernel', 'seed'], suffixes=('_a', '_b'))
    d = (np.log(p.shipped_b) - np.log(p.shipped_a)).groupby(p.kernel).mean()
    e = (p.end_to_end_b - p.end_to_end_a).groupby(p.kernel).mean()
    q = (p.end_to_end_b / p.end_to_end_a).groupby(p.kernel).mean()
    lo, hi = boot(d)
    elo, ehi = boot(e)
    print(f'  {label}: speedup ratio {np.exp(d.mean()):.4f} [{np.exp(lo):.4f}, {np.exp(hi):.4f}], '
          f'better/worse by >1% on {(d > 0.01).sum()}/{(d < -0.01).sum()} of {len(d)} kernels; '
          f'end to end {e.mean():+.0f} s [{elo:+.0f}, {ehi:+.0f}], time ratio '
          f'{p.end_to_end_b.mean() / p.end_to_end_a.mean():.2f}')


def main():
    frames = [load(s) for s in ('stage1', 'stage2')]
    r = pd.concat([f for f in frames if len(f)], ignore_index=True)
    r.to_csv(f'{X}/runs.csv', index=False)
    s1 = r[r.stage == 'stage1']
    print('Paper (T4): 5 kernels Batch TPE 309 s / 1.531, FastRR (ranking + patience + batch '
          'confirmation) 140 s / 1.531; HeCBench FastRR 1.570 / 121 s, Batch TPE 1.571 / 332 s '
          '(time ratio .36); final design without the profile 95 s per kernel.')
    for name, sel in (('LULESH/miniFE, 5 kernels (paper tab:summary)', s1.kernel.isin(FIVE)),
                      ('HeCBench, 15 kernels (paper tab:crossgpu, T4 row)', ~s1.kernel.isin(FIVE))):
        sub = s1[sel]
        if not len(sub):
            continue
        summary(sub, f'stage 1, {name}')
        a = sub[sub.method == 'tpe-batch-nocls']
        b = sub[sub.method == 'final-wave']
        if len(a) and len(b):
            compare(a, b, 'FastRR against Batch TPE')
    print('\n=== per kernel, stage 1: true speedup (geomean of seeds), end to end (mean)')
    t = s1.pivot_table(index='kernel', columns='method', values=['shipped', 'end_to_end'],
                       aggfunc={'shipped': gm, 'end_to_end': 'mean'})
    t[('LLVM16 optimum', '')] = s1.groupby('kernel').opt.first()
    print(t.round(3).to_string())
    s2 = r[r.stage == 'stage2']
    if len(s2):
        summary(s2, 'stage 2 (space-saving changes), FastRR')
        compare(s1[s1.method == 'final-wave'], s2[s2.method == 'final-wave'],
                'FastRR, stage 2 against stage 1')
        print('\n=== per kernel: FastRR true speedup and end to end, stage 1 and stage 2')
        f = r[r.method == 'final-wave'].pivot_table(index='kernel', columns='stage',
                                                    values=['shipped', 'end_to_end'],
                                                    aggfunc={'shipped': gm, 'end_to_end': 'mean'})
        print(f.round(3).to_string())


if __name__ == '__main__':
    main()
