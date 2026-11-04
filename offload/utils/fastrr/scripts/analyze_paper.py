#!/usr/bin/env python3
"""Analysis of the re-runs for the upstream version of the TACO paper (inside rr_current.sif).

usage: analyze_paper.py > paper-analysis.log

Everything is on the final system (stage 2). True speedups are the median over the
validation rounds of the same round's default time divided by the configuration's time,
with the default fallback (a run whose last confirmed or promoted speedup is below 1
ships the default). Experiments validated in different rounds are compared by kernel and
seed with kernel-clustered bootstrap 95% CIs. Shares of the optimum use the upstream grid
datasets (grids/), which a tuner can exceed.
Writes per-run tables to paper/: refs.csv, ablation.csv, search.csv, batch.csv,
incremental.csv, inputs.csv, hecbench_kernels.csv.
"""
import glob, json, os, re
import numpy as np, pandas as pd

pd.set_option('display.width', 220)
from paths import X
OUT = f'{X}/paper'
NAMES = {'final-wave': 'FastRR', 'tpe-batch-nocls': 'Batch TPE', 'final': 'FastRR without wave',
         'final-wave-norefine': 'FastRR without refinement', 'final-wave-promote': 'FastRR, promotions',
         'final-wave-4gpu': 'FastRR, four GPUs', 'tpe-mf': 'MF screening', 'tpe-60': 'TPE, 60 full',
         'bo-60': 'BO, 60 full', 'random-60': 'Random, 60 full'}
FIVE = ['lulesh-649', 'lulesh-844', 'lulesh-1201', 'minife-290', 'minife-521']
HEC = ['s3d-159', 's3d-165', 's3d-171', 's3d-183', 's3d-220', 's3d-226', 'minisweep-282',
       'minisweep-315', 'minisweep-348', 'minisweep-383', 'miniWeather-760', 'amgmk-258',
       'simplemoc-379', 'su3-41', 'xsbench-180']
rng = np.random.default_rng(0)
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))


def boot(d, n=4000):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def grid_optima():
    opt = {}
    for f in glob.glob(f'{X}/grids/*.csv'):
        g = pd.read_csv(f)
        opt[os.path.basename(f)[:-4]] = float(g.speedup.max())
    return opt


OPT = grid_optima()


def times(patterns):
    """End-to-end seconds of Kernel.Search per (app, line, method, seed) from driver logs."""
    t = {}
    for p in patterns:
        for f in glob.glob(p):
            for m in re.finditer(r'^(\S+) l(\d+) (\S+) s(\d+): (\d+)s$', open(f).read(), re.M):
                t[(m[1], int(m[2]), m[3], int(m[4]))] = float(m[5])
    return t


def true_speedups(v, keys):
    v = v[v.valid]
    base = v[v.baseline].set_index('round').duration
    out = {}
    for key in keys:
        g = v[v.key == key].set_index('round').duration
        c = g.index.intersection(base.index)
        out[key] = float(np.median(base.loc[c] / g.loc[c])) if len(c) else np.nan
    return out


def load(root, vname='validation.csv', logs=(), methods=None):
    """One row per (kernel, method, seed) of a run_tuner.py output directory."""
    tm = times(logs)
    rows = []
    for kd in sorted(glob.glob(f'{root}/*-l*')):
        app, line = os.path.basename(kd).rsplit('-l', 1)
        line = int(line)
        vf = f'{kd}/{vname}'
        if not os.path.exists(vf):
            continue
        shipped = json.load(open(vf.replace('.csv', '-shipped.json')))
        sp = true_speedups(pd.read_csv(vf), shipped)
        for key, runs in shipped.items():
            for r in runs:
                meth, seed = r.rsplit('-s', 1)
                if methods and meth not in methods:
                    continue
                rd = f'{kd}/{meth}-s{seed}'
                tr = pd.read_csv(glob.glob(f'{rd}/trace_*.csv')[0])
                meta = json.load(open(glob.glob(f'{rd}/analysis*.json')[0]))['MetaData']
                conf = tr[(tr.key == key) & tr.source.isin(['confirm', 'promote'])].speedup
                fb = len(conf) > 0 and float(conf.iloc[-1]) < 1
                e2e = tm.get((app, line, meth, int(seed)), np.nan)
                last = float(tr.wall.max()) if len(tr) else 0.0
                search = float(meta.get('search_wall_s', np.nan))
                scr = tr[tr.source.isin(['batch', 'cache', 'measured', 'repeat'])]
                rows.append(dict(kernel=f'{app}-{line}', app=app, line=line, method=meth,
                                 seed=int(seed), key=key, raw=sp[key],
                                 shipped=1.0 if fb else sp[key], fallback=fb,
                                 opt=OPT.get(f'{app}-{line}', np.nan), screens=len(scr),
                                 stop=meta.get('stop_reason'), end_to_end=e2e, setup=e2e - last,
                                 search=search, confirm=last - search, last_wall=last, dir=rd))
    return pd.DataFrame(rows)


def share(sp, opt):
    return np.clip(np.log(sp), 0, None) / np.log(opt)


def summary(r, title):
    print(f'\n=== {title}: {r.kernel.nunique()} kernels, {len(r)} runs')
    h = r[r.opt > 1.05].copy()
    h['share'] = share(h.shipped, h.opt)
    g = r.groupby('method')
    t = pd.DataFrame({'true speedup': g.shipped.apply(gm), 'screens': g.screens.mean(),
                      'setup s': g.setup.mean(), 'search s': g.search.mean(),
                      'confirm s': g.confirm.mean(), 'end to end s': g.end_to_end.mean(),
                      'share of grid optimum': h.groupby('method').share.mean(),
                      'fallbacks': g.fallback.sum()})
    print(t.rename(index=NAMES).round(3).to_string())
    return t


def compare(a, b, label, quiet=False):
    """b against a, paired by kernel and seed: speedup ratio and end-to-end difference."""
    p = a.merge(b, on=['kernel', 'seed'], suffixes=('_a', '_b'))
    d = (np.log(p.shipped_b) - np.log(p.shipped_a)).groupby(p.kernel).mean()
    e = (p.end_to_end_b - p.end_to_end_a).groupby(p.kernel).mean()
    lo, hi = boot(d)
    elo, ehi = boot(e.dropna()) if e.notna().any() else (np.nan, np.nan)
    res = dict(ratio=np.exp(d.mean()), lo=np.exp(lo), hi=np.exp(hi), better=int((d > 0.01).sum()),
               worse=int((d < -0.01).sum()), n=len(d), dt=e.mean(), dlo=elo, dhi=ehi,
               tratio=p.end_to_end_b.mean() / p.end_to_end_a.mean())
    if not quiet:
        print(f'  {label}: speedup ratio {res["ratio"]:.4f} [{res["lo"]:.4f}, {res["hi"]:.4f}], '
              f'better/worse by >1% on {res["better"]}/{res["worse"]} of {res["n"]} kernels; '
              f'end to end {res["dt"]:+.1f} s [{elo:+.1f}, {ehi:+.1f}], time ratio {res["tratio"]:.3f}')
    return res


def best_by(tr, t):
    s = tr[tr.source.isin(['batch', 'measured', 'cache', 'repeat']) & (tr.wall <= t)].speedup
    return max(1.0, float(s.max())) if len(s) else 1.0


def refs():
    r = load(f'{X}/stage2/runs', 'validation-ref.csv',
             [f'{X}/stage2/logs/*.log', f'{X}/reruns/logs/ref-*.log'])
    r.to_csv(f'{OUT}/refs.csv', index=False)
    for name, ks in (('5 LULESH/miniFE kernels (tab:summary)', FIVE),
                     ('15 HeCBench kernels (tab:crossgpu)', HEC),
                     ('all 43 kept kernels', sorted(r.kernel.unique()))):
        sub = r[r.kernel.isin(ks)]
        summary(sub, f'references, {name}')
        compare(sub[sub.method == 'tpe-batch-nocls'], sub[sub.method == 'final-wave'],
                'FastRR against Batch TPE')
    lm = r[r.app.isin(['lulesh', 'minife'])]
    summary(lm, 'references, all 17 LULESH/miniFE kernels (LOKO)')
    compare(lm[lm.method == 'tpe-batch-nocls'], lm[lm.method == 'final-wave'], 'FastRR against Batch TPE')
    oth = r[~r.app.isin(['lulesh', 'minife'])]
    summary(oth, 'references, all 26 other kernels (LOAO)')
    compare(oth[oth.method == 'tpe-batch-nocls'], oth[oth.method == 'final-wave'], 'FastRR against Batch TPE')
    print('\n=== references per kernel: true speedup (geomean of seeds), end to end (mean), grid optimum')
    t = r.pivot_table(index='kernel', columns='method', values=['shipped', 'end_to_end'],
                      aggfunc={'shipped': gm, 'end_to_end': 'mean'})
    t[('grid optimum', '')] = r.groupby('kernel').opt.first()
    print(t.round(3).to_string())
    return r


def batch_table(r):
    b = load(f'{X}/reruns/batch', logs=[f'{X}/reruns/logs/batch-*.log'])
    tpe = r[r.kernel.isin(FIVE) & (r.method == 'tpe-batch-nocls')]
    a = pd.concat([b, tpe], ignore_index=True)
    for t in (30, 60, 120):
        a[f'best{t}'] = [best_by(pd.read_csv(glob.glob(f'{d}/trace_*.csv')[0]), t) for d in a.dir]
    a.to_csv(f'{OUT}/batch.csv', index=False)
    print('\n=== tab:batch (5 kernels, three seeds; MF and TPE-60 validated in their own rounds)')
    g = a.groupby('method')
    print(pd.DataFrame({'screens': g.screens.mean(), 'best30': g.best30.apply(gm),
                        'best60': g.best60.apply(gm), 'best120': g.best120.apply(gm),
                        'shipped': g.shipped.apply(gm), 'end to end s': g.end_to_end.mean(),
                        'last wall s': g.last_wall.mean()}).rename(index=NAMES).round(3).to_string())
    compare(a[a.method == 'tpe-mf'], a[a.method == 'tpe-batch-nocls'], 'Batch TPE against MF screening')
    compare(a[a.method == 'tpe-60'], a[a.method == 'tpe-batch-nocls'], 'Batch TPE against TPE-60')
    return a


def search_table():
    s = load(f'{X}/reruns/search', logs=[f'{X}/reruns/logs/search-*.log'])
    for k in (15, 30, 60):
        col = []
        for d in s.dir:
            tr = pd.read_csv(glob.glob(f'{d}/trace_*.csv')[0])
            m = tr[tr.source.isin(['measured', 'repeat'])].head(k).speedup
            col.append(max(1.0, float(m.max())) if len(m) else 1.0)
        s[f'best{k}'] = col
    s.to_csv(f'{OUT}/search.csv', index=False)
    h = s[s.opt > 1.05].copy()
    for k in (15, 30, 60):
        h[f'share{k}'] = share(h[f'best{k}'], h.opt)
    print(f'\n=== tab:search (10 LULESH kernels, 60 proposals, three seeds; cache shared within a '
          f'kernel); shares on {h.kernel.nunique()} kernels with grid optimum > 1.05')
    g = s.groupby('method')
    print(pd.DataFrame({'true speedup': g.shipped.apply(gm), 'best measured': g.best60.apply(gm),
                        'share15': h.groupby('method').share15.mean(),
                        'share30': h.groupby('method').share30.mean(),
                        'share60': h.groupby('method').share60.mean(),
                        'end to end s (cached)': g.end_to_end.mean()}).rename(index=NAMES).round(3).to_string())
    for m in ('tpe-60', 'random-60'):
        compare(s[s.method == 'bo-60'], s[s.method == m], f'{NAMES[m]} against BO')
    print('  grid optima:', {k: round(v, 3) for k, v in sorted(OPT.items()) if k.startswith('lulesh')})
    return s


def ablations(r):
    a = load(f'{X}/reruns/ablation', logs=[f'{X}/reruns/logs/abl-*.log'])
    p = load(f'{X}/reruns/promote', logs=[f'{X}/reruns/logs/promote-*.log'])
    g = load(f'{X}/reruns/mgpu', logs=[f'{X}/reruns/logs/mgpu-*.log'])
    allr = pd.concat([a, p, g], ignore_index=True)
    allr.to_csv(f'{OUT}/ablation.csv', index=False)
    fw = r[r.method == 'final-wave']
    summary(a, 'HeCBench ablations (15 kernels)')
    compare(a[a.method == 'final'], fw[fw.kernel.isin(HEC)], 'wave-aligned team counts (FastRR against without)')
    compare(a[a.method == 'final-wave-norefine'], fw[fw.kernel.isin(HEC)], 'refinement (FastRR against without)')
    five = pd.concat([p, g, r[r.kernel.isin(FIVE)]], ignore_index=True)
    summary(five, '5 kernels: promotions, four GPUs, references')
    compare(p, fw[fw.kernel.isin(FIVE)], 'batch confirmation (FastRR against promotions)')
    compare(g, fw[fw.kernel.isin(FIVE)], 'four GPUs (one GPU against four)')
    print('\n=== per kernel, HeCBench: FastRR, ablations, Batch TPE (true speedups), grid optimum')
    k = pd.concat([r[r.kernel.isin(HEC)], a]).pivot_table(index='kernel', columns='method',
                                                          values='shipped', aggfunc=gm)
    k['grid optimum'] = [OPT.get(x, np.nan) for x in k.index]
    k.to_csv(f'{OUT}/hecbench_kernels.csv')
    print(k.round(3).to_string())
    print('  FastRR >= 0.99 of grid optimum on', int((k['final-wave'] >= 0.99 * k['grid optimum']).sum()),
          'of', len(k), '; above the optimum by > 1% on',
          list(k.index[k['final-wave'] > 1.01 * k['grid optimum']]))
    print('\n=== per kernel, 5 kernels: four GPUs and promotions, time split')
    print(five.groupby(['kernel', 'method'])[['setup', 'search', 'confirm', 'end_to_end']]
          .mean().round(1).to_string())


def incremental(r):
    rows = []
    for f in sorted(glob.glob(f'{X}/reruns/incremental/lane*/*/l*/run.json')):
        info = json.load(open(f))
        kd = os.path.dirname(f)
        v = pd.read_csv(f'{kd}/validation.csv')
        sp = true_speedups(v, [info['final']])[info['final']]
        kernel = f"{info['app']}-{info['line']}"
        ref = {m: r[(r.kernel == kernel) & (r.method == m) & (r.seed == info['seed'])]
               for m in ('final-wave', 'tpe-batch-nocls')}
        rows.append(dict(info, kernel=kernel, inc=sp,
                         **{f'{m}': float(x.shipped.iloc[0]) if len(x) else np.nan for m, x in ref.items()},
                         **{f'{m}_s': float(x.end_to_end.iloc[0]) if len(x) else np.nan for m, x in ref.items()}))
    d = pd.DataFrame(rows)
    d['bin'] = pd.cut(d.store_kernels, [-1, 0, 10, 20, 30, 43], labels=['0', '1-10', '11-20', '21-30', '31+'])
    d.to_csv(f'{OUT}/incremental.csv', index=False)
    print(f'\n=== tab:incremental: {len(d)} kernel runs in {d.lane.nunique()} lanes; '
          f'cold {int((d["mode"] == "cold").sum())}, model {int((d["mode"] == "model").sum())}; '
          f'default shipped {int((d.final != d.shipped).sum())}')
    for b, g in list(d.groupby('bin', observed=True)) + [('1+', d[d['mode'] == 'model']), ('all', d)]:
        line = f'  {b:5s}: runs {len(g):3d}, screens {g.screens.mean():6.1f}, time {g.end_to_end.mean():6.1f} s'
        for m, lab in (('final-wave', 'Exh.'), ('tpe-batch-nocls', 'TPE')):
            q = (np.log(g.inc) - np.log(g[m])).groupby(g.kernel).mean()
            e = (g.end_to_end - g[f'{m}_s']).groupby(g.kernel).mean()
            lo, hi = boot(q)
            elo, ehi = boot(e)
            line += (f' | vs {lab}: dt {e.mean():+.1f} [{elo:+.1f}, {ehi:+.1f}], ratio '
                     f'{np.exp(q.mean()):.4f} [{np.exp(lo):.4f}, {np.exp(hi):.4f}]')
        print(line)
    w = d.groupby('lane').agg(kernels=('inc', 'size'), inc_h=('end_to_end', 'sum'),
                              ref_h=('final-wave_s', 'sum'), tpe_h=('tpe-batch-nocls_s', 'sum'))
    first = d[d.pos == 0].groupby('lane').end_to_end.sum() / 3600
    print('  sequence hours per lane:')
    print((w.assign(inc_h=w.inc_h / 3600, ref_h=w.ref_h / 3600, tpe_h=w.tpe_h / 3600, first_app_h=first))
          .round(3).to_string())
    print('  mean:', round(w.inc_h.mean() / 3600, 3), 'h; first application', round(first.mean(), 3),
          'h; Batch TPE', round(w.tpe_h.mean() / 3600, 3), 'h; grid-trained FastRR', round(w.ref_h.mean() / 3600, 3), 'h')
    st = [len(pd.read_csv(f)) for f in sorted(glob.glob(f'{X}/reruns/incremental/lane*/store.csv'))]
    print('  final store rows per lane:', st, 'mean', np.mean(st))
    print('  stop reasons:', d.groupby('mode').stop.value_counts().to_dict())
    return d


def inputs():
    rows = []
    for vf in sorted(glob.glob(f'{X}/reruns/inputs/*-l*/validation.csv')):
        kd = os.path.dirname(vf)
        b, line = os.path.basename(kd).rsplit('-l', 1)
        cfg = json.load(open(f'{kd}/configs.json'))
        sp = true_speedups(pd.read_csv(vf), cfg['who'])
        for key, labels in cfg['who'].items():
            for lab in labels:
                kind, run = lab.split(':')
                meth, seed = run.rsplit('-s', 1)
                rows.append(dict(input=b, kernel=f'{b}-l{line}', kind=kind, method=meth, seed=int(seed),
                                 speedup=sp[key], trip_ratio=cfg['trip_b'] / cfg['trip_a']))
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/inputs.csv', index=False)
    t = d.pivot_table(index=['kernel', 'trip_ratio'], columns='kind', values='speedup', aggfunc=gm)
    print(f'\n=== tab:rtx-inputs (T4 column): {len(t)} kernels at a second input')
    print(t.round(3).to_string())
    print('  geomean over kernels:', {k: round(gm(t[k]), 4) for k in ('transfer', 'scaled', 'retuned')})
    for k in ('transfer', 'scaled'):
        q = np.log(t[k]) - np.log(t.retuned)
        lo, hi = boot(q)
        print(f'  {k}: ratio to retuned {np.exp(q.mean()):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}], '
              f'worse than retuned by >2% on {int((q < np.log(0.98)).sum())}, slower than default on '
              f'{int((t[k] < 1).sum())} of {len(t)}')
    print(f'  retuned slower than default on {int((t.retuned < 1).sum())}')


def cold():
    print('\n=== cold and warm state loading, one replay process per kernel')
    for st in ('stage1', 'stage2'):
        c = pd.read_csv(f'{X}/measure/cold-{st}.csv')
        print(f'  {st}: {len(c)} kernels, rc!=0 {int((c.rc != 0).sum())}; cold median {c.cold_s.median():.2f} s '
              f'(sum {c.cold_s.sum():.0f} s), warm median {c.warm_s.median():.2f} s (sum {c.warm_s.sum():.0f} s), '
              f'read {c.read_mib.sum() / 1024:.1f} GiB')
        print('   ', c.groupby('app').agg(n=('line', 'size'), cold=('cold_s', 'mean'), warm=('warm_s', 'mean'),
                                         mib=('read_mib', 'mean')).round(2).to_dict('index'))


def main():
    os.makedirs(OUT, exist_ok=True)
    r = refs()
    batch_table(r)
    search_table()
    ablations(r)
    incremental(r)
    inputs()
    cold()


if __name__ == '__main__':
    main()
