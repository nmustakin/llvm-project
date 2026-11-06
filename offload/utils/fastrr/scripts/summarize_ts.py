"""Merge training_size.py shards into the MI50 counterpart of paper Table tab:more.

Usage: summarize_ts.py <shard dir>
"""
import glob
import sys

import numpy as np
import pandas as pd

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{sys.argv[1]}/shard*.csv'))])
k = d.groupby(['kid', 'app', 'new', 'method', 'k']).share.mean().reset_index()   # mean over seeds
rng = np.random.default_rng(0)


def table(sub):
    t = sub.groupby(['method', 'k']).share.mean().unstack()
    return t.apply(lambda r: '/'.join(f'{v:.2f}' for v in r), axis=1)


def diff(sub, a, b, kk):
    x = sub[sub.k == kk].pivot(index='kid', columns='method', values='share')
    dd = (x[a] - x[b]).dropna().values
    boots = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
    return f'{dd.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]'


for name, sub in [('all kernels with headroom', k), ('new (bulk) kernels', k[k.new])]:
    print(f'{name}: {sub.kid.nunique()} kernels, {sub.app.nunique()} applications')
    print('share of the optimum after 1/5/10 evaluations:')
    print(table(sub).to_string())
    print('  LOAO - LOAOs at 5:', diff(sub, 'rank-LOAO', 'rank-LOAOs', 5))
    print('  LOAO - TPE   at 5:', diff(sub, 'rank-LOAO', 'tpe', 5))
    print('  LOAOs - TPE  at 5:', diff(sub, 'rank-LOAOs', 'tpe', 5))
    print('  LOAO - LOAOs at 10:', diff(sub, 'rank-LOAO', 'rank-LOAOs', 10))
    print()
