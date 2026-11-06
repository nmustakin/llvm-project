"""MI50 counterpart of paper Table tab:mi50more from training_size.py (rank, TPE) and
training_size2.py (regression, RGPE, GP-EI) shards.

Usage: summarize_mi50more.py <ts dir> <ts2 dir>
"""
import glob
import sys

import numpy as np
import pandas as pd

d = pd.concat([pd.read_csv(f) for a in sys.argv[1:] for f in sorted(glob.glob(f'{a}/shard*.csv'))])
k = d.groupby(['kid', 'app', 'new', 'method', 'k']).share.mean().reset_index()   # mean over seeds
rng = np.random.default_rng(0)


def cell(sub, m):
    s = sub[sub.method == m].groupby('k').share.mean()
    return '/'.join(f'{v:.2f}'.lstrip('0') for v in s.values) if len(s) else '-'


def diff(sub, a, b, kk):
    x = sub[sub.k == kk].pivot(index='kid', columns='method', values='share')
    dd = (x[a] - x[b]).dropna().values
    boots = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
    return f'{dd.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]'


print('Fold (kernels) & Ranking & Regr. & RGPE & GP-EI & TPE')
for name, sub, fold in [('LOKO', k, 'LOKO'), ('LOAO', k, 'LOAO'), ('LOAO, new', k[k.new], 'LOAO'),
                        ('LOAOs, new', k[k.new], 'LOAOs')]:
    print(f'{name} ({sub.kid.nunique()}, {sub.app.nunique()} apps) & ' + ' & '.join(
        cell(sub, m) for m in (f'rank-{fold}', f'reg-{fold}', f'rgpe-{fold}', 'gpei', 'tpe')))
for name, sub in [('all', k), ('new', k[k.new])]:
    print(f'\n{name}:')
    for a, b in [('rank-LOAO', 'reg-LOAO'), ('rank-LOAO', 'rgpe-LOAO'), ('rank-LOAO', 'gpei'),
                 ('rank-LOAO', 'tpe'), ('rank-LOAO', 'rank-LOAOs'), ('reg-LOAOs', 'rank-LOAOs'),
                 ('rank-LOKO', 'reg-LOKO')]:
        print(f'  {a} - {b}: at 5 {diff(sub, a, b, 5)}, at 10 {diff(sub, a, b, 10)}')
