"""Offline training-set-size study (paper: Section "Training-set size") on MI50.

Each target kernel's exhaustive landscape is the oracle. Methods choose one
configuration of the landscape at a time; after 1, 5, and 10 evaluations the
score is the share of the optimum log speedup, log(best so far) / log(optimum),
over kernels with an optimum above 1.05x. The default configuration is known, so
the best so far is at least 1x (share >= 0).

  rank   the tuner's bagged pairwise XGBoost rankers (Search.train_rank_models),
         fine-tuned on the kernel's own observations before each pick and
         ranked by mean + std of the z-scored scores, as RankTransferSearcher
  tpe    Optuna TPE over (binary class, NumThreads, teams fraction)

Folds for the ranking model:
  LOKO   train on all other kernels
  LOAO   train on the kernels of the other applications
  LOAOs  train on the kernels of the other applications among the original set
         (LULESH, miniFE, and the ten HeCBench proxy applications)

Usage: training_size.py <pool.csv> <out.csv> [seeds]
"""
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # the upstream tuner
import RRUtil.Search as S  # noqa: E402

warnings.filterwarnings('ignore')
pool_csv, out = sys.argv[1:3]
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]
pool = pd.read_csv(pool_csv)
# Kernels whose default launch lies outside the candidate space (every candidate
# far slower than the default) are excluded, as in the paper.
_opt = pool.groupby('kid').speedup.max()
pool = pool[pool.kid.isin(_opt[_opt >= 0.5].index)]
ORIGINAL = {'LULESH', 'MINIFE', 'XSBENCH', 'RSBENCH', 'SIMPLEMOC', 'SU3', 'HACCMK', 'AMGMK',
            'MINISWEEP', 'SW4CK', 'S3D', 'MINIWEATHER'}
K = (1, 5, 10)


def rank_run(models, cand, y, seed):
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    X = S.transfer_frame(cand)
    dc = xgb.DMatrix(X)
    base = np.array([m.predict(dc) for m, _ in models])
    seen, own_x, own_y, best = set(), [], [], []
    for _ in range(max(K)):
        if len(own_y) >= 2:
            g = S.rank_grades(np.array(own_y), levels=len(own_y))
            d = xgb.DMatrix(np.array(own_x), label=g, qid=np.zeros(len(g), int))
            sc = np.array([xgb.train(dict(p, eta=0.3, max_depth=3, subsample=1.0), d,
                                     num_boost_round=20, xgb_model=m.copy()).predict(dc)
                           for m, p in models])
        else:
            sc = base
        z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)
        acq = z.mean(0) + z.std(0) + rng.random(len(cand)) * 1e-9
        acq[list(seen)] = -np.inf
        j = int(np.argmax(acq))
        seen.add(j)
        own_x.append(X[j])
        own_y.append(math.log(y[j]))
        best.append(max(own_y))
    return best


def tpe_run(cand, y, seed):
    """TPE over the landscape: binary class, then NumThreads among that binary's,
    then the teams fraction (log2), snapped to the nearest one measured for that
    binary and NumThreads, so that every proposal is a landscape configuration."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bins = sorted(set(zip(cand.regs, cand.spill)))
    fr = np.log2(cand.teams_frac.values)
    by = {}
    for j, r in enumerate(cand.itertuples()):
        by.setdefault(bins.index((r.regs, r.spill)), {}).setdefault(int(r.NumThreads), []).append(j)
    lo, hi = float(fr.min()), float(fr.max())
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5))
    best, b = [], -np.inf
    for _ in range(max(K)):
        t = study.ask()
        bi = t.suggest_categorical('bin', sorted(by))
        nt = t.suggest_categorical(f'nt_b{bi}', sorted(by[bi]))
        f = t.suggest_float('log2_frac', lo, hi) if hi > lo else lo
        j = min(by[bi][nt], key=lambda i: abs(fr[i] - f))
        v = math.log(y[j])
        study.tell(t, v)
        b = max(b, v)
        best.append(b)
    return best


rows = []
if os.path.exists(out):
    # Resume: keep earlier results and skip the kernels already evaluated.
    rows = pd.read_csv(out).to_dict('records')
done = {r['kid'] for r in rows}
kids = [k for k, g in pool.groupby('kid') if g.speedup.max() > 1.05]
# SHARD=i/n evaluates every n-th target kernel starting at i (parallel runs).
shard = os.environ.get('SHARD')
if shard:
    i, n = map(int, shard.split('/'))
    kids = kids[i::n]
print(f'{pool.kid.nunique()} kernels, {len(kids)} with headroom', flush=True)
model_cache = {}
for n, kid in enumerate(kids):
    if kid in done:
        continue
    cand = pool[pool.kid == kid].reset_index(drop=True)
    y = cand.speedup.values
    opt = math.log(y.max())
    app = cand.app.iloc[0]
    folds = {'LOKO': pool[pool.kid != kid],
             'LOAO': pool[pool.app != app],
             'LOAOs': pool[(pool.app != app) & pool.app.isin(ORIGINAL)]}
    for seed in seeds:
        for fold, train in folds.items():
            key = (fold, app if fold != 'LOKO' else kid, seed)
            if key not in model_cache:
                model_cache[key] = S.train_rank_models(train, target_kid=None, seed=seed)
            best = rank_run(model_cache[key], cand, y, seed)
            rows += [dict(kid=kid, app=app, new=app not in ORIGINAL, method=f'rank-{fold}',
                          seed=seed, k=k, share=max(best[k - 1], 0.0) / opt) for k in K]
        best = tpe_run(cand, y, seed)
        rows += [dict(kid=kid, app=app, new=app not in ORIGINAL, method='tpe', seed=seed, k=k,
                      share=max(best[k - 1], 0.0) / opt) for k in K]
    print(f'{n + 1}/{len(kids)} kernels', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
d = pd.DataFrame(rows)
d.to_csv(out, index=False)
for subset, g in [('all', d), ('new kernels', d[d.new])]:
    t = g.groupby(['method', 'k']).share.mean().unstack().round(3)
    print(f'\n{subset} ({g.kid.nunique()} kernels): share of the optimum after 1/5/10 evaluations')
    print(t.to_string())
