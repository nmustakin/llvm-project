"""Regression model, RGPE, and GP-EI for the offline training-set-size study.

Same oracle, kernels, folds, and score as training_size.py (share of the optimum
log speedup after 1, 5, and 10 evaluations, default known), so the results join
its ranking-model and TPE rows. The searchers are the tuner's own classes,
built on the target kernel's landscape as their candidate set:

  reg    TransferModelSearcher: ExtraTrees on the training kernels plus the
         kernel's own observations (weight 20), expected improvement
  rgpe   RankTransferSearcher with rgpe=True: the ranking model for the first 5
         evaluations, then a ranking-weighted ensemble of one GP per training
         kernel, the ranker, and a GP on the own observations
  gpei   a GP on the kernel's own observations with expected improvement
         (no transfer); the first evaluation is a random configuration

Usage: training_size2.py <pool.csv> <out.csv> [seeds]   (SHARD=i/n as before)
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


class _MI50:
    Target, CUs, MaxWavesPerCU, ThreadsSMP, WarpSize, MaxThreadBlockSize = (
        'amdgcn-amd-amdhsa', 60, 40, 2560, 64, 1024)


S.set_device(_MI50)
pool_csv, out = sys.argv[1:3]
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]
pool = pd.read_csv(pool_csv)
_opt = pool.groupby('kid').speedup.max()
pool = pool[pool.kid.isin(_opt[_opt >= 0.5].index)]
ORIGINAL = {'LULESH', 'MINIFE', 'XSBENCH', 'RSBENCH', 'SIMPLEMOC', 'SU3', 'HACCMK', 'AMGMK',
            'MINISWEEP', 'SW4CK', 'S3D', 'MINIWEATHER'}
K = (1, 5, 10)


def candidates(kid):
    """The kernel's landscape as a candidate set. The pool keeps the teams
    fraction, not NumTeams; the searchers only need NumTeams to return a point,
    so it is rebuilt for a nominal trip count."""
    cand = pool[pool.kid == kid].reset_index(drop=True).copy()
    trip = 1e6
    cand['NumTeams'] = np.maximum(1, np.round(cand.teams_frac * trip / cand.NumThreads)).astype(int)
    cand['canon'] = list(zip(cand.regs, cand.spill, cand.NumThreads, cand.teams_frac.round(9)))
    return cand, trip


def base_state(s, cand, trip, seed):
    s.np, s.pd, s.seed, s.objective = np, pd, seed, 'speedup'
    s.trip, s.klass = trip, cand.klass.iloc[0]
    s.cand, s.Xc = cand, S.transfer_frame(cand)
    s.own_X, s.own_y, s.seen, s.asked = [], [], set(), None
    s.table, s.initial, s.beta, s.prior_w = None, [], 10.0, None
    s.low_teams = 0          # the upstream searcher's first-batch reservation, off offline


def run_reg(train, cand, trip, y, seed):
    s = S.TransferModelSearcher.__new__(S.TransferModelSearcher)
    base_state(s, cand, trip, seed)
    other = train[train.speedup > 0]
    s.X0, s.y0, s.own_weight = S.transfer_frame(other), np.log(other.speedup.values), 20.0
    best = []
    for _ in range(max(K)):
        (_, row), = s.ask_many(1)
        s.tell_trial(row, {'speedup': math.log(y[row.name])})
        best.append(max(s.own_y) if s.own_y else -np.inf)
    return best


_gp_cache = {}


def kernel_gps(train, seed):
    """One GP per training kernel (cached per kernel and seed), as in RGPE."""
    out = []
    rng = np.random.default_rng(seed)
    for kid, g in train[train.speedup > 0].groupby('kid'):
        key = (kid, seed)
        if key not in _gp_cache:
            X, yy = S.transfer_frame(g), np.log(g.speedup.values)
            idx = rng.choice(len(yy), min(300, len(yy)), replace=False)
            _gp_cache[key] = S.RankTransferSearcher._fit_gp(
                X[idx, :7] / np.array(S.GP_SCALE), S.RankTransferSearcher._std(yy)[idx])
        out.append(_gp_cache[key])
    return out


def run_rgpe(models, gps, cand, trip, y, seed):
    import xgboost as xgb
    s = S.RankTransferSearcher.__new__(S.RankTransferSearcher)
    base_state(s, cand, trip, seed)
    s.xgb, s.rgpe, s.switch, s.min_obs = xgb, True, 5, 32
    s.rng = np.random.default_rng(seed)
    s.models = models
    s.dc = xgb.DMatrix(s.Xc)
    s.base_scores = np.array([m.predict(s.dc) for m, _ in models])
    s.Gc = s.Xc[:, :7] / np.array(S.GP_SCALE)
    preds = [gp.predict(s.Gc, return_std=True) for gp in gps]
    s.BM = np.array([m for m, _ in preds])
    s.BS = np.array([np.maximum(sd, 1e-6) for _, sd in preds])
    s.last_weights, s.own_idx, s.refining = None, [], False
    best = []
    for _ in range(max(K)):
        (_, trial), = s.ask_many(1)
        s.tell_trial(trial, {'speedup': math.log(y[trial[0]])})
        best.append(max(s.own_y) if s.own_y else -np.inf)
    return best


def run_gpei(cand, y, seed):
    rng = np.random.default_rng(seed)
    G = S.transfer_frame(cand)[:, :7] / np.array(S.GP_SCALE)
    seen, own, best = [], [], []
    for t in range(max(K)):
        free = np.setdiff1d(np.arange(len(cand)), seen)
        if t == 0:
            j = int(rng.choice(free))
        else:
            gp = S.RankTransferSearcher._fit_gp(G[seen], S.RankTransferSearcher._std(np.array(own)))
            mu, sd = gp.predict(G[free], return_std=True)
            yz = S.RankTransferSearcher._std(np.array(own))
            j = int(free[np.argmax(S.RankTransferSearcher._ei(mu, sd, yz.max()))])
        seen.append(j)
        own.append(math.log(y[j]))
        best.append(max(own))
    return best


rows = []
if os.path.exists(out):
    rows = pd.read_csv(out).to_dict('records')
done = {r['kid'] for r in rows}
kids = [k for k, g in pool.groupby('kid') if g.speedup.max() > 1.05]
shard = os.environ.get('SHARD')
if shard:
    i, n = map(int, shard.split('/'))
    kids = kids[i::n]
model_cache = {}
for n, kid in enumerate(kids):
    if kid in done:
        continue
    cand, trip = candidates(kid)
    y = cand.speedup.values
    opt = math.log(y.max())
    app = cand.app.iloc[0]
    folds = {'LOKO': pool[pool.kid != kid],
             'LOAO': pool[pool.app != app],
             'LOAOs': pool[(pool.app != app) & pool.app.isin(ORIGINAL)]}
    new = app not in ORIGINAL
    for seed in seeds:
        for fold, train in folds.items():
            best = run_reg(train, cand, trip, y, seed)
            rows += [dict(kid=kid, app=app, new=new, method=f'reg-{fold}', seed=seed, k=k,
                          share=max(best[k - 1], 0.0) / opt) for k in K]
            key = (fold, app if fold != 'LOKO' else kid, seed)
            if key not in model_cache:
                model_cache[key] = S.train_rank_models(train, target_kid=None, seed=seed)
            best = run_rgpe(model_cache[key], kernel_gps(train, seed), cand, trip, y, seed)
            rows += [dict(kid=kid, app=app, new=new, method=f'rgpe-{fold}', seed=seed, k=k,
                          share=max(best[k - 1], 0.0) / opt) for k in K]
        best = run_gpei(cand, y, seed)
        rows += [dict(kid=kid, app=app, new=new, method='gpei', seed=seed, k=k,
                      share=max(best[k - 1], 0.0) / opt) for k in K]
    print(f'{n + 1}/{len(kids)} kernels', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
