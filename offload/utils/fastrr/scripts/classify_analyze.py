#!/usr/bin/env python3
"""Labels and rule agreement of the classification study (classify_study.py output), after
~/rr-amd-work/calib/analyze.py of the LLVM 16 study.

usage: classify_analyze.py <classify dir>

Labels each kernel by what the best configuration of its grid dataset changed:
  none      validated speedup <= 1.05
  register  the default binary spills and the best one spills less, or (after the locality
            test) the best configuration uses a different binary (VGPRs or spills)
  locality  the best configuration reads >= 25% fewer DRAM bytes
  other     gain from the runtime parameters alone
and compares them with Search.classify_bottleneck on the median baseline metrics, with the
T4 thresholds (CLASS_THRESHOLDS['nvidia']) and with the MI50 thresholds calibrated on the LLVM 16
landscapes (CLASS_THRESHOLDS['amd']); a 'compute' prediction counts as a match for 'none'.
In sample are the LULESH and miniFE kernels (on which the MI50 thresholds were fit with LLVM 16),
out of sample the HeCBench kernels. Also reports the cost of one baseline profile.
"""
import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RRUtil import Search as S  # noqa: E402


class _MI50:
    Target, CUs, MaxWavesPerCU, ThreadsSMP, WarpSize, MaxThreadBlockSize = (
        'amdgcn-amd-amdhsa', 60, 40, 2560, 64, 1024)


def label(s):
    vs = s['validated_speedup']
    bf = s['best_full']['metrics'].get('dram_read_bytes') or np.nan
    base = s['baseline_full']['metrics'].get('dram_read_bytes') or np.nan
    read_ratio = bf / base
    default_bin = tuple(s['default_binary'])
    best = s['best_screen']
    if not vs == vs:
        return 'unknown', read_ratio
    if vs <= 1.05:
        return 'none', read_ratio
    if default_bin[1] > 0 and best['spill'] < default_bin[1]:
        return 'register', read_ratio
    if read_ratio <= 0.75:
        return 'locality', read_ratio
    if (best['regs'], best['spill']) != default_bin:
        return 'register', read_ratio
    return 'other', read_ratio


def main():
    S.set_device(_MI50)
    amd = dict(S.CLASS_THRESHOLDS['amd'])
    rules = {'T4 rules': dict(S.CLASS_THRESHOLDS['nvidia']), 'MI50 rules': amd}
    rows = []
    for f in sorted(glob.glob(f'{sys.argv[1]}/*.summary.json')):
        s = json.loads(Path(f).read_text())
        lab, rr = label(s)
        pred = {}
        for name, th in rules.items():
            S.CLASS_THRESHOLDS['amd'] = th
            pred[name] = S.classify_bottleneck(s['baseline_metrics'])
        S.CLASS_THRESHOLDS['amd'] = amd
        m = s['baseline_metrics']
        rows.append(dict(kernel=f"{s['app']} l{s['line']}", app=s['app'], label=lab,
                         speedup=s['validated_speedup'], pred=pred, read_ratio=rr,
                         spill=s['default_binary'][1], dram=m.get('dram_throughput_pct', np.nan),
                         occ=m.get('achieved_occupancy_pct', np.nan),
                         valu=m.get('valu_busy_pct', np.nan),
                         prof=float(np.mean(s['profile_seconds'])), n=len(s['profile_seconds'])))
    ok = lambda p, l: p == l or (p == 'compute' and l == 'none')
    for subset, sel in [('in sample (LULESH, miniFE)', lambda r: r['app'] in ('lulesh', 'minife')),
                        ('out of sample (HeCBench)', lambda r: r['app'] not in ('lulesh', 'minife')),
                        ('all', lambda r: True)]:
        sub = [r for r in rows if sel(r) and r['label'] != 'unknown']
        counts = {l: sum(r['label'] == l for r in sub) for l in ('none', 'register', 'locality', 'other')}
        print(f'{subset}: {len(sub)} kernels, labels {counts}')
        for name in rules:
            print(f'  {name}: {sum(ok(r["pred"][name], r["label"]) for r in sub)}/{len(sub)} match')
    print()
    for r in sorted(rows, key=lambda r: (r['label'], -r['speedup'])):
        print(f"{r['kernel']:>18} {r['speedup']:5.2f}x {r['label']:>9} | T4 {r['pred']['T4 rules']:>9} "
              f"MI50 {r['pred']['MI50 rules']:>9} | spill {r['spill']:>5} dram {r['dram']:5.1f} "
              f"occ {r['occ']:5.1f} valu {r['valu']:5.1f} reads {r['read_ratio']:.2f}")
    prof = np.array([r['prof'] for r in rows])
    lm = np.array([r['prof'] for r in rows if r['app'] in ('lulesh', 'minife')])
    print(f'\nbaseline profile (4 rocprof passes): mean {prof.mean():.1f} s, median '
          f'{np.median(prof):.1f} s, max {prof.max():.1f} s over {len(prof)} kernels; '
          f'LULESH and miniFE mean {lm.mean():.1f} s' if len(lm) else '')


if __name__ == '__main__':
    main()
