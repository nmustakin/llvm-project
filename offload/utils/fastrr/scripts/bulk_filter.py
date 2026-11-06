#!/usr/bin/env python3
"""Select the kernels of a bulk application for grid datasets, as in the LLVM 16 bulk study.

usage: bulk_filter.py --stage stage2 --app h-adam [--min-ms 0.1 --max-ms 100 --max-mib 1024]

For every kernel of <stage>/kernels/<app>.json (record_app.py --keep all), keeps those with a loop
trip count, a recorded memory of at most --max-mib (the size of the input snapshot), and a default
time, replayed from the recording with the tuner's protocol (100 ms warm-up on AMD, the median of
ten launches, restored from the device copy, relocated), between --min-ms and --max-ms. Moves the
full registry to <stage>/kernels-all/<app>.json, writes the kept kernels to
<stage>/kernels/<app>.json, and appends one row per kernel to <stage>/bulk-filter.csv.
"""
import argparse, csv, json, os, re, struct, sys
from pathlib import Path

from paths import FASTRR, X, BUILDS
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402

BLOCKS_MAGIC = b'OMPRRBL2'


def memory_bytes(json_path):
    """Size of the recorded memory: the input snapshot's size (block list header or raw file)."""
    f = Path(json_path[:-len('.json')] + '.record_input')
    if not f.exists():
        return None
    with open(f, 'rb') as fd:
        head = fd.read(16)
    if len(head) == 16 and head[:6] == BLOCKS_MAGIC[:6]:  # any block format version
        return struct.unpack_from('<Q', head, 8)[0]
    return f.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='stage2')
    ap.add_argument('--app', required=True)
    ap.add_argument('--min-ms', type=float, default=0.1)
    ap.add_argument('--max-ms', type=float, default=100.0)
    ap.add_argument('--max-mib', type=float, default=1024)
    args = ap.parse_args()
    reg_path = Path(X, args.stage, 'kernels', f'{args.app}.json')
    full_path = Path(X, args.stage, 'kernels-all', f'{args.app}.json')
    full_path.parent.mkdir(exist_ok=True)
    if not full_path.exists():
        full_path.write_text(reg_path.read_text())
    registry = json.loads(full_path.read_text())
    warm = getattr(Upstream.get_device(0), 'TimingWarmupMs', 100)
    env = Upstream.environment(build=BUILDS[args.stage])
    kept, rows = {}, []
    for line, js in registry.items():
        info = json.loads(Path(js).read_text())
        trip = int(info.get('LoopTripCount') or 0)
        mem = memory_bytes(js)
        row = dict(app=args.app, line=line, trip=trip, mib=round((mem or 0) / 2**20, 1),
                   ms='', keep=0, reason='')
        if trip <= 0:
            row['reason'] = 'no trip count'
        elif mem is None or mem > args.max_mib * 2**20:
            row['reason'] = 'memory'
        else:
            import subprocess
            try:
                p = subprocess.run(
                    [f'{BUILDS[args.stage]}/bin/llvm-omp-kernel-replay', '--ignore-limits',
                     '--relocate', '--restore-from-device', '--warmup-ms', str(warm),
                     '--repetitions', '10', js], env=env, capture_output=True, text=True,
                    timeout=600, cwd=os.path.dirname(js))
                # Without --batch, the tool prints each launch's device time.
                ns = sorted(int(t) for t in
                            re.findall(r'Replay device time \(\d+\): (\d+) ns', p.stdout))
                ms = (ns[len(ns) // 2] + ns[(len(ns) - 1) // 2]) / 2e6 \
                    if ns and p.returncode == 0 else None
            except subprocess.TimeoutExpired:
                ms = None
            if ms is None:
                row['reason'] = 'replay failed'
            else:
                row['ms'] = f'{ms:.4f}'
                if args.min_ms <= ms <= args.max_ms:
                    row['keep'] = 1
                    kept[line] = js
                else:
                    row['reason'] = 'time'
        rows.append(row)
        print(f"{args.app} l{line}: trip {trip}, {row['mib']} MiB, {row['ms'] or '-'} ms, "
              f"{'keep' if row['keep'] else row['reason']}", flush=True)
    reg_path.write_text(json.dumps(kept, indent=1))
    out = Path(X, args.stage, 'bulk-filter.csv')
    new = not out.exists()
    with open(out, 'a', newline='') as fd:
        w = csv.DictWriter(fd, fieldnames=list(rows[0]) if rows else ['app'])
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f'{args.app}: kept {sorted(kept, key=int)}')


if __name__ == '__main__':
    main()
