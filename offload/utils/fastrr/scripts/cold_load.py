#!/usr/bin/env python3
"""Cold and warm state loading of one tuner replay process per kept kernel (inside rr_current.sif).

usage: cold_load.py --stage stage1 --out FILE

For every kernel in <stage>/kernels/*.json: the page cache of its recording files (and of the
directory's block store) is dropped with posix_fadvise(DONTNEED), then one replay of the
recorded configuration as the tuner runs it (--restore-from-device, plus --relocate on stage 2;
no warm-up, one repetition) is timed; a second replay right after is the warm time.
The difference is what the first replay process of a tuning run pays to read its recording.
"""
import argparse, glob, json, os, subprocess, time
from pathlib import Path

import pandas as pd

from paths import X, BUILDS
FLAGS = {'stage1': ['--restore-from-device'], 'stage2': ['--restore-from-device', '--relocate']}


def evict(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    build = BUILDS[args.stage]
    env = dict(os.environ)
    omp = f'{build}/runtimes/runtimes-bins/openmp/runtime/src'
    env['LD_LIBRARY_PATH'] = (f'{omp}:{build}/lib:{build}/lib/x86_64-unknown-linux-gnu:'
                              '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', ''))
    env['OMP_TARGET_OFFLOAD'] = 'mandatory'
    rows = []
    for reg in sorted(glob.glob(f'{X}/{args.stage}/kernels/*.json')):
        app = Path(reg).stem
        for line, js in json.loads(Path(reg).read_text()).items():
            stem = js[:-len('.json')]
            files = glob.glob(stem + '.*') + glob.glob(str(Path(js).parent / 'record_blocks.*.store'))
            size = sum(os.path.getsize(f) for f in files)
            cmd = [f'{build}/bin/llvm-omp-kernel-replay', '--ignore-limits', *FLAGS[args.stage], js]
            evict(files)
            times = []
            for _ in range(2):
                t0 = time.perf_counter()
                p = subprocess.run(cmd, env=env, capture_output=True, text=True)
                times.append(time.perf_counter() - t0)
            rows.append(dict(stage=args.stage, app=app, line=int(line), rc=p.returncode,
                             cold_s=times[0], warm_s=times[1], read_mib=size / 2**20))
            print(f'{args.stage} {app} l{line}: cold {times[0]:.2f} s, warm {times[1]:.2f} s, '
                  f'{size / 2**20:.0f} MiB read, rc {p.returncode}', flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
