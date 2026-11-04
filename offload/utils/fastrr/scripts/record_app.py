#!/usr/bin/env python3
"""Build an application with upstream clang and record it with a stage's runtime
(inside rr_current.sif).

usage: record_app.py <stage> <app> [--keep LINES|all] [--memsize GiB]

- Builds the application once into apps/<app>, with the clang of the stage 2
  build (both stages build clang from the same source, 2d927048fbe5; the
  space-saving changes are in the offload runtime and tools only).
- Records it with the runtime of the stage (stage1: rr-taco-batch, stage2:
  rr-taco-space) into <REC>/<stage>/<app>, timing the run.
- Appends the size of the whole recording to sizes.csv (per file type) before
  deleting the kernels the experiments do not tune (--keep), and writes the
  kept kernels to <stage>/kernels/<app>.json (line -> JSON path), compacting the
  block store of a stage 2 recording, and the size of what is kept to
  sizes-kept.csv.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

from paths import FASTRR, X, REC, BUILDS, HEC
RR = f'{FASTRR}/benchmarks'
FLAGS = ('-fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_75 '
         '-fopenmp-offload-mandatory -foffload-lto -Wl,--embed-bitcode')
# name: (source dirs to copy, build command, run command). The inputs are the
# paper's (the benchmark modules under RR/benchmarks); LULESH runs 5 iterations,
# because each kernel is recorded at its first launch.
APPS = {
    'lulesh': ([f'{RR}/lulesh-omp'], f'make LECPP=clang++ LEFLAGS="{FLAGS}" -j16',
               './lulesh -s 128 -i 5 -r 11 -b 1 -c 1'),
    'minife': ([f'{RR}/miniFE-omp'],
               f'make all LECPP=clang++ LECC=clang LEFLAGS="{FLAGS}" BUILDDIR=. EXEDIR=. -j16',
               './miniFE.x -nx 200 -ny 200 -nz 200'),
}
HECBENCH = {
    'xsbench': ('main -s small -m event -l 1700000 -r 1', ['xsbench-cuda']),
    'rsbench': ('main -s small -m event -l 100000', []),
    'simplemoc': ('main -s 500000 -e 128 -n 10', []),
    'su3': ('main -i 10 -l 32 -t 128 -v 3 -w 1', []),
    'minibude': (f'main --deck {HEC}/minibude-sycl/data/bm1 --wgsize 256 --iterations 2', []),
    'haccmk': ('main 1000', []),
    'amgmk': ('main', []),
    'minisweep': ('main --niterations 1', []),
    'testSNAP': ('main --nsteps 2', []),
    's3d': ('main -q -n 2 -s 4', []),
    'miniWeather': ('main', ['miniWeather-cuda']),
    'sw4ck': (f'main {HEC}/sw4ck-cuda/sw4ck.in 2', ['sw4ck-cuda']),
}
for name, (args, extra) in HECBENCH.items():
    dirs = [f'{HEC}/{name}-omp', f'{HEC}/include'] + [f'{HEC}/{d}' for d in extra]
    APPS[name] = (dirs, 'make -f Makefile.rr -j16', './' + args)
# Second inputs of the input study (RR/benchmarks/inputs): name: (source app, run arguments).
SECOND = {'s3d-s3': ('s3d', 'main -q -n 2 -s 3'),
          'su3-l24': ('su3', 'main -i 10 -l 24 -t 128 -v 3 -w 1'),
          'simplemoc-s1500k': ('simplemoc', 'main -s 1500000 -e 128 -n 10'),
          'minisweep-c884': ('minisweep', 'main --niterations 1 --ncell_x 8 --ncell_y 8 --ncell_z 4')}
for name, (src, args) in SECOND.items():
    APPS[name] = (APPS[src][0], APPS[src][1], './' + args)
APPS['miniWeather-1600'] = ([f'{HEC}/miniWeather-omp-1600', f'{HEC}/include', f'{HEC}/miniWeather-cuda'],
                            'make -f Makefile.rr -j16', './main')
APPS['minife-150'] = (APPS['minife'][0], APPS['minife'][1], './miniFE.x -nx 150 -ny 150 -nz 150')


def environment(build):
    env = dict(os.environ)
    omp = f'{build}/runtimes/runtimes-bins/openmp/runtime/src'
    libs = f'{omp}:{build}/lib:{build}/lib/x86_64-unknown-linux-gnu'
    env['PATH'] = f'{build}/bin:/usr/local/cuda/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = f'{libs}:/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
    env['LIBRARY_PATH'] = libs
    env['OMP_TARGET_OFFLOAD'] = 'mandatory'
    return env


def file_type(path):
    name = os.path.basename(path)
    if os.path.isdir(path):
        return 'dir:' + name
    return name.split('.', 1)[1] if '.' in name else name


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=BUILDS)
    ap.add_argument('app', choices=APPS)
    ap.add_argument('--keep', default='all',
                    help='comma-separated source lines, all, or none (size only)')
    ap.add_argument('--memsize', type=int, default=4, help='LIBOMPTARGET_RECORD_MEMSIZE in GiB')
    args = ap.parse_args()
    dirs, build_cmd, run_cmd = APPS[args.app]

    # Build once (stage 2 clang; identical to stage 1's).
    build_app = SECOND.get(args.app, (args.app,))[0] if args.app != 'minife-150' else 'minife'
    src_root = Path(X, 'apps', build_app)
    src = src_root / os.path.basename(dirs[0])
    if not (src / '.built').exists():
        shutil.rmtree(src_root, ignore_errors=True)
        for d in dirs:
            shutil.copytree(d, src_root / os.path.basename(d), symlinks=True,
                            ignore=shutil.ignore_patterns('*.o', 'main', 'lulesh', 'miniFE.x'))
        p = subprocess.run(build_cmd, shell=True, cwd=src, env=environment(BUILDS['stage2']),
                           capture_output=True, text=True, timeout=3600)
        Path(X, 'logs').mkdir(exist_ok=True)
        Path(X, 'logs', f'build-{build_app}.log').write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            sys.exit(f'{args.app}: build failed rc={p.returncode}')
        (src / '.built').touch()

    rec = Path(REC, args.stage, args.app)
    shutil.rmtree(rec, ignore_errors=True)
    rec.mkdir(parents=True)
    env = environment(BUILDS[args.stage])
    env.update(LIBOMPTARGET_RECORD='1', LIBOMPTARGET_RECORD_MEMSIZE=str(args.memsize << 30),
               LIBOMPTARGET_RECORD_DIR=str(rec))
    t0 = time.perf_counter()
    p = subprocess.run(run_cmd, shell=True, cwd=src, env=env, capture_output=True, text=True,
                       timeout=7200)
    seconds = time.perf_counter() - t0
    Path(X, args.stage, 'logs').mkdir(parents=True, exist_ok=True)
    Path(X, args.stage, 'logs', f'record-{args.app}.log').write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        sys.exit(f'{args.app}: recording failed rc={p.returncode}')

    # Size of the whole recording, by file type, before deleting anything.
    by_type = defaultdict(int)
    for f in rec.iterdir():
        by_type[file_type(str(f))] += du(f) if f.is_dir() else f.stat().st_size
    jsons = sorted(glob.glob(f'{rec}/*.json'), key=os.path.getmtime)
    with open(Path(X, args.stage, 'sizes.csv'), 'a') as out:
        for kind, size in sorted(by_type.items()):
            out.write(f'{args.app},{args.memsize},{len(jsons)},{seconds:.2f},{kind},{size}\n')
    print(f'{args.stage} {args.app}: {len(jsons)} kernel recordings, '
          f'{sum(by_type.values()) / 2**20:.1f} MiB, {seconds:.1f} s', flush=True)

    # Keep the first recording of each kept line.
    kept = {}
    for j in jsons:
        line = re.search(r'_l(\d+)$', json.load(open(j))['Name'])
        line = int(line.group(1)) if line else None
        if line is not None and line not in kept and args.keep != 'none' and (
                args.keep == 'all' or str(line) in args.keep.split(',')):
            kept[line] = j
    if args.keep != 'all':
        missing = set(map(int, args.keep.split(','))) - set(kept) if args.keep != 'none' else ()
        if missing:
            print(f'{args.app}: lines {sorted(missing)} were not recorded', flush=True)
        stems = {Path(j).name[:-len('.json')] for j in kept.values()}
        for f in rec.iterdir():
            if (f.is_file() and not f.name.startswith('record_blocks')
                    and f.name.split('.', 1)[0] not in stems):
                f.unlink()
        # Drop the blocks only the deleted kernels used (stage 2 block store).
        if list(rec.glob('record_blocks.*.store')):
            p = subprocess.run([f'{BUILDS[args.stage]}/bin/llvm-omp-kernel-replay',
                                '--compact-store', str(rec)], env=env, capture_output=True,
                               text=True)
            if p.returncode != 0:
                sys.exit(f'{args.app}: compacting the block store failed: {p.stderr}')
    kept_size = sum(du(f) if f.is_dir() else f.stat().st_size for f in rec.iterdir())
    with open(Path(X, args.stage, 'sizes-kept.csv'), 'a') as out:
        out.write(f'{args.app},{len(kept)},{kept_size}\n')
    Path(X, args.stage, 'kernels').mkdir(exist_ok=True)
    Path(X, args.stage, 'kernels', f'{args.app}.json').write_text(
        json.dumps({str(k): v for k, v in sorted(kept.items())}, indent=1))
    print(f'{args.app}: kept lines {sorted(kept)}', flush=True)


if __name__ == '__main__':
    main()
