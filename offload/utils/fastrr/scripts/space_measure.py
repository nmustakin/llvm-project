#!/usr/bin/env python3
"""Resources of one tuner replay process per kept kernel and stage (inside rr_current.sif).

usage: space_measure.py --stage stage1 --out FILE

For every kernel in <stage>/kernels/*.json, a replay of the recorded configuration as
the tuner runs it (--restore-from-device, plus --relocate on stage 2), with no warm-up
and one repetition, is run three times on an otherwise idle GPU. Recorded per run:
the wall time of the process (loading the recorded state and one replay), the peak
device memory of the GPU (nvidia-smi, or the sysfs VRAM counter on AMD, sampled every 20 ms, minus the idle usage),
the peak host memory (RSS) of the process, and the size of the kernel's own files.
The block store of a stage 2 recording is shared by the kernels of its directory and
is reported per application by record_app.py (sizes-kept.csv), not per kernel.
"""
import argparse, glob, json, os, subprocess, sys, threading, time
from pathlib import Path

import pandas as pd

from paths import FASTRR, X, BUILDS
sys.path.insert(0, FASTRR)
from RRUtil import Upstream  # noqa: E402
FLAGS = {'stage1': ['--restore-from-device'],
         'stage2': ['--restore-from-device', '--relocate']}


def _amd_vram_file(gpu):
    """sysfs file of the VRAM in use (bytes) on ROCm device `gpu` (KFD order)."""
    from RRUtil.GPUMonitor import AMDMonitor
    loc = AMDMonitor().gpus[int(gpu)][1]
    pci = f'{loc >> 8:02x}:{(loc >> 3) & 0x1f:02x}.{loc & 7}'
    for card in Path('/sys/class/drm').glob('card[0-9]*'):
        if not card.name[4:].isdigit():
            continue
        if os.path.realpath(card / 'device').endswith(pci):
            return card / 'device' / 'mem_info_vram_used'
    raise SystemExit(f'no DRM card for GPU {gpu} ({pci})')


_vram = {}


def gpu_mib(gpu):
    if Upstream.TARGET == 'amdgpu':
        if gpu not in _vram:
            _vram[gpu] = _amd_vram_file(gpu)
        return int(_vram[gpu].read_text()) >> 20
    out = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def measure(cmd, env, gpu):
    """Run cmd; return (rc, wall seconds, peak GPU MiB, peak RSS MiB, output)."""
    peak, done = [0], threading.Event()

    def sampler():
        while not done.is_set():
            peak[0] = max(peak[0], gpu_mib(gpu))
            time.sleep(0.02)

    thread = threading.Thread(target=sampler)
    thread.start()
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout.read()
    _, status, usage = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0
    done.set()
    thread.join()
    return os.waitstatus_to_exitcode(status), wall, peak[0], usage.ru_maxrss / 1024, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()
    build = BUILDS[args.stage]
    gpu = os.environ[os.environ.get('FASTRR_VISIBLE', 'CUDA_VISIBLE_DEVICES')]
    env = Upstream.environment(build=build)
    rows = []
    for reg in sorted(glob.glob(f'{X}/{args.stage}/kernels/*.json')):
        app = Path(reg).stem
        for line, js in json.loads(Path(reg).read_text()).items():
            stem = js[:-len('.json')]
            files = sum(os.path.getsize(f) for f in glob.glob(stem + '.*'))
            info = json.loads(Path(js).read_text())
            idle = gpu_mib(gpu)
            for run in range(args.runs):
                rc, wall, peak, rss, out = measure(
                    [f'{build}/bin/llvm-omp-kernel-replay', '--ignore-limits',
                     *FLAGS[args.stage], js], env, gpu)
                rows.append(dict(stage=args.stage, app=app, line=int(line), run=run, rc=rc,
                                 wall_s=wall, device_mib=peak - idle, host_rss_mib=rss,
                                 kernel_files_mib=files / 2**20,
                                 valloc_mib=info['VAllocSize'] / 2**20,
                                 relocated='Relocated the replay memory' in out))
                if rc != 0:
                    print(out[-2000:], flush=True)
            r = rows[-1]
            print(f'{args.stage} {app} l{line}: rc={r["rc"]} wall {wall:.2f} s, device '
                  f'{r["device_mib"]} MiB, host {r["host_rss_mib"]:.0f} MiB, files '
                  f'{r["kernel_files_mib"]:.1f} MiB, relocated {r["relocated"]}', flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
