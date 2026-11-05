"""The tuner on upstream LLVM record/replay (llvm-omp-kernel-replay).

Kernel.Search and RRUtil.Search are unchanged. This module replaces the parts
that talk to the record/replay tooling of the LLVM 16 fork:

- Recordings are upstream's: <stem>.json, .record_input, .record_output,
  .globals, the compiled .image and the IR .bc (recorded when the application
  embeds its device bitcode, -Wl,--embed-bitcode), plus .pointers and the
  directory's block store (record_blocks.*) with the space-saving changes.
- UpstreamNVIDIA.deviceCompile builds a launch-bound binary from the recorded
  IR: the kernel is extracted once (prepare_kernel_ir), its nvvm.maxntid and
  nvvm.minctasm attributes are set, and llc and ptxas compile it into
  <stem>.image, which the replay tool loads. UpstreamAMDGPU does the same for
  AMD GPUs (FASTRR_TARGET=amdgpu) with amdgpu-flat-work-group-size and
  amdgpu-waves-per-eu, llc and ld.lld.
- UpstreamKernel times configurations with the replay tool's --batch mode and
  the paper's protocol: a warm-up of Kernel.TimingWarmupMs before the first
  configuration of a process and Kernel.BatchRewarmMs before each later one,
  then the median of Kernel.TimingRepeats launches timed with device events,
  each launch restoring the recorded memory from a device copy
  (--restore-from-device). A full measurement (Kernel.sample) is a batch
  replay of one configuration per process. Upstream has no CUPTI metrics or
  NVML energy window, so metrics are skipped and energy is NaN.

The LLVM build is RR_UPSTREAM_BUILD (bin/, lib/ and the libomp directory), and
RR_UPSTREAM_REPLAY_FLAGS adds flags to every replay. FASTRR_TARGET selects the
GPU vendor (nvptx, the default, or amdgpu); on AMD, FASTRR_ROCM is the ROCm
installation (6.0 or later: the AMDGPU plugin maps the recording at its
address with hsa_amd_vmem_*) and FASTRR_AMD_ARCH the gfx target.
"""
import json
import os
import re
import subprocess
from pathlib import Path

from RRUtil import Device
from RRUtil.Benchmark import Kernel

BUILD = os.environ.get('RR_UPSTREAM_BUILD', '/data2/nmn/llvm/llvm-project-taco/build')
TARGET = os.environ.get('FASTRR_TARGET', 'nvptx')
if TARGET not in ('nvptx', 'amdgpu'):
    raise SystemExit(f'FASTRR_TARGET must be nvptx or amdgpu, not {TARGET}')
CUDA = os.environ.get('CUDA_HOME', '/usr/local/cuda')
ROCM = os.environ.get('FASTRR_ROCM', os.environ.get('ROCM_PATH', '/opt/rocm'))
ARCH = 'sm_75' if TARGET == 'nvptx' else os.environ.get('FASTRR_AMD_ARCH', 'gfx906')
if TARGET == 'amdgpu':
    # Device.AMDGPU queries the agents with $ROCM_PATH/bin/rocminfo.
    os.environ['ROCM_PATH'] = ROCM
# Offload flags of the applications: device bitcode embedded for the recordings.
_TRIPLE = {'nvptx': 'nvptx64', 'amdgpu': 'amdgcn-amd-amdhsa'}[TARGET]
OFFLOAD_FLAGS = (f'-fopenmp -fopenmp-targets={_TRIPLE} -Xopenmp-target={_TRIPLE} '
                 f'-march={ARCH} -fopenmp-offload-mandatory -foffload-lto '
                 '-Wl,--embed-bitcode')
# Extra flags of every replay, e.g. --relocate with the space-saving changes.
REPLAY_FLAGS = os.environ.get('RR_UPSTREAM_REPLAY_FLAGS', '').split()
# Recording files a work directory links to (.pointers: relocatable recordings).
RECORDING_SUFFIXES = ('.json', '.record_input', '.record_output', '.globals', '.bc',
                      '.pointers')


def tool(name):
    return f'{BUILD}/bin/{name}'


def environment(base=None, build=None):
    """Environment of the upstream LLVM build: tools, offload and libomp libraries,
    and the vendor's (CUDA or ROCm) tools and libraries."""
    build = build or BUILD
    env = dict(base or os.environ)
    omp = f'{build}/runtimes/runtimes-bins/openmp/runtime/src'
    vendor_bin, vendor_lib = ((f'{CUDA}/bin', f'{CUDA}/lib64') if TARGET == 'nvptx'
                              else (f'{ROCM}/bin', f'{ROCM}/lib'))
    env['PATH'] = f'{build}/bin:{vendor_bin}:' + env.get('PATH', '')
    libs = f'{omp}:{build}/lib:{build}/lib/x86_64-unknown-linux-gnu'
    env['LD_LIBRARY_PATH'] = f'{libs}:{vendor_lib}:' + env.get('LD_LIBRARY_PATH', '')
    if TARGET == 'amdgpu':
        env['ROCM_PATH'] = ROCM
    env['LIBRARY_PATH'] = libs + ':' + env.get('LIBRARY_PATH', '')
    env['OMP_TARGET_OFFLOAD'] = 'mandatory'
    return env


def run(cmd, cwd=None, timeout=None):
    """Run a command list in the upstream environment; return (rc, stdout, stderr).
    The environment is built per call, so that the GPU a forked replay process is
    restricted to (CUDA/ROCR_VISIBLE_DEVICES, set by Kernel._run_on_devices) applies."""
    try:
        p = subprocess.run(cmd, cwd=cwd, env=environment(), capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 'timeout', e.stdout or '', e.stderr or ''


def kernel_line(name):
    """Source line of an offload kernel name (..._l649 -> 649)."""
    m = re.search(r'_l(\d+)$', name)
    return int(m.group(1)) if m else None


def prepare_kernel_ir(bc, name, out_ll):
    """Write the IR of kernel `name` from the recorded module `bc` to `out_ll`.

    The other kernels are removed, as the fork's extractKernels did, so that each
    launch-bound compile only compiles this kernel: they are internalized and
    dropped by globaldce, together with what only they use. Every other public
    symbol stays, because the replay looks up the recorded globals in the image,
    and local symbols keep their linkage (llvm-extract --delete would make them
    external, and NVPTX only renames local symbols that PTX cannot spell)."""
    rc, out, err = run([tool('llvm-dis'), bc, '-o', '-'])
    if rc != 0:
        raise RuntimeError(f'llvm-dis {bc} failed: {err}')
    kernels = re.findall(r'^define [^\n]*(?:ptx|amdgpu)_kernel [^\n]*@([\w.$]+)\(', out, re.M)
    if name not in kernels:
        raise RuntimeError(f'kernel {name} not found in {bc}')
    public = []
    for m in re.finditer(r'^(?:@("?[^"\s]+"?) = ([^\n]*)|define ([^@\n]*)@("?[^"(\s]+"?)\()',
                         out, re.M):
        sym, prefix = (m[1], m[2]) if m[1] else (m[4], m[3])
        linkage = prefix.split()[0] if prefix.split() else ''
        if sym.strip('"') in kernels and sym.strip('"') != name:
            continue
        if linkage not in ('private', 'internal'):
            public.append(sym.strip('"'))
    api = out_ll + '.public'
    Path(api).write_text('\n'.join(public) + '\n')
    rc, _, err = run([tool('opt'), '-passes=internalize,globaldce',
                      f'-internalize-public-api-file={api}', '-S', bc, '-o', out_ll])
    os.unlink(api)
    if rc != 0:
        raise RuntimeError(f'opt failed: {err}')


_LAUNCH_BOUNDS = ('nvvm.maxntid', 'nvvm.minctasm')


def set_launch_bounds(ir, name, max_threads=None, min_teams=None):
    """IR text with the launch bounds of kernel `name` replaced: the kernel gets
    a copy of its attribute group without nvvm.maxntid and nvvm.minctasm, plus the
    given values. None keeps the recorded value."""
    return set_kernel_attributes(ir, name, dict(zip(_LAUNCH_BOUNDS, (max_threads, min_teams))))


def set_kernel_attributes(ir, name, values):
    """IR text in which kernel `name` gets a copy of its attribute group with the
    string attributes `values` ({key: value}) replaced. A value of None keeps
    the recorded attribute."""
    values = {k: v for k, v in values.items() if v is not None}
    if not values:
        return ir
    # The attribute group follows the parameters, possibly followed by metadata:
    # define ... @name(...) local_unnamed_addr #12 !guid !53 {
    define = re.search(r'^define [^\n]*@' + re.escape(name) + r'\([^\n]*\)[^\n#]*#(\d+)[^\n]*\{$',
                       ir, re.M)
    if not define:
        raise RuntimeError(f'no attribute group on kernel {name}')
    group = define.group(1)
    body = re.search(r'^attributes #' + group + r' = \{ ([^\n]*) \}$', ir, re.M).group(1)
    attrs = body
    for key, value in values.items():
        attrs = re.sub(r'"' + re.escape(key) + r'"="[^"]*" ?', '', attrs).strip()
        attrs += f' "{key}"="{value}"'
    new = str(max(int(g) for g in re.findall(r'^attributes #(\d+) ', ir, re.M)) + 1)
    ir = ir[:define.start(1)] + new + ir[define.end(1):]
    return ir + f'\nattributes #{new} = {{ {attrs} }}\n'


class UpstreamNVIDIA(Device.NVIDIA):
    """Device.NVIDIA whose binaries are built from the upstream recorded IR."""
    _ir_cache = {}

    def deviceCompile(self, original, Name, HashName, **kwargs):
        """Compile `original` (the kernel's IR, prepare_kernel_ir) with the given
        MaxThreads (nvvm.maxntid) and MinTeams (nvvm.minctasm) into
        <HashName>.image. Returns the ptxas statistics, as the fork's
        deviceCompile does."""
        key = os.path.abspath(original)
        if key not in self._ir_cache:
            self._ir_cache[key] = Path(original).read_text()
        mt, mn = kwargs.get('MaxThreads'), kwargs.get('MinTeams')
        ir = set_launch_bounds(self._ir_cache[key], Name, None if mt is None else int(mt),
                               None if mn is None else int(mn))
        Path(f'{HashName}.tune.ll').write_text(ir)
        rc, _, err = run([tool('llc'), '-O3', f'-mcpu={ARCH}', f'{HashName}.tune.ll',
                          '-o', f'{HashName}.ptx'])
        if rc != 0:
            raise RuntimeError(f'llc failed: {err}')
        rc, out, err = run([f'{CUDA}/bin/ptxas', '-v', '--gpu-name', ARCH, '-O3',
                            '--warn-on-spills', f'{HashName}.ptx', '-o', f'{HashName}.image'])
        if rc != 0:
            raise RuntimeError(f'ptxas failed: {err}')
        return self.parsePTXAS(out, err)


class UpstreamAMDGPU(Device.AMDGPU):
    """Device.AMDGPU whose binaries are built from the upstream recorded IR.

    The launch bounds are the attributes the fork's AMDGPU attribute pass set:
    MaxThreads as amdgpu-flat-work-group-size 1,MaxThreads and MinTeams as
    amdgpu-waves-per-eu (Device.AMDGPU.getWavesPerEU). The recorded bitcode is
    the linked module before the link-time optimization (OpenMPOpt and the rest
    of the O3 pipeline, which the plugin's JIT runs when it loads the recorded
    image), so opt -O3 optimizes the kernel after its attributes are set; llc
    compiles it into an object, reporting its resource usage
    (kernel-resource-usage remarks), and ld.lld links the object into the code
    object <stem>.image. Without opt the default binary of minife l169 is 5.8x
    slower than the recorded image; with it the two take the same time."""
    _ir_cache = {}

    def deviceCompile(self, original, Name, HashName, **kwargs):
        """Returns the resource usage in Device.AMDGPU's layout ({kernel: [VGPRs,
        LDS bytes, SGPRs, spill bytes per lane, 0]})."""
        key = os.path.abspath(original)
        if key not in self._ir_cache:
            self._ir_cache[key] = Path(original).read_text()
        mt, mn = kwargs.get('MaxThreads'), kwargs.get('MinTeams')
        attrs = {'amdgpu-flat-work-group-size': None if mt is None else f'1,{int(mt)}',
                 'amdgpu-waves-per-eu':
                     None if mn is None else str(self.getWavesPerEU(mt, int(mn)))}
        ir = set_kernel_attributes(self._ir_cache[key], Name, attrs)
        Path(f'{HashName}.tune.ll').write_text(ir)
        rc, _, err = run([tool('opt'), '-O3', f'-mtriple={_TRIPLE}', f'-mcpu={self.Arch}',
                          f'{HashName}.tune.ll', '-o', f'{HashName}.tune.bc'])
        if rc != 0:
            raise RuntimeError(f'opt failed: {err}')
        rc, _, err = run([tool('llc'), '-O3', f'-mtriple={_TRIPLE}', f'-mcpu={self.Arch}',
                          '-filetype=obj', '-pass-remarks-analysis=kernel-resource-usage',
                          f'{HashName}.tune.bc', '-o', f'{HashName}.o'])
        if rc != 0:
            raise RuntimeError(f'llc failed: {err}')
        rc, _, lerr = run([tool('ld.lld'), '-shared', '--no-undefined', f'{HashName}.o',
                           '-o', f'{HashName}.image'])
        if rc != 0:
            raise RuntimeError(f'ld.lld failed: {lerr}')
        os.unlink(f'{HashName}.o')
        os.unlink(f'{HashName}.tune.bc')
        return self.parseResourceUsage(err)

    def recordPrefix(self):
        # Upstream maps the recording at a fixed address (hsa_amd_vmem_*); no
        # need to disable ASLR as with the fork.
        return ''


def get_device(device_no=0):
    if TARGET == 'amdgpu':
        return UpstreamAMDGPU(None, device_no)
    return UpstreamNVIDIA.Load(None, device_no)


def stage(json_path, work):
    """A work directory for tuning one recorded kernel: links to its recording
    files and the kernel's extracted IR. Returns the kernel's file stem."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    stem = Path(json_path).name[:-len('.json')]
    shared = list(Path(json_path).parent.glob('record_blocks.*'))   # block store
    for src in [Path(json_path).with_name(stem + s) for s in RECORDING_SUFFIXES] + shared:
        dst = work / src.name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    ll = work / f'{stem}.kernel.ll'
    if not ll.exists():
        name = json.loads(Path(json_path).read_text())['Name']
        prepare_kernel_ir(str(work / f'{stem}.bc'), name, str(ll))
    return stem


class UpstreamKernel(Kernel):
    """A recorded kernel tuned with upstream llvm-omp-kernel-replay. The current
    directory must be its work directory (stage)."""
    # No CUPTI or rocprof metrics in upstream replays.
    SkipMetrics = True

    def __init__(self, device, stem, dbDir):
        from RRUtil.Search import set_device
        set_device(device)
        # The device's warm-up (100 ms on the MI50), as Kernel.__init__ sets it.
        if 'RR_TIMING_WARMUP_MS' not in os.environ:
            Kernel.TimingWarmupMs = getattr(device, 'TimingWarmupMs', Kernel.TimingWarmupMs)
        info = json.loads(Path(f'{stem}.json').read_text())
        self.Name = info['Name']
        self.recorded = True
        self.HashName = stem
        self.original = f'{stem}.kernel.ll'
        self.Profiler = device
        self.Replayer = None
        self.dbDir = dbDir
        self.profiled = False
        self.NumRegs = self.SMem = self.NumTeams = self.NumThreads = self.ExecTime = -1
        self.Metrics = {}

    def _device_workdir(self, d):
        sub = os.path.join(os.getcwd(), f'dev{d}')
        os.makedirs(sub, exist_ok=True)
        shared = [str(f) for f in Path(os.getcwd()).glob('record_blocks.*')]   # block store
        for f in [os.path.join(os.getcwd(), self.HashName + s)
                  for s in RECORDING_SUFFIXES + ('.kernel.ll',)] + shared:
            dst = os.path.join(sub, os.path.basename(f))
            if os.path.lexists(f) and not os.path.lexists(dst):
                os.symlink(os.path.realpath(f), dst)
        return sub

    def _replay_batch(self, configs, warmup_ms, restore_from_device=True):
        """One replay process timing `configs` (dicts of NumTeams/NumThreads) on the
        current binary. Returns ({position: seconds}, stdout)."""
        batch_file = f'{self.HashName}.batch.txt'
        with open(batch_file, 'w') as fd:
            for c in configs:
                fd.write(f"{int(c.get('NumTeams') or 0)} {int(c.get('NumThreads') or 0)}\n")
        rc, stdout, stderr = run(
            [tool('llvm-omp-kernel-replay'), '--batch', batch_file,
             '--repetitions', str(self.TimingRepeats), '--warmup-ms', str(warmup_ms),
             '--rewarm-ms', str(self.BatchRewarmMs),
             *(['--restore-from-device'] if restore_from_device else []),
             '--ignore-limits', *REPLAY_FLAGS, f'{self.HashName}.json'], timeout=900)
        times, current = {}, None
        for line in (stdout or '').splitlines():
            m = re.match(r'\|\|ReplayConfig\|\| index: (\d+)', line)
            if m:
                current = int(m.group(1))
                continue
            m = re.match(r'\|\|ReplayTiming\|\| duration_seconds: (\S+)', line)
            if m and current is not None:
                times[current] = float(m.group(1))
        if rc != 0 and len(times) < len(configs):
            print(f'Replay stopped after {len(times)} of {len(configs)} configurations: '
                  f'{(stderr or "").strip().splitlines()[-1:]}', flush=True)
        return times, stdout

    def sample_batch(self, configs, deviceID=0):
        """Time several launch configurations of one binary in one replay process,
        as Kernel.sample_batch does with the fork. Returns, in order, a list of
        (valid, duration_seconds). A configuration that makes the replay fail is
        marked invalid and the rest continue in a new process."""
        lb = {k: configs[0][k] for k in ('MaxThreads', 'MinTeams')
              if configs[0].get(k) is not None}
        device_handle = self._gpu_handle(deviceID)
        stats = self.Profiler.deviceCompile(self.original, self.Name, self.HashName, **lb)
        regs = stats[self.Name][0] if stats and self.Name in stats else None
        out = [None] * len(configs)
        todo = []
        for i, c in enumerate(configs):
            nt = c.get('NumThreads')
            if regs is not None and nt is not None and not self.Profiler.blockFits(regs, nt):
                out[i] = (False, None)
            else:
                todo.append(i)
        contended_retries = 0
        while todo:
            self._wait_for_exclusive_gpu(device_handle)
            times, _ = self._replay_batch([configs[i] for i in todo], self.TimingWarmupMs)
            if self._foreign_gpu_processes(device_handle) and \
                    contended_retries < self.ContendedRetries:
                contended_retries += 1
                print('GPU shared during batch replay; repeating it')
                continue
            for pos, i in enumerate(todo):
                if pos in times:
                    out[i] = (True, times[pos])
            missing = [pos for pos in range(len(todo)) if pos not in times]
            if not missing:
                break
            failed = missing[0]            # the configuration that stopped the process
            out[todo[failed]] = (False, None)
            todo = todo[failed + 1:]
        return out

    def execute(self, NumThreads=None, NumTeams=None, maxIters=9, keepMaxTeam=True,
                dropSingleTeams=False, device_handle=None, metrics=True, energy=True):
        """`maxIters` timing processes of one configuration (the binary is already
        compiled by Kernel.sample), each a warm-up and the median of TimingRepeats
        launches. Returns what the fork's Kernel.execute returns without metrics."""
        durations, contended_retries = [], 0
        self.LastMetrics = {}
        while len(durations) < maxIters:
            self._wait_for_exclusive_gpu(device_handle)
            times, _ = self._replay_batch([dict(NumThreads=NumThreads, NumTeams=NumTeams)],
                                          self.TimingWarmupMs)
            if self._foreign_gpu_processes(device_handle):
                contended_retries += 1
                print('GPU shared during timing replay; discarding sample')
                if contended_retries <= self.ContendedRetries:
                    continue
                print('GPU stayed shared; keeping contended sample')
            if 0 not in times:
                return [], [], [], [], []
            durations.append(times[0])
        return [[]], durations, [float('nan')] * len(durations), [None], [None]
