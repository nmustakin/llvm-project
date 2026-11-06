#!/usr/bin/env python3
"""Registry of the bulk HeCBench applications (the further kernels of the transfer study).

usage: bulk_registry.py --wrappers DIR --out data/bulk_hecbench.json

Reads the benchmark wrappers of the LLVM 16 bulk study (RR/benchmarks/hecbench2/H_*.py, one per
application, each with the run command of the paper) and writes, per application, the HeCBench
directories to copy (the application, ../include, and every sibling directory that its Makefile
or sources reach through ../, such as the -cuda directory whose sources an -omp port includes),
the build command (Makefile.rr, which make_rr_amd.sh writes for the GPU architecture, after
`make clean`, so that no executable of the LLVM 16 build survives), and the run
command relative to the copied application directory. Input files outside the application stay
absolute paths into FASTRR_HECBENCH. record_app.py merges the registry into its APPS.
"""
import argparse, glob, json, os, re
from pathlib import Path

from paths import HEC

CMD = re.compile(r"return f'\{HECBENCH\}/src/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+) ?(.*)'")


def siblings(app_dir):
    """Sibling directories of src/ that the application's files name through ../."""
    names = set()
    for f in Path(app_dir).iterdir():
        if f.is_file() and (f.suffix in ('.cpp', '.cc', '.c', '.h', '.hpp', '.cu', '.cuh')
                            or f.name.startswith('Makefile')):
            try:
                text = f.read_text(errors='replace')
            except OSError:
                continue
            names.update(re.findall(r'\.\./([A-Za-z0-9_.-]+)', text))
    return sorted(n for n in names if (Path(HEC) / n).is_dir() and n != Path(app_dir).name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wrappers', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    reg, skipped = {}, []
    for w in sorted(glob.glob(f'{args.wrappers}/H_*.py')):
        m = CMD.search(Path(w).read_text())
        if not m:
            skipped.append((Path(w).name, 'no command'))
            continue
        app, exe, rest = m.groups()
        src = Path(HEC) / app
        if not (src / 'Makefile.rr').exists():
            skipped.append((app, 'no Makefile.rr'))
            continue
        rest = rest.replace('{HECBENCH}/src', HEC)
        name = 'h-' + re.sub(r'-omp$', '', app)
        dirs = [str(src), f'{HEC}/include'] + [f'{HEC}/{d}' for d in siblings(src)
                                               if d != 'include']
        reg[name] = dict(dirs=dirs, build='make -f Makefile.rr clean; make -f Makefile.rr -j16',
                         run=f'./{exe} {rest}'.strip(), wrapper=Path(w).name)
    Path(args.out).write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{args.out}: {len(reg)} applications; skipped {skipped}')


if __name__ == '__main__':
    main()
