"""Upstream figures for RR-GPGPU2025/taco/extension-TACO-upstream.tex (figures-ext/upstream/).
hecbench.pdf: true speedup per HeCBench kernel (FastRR, its two ablations, Batch TPE) with the
upstream grid optimum. incremental.pdf: left, upstream online lanes by kernels in the store;
right, the LLVM 16 simulation (incremental-ranker-2026-10-28/incremental.csv), as before.
Style of incremental-ranker-2026-10-28/make_figure.py; categorical slots in fixed order.
Usage (inside rr_current.sif, after analyze_paper.py): python3 make_figures.py"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import X, SIM, FIGURES as OUT
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'lines.linewidth': 1.6, 'legend.frameon': False, 'figure.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
})


def hecbench():
    k = pd.read_csv(f'{X}/paper/hecbench_kernels.csv', index_col=0)
    k = k.sort_values('grid optimum')
    series = [('final-wave', 'FastRR'), ('final', 'FastRR without wave-aligned teams'),
              ('final-wave-norefine', 'FastRR without refinement'), ('tpe-batch-nocls', 'Batch TPE')]
    labels = [n.replace('miniWeather', 'miniW.').replace('minisweep', 'msweep').replace('simplemoc', 'SimpleMOC')
              .replace('xsbench', 'XSBench').replace('amgmk', 'AMGmk').replace('su3', 'SU3').replace('s3d', 'S3D')
              .replace('-', ' l') for n in k.index]
    fig, ax = plt.subplots(figsize=(6.9, 2.0))
    x = np.arange(len(k))
    w = 0.19
    for i, (c, lab) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, k[c] - 1, w * 0.9, bottom=1, color=SLOT[i], label=lab, linewidth=0)
    ax.scatter(x, k['grid optimum'], marker='_', s=120, color=INK, linewidths=1.4, zorder=3,
               label='Upstream grid optimum')
    ax.set_yscale('log')
    ax.set_ylim(0.95, 5)
    ax.set_yticks([1, 1.25, 1.5, 2, 3, 4], ['1', '1.25', '1.5', '2', '3', '4'])
    ax.minorticks_off()
    ax.set_xticks(x, labels, rotation=35, ha='right')
    ax.set_xlim(-0.6, len(k) - 0.4)
    ax.set_ylabel('True speedup')
    ax.grid(axis='x', visible=False)
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(lab) for _, lab in series] + [ls.index('Upstream grid optimum')]
    fig.legend([hs[i] for i in order], [ls[i] for i in order], loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(f'{OUT}/hecbench.pdf')


def incremental():
    r = pd.read_csv(f'{X}/paper/incremental.csv')
    ob = ['0', '1-10', '11-20', '21-30', '31+']
    r['bin'] = pd.cut(r.store_kernels, [-1, 0, 10, 20, 30, 43], labels=ob)
    o = r.groupby('bin', observed=True).agg(inc=('end_to_end', 'mean'), ref=('final-wave_s', 'mean'),
                                             tpe=('tpe-batch-nocls_s', 'mean'))
    s = pd.read_csv(SIM)
    sb = ['0', '1-10', '11-25', '26-50', '51-100', '101+']
    s['bin'] = pd.cut(s.store_kernels, [-1, 0, 10, 25, 50, 100, 200], labels=sb)
    h = s[s.opt > 1.05].pivot_table(index='bin', columns='policy', values='share_first_batch',
                                    aggfunc='mean', observed=False)
    lab = {'inc': 'FastRR, trained on earlier screens', 'ref': 'FastRR, trained on grid datasets',
           'tpe': 'Batch TPE (no model)'}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1))
    ax = axes[0]
    for i, (c, m) in enumerate([('inc', 'o'), ('ref', 's'), ('tpe', '^')]):
        ax.plot(range(len(ob)), o[c].values, marker=m, ms=4, color=SLOT[i], label=lab[c])
    ax.set_xticks(range(len(ob)), ob)
    ax.set_xlabel('Kernels in the store (online, upstream, 3 sequences)')
    ax.set_ylabel('End to end per kernel (s)')
    ax.set_ylim(0, 250)
    ax = axes[1]
    for i, (c, m) in enumerate([('rank1', 'o'), ('oracle1', 's'), ('tpe', '^')]):
        ax.plot(range(len(sb)), h[c].values, marker=m, ms=4, color=SLOT[i])
    ax.set_xticks(range(len(sb)), sb)
    ax.set_xlabel('Kernels in the store (simulation, LLVM 16, 20 orders)')
    ax.set_ylabel('Share of optimum, first batch')
    ax.set_ylim(0.4, 0.75)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(f'{OUT}/incremental.pdf')


if __name__ == '__main__':
    hecbench()
    incremental()
    print('wrote', OUT)
