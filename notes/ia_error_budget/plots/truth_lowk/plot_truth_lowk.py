"""The FLAMINGO II and GI ratios at large scales against the two Roman truths.

Rows: four snapshots. Left: P_II over its NLA value C1^2 P_NL; right: P_GI
over -C1 P_NL, with C1 of the NLA fit on 0.05 <= k <= 0.3, so NLA is the
dotted line at 1. Open circles: the measured FLAMINGO ratios with their
mode-scatter errors; blue: the production truth (a smoothing spline of the
measurement above k_min = 0.05, gray dotted, its edge value held below);
orange dashed: the stitched truth (NLA below k_t, the same spline above);
green dash-dot: the measurement-only truth (the same spline fitted over every
measured k, nothing held); vermillion: the linear-limit truth (per
snapshot a constant fitted to the measured points at k <= 0.07, GI and II
separately, band: its 1 sigma, joined to the spline at 0.08). Lines leaving the panel at the first two k points
are the spline's unconstrained edge.
Run from any folder: python plot_truth_lowk.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

# Figure style of the whole note: the shared plotting functions never
# change rcParams, so these lines set every font, tick and grid default.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.path.basename(HERE)
WIDTH = 7.0
PANEL_HEIGHT = 1.35
TRUTH = {'prod': dict(color='#0072B2', lw=2.0, ls='-', label='production truth (spline, held below $k_{\\rm min}$)'),
         'S1': dict(color='#E69F00', lw=1.5, ls=(0, (3.0, 1.7)), label='stitched truth (NLA below $k_t$)'),
         'lin': dict(color='#D55E00', lw=1.6, ls='-', label='linear-limit fit (constant at $k \\leq 0.07$, joined at 0.08)'),
         'meas': dict(color='#009E73', lw=1.3, ls=(0, (5.0, 1.5, 1.2, 1.5)), label='measurement only (spline over all $k$)')}
DATA = dict(fmt='o', color='0.25', ms=2.6, mfc='white', elinewidth=0.8, capsize=0, zorder=1)


def draw():
    d = json.load(open(os.path.join(HERE, 'data.json')))
    kt = np.array(d['k_truth'])
    n = len(d['panels'])
    fig, axes = plt.subplots(n, 2, figsize=(WIDTH, PANEL_HEIGHT*n), sharex=True, sharey='col')
    for i, p in enumerate(d['panels']):
        for j, (key, lab) in enumerate((('EE', 'II'), ('dE', 'GI'))):
            ax = axes[i, j]
            ax.axhline(1.0, color='0.1', ls=':', lw=1.0)
            ax.axvline(d['kmin'], color='0.5', ls=':', lw=0.9)
            ax.errorbar(p['k'], p[key], yerr=p[key + '_err'], **DATA)
            c, ce = p[f'lin_{key}_c'][:2]
            ax.fill_between([0.008, d['klin']], c - ce, c + ce, color='#D55E00', alpha=0.15, lw=0, zorder=0)
            for v, st in TRUTH.items():
                ax.plot(kt, p[f'{v}_{key}'], color=st['color'], lw=st['lw'], ls=st['ls'], zorder=3)
            ax.set_xscale('log')
            ax.set_xlim(0.008, 1.2)
            ax.text(0.03, 0.88, f'{lab}, $z = {p["z"]:g}$', transform=ax.transAxes, fontsize=9, va='top')
    axes[0, 0].set_ylim(0.3, 2.4)
    axes[0, 1].set_ylim(0.7, 1.55)
    for ax in axes[-1]:
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
    for i in range(n):
        axes[i, 0].set_ylabel(r'$P_{\rm II}/(C_1^2 P_{\rm NL})$', fontsize=9)
        axes[i, 1].set_ylabel(r'$P_{\rm GI}/(-C_1 P_{\rm NL})$', fontsize=9)
    h = [Line2D([], [], marker='o', ls='none', color='0.25', mfc='white', ms=3)] + \
        [Line2D([], [], color=st['color'], lw=st['lw'], ls=st['ls']) for st in TRUTH.values()]
    lab = ['FLAMINGO measured ($\\pm$ mode scatter)'] + [st['label'] for st in TRUTH.values()]
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=1.2, fontsize=8.5)
    fig.subplots_adjust(hspace=0.08, wspace=0.28)
    fig.savefig(os.path.join(HERE, NAME + '.pdf'))
    fig.savefig(os.path.join(HERE, NAME + '.png'), dpi=300)


if __name__ == '__main__':
    draw()
