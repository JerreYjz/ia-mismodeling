"""The 3x2pt test of the separate cuts, one cut lowered at a time.

Top: the figure of bias of the minimizer's MAP in (Omega_m, S8), 3x2pt,
every parameter free, CMB prior; dotted: 0.3. Bottom: the figure of merit
of (Omega_m, S8) in thousands. Left: the cosmic-shear cut lowered with
gamma_t at 0.5 (its 2x2pt cut). Right: the gamma_t cut lowered with cosmic
shear at its own cut (0.35 on the stitched truth, 0.6 on the E minus B
truth). NLA orange squares, TATT blue circles; stitched truth solid filled,
E minus B dashed open; a gray ring marks a fit with a parameter at a prior
bound. Run from any folder: python plot_cut_combo.py
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

# models: color and marker (NLA orange squares, TATT blue circles, as in
# every figure of the note); truths: stitched solid filled, E minus B dashed open
MODEL = {'NLA': dict(color='#E69F00', marker='s', lw=1.5), 'TATT': dict(color='#0072B2', marker='o', lw=2.0)}
TRUTH = {'S1': dict(ls='-', fill=True, name='stitched'), 'EB': dict(ls='--', fill=False, name='E minus B')}
# a fit with a parameter at a prior bound: a gray ring around the point
BOUND_STYLE = dict(marker='o', ms=9, mfc='none', mec='0.45', mew=0.9, ls='none', clip_on=False)
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}


def style(m, tr):
    c, t = MODEL[m], TRUTH[tr]
    return dict(color=c['color'], lw=c['lw'] if t['fill'] else 1.3, ls=t['ls'], marker=c['marker'], ms=3.8,
                mfc=c['color'] if t['fill'] else 'white', mec=c['color'], clip_on=False)


def draw(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex='col', sharey='row',
                             gridspec_kw={'hspace': 0.08, 'wspace': 0.08})
    for j, scan in enumerate(('xi', 'gt')):
        for m in ('NLA', 'TATT'):
            for tr in ('S1', 'EB'):
                pts = data['lines'][f'{m}|{tr}|{scan}']
                x = np.array([p['cut'] for p in pts])
                for i, q in enumerate(('fob', 'fom')):
                    y = np.array([np.nan if p[q] is None else p[q] for p in pts])
                    axes[i, j].plot(x, y, **style(m, tr))
                    b = np.array([len(p['at_bound']) > 0 for p in pts])
                    if b.any():
                        axes[i, j].plot(x[b], y[b], **BOUND_STYLE)
        axes[0, j].axhline(data['threshold'], **THRESHOLD_STYLE)
        axes[0, j].grid(False, axis='x')
        axes[1, j].grid(False, axis='x')
    axes[0, 0].set_ylim(0, 0.5)
    axes[1, 0].set_ylim(35, 65)
    axes[0, 0].set_ylabel('figure of bias, 3x2pt', fontsize=8)
    axes[1, 0].set_ylabel(r'FoM$(\Omega_m, S_8)$ / 1000', fontsize=8)
    axes[1, 0].set_xlabel(r'cosmic-shear cut $K_0$ [$h\,\mathrm{Mpc}^{-1}$]')
    axes[1, 1].set_xlabel(r'$\gamma_t$ cut $K_0$ [$h\,\mathrm{Mpc}^{-1}$]')
    axes[0, 0].set_title(r'cosmic-shear cut lowered, $\gamma_t$ at 0.5', fontsize=9)
    axes[0, 1].set_title('$\\gamma_t$ cut lowered, cosmic shear at its own cut', fontsize=9)
    h = [Line2D([], [], **style(m, tr)) for m in ('NLA', 'TATT') for tr in ('S1', 'EB')]
    lab = [f"{m}, {TRUTH[tr]['name']}" for m in ('NLA', 'TATT') for tr in ('S1', 'EB')]
    h.append(Line2D([], [], **BOUND_STYLE))
    lab.append('parameter at a prior bound')
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, axes[0, 0].get_position().y1 + 0.05), ncol=5,
               frameon=False, fontsize=7, handlelength=2.6, columnspacing=1.0)
    return fig


def main():
    with open(os.path.join(HERE, 'data.json')) as f:
        data = json.load(f)
    fig = draw(data)
    fig.savefig(os.path.join(HERE, NAME + '.pdf'))
    fig.savefig(os.path.join(HERE, NAME + '.png'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
