"""Goodness of fit of the halo models to the FLAMINGO IA spectra, from data.json alone.

Rows: the TATT form (A2, b_TA free) and the NLA form (A2 = b_TA = 0) of each
model. Columns, as in the HMc review: the summed chi2_min/nu over the nine
core snapshots against k_max; chi2_min/nu per snapshot; the goodness-of-fit
cut k_max (last k_max before two consecutive p < 0.05) per snapshot redshift.
Models: A1 terms on P_NL (gray, triangles), on I_1 P_lin (HMc, purple
diamonds), with a free share alpha of P_NL (orange squares) and with the
nonlinear halo bias (thick blue circles). Dotted: chi2/nu = 1.

Run from any folder: python plot_gof.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

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

# one encoding per halo-model family, the same in every figure of the note
STYLE = {
    'base': {'color': '0.45', 'lw': 1.0, 'marker': '^'},
    'hmc': {'color': '#CC79A7', 'lw': 1.1, 'marker': 'D'},
    'alpha': {'color': '#E69F00', 'lw': 1.5, 'marker': 's'},
    'nlb': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
}
LABEL = {'base': r'$P_{\rm NL}$', 'hmc': r'HMc, $I_1P_{\rm lin}$', 'alpha': r'$\alpha$ free',
         'nlb': r'$\beta_{\rm NL}$'}
MARKER_SIZE = 3.6
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
# full page width [in] and the height of one row [in]
WIDTH, ROW_HEIGHT = 7.0, 2.1
FORMS = (('TATT', 'TATT form'), ('NLA', 'NLA form'))
# y range of the chi2 panels and the k ticks [h/Mpc]
YLIM = (0.4, 900.0)
KTICKS = (0.2, 0.5, 1.0, 2.0)


def as_array(values):
    '''A list of numbers and None as a float array, NaN (a gap) for None.'''
    return np.array([np.nan if v is None else float(v) for v in values])


def kw(key, thin=False):
    st = STYLE[key]
    if thin:
        return {'color': st['color'], 'lw': 0.6*st['lw'], 'alpha': 0.55}
    return {'color': st['color'], 'lw': st['lw'], 'marker': st['marker'], 'ms': MARKER_SIZE,
            'mfc': st['color'], 'mec': st['color']}


def k_axis(ax, which='x'):
    axis = ax.xaxis if which == 'x' else ax.yaxis
    axis.set_major_locator(FixedLocator(KTICKS))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def draw(data):
    kg = as_array(data['kgrid'])
    zs = np.array(data['zs'], float)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, ROW_HEIGHT*2),
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.45})
    for r, (form, name) in enumerate(FORMS):
        a_sum, a_snap, a_cut = axes[r]
        for key in STYLE:
            line = data['lines'].get(f'{form}|{key}')
            if line is None:
                continue
            a_sum.plot(kg, as_array(line['summed']), **kw(key))
            for row in line['per_snapshot']:
                a_snap.plot(kg, as_array(row), **kw(key, thin=True))
            a_cut.plot(as_array(line['cut']), zs, **kw(key))
        for ax in (a_sum, a_snap):
            ax.axhline(1.0, **THRESHOLD_STYLE)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(*YLIM)
            ax.set_xlim(0.15, 3.3)
            k_axis(ax)
            ax.grid(False, axis='x')
        a_cut.set_xscale('log')
        a_cut.set_xlim(0.1, 2.0)
        a_cut.set_ylim(0.0, 2.25)
        k_axis(a_cut)
        a_sum.set_ylabel(r'$\sum\chi^2_{\min}/\sum\nu$')
        a_snap.set_ylabel(r'$\chi^2_{\min}/\nu$ per snapshot')
        a_cut.set_ylabel(r'$z$')
        a_cut.yaxis.set_label_position('left')
        # the form's name on the right of the row
        a_cut.text(1.04, 0.5, name, transform=a_cut.transAxes, rotation=90, va='center', ha='left',
                   fontsize=10)
        if r == 0:
            for ax in axes[0]:
                ax.tick_params(labelbottom=False)
        else:
            a_sum.set_xlabel(r'$k_{\max}$ [$h\,{\rm Mpc}^{-1}$]')
            a_snap.set_xlabel(r'$k_{\max}$ [$h\,{\rm Mpc}^{-1}$]')
            a_cut.set_xlabel(r'goodness-of-fit cut $k_{\max}$ [$h\,{\rm Mpc}^{-1}$]')
    handles = [Line2D([], [], **kw(k)) for k in STYLE]
    labels = [LABEL[k] for k in STYLE]
    # the legend above the grid, outside every panel: it cannot touch data
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=4,
               frameon=False, fontsize=9)
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
