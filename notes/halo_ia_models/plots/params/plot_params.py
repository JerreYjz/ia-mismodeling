"""Fitted parameters of the halo models against the FLAMINGO fit range, from data.json alone.

Columns: the snapshots z = 0.3, 1 and 2. Rows: A1, b_TA and the central
1-halo amplitude a_c of the TATT form; alpha of the alpha-free model (TATT
form solid, NLA form dashed); A1 of the NLA form. Models as in the GOF
figure: A1 terms on P_NL (gray triangles), HMc (purple diamonds), alpha
free (orange squares), beta_NL (thick blue circles). Open markers: the
parameter is within 1 percent of its bound (b_TA >= 0, 0 <= a_c <= 1,
-1 <= alpha <= 2). Dotted: a_c = 1 and alpha = 0, 1.

Run from any folder: python plot_params.py
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

STYLE = {
    'base': {'color': '0.45', 'lw': 1.0, 'marker': '^'},
    'hmc': {'color': '#CC79A7', 'lw': 1.1, 'marker': 'D'},
    'alpha': {'color': '#E69F00', 'lw': 1.5, 'marker': 's'},
    'nlb': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
}
LABEL = {'base': r'$P_{\rm NL}$', 'hmc': r'HMc, $I_1P_{\rm lin}$', 'alpha': r'$\alpha$ free',
         'nlb': r'$\beta_{\rm NL}$'}
MARKER_SIZE = 3.6
DASH = (4.5, 2.5)
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
WIDTH, ROW_HEIGHT = 6.4, 1.15
# rows: (form, parameter, models drawn, y label, y range, reference lines)
ROWS = (
    ('TATT', 'A1', ('base', 'hmc', 'alpha', 'nlb'), r'$A_1$, TATT form', (0.0, 9.0), ()),
    ('TATT', 'bTA', ('base', 'hmc', 'alpha', 'nlb'), r'$b_{\rm TA}$', (-0.1, 2.1), ()),
    ('TATT', 'a_c', ('base', 'hmc', 'alpha', 'nlb'), r'$a_c$', (-0.1, 1.15), (1.0,)),
    ('both', 'alpha', ('alpha',), r'$\alpha$', (-1.1, 2.1), (0.0, 1.0)),
    ('NLA', 'A1', ('base', 'hmc', 'alpha', 'nlb'), r'$A_1$, NLA form', (0.0, 9.0), ()),
)
KTICKS = (0.2, 0.5, 1.0, 2.0)


def as_array(values):
    '''A list of numbers and None as a float array, NaN (a gap) for None.'''
    return np.array([np.nan if v is None else float(v) for v in values])


def kw(key, dashed=False):
    st = STYLE[key]
    ls = (0, (DASH[0]/st['lw'], DASH[1]/st['lw'])) if dashed else '-'
    return {'color': st['color'], 'lw': st['lw'], 'ls': ls}


def draw(data):
    kg = as_array(data['kgrid'])
    zs = data['zs']
    fig, axes = plt.subplots(len(ROWS), len(zs), figsize=(WIDTH, ROW_HEIGHT*len(ROWS)), sharex=True,
                             gridspec_kw={'hspace': 0.08, 'wspace': 0.08})
    for r, (form, par, models, ylab, ylim, refs) in enumerate(ROWS):
        for c, z in enumerate(zs):
            ax = axes[r, c]
            for ref in refs:
                ax.axhline(ref, **THRESHOLD_STYLE)
            forms = ('TATT', 'NLA') if form == 'both' else (form,)
            for f in forms:
                for key in models:
                    line = data['lines'].get(f'{f}|{key}|{z:g}')
                    if line is None:
                        continue
                    y = as_array(line[par])
                    bound = np.array(line['at_bound'][par], bool)
                    ax.plot(kg, y, **kw(key, dashed=(f == 'NLA' and form == 'both')))
                    st = STYLE[key]
                    ax.plot(kg[~bound], y[~bound], ls='none', marker=st['marker'], ms=MARKER_SIZE,
                            color=st['color'])
                    ax.plot(kg[bound], y[bound], ls='none', marker=st['marker'], ms=MARKER_SIZE,
                            mfc='white', mec=st['color'])
            ax.set_xscale('log')
            ax.set_xlim(0.15, 3.3)
            ax.xaxis.set_major_locator(FixedLocator(KTICKS))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(False, axis='x')
            ax.set_ylim(*ylim)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=10)
            else:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(rf'$z = {z:g}$', fontsize=10)
            if r == len(ROWS) - 1:
                ax.set_xlabel(r'$k_{\max}$ [$h\,{\rm Mpc}^{-1}$]')
    handles = [Line2D([], [], marker=STYLE[k]['marker'], ms=MARKER_SIZE, **kw(k)) for k in STYLE]
    labels = [LABEL[k] for k in STYLE]
    handles.append(Line2D([], [], marker='s', ms=MARKER_SIZE, **kw('alpha', dashed=True)))
    labels.append(r'$\alpha$, NLA form')
    handles.append(Line2D([], [], ls='none', marker='o', ms=MARKER_SIZE, mfc='white', mec='0.2'))
    labels.append('at a bound')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.925), ncol=6,
               frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=2.4)
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
