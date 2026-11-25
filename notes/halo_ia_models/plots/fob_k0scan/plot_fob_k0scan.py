"""Roman figure of bias of the halo models against the xi_+- cut, from data.json alone.

Roman with the CMB prior (n_s, Omega_b h^2, H0), every parameter free, the
Fisher shift of (Omega_m, S8) on the CP truth. Panels, top to bottom: cosmic
shear in the TATT form and in the NLA form, 3x2pt (gamma_t at 0.5, w at 0.2)
in the TATT and the NLA form. Lines: A1 terms on P_NL (gray triangles), HMc
(purple diamonds), alpha free (orange squares), alpha of FLAMINGO (orange,
dashed, open squares), beta_NL (thick blue circles). Dotted: 0.3.

Run from any folder: python plot_fob_k0scan.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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

STYLE = {
    'base': {'color': '0.45', 'lw': 1.0, 'marker': '^'},
    'hmc': {'color': '#CC79A7', 'lw': 1.1, 'marker': 'D'},
    'alpha': {'color': '#E69F00', 'lw': 1.5, 'marker': 's'},
    'alphaf': {'color': '#E69F00', 'lw': 1.5, 'marker': 's', 'dashed': True},
    'nlb': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
}
LABEL = {'base': r'$P_{\rm NL}$', 'hmc': r'HMc', 'alpha': r'$\alpha$ free',
         'alphaf': r'$\alpha$ of FLAMINGO', 'nlb': r'$\beta_{\rm NL}$'}
PANELS = (('cs', 'TATT', 'shear, TATT'), ('cs', 'NLA', 'shear, NLA'),
          ('3x2', 'TATT', '3x2pt, TATT'), ('3x2', 'NLA', '3x2pt, NLA'))
MARKER_SIZE = 3.6
DASH = (4.5, 2.5)
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
WIDTH, PANEL_HEIGHT = 3.5, 1.2


def as_array(values):
    '''A list of numbers and None as a float array, NaN (a gap) for None.'''
    return np.array([np.nan if v is None else float(v) for v in values])


def kw(key):
    st = STYLE[key]
    dashed = st.get('dashed', False)
    ls = (0, (DASH[0]/st['lw'], DASH[1]/st['lw'])) if dashed else '-'
    return {'color': st['color'], 'lw': st['lw'], 'ls': ls, 'marker': st['marker'], 'ms': MARKER_SIZE,
            'mfc': 'white' if dashed else st['color'], 'mec': st['color']}


def check_clear(fig, axes, artist):
    '''Raise when the legend overlaps a panel or the x label.'''
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = artist.get_window_extent(r)
    for ax in axes:
        if box.overlaps(ax.get_window_extent()):
            raise RuntimeError(f'{NAME}: the legend overlaps a panel')
    if box.overlaps(axes[-1].xaxis.label.get_window_extent(r)):
        raise RuntimeError(f'{NAME}: the legend overlaps the x label')


def draw(data):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(WIDTH, PANEL_HEIGHT*len(PANELS)), sharex=True,
                             sharey=True, gridspec_kw={'hspace': 0.06})
    top = 0.0
    for ax, (probe, form, name) in zip(axes, PANELS):
        k0 = np.array(data['K0'][probe], float)
        ax.axhline(float(data['threshold']), **THRESHOLD_STYLE)
        for key in STYLE:
            y = as_array(data['lines'][f'{probe}|{form}|{key}']['fob'])
            ax.plot(k0, y, clip_on=False, **kw(key))
            top = max(top, np.nanmax(y))
        ax.grid(False, axis='x')
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(name, rotation=90, fontsize=9, labelpad=5)
    axes[0].set_ylim(0.0, 1.08*top)
    axes[-1].set_xlabel(r'$K_0$ of the $\xi_\pm$ cut [$h\,{\rm Mpc}^{-1}$]')
    axes[-1].set_xlim(0.13, 0.62)
    fig.text(0.0, 0.5, 'figure of bias', rotation=90, ha='right', va='center', fontsize=11)
    handles = [Line2D([], [], **kw(k)) for k in STYLE]
    # the legend under the x label: measured after drawing, so it cannot touch it
    fig.canvas.draw()
    lab = axes[-1].xaxis.label.get_window_extent(fig.canvas.get_renderer())
    y = (lab.y0 - 4*fig.dpi/72)/fig.bbox.height
    leg = fig.legend(handles, [LABEL[k] for k in STYLE], loc='upper center', ncol=3, frameon=False,
                     fontsize=8, bbox_to_anchor=(0.5, y), handlelength=2.6, columnspacing=1.0)
    check_clear(fig, axes, leg)
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
