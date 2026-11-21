"""Redraw this figure from data.json alone (bias_k).

Scale dependence of the galaxy bias of the FLAMINGO L1_m8 galaxies at
z = 1.5, 2 and 3, one panel per redshift: b_gm/b1 = P_gm/(b1 P_m) (solid,
filled markers), the bias gamma_t sees, and b_gg/b1 = sqrt(P_gg - V/N)/(b1
sqrt(P_m)) (dashed, open markers), the bias w sees, with b1 the bias at
0.01 < k < 0.05 h/Mpc and P_m the total matter. The dotted horizontal line is
linear bias; the vertical dotted lines are k_max = K0/D(z) of the w cut (0.2)
and of the gamma_t cut (0.5).

Run from any folder: python plot_<name>.py
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

# the two biases keep one encoding: color, line width, marker and line style
STYLES = {'b_gm': {'color': '#0072B2', 'lw': 2.0, 'marker': 'o', 'ls': '-', 'mfc': '#0072B2'},
          'b_gg': {'color': '#E69F00', 'lw': 1.4, 'marker': 's', 'ls': (0, (3.2, 1.8)), 'mfc': 'white'}}
LABEL = {'b_gm': r'$b_{\rm gm}/b_1$ (for $\gamma_t$)', 'b_gg': r'$b_{\rm gg}/b_1$ (for $w$)'}
MARKER_SIZE = 2.6
MARK_EVERY = 3
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
WIDTH = 3.5
PANEL_HEIGHT = 1.25
CUT_NAME = {'w': r'$w$', 'gt': r'$\gamma_t$'}
def line_points(ax):
    '''Display coordinates of points densely sampled along every line drawn
    in ax (reference lines included), inside the axes only.

    Each line's own transform maps its data to the display (an axhline keeps
    x in axes units), and segments are sampled in display space, where a
    segment is straight on any axis scale.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        if len(xy) == 0:
            continue
        disp = line.get_transform().transform(xy)
        ok = np.all(np.isfinite(disp), axis=1)
        pts.append(disp[ok])
        t = np.linspace(0.0, 1.0, 60)[:, None]
        for i in range(len(disp) - 1):
            if ok[i] and ok[i + 1]:
                pts.append(disp[i] + t*(disp[i + 1] - disp[i]))
    for coll in ax.collections:   # error bars and scatter points
        for path in coll.get_paths():
            v = coll.get_transform().transform(path.vertices) if len(path.vertices) else path.vertices
            if len(v):
                pts.append(v[np.all(np.isfinite(v), axis=1)])
    pts = np.vstack(pts)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[inside]


def touches_lines(box, axes):
    '''True when a display box, padded by 3% / 8%, contains a drawn point.'''
    padded = box.expanded(1.03, 1.08)
    for ax in axes:
        pts = line_points(ax)
        hit = ((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
               & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1))
        if np.any(hit):
            return True
    return False


def check_clear(fig, axes, artists):
    '''Raise when a legend or text touches a drawn line or another artist.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'{NAME}: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{NAME}: artists {i} and {j} overlap')



def shared_ylabel(fig, axes, text, pad_pt=4.0, fontsize=11):
    '''One y label for a stack of panels, placed left of the widest y tick
    label (measured after drawing), so it can never overlap a tick label.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(t.get_window_extent(renderer).x0 for ax in axes
               for t in ax.get_yticklabels() if t.get_text() and t.get_visible())
    x = (left - pad_pt*fig.dpi/72.0)/fig.bbox.width
    top = axes[0].get_position().y1
    bottom = axes[-1].get_position().y0
    return fig.text(x, 0.5*(top + bottom), text, rotation=90, ha='right', va='center',
                    fontsize=fontsize)



def draw(data):
    zs = data['zs']
    fig, axes = plt.subplots(len(zs), 1, figsize=(WIDTH, PANEL_HEIGHT*len(zs)), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.08})
    texts = []
    for ax, z in zip(axes, zs):
        row = data['rows'][z]
        k = np.array(row['k'])
        ax.axhline(1.0, **THRESHOLD_STYLE)
        for name, st in STYLES.items():
            ax.plot(k, row[name], color=st['color'], lw=st['lw'], ls=st['ls'], marker=st['marker'],
                    ms=MARKER_SIZE, mfc=st['mfc'], mec=st['color'], markevery=MARK_EVERY)
        for c, km in row['kmax'].items():
            ax.axvline(km, **THRESHOLD_STYLE)
        ax.set_xscale('log')
        ax.grid(False, axis='x')
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(f'$z = {z}$', rotation=90, fontsize=8, labelpad=5)
    axes[0].set_ylim(0.9, 2.5)
    axes[-1].set_xlim(0.03, 3.0)
    axes[-1].set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]')
    for ax, z in zip(axes, zs):
        for c, km in data['rows'][z]['kmax'].items():
            texts.append(ax.text(km*1.06, 2.42, CUT_NAME[c], fontsize=7, va='top', ha='left'))
    handles = [Line2D([], [], color=st['color'], lw=st['lw'], ls=st['ls'], marker=st['marker'], ms=MARKER_SIZE,
                      mfc=st['mfc'], mec=st['color']) for st in STYLES.values()]
    leg = axes[0].legend(handles, [LABEL[n] for n in STYLES], loc='lower left', bbox_to_anchor=(0.0, 1.03),
                         ncol=2, frameon=False, borderaxespad=0.0, fontsize=7)
    check_clear(fig, list(axes), [leg, *texts])
    shared_ylabel(fig, axes, 'bias over its large-scale value')
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
