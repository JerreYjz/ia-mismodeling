"""Redraw this figure from data.json alone (gbias_bins).

The galaxy-bias figure of bias that one lens bin brings in, against that
bin's cut K0 (all other cuts at 0.1): the piece from its w rows (top) and
its gamma_t rows (bottom; lens bin 8 has no source behind it), fitted with
linear bias (left) and with the one-loop bias model, b2 free (right).
Lens bins by color (dark to light with redshift), marker and line width.
The dotted line is 0.3. NLA, stitched truth, xi_+- at 0.2, CMB prior.

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

# lens bins 1..8: a sequential colorblind-safe ramp, plus marker and width
COLORS = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a']
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
WIDTHS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]
MARKER_SIZE = 3.2
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
WIDTH = 6.4
HEIGHT = 4.2
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



def draw(data):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
    cuts = data['cuts']
    for r, probe in enumerate(('w', 'gt')):
        for c, model in enumerate(('linear', 'oneloop')):
            ax = axes[r, c]
            ax.axhline(0.3, **THRESHOLD_STYLE)
            for i in range(1, 9):
                v = data['lines'].get(f'{model}|{probe}|{i}')
                if v is None:
                    continue
                ax.plot(cuts, np.clip(v, 1e-3, None), color=COLORS[i - 1], marker=MARKERS[i - 1],
                        lw=WIDTHS[i - 1], ms=MARKER_SIZE, clip_on=True)
            ax.set_yscale('log')
            ax.grid(False, axis='x')
    axes[0, 0].set_ylim(0.01, 12)
    axes[0, 0].set_xlim(0.08, 0.82)
    for ax in axes[1]:
        ax.set_xlabel(r'cut $K_0$ of that bin')
    axes[0, 0].set_title('linear bias', fontsize=9)
    axes[0, 1].set_title(r'one-loop bias, $b_2$ free', fontsize=9)
    for ax, t in zip(axes[:, 1], (r'$w$', r'$\gamma_t$')):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(t, rotation=90, fontsize=9, labelpad=6)
    fig.text(0.04, 0.5, 'figure of bias of one lens bin', rotation=90, ha='center', va='center', fontsize=10)
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i], lw=WIDTHS[i], ms=MARKER_SIZE) for i in range(8)]
    labels = [f'bin {i + 1} ($z = {z:.2f}$)' for i, z in enumerate(data['zbin'])]
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False,
                     fontsize=7)
    check_clear(fig, list(axes.ravel()), [leg])
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
