"""Redraw this figure from data.json alone (gbias_configs).

The trade between precision and galaxy-bias error for six configurations of
the gamma_t and w cuts (letters, defined in the note's table): the figure
of bias against sigma(S8), for the galaxy bias alone (top) and for IA and
galaxy bias together (bottom). Filled circles: linear bias; open squares:
the one-loop model, b2 free. The dotted line is 0.3. NLA, stitched truth,
xi_+- at 0.2, CMB prior.

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

STYLES = {'linear': {'color': '#0072B2', 'marker': 'o', 'mfc': '#0072B2', 'ms': 5.0},
          'oneloop': {'color': '#E69F00', 'marker': 's', 'mfc': 'white', 'ms': 5.0}}
LABEL = {'linear': 'linear bias', 'oneloop': r'one-loop bias, $b_2$ free'}
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
WIDTH = 3.5
PANEL_HEIGHT = 1.55
CORNERS = ('upper left', 'upper right', 'lower right', 'lower left', 'upper center')
HEADROOM_STEP = 0.08
HEADROOM_TRIES = 12
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


def place_legend(fig, ax, handles, labels, axes, others=(), above_ncol=2,
                 max_headroom=HEADROOM_TRIES, **kw):
    '''Put the legend where it touches no line and none of the artists in
    others, trying in order: the corners of ax; above ax, outside the panel,
    in above_ncol columns (the usual home in short stacked panels); the
    corners again while the top of the y range grows by the least amount,
    at most max_headroom steps. Raise when nothing works.'''
    def clear(leg):
        try:
            check_clear(fig, axes, [leg, *others])
            return True
        except RuntimeError:
            leg.remove()
            return False

    for loc in CORNERS:
        leg = ax.legend(handles, labels, loc=loc, frameon=False, **kw)
        if clear(leg):
            return leg
    leg = ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.02),
                    ncol=above_ncol, frameon=False, borderaxespad=0.0, **kw)
    if clear(leg):
        return leg
    lo, hi = ax.get_ylim()
    for attempt in range(1, max_headroom + 1):
        ax.set_ylim(lo, hi + attempt*HEADROOM_STEP*(hi - lo))
        for loc in CORNERS:
            leg = ax.legend(handles, labels, loc=loc, frameon=False, **kw)
            if clear(leg):
                return leg
    ax.set_ylim(lo, hi)
    raise RuntimeError(f'{NAME}: no legend position clears the data')




# label offsets tried in order [points]: right-above, right-below, left-above, left-below, above, below
OFFSETS = [(4, 3, 'left', 'bottom'), (4, -3, 'left', 'top'), (-4, 3, 'right', 'bottom'),
           (-4, -3, 'right', 'top'), (0, 6, 'center', 'bottom'), (0, -6, 'center', 'top')]


def place_label(fig, ax, x, y, text, color, others):
    """A point's letter at the first offset that overlaps no earlier label of the figure."""
    fig.canvas.draw()
    prev = [t.get_window_extent() for t in others]
    for dx, dy, ha, va in OFFSETS:
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7, color=color,
                        ha=ha, va=va)
        fig.canvas.draw()
        bb = t.get_window_extent()
        if not any(bb.overlaps(b) for b in prev):
            return t
        t.remove()
    raise RuntimeError(f'{NAME}: no free place for label {text}')


def draw(data):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2*PANEL_HEIGHT), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.08})
    texts = []
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.05, 3.0)
    axes[1].set_xlim(4.0, 7.6)
    for ax, q in zip(axes, ('gb', 'total')):
        ax.axhline(0.3, **THRESHOLD_STYLE)
        for model, st in STYLES.items():
            for name, lab in data['label'].items():
                p = data['points'][f'{model}|{name}']
                ax.plot(1e3*p['sigma_S8'], p[q], ls='none', color=st['color'], marker=st['marker'],
                        mfc=st['mfc'], mec=st['color'], ms=st['ms'], mew=1.2)
                texts.append(place_label(fig, ax, 1e3*p['sigma_S8'], p[q], lab, st['color'], texts))
        ax.set_yscale('log')
        ax.grid(False, axis='x')
        ax.yaxis.set_label_position('right')
        ax.set_ylabel('galaxy bias' if q == 'gb' else 'IA + galaxy bias', rotation=90, fontsize=8, labelpad=5)
    axes[1].set_xlabel(r'$10^3\,\sigma(S_8)$')
    handles = [Line2D([], [], ls='none', color=st['color'], marker=st['marker'], mfc=st['mfc'], mec=st['color'],
                      ms=st['ms'], mew=1.2) for st in STYLES.values()]
    leg = axes[0].legend(handles, [LABEL[m] for m in STYLES], loc='lower left', bbox_to_anchor=(0.0, 1.03),
                         ncol=2, frameon=False, borderaxespad=0.0, fontsize=7)
    check_clear(fig, list(axes), [leg])
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{NAME}: labels {texts[i].get_text()} and {texts[j].get_text()} overlap')
    fig.text(0.0, 0.5, 'figure of bias', rotation=90, ha='right', va='center', fontsize=10)
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
