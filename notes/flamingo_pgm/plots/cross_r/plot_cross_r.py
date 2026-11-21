"""Redraw this figure from data.json alone (cross_r).

Galaxy-matter cross-correlation coefficient of the FLAMINGO L1_m8 galaxies,
r = P_gm/sqrt(P_gg P_dm), at z = 1.5, 2 and 3 (color, line width and marker).
Solid with filled markers: Poisson shot noise V/N subtracted from P_gg and
P_dm, stitched over the folds; dashed with open markers: nothing subtracted,
fold 1 only (k up to a quarter of its Nyquist frequency). The dotted line is
r = 1.

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

# the redshifts keep one encoding: color, line width and marker
STYLES = {'1.5': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
          '2': {'color': '#E69F00', 'lw': 1.7, 'marker': 's'},
          '3': {'color': '#CC79A7', 'lw': 1.2, 'marker': '^'}}
MARKER_SIZE = 2.8
MARK_EVERY = 3
DASH = (3.2, 1.8)
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
WIDTH = 3.5
HEIGHT = 2.5
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



def draw(data):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    ax.axhline(1.0, **THRESHOLD_STYLE)
    for z in data['zs']:
        row, st = data['rows'][z], STYLES[z]
        ax.plot(row['k'], row['r_poisson'], color=st['color'], lw=st['lw'], marker=st['marker'], ms=MARKER_SIZE,
                markevery=MARK_EVERY)
        ax.plot(row['k_raw'], row['r_raw'], color=st['color'], lw=st['lw'], marker=st['marker'], ms=MARKER_SIZE,
                mfc='white', markevery=MARK_EVERY, ls=(0, (DASH[0]/st['lw'], DASH[1]/st['lw'])))
    ax.set_xscale('log')
    ax.set_xlim(0.03, 6.0)
    ax.set_ylim(0.7, 1.55)
    ax.grid(False, axis='x')
    ax.set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]')
    ax.set_ylabel(r'$r = P_{\rm gm}/\sqrt{P_{\rm gg} P_{\rm dm}}$')
    handles = [Line2D([], [], color=st['color'], lw=st['lw'], marker=st['marker'], ms=MARKER_SIZE)
               for st in STYLES.values()]
    labels = [f'$z = {z}$' for z in STYLES]
    handles += [Line2D([], [], color='0.3', lw=1.2), Line2D([], [], color='0.3', lw=1.2, ls=(0, (DASH[0]/1.2, DASH[1]/1.2)))]
    labels += ['shot noise subtracted', 'nothing subtracted']
    place_legend(fig, ax, handles, labels, [ax], fontsize=7, ncol=1)
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
