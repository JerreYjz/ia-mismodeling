"""Redraw corr_overlay.pdf and corr_overlay.png from data.json alone.

How wrong NLA and TATT are against the production FLAMINGO truth, over the
error budget for random errors correlated in k (corr_bands). Gray band and
black dashed edges: +-eps_95 for correlation lengths ell = 1 and 0.3 in
ln k, the rms of a random fractional error confined to one k range around k
(a Gaussian of width 0.3 in ln k) at which 95% of such errors keep Omega_m
and S8 each within 0.3 of their marginalised errors (model agnostic).
Lines with markers: P^model/P^truth - 1 of each model fitted at the fiducial
cosmology (IA parameters only), the mean over the core snapshots
0.2 <= z <= 2; thin lines in the model's color: its smallest and largest
value over those snapshots. Panels: GI and II, with no cut and with the cut
k_UV = 0.3/D(z), each model fitted with the same cut. Linear y axis to
+-30%; larger values run off the panel edge.

Run from any folder: python plot_corr_overlay.py
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

# The IA models keep one encoding in every note: color, line width and
# marker, so they stay apart in grayscale.
MODEL_STYLE = {
    'TATT': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
    'NLA': {'color': '#E69F00', 'lw': 1.5, 'marker': 's'},
    'HMc': {'color': '#CC79A7', 'lw': 1.1, 'marker': 'D'},
}
# marker size [pt]
MARKER_SIZE = 3.6
# dash and gap lengths [pt]; matplotlib scales a dash pattern by the line
# width, so dividing by it keeps the dashes visible on the thinnest line
DASH = (4.5, 2.5)
# thresholds and references: thin, black, dotted, no legend entry
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
# one column of a two-column paper [in]; the height of one stacked panel [in]
WIDTH = 3.5
PANEL_HEIGHT = 1.2
# the legend corners tried in order, then the y headroom grown in steps of
# HEADROOM_STEP (fraction of the range) at most HEADROOM_TRIES times
CORNERS = ('upper left', 'upper right', 'lower right', 'lower left', 'upper center')
HEADROOM_STEP = 0.08
HEADROOM_TRIES = 12


def line_style(model, variant):
    '''Plot keywords for a model: the base variant solid with filled
    markers, the other variant dashed with open markers.'''
    st = MODEL_STYLE[model]
    dashed = variant != 'base'
    ls = (0, (DASH[0]/st['lw'], DASH[1]/st['lw'])) if dashed else '-'
    return {'color': st['color'], 'lw': st['lw'], 'ls': ls, 'marker': st['marker'],
            'ms': MARKER_SIZE, 'mfc': 'white' if dashed else st['color'], 'mec': st['color']}


def as_array(values):
    '''A list of numbers and None as a float array, NaN (a gap) for None.'''
    return np.array([np.nan if v is None else float(v) for v in values])


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


def trim_top_tick(axes, margin=0.04):
    '''In a stack of panels sharing y, drop the labeled y ticks within margin
    (fraction of the range) of the top, so the top label of one panel never
    collides with the bottom label of the panel above. Call after the final
    set_ylim. Linear axes only.'''
    lo, hi = axes[0].get_ylim()
    ticks = [t for t in axes[0].get_yticks() if lo - 1e-12 <= t <= hi - margin*(hi - lo)]
    for ax in axes:
        ax.set_yticks(ticks)
        ax.set_ylim(lo, hi)


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



from matplotlib.patches import Patch  # noqa: E402


PANELS = [('none', 'GI', 'GI, no cut'), ('none', 'II', 'II, no cut'),
          ('K0/D 0.3', 'GI', r'GI, $K_0 = 0.3$'), ('K0/D 0.3', 'II', r'II, $K_0 = 0.3$')]
# the budget: ell = 1 shaded gray (TATT keeps the blue), ell = 0.3 black dashed
BAND = '0.75'
EDGE = {'color': 'black', 'lw': 1.0, 'ls': (0, (4.5, 2.5)), 'zorder': 1.6}
# the y range [fraction]; larger values run off the panel
YMAX = 0.3


def draw(data):
    k = as_array(data['k_c'])
    kd = as_array(data['k_dense'])
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 4*1.3 + 0.9), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.06})
    for ax, (cut, X, name) in zip(axes, PANELS):
        b = data['eps95'][cut][X]
        e = as_array(b['1'])
        ax.fill_between(k, -e, e, color=BAND, lw=0, zorder=1)
        v = as_array(b['0.3'])
        ax.plot(k, v, **EDGE)
        ax.plot(k, -v, **EDGE)
        for model in ('NLA', 'TATT'):
            st = MODEL_STYLE[model]
            err = np.array(data['errors'][cut][model][X])      # (n core z, k_dense)
            for edge in (err.min(0), err.max(0)):
                ax.plot(kd, edge, color=st['color'], lw=0.6, zorder=2)
            ax.plot(kd, err.mean(0), markevery=3, zorder=3, **line_style(model, 'base'))
        ax.axhline(0.0, **THRESHOLD_STYLE)
        ax.set_xscale('log')
        ax.grid(False, axis='x')
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(name, rotation=90, fontsize=8.5, labelpad=5)
    axes[-1].set_xlim(0.0095, 10.5)
    for ax in axes:
        ax.set_ylim(-YMAX, YMAX)
        ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax.set_yticklabels(['-20%', '-10%', '0', '10%', '20%'], fontsize=7.5)
        ax.minorticks_off()
    axes[-1].set_xticks([0.01, 0.1, 1.0, 10.0])
    axes[-1].set_xticklabels(['0.01', '0.1', '1', '10'])
    axes[-1].set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]')
    handles = [Patch(color=BAND, lw=0), Line2D([], [], **{k_: v_ for k_, v_ in EDGE.items()
                                                          if k_ != 'zorder'})]
    for m in ('TATT', 'NLA'):
        st = MODEL_STYLE[m]
        handles.append((Line2D([], [], color=st['color'], lw=0.6),
                        Line2D([], [], **line_style(m, 'base'))))
    labels = [r'budget, random error, $\ell = 1$', r'budget, $\ell = 0.3$',
              'TATT error (mean, range over $z$)', 'NLA error (mean, range over $z$)']
    place_legend(fig, axes[0], handles, labels, list(axes), above_ncol=2, fontsize=7.5,
                 max_headroom=0, handler_map={tuple: HandlerTuple(ndivide=1)})
    shared_ylabel(fig, axes, r'fractional error')
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
