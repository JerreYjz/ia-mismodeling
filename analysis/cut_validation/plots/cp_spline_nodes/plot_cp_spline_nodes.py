"""Where the bias of NLA on the CP truth comes from in k: response and error per B-spline node.

Columns: cosmic shear at K0 = 0.2 and 0.4, 3x2pt (gamma_t at 0.5) at 0.2;
every parameter free, CMB prior. Nodes: cubic B-splines in ln k, 3 per
decade (marker at each node's centre). Top: the S8 shift (in sigma) of a
unit fractional error of the spectrum at every snapshot. Middle: the model
error 1 - P^model/P^truth after the amplitude (A1, eta1) part is projected
out, at z = 1 (thick) and z = 0.5, 1.5 (thin). Bottom: each node's S8
contribution (markers) and their running sum over k (black). Blue GI,
vermillion II E modes, reddish purple II B modes (dashed).

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


SPEC = {'GI': dict(color='#0072B2', ls='-', marker='s', label='GI'),
        'II': dict(color='#D55E00', ls='-', marker='o', label='II, E'),
        'IIB': dict(color='#CC79A7', ls=(0, (3, 1.5)), marker='^', label='II, B')}
TITLE = {'cs 0.2': r'cosmic shear, $K_0=0.2$', 'cs 0.4': r'cosmic shear, $K_0=0.4$',
         '3x2 gt0.5 0.2': r'3x2pt, $\gamma_t$ 0.5, $K_0=0.2$'}
MODEL = 'NLA'


def draw(data):
    cols = list(data['cuts'])
    fig, axes = plt.subplots(3, len(cols), figsize=(7.0, 4.6), sharex=True, sharey='row',
                             gridspec_kw={'hspace': 0.08, 'wspace': 0.08})
    for j, c in enumerate(cols):
        L = data['lines'][f'{c}|{MODEL}']
        k = np.array(L['k_node'])
        for X in L['spectra']:
            st = SPEC[X]
            axes[0, j].plot(k, L['response_S8'][X], color=st['color'], ls=st['ls'], marker=st['marker'],
                            ms=2.8, lw=1.2)
            for z, e in L['error'][X].items():
                thick = abs(float(z) - 1.0) < 0.05
                axes[1, j].plot(k, e, color=st['color'], ls=st['ls'], lw=1.5 if thick else 0.7,
                                marker=st['marker'] if thick else None, ms=2.8, alpha=1 if thick else 0.6)
            axes[2, j].plot(k, L['contrib_S8'][X], ls='none', color=st['color'], marker=st['marker'], ms=3.2,
                            mfc='white' if X == 'IIB' else st['color'])
        axes[2, j].step(k, L['running_S8'], where='mid', color='black', lw=1.4)
        axes[2, j].axhline(L['total_S8'], **THRESHOLD_STYLE)
        axes[0, j].set_title(TITLE[c], fontsize=8)
        for a in axes[:, j]:
            a.axhline(0.0, color='0.6', lw=0.6)
            a.grid(False, axis='x')
        axes[2, j].set_xscale('log')
        axes[2, j].set_xlim(0.008, 12)
        axes[2, j].set_xticks([0.01, 0.1, 1.0])
        axes[2, j].set_xticklabels(['0.01', '0.1', '1'])
        axes[2, j].set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]')
    axes[0, 0].set_ylabel(r'response, $\Delta S_8/\sigma$', fontsize=9)
    axes[1, 0].set_ylabel('error', fontsize=9)
    axes[2, 0].set_ylabel(r'node $\Delta S_8/\sigma$', fontsize=9)
    axes[1, 0].set_ylim(-0.3, 1.15)
    h = [Line2D([], [], color=st['color'], ls=st['ls'], marker=st['marker'], ms=3, lw=1.2) for st in SPEC.values()]
    h += [Line2D([], [], color='black', lw=1.4), Line2D([], [], **THRESHOLD_STYLE)]
    leg = fig.legend(h, [st['label'] for st in SPEC.values()] + ['running sum', 'total (direct)'],
                     loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=5, frameon=False, fontsize=7.5)
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
