"""The halo-model variants on the stitched and linear-limit truths.

Rows: cosmic shear and 3x2pt (every parameter free, CMB prior), each on the
stitched and on the linear-limit truth. Left: the figure of bias of the
minimizer's MAP. Middle: the largest single piece of it (bias_pieces.py
"zshape": the amplitude error of every snapshot, or the scale dependence of
GI or II in one of k < 0.06, 0.06 to 0.3, above 0.3 h/Mpc), the bias a cut
would carry if the pieces did not cancel. Right: the largest piece of scale
dependence alone (the amplitude piece left out), the part a scale cut acts on. NLA family orange squares, TATT
family blue circles; solid filled: NLA, TATT; dashed open: alpha free;
dotted with crosses: alpha at FLAMINGO's values. Dotted gray: 0.3.
Run from any folder: python plot_hm_cancel.py
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



PIECE_STYLE = {
    'z|amplitude': dict(color='#009E73', lw=1.6, ls='-', marker='D', label='amplitude (all $k$)'),
    'II|0|0.06': dict(color='#D55E00', lw=1.2, ls=':', marker='v', label=r'II, $k<0.06$'),
    'II|0.06|0.3': dict(color='#D55E00', lw=1.4, ls='-', marker='s', label=r'II, 0.06 to 0.3'),
    'II|0.3|inf': dict(color='#D55E00', lw=1.2, ls='--', marker='^', label=r'II, $k>0.3$'),
    'GI|0|0.06': dict(color='#0072B2', lw=1.0, ls=':', marker='v', label=r'GI, $k<0.06$'),
    'GI|0.06|0.3': dict(color='#0072B2', lw=1.1, ls='-', marker='s', label=r'GI, 0.06 to 0.3'),
    'GI|0.3|inf': dict(color='#0072B2', lw=1.0, ls='--', marker='^', label=r'GI, $k>0.3$'),
}
TOTAL_STYLE = dict(color='black', lw=2.2, ls='-', marker='o', label='total (MAP)')
MODEL_NAME = {'NLA': 'NLA', 'TATT': 'TATT', 'NLAa': r'NLA + $\alpha$', 'TATTa': r'TATT + $\alpha$',
              'NLAf': r'NLA, $\alpha_{\rm F}$', 'TATTf': r'TATT, $\alpha_{\rm F}$'}
PROBE_NAME = {'cs': 'cosmic shear', '3x2': '3x2pt'}


def kw(st, open_marker=True):
    return dict(color=st['color'], lw=st['lw'], ls=st['ls'], marker=st['marker'], ms=3.4,
                mfc='white' if open_marker else st['color'], mec=st['color'], clip_on=False)


FAMILY = {'NLA': dict(color='#E69F00', marker='s', lw=1.5), 'TATT': dict(color='#0072B2', marker='o', lw=2.0)}
VARIANT = {'': dict(ls='-', fill=True), 'a': dict(ls='--', fill=False), 'f': dict(ls=':', fill=None)}
TRUTH_NAME = {'S1': 'stitched', 'lin': 'linear limit', 'prod': 'production'}


def mstyle(m):
    fam, var = (m[:-1], m[-1]) if m[-1] in 'af' else (m, '')
    f, v = FAMILY[fam], VARIANT[var]
    marker = 'x' if v['fill'] is None else f['marker']
    return dict(color=f['color'], lw=f['lw'] if var == '' else 1.3, ls=v['ls'], marker=marker,
                ms=3.6 if marker != 'x' else 4.0, mfc=f['color'] if v['fill'] else 'white', mec=f['color'],
                clip_on=False)


def draw(data):
    K = as_array(data['K0'])
    rows = [(p, tr) for p in data['probes'] for tr in data['truths']]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.0, 1.3*len(rows) + 0.5), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.08, 'wspace': 0.12})
    top = 0.0
    for i, (p, tr) in enumerate(rows):
        for j, what in enumerate(('total', 'largest', 'largest_shape')):
            ax = axes[i, j]
            ax.axhline(data['threshold'], **THRESHOLD_STYLE)
            for m in data['models']:
                y = as_array(data['lines'][f'{p}|{m}|{tr}'][what])
                if np.any(np.isfinite(y)):
                    top = max(top, np.nanmax(y))
                ax.plot(K, y, **mstyle(m))
            ax.grid(False, axis='x')
            ax.text(0.04, 0.93, f'{PROBE_NAME[p]}, {TRUTH_NAME[tr]}', transform=ax.transAxes, fontsize=7,
                    va='top')
    for ax in axes.flat:
        ax.set_ylim(0, max(0.6, 1.08*top))
    trim_top_tick(list(axes.flat))
    axes[0, 0].set_title('total (MAP)', fontsize=9)
    axes[0, 1].set_title('largest piece', fontsize=9)
    axes[0, 2].set_title('largest scale-dependence piece', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(r'$K_0$ [$h\,\mathrm{Mpc}^{-1}$]')
    for ax in axes[:, 0]:
        ax.set_ylabel('figure of bias', fontsize=8)
    h = [Line2D([], [], **mstyle(m)) for m in data['models']]
    fig.legend(h, [MODEL_NAME[m] for m in data['models']], loc='lower center', bbox_to_anchor=(0.5, axes[0, 0].get_position().y1 + 0.03),
               ncol=6, frameon=False, fontsize=7.5, handlelength=2.6, columnspacing=1.0)
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
