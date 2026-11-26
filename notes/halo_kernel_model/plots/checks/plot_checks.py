"""Physical checks of the fitted saturated kernel gradient A (per snapshot, for three k_max) and
of the large-scale alignment kernels per mass bin against z.

Redraws checks.pdf and checks.png from data.json alone:
    python plot_checks.py
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



# one encoding per model, the same in every figure of the note: color, line
# width and marker (and dashes with open markers for the HMk variant without
# the kernel gradient), so the lines stay apart in grayscale
MODEL_STYLE = {
    'tatt': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
    'nla': {'color': '#E69F00', 'lw': 1.5, 'marker': 's'},
    'lin': {'color': '0.45', 'lw': 1.0, 'marker': '^'},
    'emu': {'color': '#56B4E9', 'lw': 1.1, 'marker': 'v'},
    'tab': {'color': '#CC79A7', 'lw': 1.1, 'marker': 'P'},
    'w': {'color': '#009E73', 'lw': 1.8, 'marker': 'D', 'dashed': True},
    'wg': {'color': '#009E73', 'lw': 1.8, 'marker': 'D'},
    'wz': {'color': '#009E73', 'lw': 1.0, 'marker': 'o', 'dashdot': True},
    'ctrl': {'color': '0.1', 'lw': 0.9, 'marker': 'x'},
}


def kw(key, thin=False, markers=True):
    '''Plot keywords of a model (thin: per-snapshot lines, no markers).'''
    st = MODEL_STYLE[key]
    lw = 0.7 if thin else st['lw']
    ls = (0, (DASH[0]/lw, DASH[1]/lw)) if st.get('dashed') else '-'
    if st.get('dashdot'):
        ls = (0, (DASH[0]/lw, 1.5/lw, 1.0/lw, 1.5/lw))
    out = {'color': st['color'], 'lw': lw, 'ls': ls}
    if markers and not thin:
        out.update(marker=st['marker'], ms=MARKER_SIZE, mec=st['color'],
                   mfc='white' if st.get('dashed') or st.get('dashdot') else st['color'])
    if thin:
        out['alpha'] = 0.75
    return out


def k_axis(ax, ticks=(0.1, 0.2, 0.5, 1.0, 2.0)):
    from matplotlib import ticker
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


KM = (('0.52', {'color': '#E69F00', 'lw': 1.2, 'marker': 's', 'dashed': True}),
      ('0.94', {'color': '#56B4E9', 'lw': 1.4, 'marker': 'v'}),
      ('1.26', {'color': '#009E73', 'lw': 1.8, 'marker': 'D'}))
BIN_COLORS = ('#440154', '#3b528b', '#21918c', '#5ec962', '#c8a400', '#E69F00')
BIN_MARKERS = ('o', 's', 'D', '^', 'v', 'P')


def st(s):
    lw = s['lw']
    ls = (0, (DASH[0]/lw, DASH[1]/lw)) if s.get('dashed') else '-'
    return {'color': s['color'], 'lw': lw, 'ls': ls, 'marker': s['marker'], 'ms': MARKER_SIZE,
            'mec': s['color'], 'mfc': 'white' if s.get('dashed') else s['color']}


def draw(data):
    z = as_array(data['zs'])
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3*1.3), sharex=True, gridspec_kw={'hspace': 0.08})
    a = axes[0]
    a.axhline(0.0, color='0.6', lw=0.6)
    for i, (km, s) in enumerate(KM):
        a.errorbar(z + 0.04*(i - 1), as_array(data['A'][km]), as_array(data['A_err'][km]),
                   capsize=0, clip_on=False, **st(s))
    a.set_ylabel(r'$A$', fontsize=10)
    a.set_ylim(-0.05, 0.62)
    h = [Line2D([], [], **st(s)) for _, s in KM]
    place_legend(fig, a, h, [rf'$k_{{\max}} = {km}$' for km, _ in KM], list(axes), fontsize=7.5, max_headroom=0)
    e = data['lgm_edges']
    labs = [f'{max(e[i], 11.3):.1f} to {min(e[i + 1], 15):.1f}' if i else f'< {e[1]:.1f}' for i in range(len(e) - 1)]
    for ax, key, name in ((axes[1], 'kappa_c', r'$\kappa_c$'), (axes[2], 'kappa_s', r'$\kappa_s$')):
        K = np.array([as_array(r) for r in data[key]])
        hs = []
        for j in range(K.shape[1]):
            s = {'color': BIN_COLORS[j], 'lw': 1.1, 'marker': BIN_MARKERS[j], 'ms': 2.8}
            ax.plot(z, K[:, j], **s, mec=s['color'])
            hs.append(Line2D([], [], **s, mec=s['color']))
        ax.set_ylabel(name, fontsize=10)
        ax.grid(False, axis='x')
        if key == 'kappa_c':
            ax.legend(hs, labs, loc='lower center', ncol=3, frameon=False, fontsize=6.5,
                      bbox_to_anchor=(0.5, 0.0), columnspacing=0.8)
    axes[1].set_ylim(-0.75, 2.6)
    axes[2].set_ylim(0.0, 1.5)
    axes[-1].set_xlabel('$z$')
    axes[-1].set_xlim(-0.1, 4.1)
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
