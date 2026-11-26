"""FLAMINGO P_hE, P_EE, P_BB at z = 0.3 and 1 with the HMk prediction (no fitted parameter),
HMk with the kernel gradient fitted to k <= kmax_fit, and the 2-halo and 1-halo parts;
residual rows in units of the FLAMINGO error.

Redraws spectra.pdf and spectra.png from data.json alone:
    python plot_spectra.py
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


NAMES = (r'$-P_{hE}$', r'$P_{EE}$', r'$P_{BB}$')
WIDE = 7.0


def draw(data):
    zs = list(data['panels'])
    fig, axes = plt.subplots(2*len(zs), 3, figsize=(WIDE, 2.55*len(zs)), sharex=True,
                             gridspec_kw={'height_ratios': [2.3, 1.0]*len(zs), 'hspace': 0.07, 'wspace': 0.3})
    final = {'color': '#009E73', 'lw': 1.8}
    zero = {'color': '#009E73', 'lw': 1.2, 'ls': (0, (DASH[0]/1.2, DASH[1]/1.2))}
    two = {'color': '#0072B2', 'lw': 0.9, 'ls': ':'}
    one = {'color': '#E69F00', 'lw': 0.9, 'ls': '-.'}
    for r, z in enumerate(zs):
        p = data['panels'][z]
        k = as_array(p['k'])
        for c in range(3):
            sg = -1.0 if c == 0 else 1.0
            a, b = axes[2*r, c], axes[2*r + 1, c]
            y, e = sg*as_array(p['y'][c]), as_array(p['err'][c])
            a.errorbar(k, y, e, fmt='o', ms=2.4, color='k', lw=0.7, zorder=5)
            a.plot(k, sg*as_array(p['final'][c]), **final)
            a.plot(k, sg*as_array(p['zero'][c]), **zero)
            if c < 2:
                a.plot(k, sg*as_array(p['two_halo'][c]), **two)
            a.plot(k, sg*as_array(p['one_halo'][c]), **one)
            a.set_yscale('log')
            pos = y[np.isfinite(y) & (y > 0)]
            a.set_ylim(0.3*pos.min(), 2.0*pos.max())
            a.grid(False, axis='x')
            b.axhline(0.0, color='0.3', lw=0.6)
            b.fill_between([0.04, 4.0], -1, 1, color='0.9', lw=0)
            b.plot(k, (as_array(p['final'][c]) - as_array(p['y'][c]))/e, **final)
            b.plot(k, (as_array(p['zero'][c]) - as_array(p['y'][c]))/e, **zero)
            b.set_ylim(-12, 12)
            b.grid(False, axis='x')
            k_axis(b, (0.1, 0.3, 1.0, 3.0))
            b.axvline(float(data['kmax_fit']), **THRESHOLD_STYLE)
            if c == 0:
                a.set_ylabel(f'$z = {z}$\n' + r'[$(h^{-1}{\rm Mpc})^3$]', fontsize=9)
                b.set_ylabel(r'$\Delta/\sigma$', fontsize=9)
            a.set_title(NAMES[c], fontsize=10) if r == 0 else None
            if r == len(zs) - 1:
                b.set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]')
    axes[-1, 0].set_xlim(0.045, 3.3)
    handles = [Line2D([], [], marker='o', ms=2.4, color='k', ls='none'), Line2D([], [], **final),
               Line2D([], [], **zero), Line2D([], [], **two), Line2D([], [], **one)]
    labels = ['FLAMINGO', r'HMk + $A$', 'HMk (0 par.)', '2-halo', '1-halo (measured)']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=5, frameon=False, fontsize=8)
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
