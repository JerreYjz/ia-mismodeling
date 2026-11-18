"""Redraw ia_redblue.pdf and ia_redblue.png from data.json alone.

The halo model with red and blue galaxies, fitted jointly to the nine core
FLAMINGO snapshots (k <= 0.52 h/Mpc). Top: the effective linear-alignment
amplitude against redshift. Black points with error bars: A1 of the
per-snapshot halo-model fit (free share of nonlinear power, no satellites).
Lines: the effective amplitude of each joint model, with chi2/nu in the
legend: one population with a power law in (1 + z) (blue, circles) and with A1
proportional to the galaxy bias (green, squares), no colour split; the
halo-based colour split, every galaxy aligned like its host halo with blue
galaxies scaled by kappa_b (pink, diamonds; at kappa_b = 1 it is the green
model), and red and blue centrals with constant amplitudes as Fortuna et al.
2021 (vermillion, down triangles). Bottom: the red (vermillion) and blue (blue)
fractions of the sample, centrals solid, satellites dashed, from the
halo-quenching fits of Zu and Mandelbaum 2016 and Tinker et al. 2013 on the
FLAMINGO HOD.

Run from any folder: python plot_ia_redblue.py
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


TRUTHS = ('prod', 'S1')
TRUTH_LABEL = {'prod': 'production', 'S1': 'stitched'}
SHADE = '0.9'


STYLE = {
    '1p': {'color': '#0072B2', 'lw': 2.0, 'ls': '-', 'marker': 'o', 'label': r'no split: $A_1 X^{\eta_1}$'},
    '1pB': {'color': '#009E73', 'lw': 1.6, 'ls': (0, (5, 2)), 'marker': 's', 'label': r'no split: $A_1 b_g$'},
    'rbS': {'color': '#CC79A7', 'lw': 1.6, 'ls': (0, (6, 2, 1, 2)), 'marker': 'D', 'label': r'split, halo-based ($\kappa_b$ free)'},
    'rbF21': {'color': '#D55E00', 'lw': 1.2, 'ls': (0, (8, 2.5)), 'marker': 'v', 'label': 'split, F21 (constant amplitudes)'},
}
FRAC = {'f_rc': ('#D55E00', '-', 'red centrals'), 'f_bc': ('#0072B2', '-', 'blue centrals'),
        'f_rs': ('#D55E00', (0, (4, 2)), 'red satellites'), 'f_bs': ('#0072B2', (0, (4, 2)), 'blue satellites')}


def draw(data):
    z = as_array(data['zs'])
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2*1.6 + 1.3), sharex=True,
                             gridspec_kw={'hspace': 0.07, 'height_ratios': [1.4, 1.0]})
    ax = axes[0]
    handles, labels = [], []
    for m, st in STYLE.items():
        h, = ax.plot(z, as_array(data['a1_joint'][m]), color=st['color'], lw=st['lw'], ls=st['ls'],
                     marker=st['marker'], ms=3.4, zorder=2)
        handles.append(h)
        labels.append(st['label'] + f" ({data['chi2_nu'][m]:.1f})")
    eb = ax.errorbar(z, as_array(data['a1_snap']), yerr=as_array(data['a1_snap_sig']), fmt='o', color='black',
                     ms=3.6, mfc='white', lw=1.0, capsize=1.5, zorder=3)
    handles.insert(0, eb)
    labels.insert(0, 'per-snapshot fit')
    ax.set_ylim(0, 7.5)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel(r'effective $A_1$', fontsize=9)
    leg = ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(-0.02, 1.04), ncol=1,
                    frameon=False, borderaxespad=0.0, fontsize=7.0, title=r'$\chi^2/\nu$ of the joint fit in brackets',
                    title_fontsize=7.0)
    ax2 = axes[1]
    for key, (c, ls, lab) in FRAC.items():
        ax2.plot(z, as_array(data['fractions'][key]), color=c, ls=ls, lw=1.5 if ls == '-' else 1.2, label=lab)
    ax2.set_ylim(0, 0.9)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_ylabel('fraction of sample', fontsize=9)
    leg2 = place_legend(fig, ax2, *ax2.get_legend_handles_labels(), [ax2], above_ncol=2, fontsize=7.0,
                        max_headroom=0, ncol=2)
    for a_ in axes:
        a_.grid(False)
        a_.minorticks_off()
    axes[-1].set_xlim(0.1, 2.1)
    axes[-1].set_xlabel(r'$z$')
    check_clear(fig, list(axes), [leg, leg2])
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
