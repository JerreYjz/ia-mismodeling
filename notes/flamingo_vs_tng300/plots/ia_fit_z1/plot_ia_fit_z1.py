"""Redraw ia_fit_z1.pdf and ia_fit_z1.png from data.json alone.

The best NLA and TATT fits to the FLAMINGO L1_m8 IA power spectra at z = 1,
in shear units at mu = 0. Top row: the density-shape spectrum -P_dE (left)
and the shape-shape spectrum P_EE (right) against k: the measured points
with their errors and the two models on one shared log axis. Bottom row:
the percent difference 100 (model - FLAMINGO) / FLAMINGO at the measured
bins, with the measurement error as a gray band. Each model is solid with
filled markers inside its fit range and dashed with open markers outside.

Run from any folder: python plot_ia_fit_z1.py
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
# marker, so they stay apart in grayscale. Drawn in this order (the thick
# TATT line under the thin NLA line).
MODEL_STYLE = {
    'TATT': {'color': '#0072B2', 'lw': 2.2, 'marker': 'o'},
    'NLA': {'color': '#E69F00', 'lw': 1.5, 'marker': 's'},
}
# the measured points: black circles with error bars
DATA_STYLE = {'color': '#000000', 'marker': 'o', 'ms': 2.8}
# marker size of the models in the percent panels [pt]
MARKER_SIZE = 3.4
# dash and gap lengths [pt]; matplotlib scales a dash pattern by the line
# width, so dividing by it keeps the dashes visible on the thinnest line
DASH = (4.5, 2.5)
# error bar line width [pt]
ERR_LW = 0.9
# thresholds and references: thin, black, dotted, no legend entry
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.1}
# the band of the measurement error in the percent panels
BAND_COLOR = '0.82'
# full-width strip [in]: the two spectra sit side by side
WIDTH = 6.4
# heights of the spectra row and of the percent row [in]
ROW_HEIGHTS = (2.3, 1.6)
# drawn ranges: k [h/Mpc], P [(Mpc/h)^3], percent difference
K_RANGE = (0.018, 4.8)
P_RANGE = (8e-2, 6e2)
PCT_RANGE = (-50.0, 50.0)


def line_points(ax):
    '''Display coordinates of points densely sampled along every line drawn
    in ax (reference lines and error bars included), inside the axes only.
    Filled bands are not sampled: text may sit on a band.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        if len(xy) == 0:
            continue
        disp = line.get_transform().transform(xy)
        ok = np.all(np.isfinite(disp), axis=1)
        pts.append(disp[ok])
        if line.get_linestyle() in ('None', '', ' '):
            continue
        t = np.linspace(0.0, 1.0, 60)[:, None]
        for i in range(len(disp) - 1):
            if ok[i] and ok[i + 1]:
                pts.append(disp[i] + t*(disp[i + 1] - disp[i]))
    t = np.linspace(0.0, 1.0, 40)[:, None]
    for coll in ax.collections:   # error bars
        if not isinstance(coll, matplotlib.collections.LineCollection):
            continue
        for path in coll.get_paths():
            if len(path.vertices) < 2:
                continue
            v = coll.get_transform().transform(path.vertices)
            v = v[np.all(np.isfinite(v), axis=1)]
            for i in range(len(v) - 1):
                pts.append(v[i] + t*(v[i + 1] - v[i]))
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


def dashes(lw):
    '''The dash pattern of a line of width lw.'''
    return (0, (DASH[0]/lw, DASH[1]/lw))


def pieces(k, lo, hi):
    '''Index arrays that split k into below the fit range, inside it and
    above it; neighbors share their boundary point, so the line is unbroken.'''
    inside = np.where((k >= lo*(1 - 1e-9)) & (k <= hi*(1 + 1e-9)))[0]
    first, last = inside[0], inside[-1]
    return np.arange(0, first + 1), inside, np.arange(last, len(k))


def draw_model_curve(ax, k, p, style, lo, hi):
    '''A model spectrum: solid inside the fit range, dashed outside.'''
    below, inside, above = pieces(k, lo, hi)
    kw = {'color': style['color'], 'lw': style['lw']}
    ax.plot(k[inside], p[inside], ls='-', **kw)
    for part in (below, above):
        if len(part) > 1:
            ax.plot(k[part], p[part], ls=dashes(style['lw']), **kw)


def draw_model_percent(ax, k, pct, style, lo, hi):
    '''The percent difference of a model at the measured bins: solid with
    filled markers inside the fit range, dashed with open markers outside.'''
    below, inside, above = pieces(k, lo, hi)
    kw = {'color': style['color'], 'lw': style['lw']*0.6, 'marker': style['marker'],
          'ms': MARKER_SIZE, 'mec': style['color'], 'mew': 0.8}
    for part in (below, above):
        if len(part) > 1:
            ax.plot(k[part], pct[part], ls=dashes(style['lw']*0.6), mfc='white', **kw)
    ax.plot(k[inside], pct[inside], ls='-', mfc=style['color'], **kw)


def draw(data):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, sum(ROW_HEIGHTS)), sharex=True, sharey='row',
                             gridspec_kw={'hspace': 0.06, 'wspace': 0.05,
                                          'height_ratios': ROW_HEIGHTS})
    d = data['data']
    k = np.array(d['k_h_per_Mpc'])
    # the measured bins inside the drawn k range: a line to a bin outside it
    # would enter the panel from the edge
    shown = (k >= K_RANGE[0]) & (k <= K_RANGE[1])
    lo = float(data['kmin_fit'])
    # the two columns: key in data.json, sign that makes the spectrum
    # positive, panel name
    columns = (('dE', -1.0, r'$-P_{\delta E}$'), ('EE', 1.0, r'$P_{EE}$'))
    names = []
    for j, (key, sign, name) in enumerate(columns):
        top, bottom = axes[0, j], axes[1, j]
        p, sig = sign*np.array(d['P' + key]), np.array(d['sig_' + key])
        for model, style in MODEL_STYLE.items():
            m = data['models'][model]
            hi = float(m['kmax_fit'])
            draw_model_curve(top, np.array(m['curve_k_h_per_Mpc']),
                             sign*np.array(m['curve_P' + key]), style, lo, hi)
            draw_model_percent(bottom, k[shown], np.array(m['pct_' + key])[shown], style, lo, hi)
        # an error bar that reaches zero is cut just below the drawn range,
        # so it stays a finite segment for the collision check
        lower = np.minimum(sig, p - 0.5*P_RANGE[0])
        top.errorbar(k, p, [lower, sig], fmt=DATA_STYLE['marker'], ms=DATA_STYLE['ms'],
                     color=DATA_STYLE['color'], elinewidth=ERR_LW, capsize=0, zorder=3)
        top.set_xscale('log')
        top.set_yscale('log')
        top.set_xlim(*K_RANGE)
        top.set_ylim(*P_RANGE)
        top.grid(False, axis='x')
        names.append(top.text(0.96, 0.93, name, transform=top.transAxes, ha='right', va='top',
                              fontsize=11))
        # the measurement error as a band around zero
        band = 100.0*sig/np.abs(p)
        bottom.fill_between(k[shown], -band[shown], band[shown], color=BAND_COLOR, lw=0,
                            zorder=0)
        bottom.axhline(0.0, **THRESHOLD_STYLE)
        bottom.set_ylim(*PCT_RANGE)
        bottom.set_yticks([-40, -20, 0, 20, 40])
        bottom.grid(False, axis='x')
        bottom.set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]')
    axes[0, 0].set_ylabel(r'$P(k)$ [$({\rm Mpc}/h)^3$]')
    axes[1, 0].set_ylabel('percent difference')
    # one legend, in one row above the panels: the data, the models with
    # their fit ranges, and the outside-the-fit-range encoding of both models
    handles = [Line2D([], [], ls='none', marker=DATA_STYLE['marker'], ms=DATA_STYLE['ms'] + 0.8,
                      color=DATA_STYLE['color'])]
    labels = ['FLAMINGO']
    order = [model for model in ('NLA', 'TATT') if model in MODEL_STYLE]
    for model in order:
        st = MODEL_STYLE[model]
        handles.append(Line2D([], [], color=st['color'], lw=st['lw'], marker=st['marker'],
                              ms=MARKER_SIZE, mfc=st['color'], mec=st['color']))
        labels.append(rf"{model}, $k \leq {data['models'][model]['kmax_fit']:.2f}$")
    handles.append(tuple(Line2D([], [], color=MODEL_STYLE[m]['color'], lw=MODEL_STYLE[m]['lw'],
                                ls=dashes(MODEL_STYLE[m]['lw']), marker=MODEL_STYLE[m]['marker'],
                                ms=MARKER_SIZE, mfc='white', mec=MODEL_STYLE[m]['color'])
                         for m in order))
    labels.append('outside the fit range')
    leg = axes[0, 0].legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.02),
                            ncol=len(handles), frameon=False, borderaxespad=0.0, fontsize=9,
                            handletextpad=0.5, columnspacing=1.1, handlelength=2.1,
                            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
    check_clear(fig, list(axes[0]), [leg])
    for ax, name in zip(axes[0], names):
        check_clear(fig, [ax], [name])
    fig.align_ylabels(axes[:, 0])
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
