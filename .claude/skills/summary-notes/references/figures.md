# Figures: every rule, with the reason

Read this before writing or changing any figure. The working template
is `../assets/plot_template.py`; copy it rather than starting from an
empty file, because it already carries the style block and the
legend checks.

## 1. The plot-folder contract

Each figure lives in `notes/<note>/plots/<name>/`:

| file | what it is |
|---|---|
| `data.json` | everything the figure draws, plus an `about` string: what the numbers are, their units, and the script or notebook cell that wrote them |
| `plot_<name>.py` | redraws the figure from `data.json` alone |
| `<name>.pdf` | the file the note includes (vector) |
| `<name>.png` | dpi 300, for looking at, for slides and for GitHub |

The numbers get into `data.json` through a **generator script** in the
note's `plots/` folder (for example `plots/make_figure_data.py`) that
reads the analysis outputs and writes every folder's `data.json`. It
is the only place that knows where the pipeline's files are.

**Why the split.** The expensive part (the chains, the fits, the
Fisher matrices) runs once; the drawing is redone ten times while the
note is reviewed. If the plot script reads the pipeline's own files,
every restyle depends on the pipeline still existing in the same
state. With `data.json` in between, the figure is a small, frozen,
reviewable object, and `git diff data.json` shows exactly which
numbers moved when the analysis is rerun.

**The test.** Copy the folder to an empty directory and run
`python -I -W error::RuntimeWarning plot_<name>.py` there (the `-I`
flag stops Python from importing anything from the current folder, so
a hidden dependency on the repository fails loudly). Then open the PNG
and look at it.

What goes into `data.json`:

- the arrays drawn, with their units in the key name or in `about`
  (`"kmax_h_per_Mpc"`, `"sigma_S8"`);
- the reference values drawn as lines (thresholds) under their own
  keys, so the plot script never hard-codes a result;
- values the caption quotes but the figure does not draw, when they
  belong to the same computation (the case-B numbers next to case A);
- `null` for a missing point (the script draws a gap, never a zero).

What never goes into `data.json`: styling. Colors and line widths
belong to the plot script, so a restyle never touches the data file.

## 2. The house rcParams block

Every plot script, and every notebook cell that draws a figure for a
note or paper, carries this block verbatim:

```python
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
```

**Why each line.** STIX text matches the Times-like body font of a
journal, so labels do not look pasted in. Ticks only on the bottom and
left keep the frame quiet. The grid is on but drawn with zero width:
it exists for the `ax.grid(False, axis='x')` style toggles without
ever showing. `bbox = 'tight'` trims white margins, so the
`\includegraphics` width is the width of the ink.

Never change rcParams inside a helper function; set them once at the
top. A helper that changes a global default changes every later figure
of the same session.

## 3. Layout

- **Width.** One column (3.5 in) by default. A full-width strip
  (about 6.4 in) only for a grid that must sit side by side, and then
  short: a strip of the page, never a full page.
  *Why:* the final home is a two-column paper. A figure designed at
  full width and shrunk into a column has unreadable labels.
- **Stacked panels.** Wide and short (1.1 to 1.3 in each), stacked
  vertically with a thin gap (`hspace` about 0.06, never 0), sharing
  the x axis (tick labels on the bottom panel only), one shared y range
  and one shared y label (`fig.supylabel`).
  *Why:* stacked panels with one y range let the eye compare heights
  across panels directly; separate ranges invite wrong comparisons.
- **Panel names** go inside the panel (a corner free of data) or as a
  rotated right-hand label (`ax.yaxis.set_label_position('right')`),
  never as a title unless the caption cannot carry it.
- **Different natural ranges.** Divide each panel by one round
  constant written inside the panel ("x 10"), so all panels share one
  axis. Separate ranges give a poor aspect ratio.
- **Sibling figures** (same axes, two scenarios) become rows of one
  grid.
- **Crowding.** Draw only the variants the text discusses. Values that
  are 0 or 1 by construction go into the caption. One panel with
  redundant encodings beats several panels when curves stay apart.
- **Axes.** Linear unless the values span decades. Axis labels carry
  units in brackets: `$k$ [$h\,{\rm Mpc}^{-1}$]`. A categorical x axis
  keeps cases close.
- **Thresholds** (0.3 sigma, chi2 = 1) are a thin black dotted line
  with no legend entry; the caption names them.

## 4. Color and redundant encoding

**Palette: Okabe-Ito**, colorblind safe. Never red together with green
(its green, `#009E73`, is not used).

| name | hex | typical use |
|---|---|---|
| blue | `#0072B2` | the main model / the reference layout |
| orange | `#E69F00` | the comparison model |
| reddish purple | `#CC79A7` | a third model |
| sky blue | `#56B4E9` | a fourth category |
| vermillion | `#D55E00` | a fifth category (never with green) |
| black / grays | `#000000`, `0.5` | references and external results |

For the IA project the models are fixed across every note, so a reader
never relearns them:

| model | color | line width | marker |
|---|---|---|---|
| TATT | `#0072B2` | 2.2 | circle `o` |
| NLA | `#E69F00` | 1.5 | square `s` |
| HMc (halo model) | `#CC79A7` | 1.1 | diamond `D` |

and the variants of one model are line styles: the baseline solid with
filled markers, a variant (with a nuisance, another covariance) dashed
with open markers. A dash pattern must stay visible on the thinnest
line, so divide the dash lengths by the line width:
`ls = (0, (4.5/lw, 2.5/lw))`.

**Never color alone.** Every category differs in at least two of:
color, line width, marker, line style. *Why:* the grayscale test.
Convert the PNG to gray (`convert x.png -colorspace Gray g.png`, or
look at it with the screen's grayscale filter) and check that every
line can still be named.

## 5. Legends, labels and references

- Legends `frameon=False`; no `bbox` on text; no `\%` in labels
  without usetex (it prints the backslash; write "percent" or `%`).
- **A legend, label or text never touches a data line, a marker or a
  reference line, and this is enforced in code.** The template's
  `place_legend` samples every drawn line in display coordinates
  (`line_points`), tries the free corners in order, then grows the y
  headroom by the least amount, and raises `RuntimeError` when nothing
  clears. Never rely on a visual check alone: what clears at one size
  overlaps at another.
- Legend entries show the markers and line widths that are actually
  drawn; for filled-versus-open pairs use a tuple handle with both.
- Two legends (models above, variants below) sit close together.
- A reference far above the data (more than 1.5 times the highest
  point) is not drawn on a stretched axis; an in-panel text quotes it,
  and the text is included in the collision check.
- Legend labels name things plainly ("TATT", "with the white-II
  nuisance"), never with internal code names (`TATTw_A`).

## 6. Captions

- Short statements, never questions. Say what is drawn, what is
  omitted, and how the encodings read ("Color, line width and marker:
  the model; solid, filled: case A; dashed, open: with the white-II
  nuisance; dotted: FoB = 0.3").
- No value the reader takes from the plot. No number to more digits
  than the plot resolves.
- Symbols are defined in the text; the caption only uses them.
- A caption of a figure with a scale-cut axis carries the same clause
  every time it appears (defined once as a macro, e.g. `\kuvclause`).

## 7. Tables become figures

A table of many numbers becomes stacked panels with one line per entry
against the natural variable (the cut, the redshift, the volume). The
full table stays in `data.json`. A table that a figure duplicates is
removed. Only short didactic tables remain: a list of definitions, the
five recommended configurations.

**Order of the story.** A model or prior that fails the accuracy test
gets no precision figure (its precision means nothing). Show the
accuracy first, then the precision of what survives.

## 8. Looking at the PNG

After every redraw, open the PNG (the Read tool shows images). Check:
labels readable at 100%; no text on a line; the legend matches what is
drawn; the panels share the range the caption claims; nothing is cut
off by the tight bounding box; in grayscale, every line identifiable.
