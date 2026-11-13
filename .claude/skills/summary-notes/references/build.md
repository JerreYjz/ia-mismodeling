# Building and checking (this machine)

## Tools

- LaTeX: `/Library/TeX/texbin/pdflatex` (MacTeX).
- Python for figures: the cocoa environment's interpreter,
  `~/cocoa/Cocoa/.local/bin/python` (numpy, scipy, matplotlib, CAMB,
  FAST-PT). The system `python3` has no numpy.
- Scratch: the session scratchpad. Never build inside the repository:
  the `.aux`, `.log` and `.out` files must not be committed.

## Redraw test for one figure

```bash
SCR=$(mktemp -d)
cp -R notes/<note>/plots/<name> "$SCR"/
cd "$SCR/<name>" && ~/cocoa/Cocoa/.local/bin/python -I -W error::RuntimeWarning plot_<name>.py
```

It must finish without error and write `<name>.pdf` and `<name>.png`.
Then copy the two images back (or run the script in place: it writes
next to itself) and **look at the PNG**.

## Building the note

`assets/build_note.sh notes/<note>` does all of this:

1. copies `<note>.tex` (and any appendix `.tex`) and `plots/` to a
   scratch folder;
2. runs `pdflatex -interaction=nonstopmode -halt-on-error` three times
   (references and the table of contents settle on the third pass);
3. counts in the log: lines starting with `!` (errors), `undefined`
   (references or citations), `too large` (a float that does not fit);
   all three must be zero. `Overfull \hbox` lines are reported, and
   anything above 5pt is fixed;
4. greps the source for em-dashes (`---`, and ` -- ` used as
   punctuation) and for TODO;
5. copies `<note>.pdf` back next to the source.

## Looking at the result

Render the pages you changed to PNG and look at them (the Read tool
displays PDFs page by page with `pages`). Check: no float alone on a
mostly empty page, no figure far from its first mention, the Summary
fits in two pages.

## Committing

The commit carries the `.tex`, the `.pdf`, `plots/make_figure_data.py`
and every `plots/<name>/` folder (data.json, script, PDF, PNG).
