---
name: summary-notes
description: House style and workflow for Yijie's summary PDFs (LaTeX notes, work summaries, memos for collaborators, result write-ups) and for every figure in them, modeled on Vivian Miranda's notes in SBU-COSMOLIKE/lssty1xdesi. Use whenever a summary PDF, note, report or memo is written, revised or "cleaned up" from a work log; whenever a plot or figure is made for a note, a paper or a talk (matplotlib scripts, notebook plotting cells, captions); and whenever results tables are turned into figures. Covers the note anatomy (Summary first, the fiducial frozen in the note, appendices, Open issues), the numbers block, the plot-folder contract (data.json, a standalone plot script, PDF and PNG), the house rcParams block, the colorblind-safe palette with redundant encodings, legend placement checked in code, captions, and the build-and-check recipe on this machine.
---

# Summary notes and their figures

This skill is how a result becomes something a collaborator can read
in ten minutes and trust. It is copied, with explanations added, from
the way Vivian Miranda writes the notes of the lssty1xdesi project
(`github.com/SBU-COSMOLIKE/lssty1xdesi`, folder `notes/`, and its
`.claude/skills/paper-figures`). The rules came from repeated
corrections by a PI; each one is here because breaking it once cost a
round of review. Read the **why** under each rule: when a new situation
is not covered, the reason tells you what to do.

Two products, always together:

1. **The note**: a LaTeX document, `notes/<note>/<note>.tex`, built to
   `<note>.pdf`. It describes what was done: the question, the method,
   the result and its limits.
2. **Its figures**: one folder per figure,
   `notes/<note>/plots/<name>/`, that redraws the figure from saved
   numbers alone.

Details live in three references (read the one you need before
writing):

| topic | file |
|---|---|
| every figure rule, the rcParams block, palette, encodings, legend checks, captions | `references/figures.md` |
| note anatomy, the numbers block, writing rules, turning a work log into a note | `references/notes.md` |
| building the PDF, checking the log, the redraw test, looking at every PNG | `references/build.md` |
| a complete plot script to copy | `assets/plot_template.py` |
| a note skeleton to copy | `assets/note_template.tex` |
| the build-and-check script | `assets/build_note.sh` |

A worked example of everything here is the IA scale-cut note,
`~/cocoa/ia_mismodeling/notes/scale_cuts_summary/` (repository
`JerreYjz/ia-mismodeling`): it was rebuilt with this skill from a
work log (`analysis/roman_dchi2/report/summary.pdf`), so comparing the
two shows every rule applied.

## The workflow

Follow these steps in order. Steps 1 to 3 are where most of the time
should go; a note written before its figures are right gets rewritten.

1. **Decide the one question and the answer.** Write them as two
   sentences before anything else. If you cannot, the note is not
   ready: you are still exploring, and the result belongs in a log,
   not a note.
2. **Choose the figures.** One figure per claim of the Summary. Every
   large table becomes a figure (stacked panels, one line per entry
   against the natural variable). Draw only what the text discusses.
3. **Export the data, then draw.** For each figure, write `data.json`
   (with an `about` string naming its producer) from the analysis
   outputs with a generator script, then write `plot_<name>.py` from
   `assets/plot_template.py`. Run the redraw test
   (`references/build.md`) and **look at every PNG**.
4. **Write the note**: Summary first (one to two pages), then the
   fiducial setup, then one section per claim, then appendices
   (method details, numerical checks, where the code is, Open issues).
   Every number goes through the numbers block at the top.
5. **Build and check** in a scratch folder: pdflatex three times, zero
   errors, zero undefined references, no float too large. Copy the PDF
   back.
6. **Revise once for length**: cut about 20% of the prose. Say each
   thing once, remove repeated caveats first, prefer a figure to a
   paragraph.

## The core rules, with their reasons

**A note describes what was done, not what should be done.** No TODO,
no "should", "will" or "needs" in the body; no Recommendation section
mixed into the results. *Why:* a note is read months later, after the
plan changed; future-tense text goes stale silently, while a record of
what was done stays true. Decisions and recommendations live in the
project skill (or, for a paper note, in one clearly labeled final
section "Recommendation" that the reader can skip). Open items go into
an "Open issues" appendix, one to three sentences each.

**Summary first, short, complete.** The Summary states the question,
the method in one paragraph, the result with its numbers, and the
limits. A collaborator who reads only the Summary must leave with the
right conclusion. *Why:* most readers stop there; the body is for the
reader who wants to check.

**Freeze the fiducial in the note.** A section "Fiducial setup" lists
the cosmology, the data vector, the covariance, the masks and the
fitting choices, with the values. *Why:* the yaml files and scripts
change; the note must still say what its numbers were computed at.

**Every number has one home: the numbers block.** Numbers that come
from results are LaTeX macros defined at the top, each with a comment
naming the file (and key) it comes from. Results that will be
recomputed use `\refresh{...}`, so a whole block can be printed in red
while it is stale. *Why:* a number typed in three places will be
updated in two.

**Figures, not tables.** No table of many numbers in any note; it
becomes a compact figure, and the full values stay in the figure's
`data.json`. Only short didactic tables survive (a definitions table,
a five-row list of recommended choices). *Why:* the eye compares lines
instantly and compares columns of digits slowly and wrongly; the
digits are still one file away for whoever needs them.

**Every figure redraws from its folder alone.** The plot-folder
contract: `data.json`, `plot_<name>.py`, `<name>.pdf`, `<name>.png`
(dpi 300). Copying the folder anywhere and running the script redraws
the figure without the pipeline, cosmolike, CAMB or the repository.
*Why:* figures get restyled many times after the expensive run that
made their numbers; the restyle must never need the run again.

**One house style, set by one rcParams block.** Copied verbatim into
every plot script (`references/figures.md`). One-column width (3.5 in)
by default; stacked, wide, short panels sharing x and one y range.
*Why:* consistency across a document is what makes it look finished,
and a two-column paper is the final destination of most figures.

**Colorblind-safe colors, and never color alone.** Okabe-Ito colors,
never red together with green; every category also differs in line
width, marker, or line style. Check: the PNG in grayscale still lets
every line be identified. *Why:* about one reader in twelve is
colorblind, and papers get printed in gray.

**Legends and labels never touch data, enforced in code.** The plot
script samples every drawn line in display coordinates and raises an
error if a legend or text box touches one; it tries other corners and
then grows the headroom by the least amount. *Why:* "looks fine on my
screen" fails at another size; a check in code fails loudly instead.

**Captions are statements.** Say what is drawn, what is omitted, and
how to read the encodings; never a question, never a value the reader
reads off the plot, never a definition already in the text. *Why:*
the caption is read with the figure, out of the flow of the text; it
must be enough to decode the figure, not to repeat the paragraph.

**Define each symbol once, where it is first used; cite sections by
title.** Never name a method by another paper's equation number: give
it a descriptive name, the small equation, and the arXiv ID once.
*Why:* readers jump into the middle; an undefined symbol or "Eq. 10 of
X" sends them away from the note.

## Yijie's own rules (override Vivian's where they differ)

- **No em-dashes**, in notes, captions, READMEs and code comments. Use
  commas, colons or a new sentence. In LaTeX, never write `---` or a
  spaced `--` as punctuation; write numeric ranges as "0.2 to 0.4"
  (or `0.2$-$0.4` only inside a compact table).
- Bias is reported in units of the posterior sigma, against the 0.3
  sigma reference (DES convention) in the 2D (Omega_m, S_8) plane, as
  the figure of bias; the matched-model control is always shown or
  quoted next to a mismatched result.
- Notes, results and code for a project live in that project's
  repository (for the IA paper, `~/cocoa/ia_mismodeling`), never only
  in Downloads or a scratch folder.
- **Check the physical sensibility of every result before it is shown,
  in the same run that produces it.** For each fit or model output:
  compare every fitted parameter with its natural size and allowed
  range, and flag any that sits at a prior bound; check that it does
  not drift with the fit range, the snapshot or the starting point;
  check each model term's sign, its limits and that its scale
  dependence matches the residual it absorbs; look at the model just
  outside the fit range. Report the outcome next to the result, and
  when a check fails, lead with that. *Why:* a good chi2 with an
  unphysical parameter is a failed fit. A halo-model fit was once
  plotted and reported with a central 1-halo amplitude about 100 times
  its physical maximum, because only the goodness of fit had been
  checked; the amplitude was compensating for power missing from
  another term.

## Checklist before calling a note done

- [ ] The question and the answer are the first two sentences of the
      Summary.
- [ ] Every figure folder passes the redraw test, and you looked at
      every PNG (and once in grayscale).
- [ ] No large table; no number typed outside the numbers block.
- [ ] Every fitted parameter and model term passed the physical
      sensibility check (natural size, bounds, stability, limits), and
      the note says so where the result is quoted.
- [ ] Build log: 0 errors, 0 undefined references, 0 "too large".
- [ ] No em-dash, no TODO, no "should/will/needs" in the body.
- [ ] Open issues appendix holds every known limitation, once.
- [ ] The PDF is copied next to the `.tex`, and the repository commit
      includes the `.tex`, the PDF and every `plots/` folder.
