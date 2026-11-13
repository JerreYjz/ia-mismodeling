# Notes: anatomy, numbers, writing

## 1. Anatomy of a note

```
notes/<note>/
  <note>.tex          the source (appendices may be separate .tex files)
  <note>.pdf          the built note, committed next to the source
  plots/
    make_figure_data.py   reads the analysis outputs, writes every data.json
    <name>/               one folder per figure (references/figures.md)
```

The sections, in this order:

1. **Summary.** One to two pages. Paragraphs with bold run-in heads
   (`\paragraph{The question.}`, `\paragraph{The answer.}`, ...). The
   first two sentences are the question and the answer. Numbers appear
   here only through macros.
2. **Fiducial setup** (or "Setup and definitions"). Everything a
   reader needs to reproduce a number: the simulation and its
   conversion, the data vector, the covariance, the masks, what is fit
   and how, the statistic reported (with its equation). Frozen values,
   not pointers to yaml files.
3. **One section per claim of the Summary**, each built around one
   figure: the claim as the first sentence, the figure, how to read it,
   what it means. Short paragraphs.
4. **Recommendation** (only in a decision note): the choice, the
   wording for the paper. Clearly separated so it can be skipped.
5. **Appendices**: method details (derivations a specialist wants),
   numerical checks (what was validated, to what precision), "Where the
   code and results are" (a short table: result, script, output file),
   and **Open issues** (one to three sentences each).

New material goes into an appendix first and reaches the Summary only
when it changes a headline statement or figure.

## 2. The numbers block

At the top of the `.tex`, before `\begin{document}`:

```latex
% =====================================================================
% Numbers
% =====================================================================
% Every number the text quotes is defined here, with the file it comes
% from. Part 1: results (move when the analysis is rerun). Part 2:
% inputs (survey settings, priors), which do not.
\newif\ifnumbersfinal
\numbersfinaltrue
\newcommand{\refresh}[1]{\ifnumbersfinal#1\else\textcolor{red}{#1}\fi}
% Part 1. FoB at K_0 = 0.35, NLA + white nuisance, example1 cov
% (analysis/roman_dchi2/v3/mask_scan_ex1_D1.json, NLAw_A.FoB)
\newcommand{\FoBNLAwhite}{\refresh{0.22}}
```

Rules:
- A macro name says what the number is (`\FoBNLAwhite`, not `\numA`).
- The comment names the file and the key. Someone rerunning the
  analysis must be able to refresh every macro without reading the
  text.
- Set `\numbersfinalfalse` while a rerun is in progress: every
  `\refresh` value prints red, and a red line under the title says so.
- List under the block the **statements** of the text that rest on the
  numbers ("NLA needs the tighter cut in every configuration with the
  white nuisance"), so a refresh also re-checks the sentences, not just
  the digits.

## 3. Writing rules

- **What was done, in the past or present tense.** No TODO, no
  "should", "will", "needs" in the body. Open items go to Open issues.
- **Say each thing once.** A caveat stated in the setup is not
  repeated in every section. Remove repeated caveats first when
  cutting.
- **Define every symbol where it is first used**, once. Write `A_s`,
  `S_8`, `\Omega_m` the same way everywhere; one macro per object
  (`\newcommand{\kuv}{k_{\rm UV}}`).
- **Name methods descriptively**: "the IR/UV scale cut
  (arXiv:2603.13031)", never "Eq. 10 of Saraivanov et al.".
- **Cite sections by title**: "(Section~\ref{sec:white}, The white II
  excess)", never "App. C".
- **Didactic means the reasoning is shown**: when a result is
  surprising, give the two-line derivation that makes it obvious (a
  white spectrum is a constant C_ell, so xi_- = 2C/(pi theta^2) at every
  angle, which is why no k cut removes it). This is what makes a note
  reusable by a student.
- **The history stays out.** The note describes the method as it is;
  an earlier wrong version is mentioned only if a reader could
  otherwise repeat the mistake, and then in one sentence.
- **No em-dashes** (Yijie). Ranges as "0.2 to 0.4".
- Each paragraph opens with its claim; the evidence follows.

## 4. From a work log to a note (the cleanup recipe)

Work logs accumulate: a summary written day by day has every table,
every intermediate convention, and conclusions that later runs
revised. The cleanup is:

1. **Find the latest state of each claim.** Later documents override
   earlier ones (in the IA example: the 10 November strategies note,
   with the production covariance, overrides the 9 November summary,
   which used a Gaussian covariance). Make a list: claim, latest
   number, source file.
2. **Write the two sentences** (question, answer) from that list.
3. **Map each surviving claim to one figure.** Tables of scans become
   line plots against the scanned variable.
4. **Move the rest to appendices or drop it.** Intermediate versions
   (v1 fits, superseded conventions) are dropped; the checks that
   validate the pipeline go to "Numerical checks"; the file inventory
   becomes "Where the code and results are".
5. **Rewrite, do not edit.** Editing a log keeps its order (the order
   things were tried); a note needs the order of the argument.
6. Keep the old PDFs in the repository (`report/`) as the record; the
   note links to them for anything it dropped.

## 5. Length and look

- A summary for collaborators: as few pages as possible without
  looking compact. White space, short sections, figures at a readable
  size (one column, `width=3.5in`, centered).
- `11pt`, `article`, `geometry` margin 1in, `booktabs` for the few
  tables, `caption` with `font=small,labelfont=bf`, `hyperref` with
  `hidelinks`. Floats `[tbp]`, never `[H]` (it leaves half-empty pages).
- A revision cuts the prose by about 20%.
