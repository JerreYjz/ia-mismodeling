#!/usr/bin/env bash
# Build a note in a scratch folder and check it.
#
#   build_note.sh <note folder>        e.g. build_note.sh notes/scale_cuts_summary
#
# Copies <note>.tex (plus any other .tex in the folder) and plots/ to a
# temporary folder, runs pdflatex three times, counts errors, undefined
# references and floats that are too large (all must be 0), reports
# overfull boxes, checks the source for em-dashes and TODO, and copies the
# PDF back next to the source. Exit status 1 when a check fails.
set -u
DIR=$(cd "$1" && pwd)
NOTE=$(basename "$DIR")
TEX="$DIR/$NOTE.tex"
[ -f "$TEX" ] || { echo "no $TEX"; exit 1; }
SCR=$(mktemp -d)
cp "$DIR"/*.tex "$SCR"/
[ -d "$DIR/plots" ] && cp -R "$DIR/plots" "$SCR"/
cd "$SCR"
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error "$NOTE.tex" > /dev/null 2>&1
done
LOG="$NOTE.log"
errors=$(grep -c '^!' "$LOG")
undefined=$(grep -ci 'undefined' "$LOG")
toolarge=$(grep -ci 'too large' "$LOG")
overfull=$(grep -c '^Overfull' "$LOG")
dashes=$(grep -nHE -- '---| -- ' "$DIR"/*.tex | grep -vE ':[0-9]+:[[:space:]]*%' | wc -l | tr -d ' ')
todo=$(grep -nH 'TODO' "$DIR"/*.tex | wc -l | tr -d ' ')
pages=$(grep -o 'Output written on .*' "$LOG")
echo "$pages"
echo "errors=$errors undefined=$undefined too_large=$toolarge overfull=$overfull em_dashes=$dashes todo=$todo"
grep '^Overfull' "$LOG" | sed 's/^/  /' | head -20
status=0
[ "$errors" = 0 ] && [ "$undefined" = 0 ] && [ "$toolarge" = 0 ] && [ "$dashes" = 0 ] && [ "$todo" = 0 ] || status=1
if [ -f "$NOTE.pdf" ] && [ "$errors" = 0 ]; then
  cp "$NOTE.pdf" "$DIR/$NOTE.pdf"
  echo "copied $NOTE.pdf to $DIR"
fi
[ $status = 0 ] || { echo "CHECK FAILED (log: $SCR/$LOG)"; grep -A3 '^!' "$LOG" | head -20; }
exit $status
