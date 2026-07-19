# WHest Technical Write-Up

This directory contains an arXiv-style LaTeX draft for the WHest technical write-up.

The template is a cleaned, local version of the article/preprint pattern used in your ML-theory arXiv sources, especially the `Foundations of Top-k Decoding For Language Models` source: single-column `article`, `natbib`, theorem environments, compact math macros, `booktabs`, and a contribution-roadmap structure. I did not copy paper text; the draft uses the template and style cues only.

## Current Target

- Phase: Phase 1
- Graded submission: `316017`
- Artifact described by the logs: `submission-algo26-fire05-2026-07-12.tar.gz`
- Public result: adjusted final-layer score `9.121856491923e-8`, raw final-layer MSE `3.724918391868e-7`, mean multiplier `0.2416126721`, mean effective compute `65.719G`, `0/50` public failures.

The PDF should correspond to exactly one successfully graded submission. Keep the visible submission ID centralized in `main.tex` via `\submissionid`; if a later graded non-failed submission becomes the target, update that macro and the results table together.

Note that the current root `estimator.py` may have moved on to a comparison baseline. This draft is written from the bench logs for submission `316017`, not from whatever estimator happens to be checked in later.

## Build

```bash
cd writeup
make pdf
```

Figures used in the PDF are reproducible from `figures.ipynb`. Run the notebook
from this directory to regenerate all files referenced as `figures/*.png` before
building the PDF.

If `latexmk` is unavailable:

```bash
cd writeup
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Build outputs are ignored by `writeup/.gitignore`. The root `.whestignore` also excludes `writeup/` so folder-mode WHest packaging does not upload the paper sources.