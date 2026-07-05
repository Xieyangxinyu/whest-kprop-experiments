---
name: whest-submission-review
description: "Use when: preparing WHest AIcrowd submission, package, submit, requirements.txt, shipped weights, multi-file estimator folder, reproducibility, .whestignore, local/subprocess parity, or pre-submission checklist review."
argument-hint: "estimator path or submission artifact"
---

# WHest Submission Review

## When to Use

Use this skill before packaging or submitting a WHest estimator, or when investigating a local-vs-submission mismatch.

## Review Workflow

1. Load [submission-checklist.md](./references/submission-checklist.md).
2. Run contract validation and a small fixed-seed local run.
3. Run the same fixed-seed check under the subprocess runner and compare scores.
4. Check budget, time, residual-wall-time, and error fields for every tested MLP.
5. Confirm reproducibility rules: no laptop-only paths, no network calls, no time-based seeds, dependencies declared.
6. Package the correct target: single file for [estimator.py](../../../estimator.py), folder for helper modules, weights, or `requirements.txt`.
7. Inspect the tarball before submit when the packaging surface is non-trivial.

## Must-Catch Submission Mistakes

- Missing dependencies in `requirements.txt` for any non-standard imports.
- Reading files outside `SetupContext.submission_dir` or `SetupContext.scratch_dir`.
- Local-only globals, RNG state, or custom seeds that differ across subprocesses.
- Shipping only `estimator.py` when helper modules or weights are required.
- Over-budget or timed-out MLPs hidden inside a mean score.

## Source Docs

- [docs/how-to/pre-submission-checklist.md](../../../docs/how-to/pre-submission-checklist.md)
- [docs/getting-started/stage-5-package.md](../../../docs/getting-started/stage-5-package.md)
- [docs/how-to/ship-weights.md](../../../docs/how-to/ship-weights.md)
- [docs/how-to/validate-run-package.md](../../../docs/how-to/validate-run-package.md)
