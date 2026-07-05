# WHest Repo Instructions

Use the repo skills under `.github/skills/` whenever the task touches WHest, estimator changes, score reports, FLOP budgets, submissions, experiment design, handoffs, or docs hygiene.

## Working Defaults

- Prefer `uv run ...` for Python commands when the local environment supports it.
- Treat `estimator.py` as the primary submission surface. Keep changes minimal, portable, and compatible with subprocess/package validation.
- For estimator work, confirm the contract before editing: `predict(self, mlp, budget)` returns a finite flopscope array shaped `(mlp.depth, mlp.width)` containing post-ReLU mean activations for every layer.
- Use `flopscope.numpy as fnp` for FLOP-tracked numerical work inside estimators unless a repo example shows a safe alternative.
- Optimize for `adjusted_final_layer_score`; raw final-layer MSE, all-layer MSE, FLOPs used, residual wall time, and budget exhaustion are supporting diagnostics.
- Before comparing variants, establish a fixed-seed or fixed-dataset baseline and record enough command/output context to reproduce the comparison.
- Keep benchmark notes and submission learnings in `bench_logs/`; avoid scattering one-off conclusions in unrelated files.
- Do not commit secrets, local tokens, package artifacts, or machine-specific paths.

## Useful Skills

- `whest-competition-context`: baseline contest contract and scoring mental model.
- `whest-estimator-debugging`: validation failures, wrong shapes, NaN/Inf, budget errors, subprocess mismatches.
- `whest-flop-optimization`: score and FLOP tradeoffs, profiling, estimator optimization.
- `whest-experiment-design`: structure a scoring experiment before editing or running sweeps.
- `whest-submission-review`: pre-submission packaging and reproducibility checks.
- `whest-submission-learning-loop`: capture public leaderboard results and avoid repeating failed ideas.
- `whest-handoff`: preserve long-running optimization context across sessions.
- `repo-doc-hygiene`: find orphaned or stale documentation and skill references.