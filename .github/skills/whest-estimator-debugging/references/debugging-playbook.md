# Debugging Playbook

## First Commands

Use `uv run` from the repo root unless the user is already inside the managed environment.

```bash
uv run whest validate --estimator estimator.py
uv run whest run --estimator estimator.py --runner local --debug --n-mlps 3
uv run whest run --estimator estimator.py --runner local --debug --fail-fast --n-mlps 3
```

For quick pure-Python iteration:

```bash
uv run python estimator.py
uv run python estimator.py --baseline mean_propagation
```

## Symptom to Fix Map

- Wrong shape: return `fnp.ndarray` shape `(mlp.depth, mlp.width)`.
- Non-finite values: add numeric guards/clipping; validate finiteness.
- Class not found: class must be named `Estimator` unless `--class` is passed.
- Signature mismatch: use `predict(self, mlp, budget)`.
- Import error: add dependencies to `requirements.txt`; prefer `flopscope.numpy as fnp`.
- `PREDICT_ERROR`: rerun with `--debug --fail-fast` and fix the traceback line.
- `SETUP_TIMEOUT`: move heavy setup to shipped artifacts, scratch cache, or predict-time logic.
- `budget_exhausted`: analytical flopscope budget tripped before an operation ran.
- `combined_budget_exhausted`: `effective_compute = flops_used + lambda * residual_wall_time_s` exceeded budget after Python overhead was counted.
- `time_exhausted` or residual wall-time exhaustion: check Python loops, external libraries, print spam, and unbounded operations.

## Report Fields to Inspect

Use JSON/plain output when needed:

```bash
uv run whest run --estimator estimator.py --runner local --format json --n-mlps 3
```

Inspect:

- `adjusted_final_layer_score`
- `final_layer_mse`
- `all_layers_mse`
- `mean_score_multiplier`
- per-MLP `flops_used`, `effective_compute`, `residual_wall_time_s`
- per-MLP `budget_exhausted`, `combined_budget_exhausted`, `time_exhausted`
- per-MLP `error`, `error_code`, `traceback`

## Local vs Subprocess Mismatch

Run the same fixed-seed checks:

```bash
uv run whest run --estimator estimator.py --runner local --seed 42 --n-mlps 3
uv run whest run --estimator estimator.py --runner subprocess --seed 42 --n-mlps 3
```

If they differ, suspect global mutable state, RNG seeding, hidden filesystem assumptions, imports that only work in the current process, or setup/predict side effects.

## Debugger Tips

Use `breakpoint()` inside `predict()` with local runner. If using `pdb.set_trace()`, add `--format plain` to avoid live-display interference.
