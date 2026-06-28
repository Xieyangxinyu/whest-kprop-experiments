# Submission Checklist

## Correctness Gates

```bash
uv run whest validate --estimator estimator.py
uv run whest run --estimator estimator.py --runner local --seed 42 --n-mlps 3
uv run whest run --estimator estimator.py --runner subprocess --seed 42 --n-mlps 3
```

Local and subprocess scores should be close for the same seed and MLP count. Large differences usually mean hidden state, RNG, imports, filesystem assumptions, or process-isolation bugs.

## Budget Gates

For every tested MLP, confirm:

- `budget_exhausted` is false.
- `combined_budget_exhausted` is false.
- `time_exhausted` is false.
- `residual_wall_time_exhausted` is false when relevant.
- `flops_used` and `effective_compute` have comfortable headroom under `flop_budget`.

## Reproducibility Gates

- Declare every non-standard dependency in `requirements.txt`.
- Do not read from arbitrary local paths. Shipped files must be loaded from `SetupContext.submission_dir`; scratch/cache files from `SetupContext.scratch_dir`.
- No network calls in `setup()` or `predict()`.
- No time-based seeds or participant-chosen seeds. Use `mlp.seed` in `predict()` and `ctx.seed` in `setup()`.
- Avoid global mutable state that changes across MLPs unless it is deliberate, deterministic, and subprocess-safe.

## Packaging

Single-file common case:

```bash
uv run whest package --estimator estimator.py --output submission.tar.gz
```

Folder package for helper modules, weights, or `requirements.txt`:

```bash
uv run whest package --estimator . --output submission.tar.gz
```

Inspect when non-trivial:

```bash
tar tf submission.tar.gz
```

The artifact should contain `estimator.py`, `manifest.json`, and any intended helper files. Credential files are excluded, and `.whestignore` can exclude scratch/large artifacts.

## Submit

```bash
uv run whest login
uv run whest submit --estimator estimator.py --watch
```

For a prebuilt artifact:

```bash
uv run whest submit submission.tar.gz --watch
```
