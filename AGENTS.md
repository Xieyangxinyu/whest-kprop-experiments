# AGENTS.md

## Commands

```bash
# Install / sync
uv sync --group dev

# Lint
uv run ruff check .

# Test
uv run pytest tests/ -v

# Stage 1 — local estimator run (prints MC convergence table)
uv run python estimator.py

# Stage 2 — contract validation (shapes, types, finite values)
uv run whest validate --estimator estimator.py

# Stage 3 — score against public MLPs (in-process)
uv run whest run --estimator estimator.py --n-mlps 2          # quick smoke
uv run whest run --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
    --split mini --runner local                                # full mini split

# Stage 4 — subprocess runner (catches state-bleed, RNG reuse)
uv run whest run --estimator estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
    --split mini --runner subprocess

# Diagnostics
uv run whest doctor
```

CI order: `ruff check` → `python estimator.py` → examples smoke → `whest validate` → `whest run --n-mlps 2` → `pytest`.

## Estimator contract

- `estimator.py` is the submission surface. `Estimator` extends `BaseEstimator`.
- `predict(self, mlp, budget)` must return a **finite** `flopscope.numpy` ndarray of shape `(mlp.depth, mlp.width)` — post-ReLU mean activations per layer.
- Use `flopscope.numpy as fnp` for all FLOP-tracked numerical work. Plain numpy is not tracked.
- `setup(self, ctx)` receives `SetupContext` with `ctx.seed`, `ctx.submission_dir`. Load shipped artifacts (e.g. `sobol_points.npz`) from `ctx.submission_dir`.
- Optimize for `adjusted_final_layer_score`; raw MSE and FLOPs are supporting diagnostics.

## Key files

| Path | Role |
|---|---|
| `estimator.py` | Main estimator — the only file shipped by default |
| `local_engine.py` | Stage 1 pedagogical harness; **excluded** from submissions via `.whestignore` |
| `sobol_points.npz` | Shipped Sobol QMC points; loaded in `setup()` or `_load_sobol_points()` |
| `examples/` | Numbered curriculum (01–08, 34, 40); run with `uv run python examples/NN_name.py` |
| `tests/` | README drift gate (`bash-test` blocks), `local_engine` unit + parity tests |
| `.whestignore` | Controls submission tarball contents — 50-file cap |
| `.claude/skills/` | 8 domain skills (competition context, debugging, FLOP optimization, etc.) |

## Gotchas

- `local_engine.py` must NOT import from `whestbench` — drift detection is intentional.
- README ````bash-test` fenced blocks are executed by `test_readme_commands.py` — keep them passing.
- `setuptools` `py-modules` is pinned to `["estimator", "local_engine"]` in `pyproject.toml`; adding new top-level modules requires updating it.
- `sobol_points.npz` must be next to `estimator.py` for local runs; for submissions it's loaded from `ctx.submission_dir`.
- To ship extra files (weights, modules), point `--estimator` at a folder instead of a file.
- Python 3.10 required (`.python-version`).
