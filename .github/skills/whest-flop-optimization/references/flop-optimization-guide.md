# FLOP Optimization Guide

## Goal

Optimize `adjusted_final_layer_score`, not only raw MSE. Once compute is at or below the 0.1 multiplier floor, further FLOP savings do not improve score unless accuracy is preserved or improved.

## Baseline Commands

```bash
uv run whest run --estimator estimator.py --runner local --seed 42 --n-mlps 3 --profile
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner local
```

Use fixed seeds or fixed datasets for comparisons.

## FLOP Cost Rules

Free in flopscope:

- `fnp.zeros`, `fnp.ones`, `fnp.eye`, `fnp.array`
- `fnp.reshape`, `fnp.transpose`
- indexing/slicing
- `fnp.concatenate`, `fnp.stack`

Costs FLOPs:

- pointwise math: roughly output element count
- reductions: input size
- random samplers: calibrated per sample
- matmul/einsum: shape-dependent, usually dominant

For width 256, matrix-matrix operations are much more expensive than matrix-vector operations. Diagonal propagation is `O(width^2)` per layer; full covariance is `O(width^3)` per layer.

## flopscope Patterns

Use:

```python
import flopscope as flops
import flopscope.numpy as fnp
```

Operators on `fnp.ndarray` are tracked, so `+`, `*`, `/`, and `@` are fine.

For covariance updates, prefer symmetry-aware einsum:

```python
cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
```

Avoid chained matmuls like `w.T @ cov @ w`; they can lose symmetry information and inflate downstream cost/warnings.

## ReLU Gaussian Moment Pattern

For `z ~ N(mu, sigma^2)`:

```python
alpha = mu_pre / sigma_pre
mean = mu_pre * flops.stats.norm.cdf(alpha) + sigma_pre * flops.stats.norm.pdf(alpha)
```

This approximation is exact at the first layer and approximate later. Errors tend to compound with depth; compare `all_layers_mse` and final-layer MSE to diagnose where propagation drifts.

## Residual Wall Time

Python control flow and untracked external calls can raise `residual_wall_time_s`, which feeds `effective_compute`. Avoid heavy Python loops, print spam inside `predict()`, and expensive non-flopscope libraries unless they are truly worth the residual penalty and packaging surface.
