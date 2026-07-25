---
name: whest-flop-optimization
description: "Use when: optimizing WHest estimator score, FLOP budget, flopscope cost, matmul/einsum cost, residual wall time, mean propagation, covariance propagation, Monte Carlo sampling, algorithm ideas, or adjusted_final_layer_score tradeoffs."
argument-hint: "estimator code, score report, or optimization goal"
---

# WHest FLOP Optimization

## When to Use

Use this skill when improving an estimator that already runs, especially if the user mentions score, speed, budget, flops, profile, matmul, covariance, sampling, or optimization.

## Optimization Workflow

1. Load [flop-optimization-guide.md](./references/flop-optimization-guide.md).
2. Establish the target metric: `adjusted_final_layer_score`.
3. Gather a baseline run with enough fixed seeds or a fixed dataset to compare meaningfully.
4. Inspect `flops_used`, `effective_compute`, `mean_score_multiplier`, `residual_wall_time_s`, raw `final_layer_mse`, and `all_layers_mse`.
5. Profile operation cost before changing algorithms. Contractions (`matmul`, `einsum`, `dot`, relevant `linalg`) share one cost engine and usually dominate, but under the 0.9.x cost model access-tier ops (`take`, `sort`/`argsort`, fancy indexing, 3-arg `where`) and dtype promotion can rival them — read the per-op breakdown and `resolved_dtype`, don't assume.
6. Prefer repo-established flopscope patterns over untracked NumPy/Torch shortcuts.
7. Validate after every meaningful optimization and compare against the prior fixed-seed score.

## flopscope 0.9.x Cost-Model Notes

- Dependency line: flopscope `0.9.x` / whestbench `0.13.x` (see `pyproject.toml`). The billing model is `charged = int(flop_cost × dtype_rate × complex_factor × weight)`, weights `{0, 1, 4, 16}` — full rules in the reference guide; authoritative source is flopscope `docs/reference/cost-model.md`.
- **Every byte written is metered.** The old "data logistics are free" rule is retired: `take`/`take_along_axis`/fancy indexing bill 4/elem, `concatenate`/`stack`/`reshape`/`copy`/`ones` bill 1/elem, 3-arg `where` bills at the gather rate. Only views, basic slicing, and `zeros`/`empty` are free.
- **Billing is dtype-aware.** 64-bit dtypes (float64, int64) bill 2× — mixed `f32 @ f64` promotes and bills wide, and integer/bool reductions accumulate in int64. Keep the whole pipeline float32; cast inputs once.
- Any pre-0.9 measurement of packing, gather, Strassen, or dtype tricks in old logs and memories is suspect — re-measure under the current build before relying on it.

## Default Engineering Bias

- Stay under budget with headroom; over-budget MLPs are zeroed.
- Accuracy matters once compute is at or below the 0.1 multiplier floor.
- Float32 end-to-end; audit `resolved_dtype` on the dominant ops.
- Dense-by-default matmuls; gather-based sparse packing loses at current prices unless row nnz is far below the measured crossover (~75 at width 256).
- ReLU via `fnp.maximum(x, 0.0)`, never 3-arg `where`.
- Use diagonal approximations when full covariance cost does not buy enough raw-MSE improvement.
- Use symmetry-aware `fnp.einsum` for covariance updates instead of chained matmuls.
- Keep random estimators reproducible with `mlp.seed` and `SetupContext.seed`.

## Source Docs

- [docs/how-to/manage-flop-budget.md](../../../docs/how-to/manage-flop-budget.md)
- [docs/how-to/performance-tips.md](../../../docs/how-to/performance-tips.md)
- [docs/reference/code-patterns.md](../../../docs/reference/code-patterns.md)
- [docs/concepts/scoring-model.md](../../../docs/concepts/scoring-model.md)
