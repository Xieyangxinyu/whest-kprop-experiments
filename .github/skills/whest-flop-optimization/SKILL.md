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
5. Profile operation cost before changing algorithms; in flopscope `0.8.x`, contractions (`matmul`, `dot`, `tensordot`, `einsum`, relevant `linalg`) share one cost engine and usually dominate analytical FLOPs. Python loops and non-flopscope work can dominate residual wall time.
6. Prefer repo-established flopscope patterns over untracked NumPy/Torch shortcuts.
7. Validate after every meaningful optimization and compare against the prior fixed-seed score.

## Phase 1 flopscope / whestbench Notes

- Current Phase 1 dependency line is flopscope `0.8.x` / whestbench `0.12.x` (this repo pins release candidates or newer in `pyproject.toml`). Re-check assumptions after dependency bumps.
- Cost principle: computation on values costs FLOPs; data logistics such as slicing, reshaping, stacking, concatenation, and gathering are intended to be 0-FLOP data movement.
- Residual wall time is meant to cover participant Python and untracked work, not framework/client-server plumbing. Still inspect `residual_wall_time_s` because local and grader timing can differ.
- Dtype/packing exploits are unstable policy ground: organizers have acknowledged complex dtype undercounting and said affected submissions may be re-evaluated after cost fixes. Treat complex/float64/bitpacking wins as provisional and record them separately from algorithmic accuracy wins.

## Default Engineering Bias

- Stay under budget with headroom; over-budget MLPs are zeroed.
- Accuracy matters once compute is at or below the 0.1 multiplier floor.
- Use diagonal approximations when full covariance cost does not buy enough raw-MSE improvement.
- Use symmetry-aware `fnp.einsum` for covariance updates instead of chained matmuls.
- Keep random estimators reproducible with `mlp.seed` and `SetupContext.seed`.

## Source Docs

- [docs/how-to/manage-flop-budget.md](../../../docs/how-to/manage-flop-budget.md)
- [docs/how-to/performance-tips.md](../../../docs/how-to/performance-tips.md)
- [docs/reference/code-patterns.md](../../../docs/reference/code-patterns.md)
- [docs/concepts/scoring-model.md](../../../docs/concepts/scoring-model.md)
