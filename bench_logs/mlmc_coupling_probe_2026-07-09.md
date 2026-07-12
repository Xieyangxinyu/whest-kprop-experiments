# MLMC Level-Coupling Feasibility Probe - 2026-07-09

## Question

Memory (variance-reduction-tapped) flagged MLMC as the one variance lever not yet
disproven: "only nonlinear-coupled (MLMC) could plausibly work." Does a cheaper
coarse level exist that stays coupled to the full network THROUGH the ReLU gates,
so the increment variance is small enough for a net MLMC win?

## Design tested

2-level, width-truncation coarse: coarse = full network with the lowest-variance
`(1-keep_frac)` neurons/layer zeroed after ReLU, sharing the fine samples/gates.
Kill-gate metric: `R = Var[fine - coarse]/Var[fine]` on final-layer per-sample
activations. 2-level MLMC cost vs plain MC scales `1/(sqrt(rho_c)+sqrt(R))^2`
with `rho_c ~ keep_frac^2` (matmul ~ width^2). Need R << 0.1 for any win because
the increment still requires a full fine pass.

## Results (build_mlp seeds 0-4, N=12000)

| keep_frac | rho_c | R_agg range | implied speedup |
|---|---|---|---|
| 0.70 | 0.49 | 0.70 - 0.82 | 0.39 - 0.42x |
| 0.50 | 0.25 | 0.80 - 0.91 | 0.47 - 0.51x |

R is 0.7-0.9, not << 0.1. Dropping even the lowest-variance 30% of neurons flips
enough downstream gates that the correction is ~80% as variable as the full
signal. Implied speedup < 1 (MLMC slower than plain MC).

## Conclusion / Decision

MLMC is DEAD for the same reason as every other variance/matmul lever: the
final-layer per-sample fluctuation is a rough nonlinear function dominated by gate
combinatorics, and no cheap coarse model tracks it (linear CV gave R~0.94; this
width-truncation gives R~0.8). Even a hypothetical free coarse (rho_c->0) caps at
~1.26x speedup at R=0.8, which MLMC orchestration wall-time would erase.

Do NOT implement MLMC. This closes the last open lever. 315416 stands as the
robust optimum. Do not retry MLMC without a fundamentally different, gate-preserving
coarse level (none identified).
