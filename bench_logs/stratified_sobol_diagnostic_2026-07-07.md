# Stratified-Sobol Diagnostic - 2026-07-07

## Idea (Idea 2)

Augment the antithetic-Sobol block by Latin-hypercube **stratifying** (not
rotating) the input Gaussian along the top-K input-side singular directions of
the expected-gate map `E = W0 diag(p0) ... W_{L-1} diag(p_{L-1})`,
`p_l = Phi(alpha_l)`. Distinct from the previously-rejected `sens30/sens31/w0`
**rotation** family: stratification is an unbiased variance reducer (projections
onto orthonormal dirs of N(0,I) are iid N(0,1); orthogonal complement kept from
the baseline Sobol cloud, so the A/B isolates the stratification effect).

## Command

`uv run python scripts/stratified_sobol_ab_test.py` — pure-NumPy offline,
width 256 depth 32 He-init, N=40960 antithetic (submission scale), seeds 0-9,
final-layer MSE vs ~1.05M-sample MC reference. K in {2, 8, 16}.

## Result (geomean stratified/baseline MSE ratio, <1 = helps)

| K  | seeds 0-4 | seeds 5-9 |
|----|-----------|-----------|
| 2  | 0.742     | 1.234     |
| 8  | 0.752     | 1.482     |
| 16 | 0.849     | 1.911     |

Per-seed extremely noisy: seed 3 K=16 = 4.88x worse, seed 6 K=16 = 4.65x worse,
seed 9 K=2 = 2.50x worse. Expected-gate map is very low rank (top-16 = 99.9%
of E's spectral energy) yet stratifying it does not help.

## Interpretation

Same seed-group instability the logs already recorded for `sens30/sens31`
rotation (0-4 improve, 5-9 regress). The expected-gate map's low-rank energy is
in the MEAN path, not the final-layer VARIANCE; after 31 ReLU folds the scored
integrand is high-dimensional/rough (see qmc_tail_diagnostic), so the
expected-gate linear subspace is the wrong basis for variance reduction. K/gate
tuning cannot fix a miscalibrated subspace.

## Decision

REJECT before any `estimator.py` change or `whest run`. Fails the
pre-registered "improve BOTH seed groups beyond noise" criterion. Do not retry
expected-gate-aligned sampling geometry (rotation OR stratification) at the
40960 envelope without a subspace calibrated to final-layer VARIANCE, not the
expected-gate mean path. Kept `scripts/stratified_sobol_ab_test.py` for reuse.
