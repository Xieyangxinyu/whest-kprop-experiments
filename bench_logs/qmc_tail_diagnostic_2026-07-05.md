# QMC Tail Diagnostic - 2026-07-05

External review claimed plain Sobol is "quietly defeated" by a lognormal
(NLN) tail in the final-layer integrand and recommended: Owen-scrambled
replicates, importance-sampling mean-shift before the Sobol map, coordinate
rotation by Jacobian importance, and a doubly-robust NLN plug-in + residual
split. Before implementing any of that, we ran the review's own gating
diagnostic.

## Command

`uv run python scripts/qmc_tail_diagnostic.py` (pure-NumPy offline diagnostic,
width=256 depth=32 He-init MLPs, seeds 0 and 1; shipped
`sobol_points.npz`, antithetic pairing exactly like the submission;
reference = 1,048,576-sample MC).

## Results

Final-layer MSE vs reference, effective n = 1,024 -> 40,960:

- Seed 0: Sobol 2.50e-5 -> 9.88e-7 (slope -0.47); plain MC 3.01e-5 -> 1.45e-6 (slope -0.43).
- Seed 1: Sobol 8.09e-5 -> 1.03e-6 (slope -0.58); plain MC 3.80e-5 -> 1.20e-6 (slope -0.49).
- sigma_h = std of 2*log||a^(31)|| ~= 0.37 on both seeds; lognormal ESS
  penalty exp(sigma_h^2) ~= 1.14.
- Max-single-sample share of a neuron's final-layer mean: median 0.006%,
  p95 7-16%, max 100% (a handful of nearly-dead neurons are carried by
  one sample).

## Interpretation

1. The heavy-tail premise is mostly false at this width/depth. sigma_h ~=
   0.37 is a mild lognormal; the tail costs ~14% effective samples, not
   orders of magnitude. Importance-sampling tilts (radial or directional)
   and the doubly-robust NLN plug-in target a problem we largely do not
   have; the IS weights would add variance to the well-behaved bulk.
2. The rate-collapse observation is real but has a different cause. Sobol's
   error slope at layer 32 is ~-0.5, barely better than MC — after 31 ReLU
   folds the integrand is effectively high-dimensional and rough, so QMC
   equidistribution buys little at the final layer. Most of our Sobol gain
   is at early layers (which do not score).
3. The actionable tail is the classification tail, not the lognormal tail:
   the neurons whose mean is dominated by one sample are exactly the
   borderline dead/on neurons the leaderboard-confirmed pilot refinements
   (314954, 314957) already target.

## Decision

- Reject: IS mean-shift tilt, radial tilt, doubly-robust NLN plug-in for
  the current 256x32 regime. Do not retry without evidence that challenge
  MLPs have sigma_h >> 0.4 (worth one measurement on public MLPs).
- Worth a cheap test: rotate input coordinates per-MLP so the first Sobol
  coordinates align with the top singular directions of W0 (rotation
  invariance makes this exact; cost ~one extra 256x256 matmul per sample
  batch, ~4% budget). Could recover some QMC rate.
- Worth a cheap test: RQMC-style scramble replicates only as an error-bar
  diagnostic offline, not in the submission (samples are budget-capped).
- Reinforces: keep investing in borderline dead/on refinement — that is
  where single-sample-dominated neurons live.
