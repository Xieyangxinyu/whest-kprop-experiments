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

## Rotation A/B Test (same day, follow-up)

`uv run python scripts/rotation_ab_test.py` — full 32-layer forward (no
active-set machinery), 40,960 antithetic Sobol samples, final-layer MSE vs
1M-sample MC reference, seeds 0-4.

| seed | baseline | W0-SVD rotated | ratio | random-rotation ratios |
|------|----------|----------------|-------|------------------------|
| 0 | 9.89e-7 | 5.26e-7 | 0.53 | 1.24, 0.68, 2.13 |
| 1 | 1.03e-6 | 2.02e-7 | 0.20 | 0.99, 0.90, 0.60 |
| 2 | 1.16e-6 | 6.77e-7 | 0.59 | 0.38, 0.67, 0.52 |
| 3 | 3.48e-7 | 8.21e-7 | 2.36 | 5.66, 3.23, 3.77 |
| 4 | 2.46e-6 | 8.52e-7 | 0.35 | 0.27, 0.54, 0.12 |

Geomean ratio: W0-SVD 0.55, random rotations 0.88.

Reading: the W0-SVD rotation looks better than baseline AND better than
random rotations, but the control exposes heavy seed-level noise — random
rotations also "help" on most seeds, and seed 3 punishes every rotation
(its baseline draw is likely just lucky). 4-of-5 seeds winning with a
0.55 geomean is promising but below the bar given (a) the noise floor
shown by the control and (b) history: Sobol seed selection and
sphere/radial tricks both looked good locally and failed on leaderboard.

Verdict: inconclusive-promising. Before any submission: (1) rerun on 10+
seeds with per-seed paired ratios, (2) test inside the real active-set
estimator via eval_variants.py (the fold machinery may interact), (3)
account for the added cost (~5.4 GFLOP rotation matmul + SVD, ~2% of
budget). Treat as sample-geometry tuning — the exact category that has
overfit locally before.

## Rotation Inside eval_variants.py (same day, final)

Added `rotate_w0` variants to `eval_variants.py` (W0-SVD rotation of the
Sobol half-points, rotation matmul flopscope-tracked; the SVD itself ran
untracked, so real cost is slightly higher than shown). Champion config
`l29+30 borderline 4/-4` with and without rotation, both seed groups.

At N_SAMPLES=16384 (diagnostic scale): rotation wins clearly.
- seeds 0-4: score 2.86e-7 vs 3.30e-7 (rot better, -13%)
- seeds 5-9: score 2.95e-7 vs 3.22e-7 (rot better, -8%)

At N_SAMPLES=40960 (submission scale): rotation LOSES.
- seeds 0-4: score 5.15e-7 vs 5.10e-7 (rot worse; raw final MSE slightly
  better, 1.152e-6 vs 1.166e-6, but +1% util eats it)
- seeds 5-9: score 3.13e-7 vs 2.66e-7 (rot clearly worse, raw final MSE
  worse too: 6.97e-7 vs 6.04e-7)

This is the same signature as sphere/radial sampling: helps at 16384,
loses at 40960. The likely mechanism: the shipped Sobol net's later
coordinates are good enough at 40960 points that rotation's realignment
no longer buys variance, while on seeds 5-9 it actively hurts (rotation
changes which 256 of the point set's coordinates are used, effectively a
different sample draw — seed-level luck, not structure). All-layers MSE
improves with rotation at both scales, but the leaderboard scores only
the final layer.

DECISION: reject W0-SVD rotation for the 40960-sample submission config.
Do not retry without evidence at submission scale. This closes out the
last surviving idea from the external QMC review — all of its proposals
are now measured rejects for this regime.
- Worth a cheap test: RQMC-style scramble replicates only as an error-bar
  diagnostic offline, not in the submission (samples are budget-capped).
- Reinforces: keep investing in borderline dead/on refinement — that is
  where single-sample-dominated neurons live.
