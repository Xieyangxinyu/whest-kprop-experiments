# Algorithm 30 floor-mode: build, gate, package - 2026-07-18

Surface: submissions/algo30-floormode/ (copy of graded 316405 bytes with
the sample rule replaced); artifact = shipped sobol_points.npz (re-bake
withdrawn 07-17); pilot fractions stock (retune killed 07-18). Package:
submission-algo30-floormode.tar.gz (estimator.py, sobol_points.npz,
requirements.txt, LICENSE, manifest.json).

## Design

Replaces `_choose_samples` (variance rule, N in [30720, 61440]) with a
deterministic closed-loop FLOP-metered fill toward the 0.1 clamp:

1. Base block N=10240 (Sobol prefix, refine=True), meter the flopscope
   counter delta (delta-based -> robust to reset semantics).
2. Fill blocks (refine=False) sized from a slope model until metered
   flops reach `_FLOOR_TARGET_FRACTION = 0.085 * budget`; first fill at
   0.97 of headroom, later fills sized by the MEASURED fill slope
   (self-correcting); `_MIN_FILL_SAMPLES = 128`, max 4 blocks.
3. Rows merged sample-count-weighted across blocks.

Because the meter reads the counter the grader prices with, N
self-adjusts to grader-side pricing (the divergence absorbs into N, not
the multiplier) — this is the pricing-immunity mechanism, plus the 0.1
clamp itself.

### Why 0.085, not 0.096 (effective-compute correction, load-bearing)

Local `whest run` exposed that effective compute adds residual wall at
the local lambda 1e11 eq/s (local multiplier ~0.5 - meaningless). The
REMOTE calibration (07-14 learnings): lambda ~5.8e9 eq/s, residual =
concat class ~2.2 FLOP-eq/B ~= 1.2e10 eq at fleet N~53k, i.e. ~14.2% on
top of metered flops, ~proportional to samples. Flops target 0.085B ->
effective ~0.0971B, ~3% margin under the clamp. Sitting at the clamp
absorbs the residual ratio into the flat product, so the margin (not
the residual) is the only score cost (~2%).

### Fill-slope calibration (new pricing fact)

Fill blocks price ~10% HIGHER per sample than the base block: their
halves (~1-3k rows) sit below `_DENSE_STRASSEN_MIN_ROWS = 4096`, losing
the Strassen discount (measured ratio 1.105/1.111 on nets 1/5).
`_FILL_SLOPE_RATIO = 1.10` corrects the base-derived slope. Also true
out-of-block overhead (_initial_structure) is ~1.5e7 flops, NOT the
fleet-fitted F0~1e9 (that fit bundled per-block fixed costs);
`_F0_EST = 2e7`. Naive slope (F0=1e9, no ratio) overshot the target
+2%, landing effective at 0.0992 - the two constants fix landing.

## 32-net gate (final, .tmp/router_opt/floormode_gate.{py,pkl})

- Flops: ALL 32 nets C = 0.0850B (4-digit exact); effective implied
  0.0971B; 3 blocks (base + fill + top-up ~130-270 samples).
- Determinism: repeat on net 1 bit-identical (predictions + flop count).
- N: mean 12872 (11968-14372). Wall: ~4.5 s/net local (vs ~18-20 s
  316405-class) -> grader jobs ~3-4 min, outside the 15-min reaper and
  60s-cap regimes.
- Subprocess parity: bit-identical mse+flops vs local runner (seed 42);
  validate passed; no budget/time exhaustion anywhere.
- Raw fleet (nets 0-31): 2.2385e-6 -> naive projection 1.897e-7 (+43%
  vs 316405). This number is REALIZATION-LUCK dominated: nets 14/19/26
  carry 41% of fleet mean; scrambled-artifact replicates on nets 14/19
  show the shipped prefix draws 2-15x worse than the realization mean
  on this net set at floor N (net 14: 9.7e-6 shipped vs 1.8-5.3e-6
  scrambled; net 19: 11.3e-6 vs 0.75-6.8e-6). Variance-model
  expected-realization raw ~1.53e-6 -> projected ~1.30e-7 (-2% vs
  316405) - consistent with the flat-product theory.

## Honest score expectation

Expected-realization: ~FLAT vs 316405 (1.30-1.35e-7 class). Realized on
the 50-net private set: wide band (+/-20-30%) because fleet raw at
N~13k is a single shared QMC-prefix realization with heavy per-net
tails (df~1; same as the re-bake screen/holdout inversion). The
submission's case is NOT score: it is (a) pricing-divergence immunity
(clamp + self-metering), (b) ~4x shorter jobs outside both death
regimes. Its risk is a bad-luck draw grading +20-30%.

## Status

Packaged and validated; NOT submitted (user gate). Constants of record:
base 10240, target 0.085, safety 0.97, slope ratio 1.10, F0 2e7,
min fill 128, max blocks 4.
