# Algorithm 19: sample-scaling investigation - 2026-07-11

Evidence: `algorithm19_sample_scaling.ipynb` (executed; 8 public-mini MLPs x
6 sample counts through the 315824 surface, exact GT, forced-N harness with
Gaussian-antithetic extension beyond the 61440 artifact cap).

## Question

Finer row buckets cut cost-per-sample ~25%; multiplier sits at 0.36 vs the
0.10 floor. Does the optimal sample count move up, and does a closed-form
per-MLP `N* = sqrt(v*F0'/(b*c))` beat the shipped
`clip(49152*sqrt(V/V_ref), 30720, 61440)` heuristic?

## Result: NO on both counts (allocation axis near-optimal)

Pooled suite fit (per-MLP fits were single-realization noise-limited; two of
eight fit negative bias floors): `b=1.63e-7, v=1.51e-2, F0=5.1e8, c=1.79e6`
FLOPs/sample.

- Kill criterion not hit: variance is still ~60% of MSE at N=61440.
- BUT with the grader-corrected fixed cost (charged residual
  `lambda*R ~= 1.9e10` FLOP-equivalents, inferred from 315824's mult 0.3621 vs
  ~8e10 analytic FLOPs), suite `N* ~= 31,700` - at the BOTTOM of the current
  clamp range. Predicted gain from re-allocating: ~2%, inside measurement
  noise. N=122,880 predicted to LOSE ~35% adjusted despite better raw MSE.
- FLOPs-only analysis is misleading here: analytic F0 is ~1% of the bill, so
  the naive closed form says N*~5k; the grader residual (40x analytic F0) is
  what holds the optimum up. Same machine-transfer trap as the 315844
  postmortem, applied in the opposite direction.

## Decisions

- Do NOT pursue sample-count increases or per-MLP N* allocation; ceiling ~2%.
- The fit exposes the two real levers:
  1. Bias floor `b=1.63e-7` - ~40%+ of grader raw MSE (3.72e-7); no N buys it.
     Sources: analytic filler rows are exact-zero contributors (final-layer
     scored only), so this is classification/fold/dead-correction bias in the
     final row.
  2. Charged residual `lambda*R ~= 1.9e10` (~19% of effective compute) -
     grader-side Python overhead is now material; residual trimming is worth
     ~1.9 points of multiplier at zero accuracy cost IF trimmable.
- Policy B (empirical variance from base block) inconclusive: v_emp/v_fit
  scattered 0.34-2.66, confounded by antithetic pair-splitting in contiguous
  sub-block slices. Not worth fixing given the 2% ceiling.

## Do not retry without new evidence

- "More samples because multiplier headroom" - falsified by the bias floor +
  residual-corrected optimum.
- Max-squared hardness allocation (rejected 2026-07-10) stays rejected; this
  analysis explains why: it over-spends exactly where b dominates.

## UPDATE (later 2026-07-11): hard-tail refinement tested and FALSIFIED as calibrated

Evidence: `hard_mlp_sample_frontier.ipynb` (executed, exact-run confirmations).

- The hard TAIL is real and variance-limited: all 6 hardest of 32 mini nets
  improve raw MSE 2.3-9.8x from 30.7k to 122.9k samples. mlp 14 exact-run
  confirmed adjusted 5.39e-7 -> 2.38e-7 (-56%) at N=89,926 under grader
  pricing. Budget exhaustion ceiling ~133k samples; CAP=92,160 safe (<=71%),
  122,880 = 85-93% (needs grader residual confirmation first).
- BUT the refined rule N = clip(s*sqrt(V_emp), 30720, 92160) FAILED
  validation: V_emp measured via interleaved Sobol sub-blocks UNDERESTIMATES
  3x on mlp 18 (1.44e-2 vs ~4.4e-2 implied by its measured curve), sending
  the second-biggest win to the floor (+6.5%); eval-set mean +1.7% WORSE.
  The 2026-07-07 Path-C fidelity (Pearson 0.9999) does NOT transfer to this
  sub-block construction - suspected QMC correlation artifact.
- Calibration is immaterial: best-s search plateau ~20x wide within 2%.
- Next signal candidates before any estimator change: (a) pair-level variance
  over all 15,360 antithetic pairs (no sub-blocks); (b) hybrid
  max(V_emp, rescaled V_analytic); (c) fallback = conservative cap raise only
  for analytic-forecast-capped nets (mlp-14 type).
- Easy-net no-harm gate also FAILED for the V_emp rule: mean +26.5%, worst
  +116.9% (net 30 mis-ranked upward). Signal fails in both directions.

## UPDATE 2: 82k-cap fallback (analytic signal unchanged) - REGRESSION,
## and the reason is the SOBOL FLIP (2026-07-11, hard_mlp_sample_frontier.ipynb)

13/100 mini nets saturate the 61,440 cap (max uncapped demand 81,903). Exact
runs at 61,440 vs uncapped-demand for all 13: mean adjusted +41.3% WORSE,
suite ~-13%. DECOMPOSITION: the 5 near-zero-move noise-control nets ALL
regressed (+3..+124%) - systematic, not noise. Cause: one sample past 61,440
flips the whole ~30k continuation block from Sobol to Gaussian (artifact edge).
=> The QMC advantage on the continuation block is worth ~30-50% adjusted;
   ANY cap raise REQUIRES shipping more Sobol half-points first (float16:
   61,440 half-points ~31MB, artifact ~46MB < 52MB cap -> 122,880 all-Sobol).
=> Nets 11/14/84 still gained -50..-69% THROUGH the Gaussian penalty - lower
   bounds on all-Sobol gains; the capped tail is real, the tail QUALITY was
   the blocker.
=> Also retro-corrects round 1: 92k/123k measurements carried the same
   penalty; variance-limited conclusions stand, marginal-N economics were
   understated. "Gaussian tail is fine" claim WITHDRAWN for marginal dN.
NEXT: bake extended fp16 Sobol artifact, re-run this exact experiment with
all-Sobol continuation, then decide the cap.

## UPDATE 3 (final): all-Sobol re-run - cap raise is a WASH; line CLOSED

Baked `sobol_points_ext61440_fp16.npz` (29.0MB, 61,440 half-points = 122,880
all-Sobol samples; existing prefix bit-exact + fresh scrambled block seed
20260711 - original scramble seed unrecoverable, replicated scrambled QMC is
the fallback). Re-ran the 13 capped nets with all-Sobol continuation:

- 13-net mean adjusted 3.3494e-7 -> 3.3415e-7 (-0.2%), suite ~0.1%: INSIDE
  NOISE. Cap raise with the analytic signal buys nothing - MSE gain and
  multiplier cost cancel at the margin, as the pooled N* analysis predicted.
- The Gaussian-round per-net "wins" were NOISE: net 11 flipped -50% -> +74%,
  net 14 -69% -> +5%. Single-realization MSE noise on hard nets is 30-70%.
  RULE: only >=13-net aggregates (or replications) are decision-grade.
- fp16 quantization VALIDATED FREE: all 61,440 baselines within +-0.2% of the
  fp32 artifact. The 29MB fp16 artifact is available if a future signal
  genuinely identifies 2x-sample nets.
- Sobol-flip confirmed: with all-Sobol continuation the noise-control nets
  scatter +-3..11% around zero (vs uniformly +17..+124% with Gaussian tail).

DECISION: allocation axis closed from both ends (signal replacement failed
both gates; cap raise with trusted signal is a wash). Reconfirms the
2026-07-07 allocation ceiling on the 315824 surface with exact public GT.
Do not retry cap/allocation changes without a fundamentally better variance
signal AND replication-grade measurement.
