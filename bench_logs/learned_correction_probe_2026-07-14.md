# Learned final-row correction probe - 2026-07-14

## Hypothesis (pre-registered)

Hypothesis: the per-neuron error of the shipped estimator's final row
(gt - predicted) is partially predictable OUT-OF-NET from features
computable inside predict() (diagonal analytic moments, full-cov anchor,
final alpha, the prediction itself), so a correction fitted on public GT
cuts fleet raw final-layer MSE at ~zero compute cost.

Ceiling analysis (pre-registered): the estimator is deterministic given the
fixed Sobol artifact; the cross-net-systematic error component is bounded by
the bias-like structure - the algo25 calibration fitted a = 2.527e-8 = 6.8%
of raw 3.725e-7. Expect <= ~7% unless fixed-point-set QMC error is itself
cross-net systematic. Calibrating constants on public data is the
established-legitimate _VAR_REF class.

Prediction: OOS fleet raw cut in the 3-7% range if bias dominates the
predictable part; >7% only if QMC structure is learnable.
Baseline: algo21/315892 estimator.py (byte-exact), in-process predicts,
100 mini nets, budget 2.72e11, fixed Sobol artifact.
Variant: post-hoc per-neuron correction models fitted on (features -> error),
leave-nets-out CV (10 folds by net).
Scientific parameters: feature set; model class (alpha-binned means, ridge,
gradient boosting if available).
Nuisance: fold count, bin edges, ridge lambda.
Fixed: nets 0-99 mini, N chosen by the shipped rule, no estimator edits.
Success threshold: SHIP-track if OOS fleet raw cut >= 5% and stable across
folds (every-fold improvement). MARGINAL 3-5%: adopt only if the correction
is a trivial artifact (e.g. per-alpha-bin constants). KILL < 3%.
Kill criteria: OOS cut < 3%, or gains concentrated in < 20% of nets
(idiosyncratic overfit), or correction requires features unavailable in
predict().

## Results

Capture: 100 mini nets, in-process natural predicts (17s/net), fleet raw
final-layer MSE 4.392e-7 (mini split; leaderboard set is 3.725e-7).
25,600 (neuron, net) rows; features = pred, diag mean/alpha, full-cov
anchor mean/alpha, differences, |alpha|, saturation flags, net-level
pred mean/std. Target gt - pred. 10-fold leave-nets-out CV, per-net MSE
aggregation matching the scoring.

| model | OOS fleet raw cut | net-win |
|---|---|---|
| alpha-binned constants | -1.19% (WORSE) | 30% |
| alpha-binned linear-in-pred | -1.63% (WORSE) | 28% |
| ridge (all features) | +1.09% | 47% |
| gradient boosting | -1.12% (WORSE) | 41% |

## Verdict: KILL (bar was >= 5% ship / < 3% kill; best model +1.09%)

- The cross-net predictable component of the final-row error is ~1% -
  BELOW even the 6.8% bias-fraction ceiling: the bias itself is per-net
  idiosyncratic, not feature-predictable (binned corrections transfer
  NEGATIVELY across nets - per-alpha-bin error means are net-specific).
- This closes the last cheap VALUE lever flagged in
  cheap_forward_scoping_2026-07-14.md and directly confirms the
  deep-band memory's "per-net idiosyncratic bias floor" as
  learned-correction-proof, not just analytics-proof.
- Do NOT retry: learned post-hoc corrections from analytic features
  (any model class); per-alpha-bin corrections fitted on public GT.
  New evidence required: a feature family that is per-net adaptive
  (e.g., computed from the net's own samples) with demonstrated
  cross-net transfer.

Scripts: session scratchpad correction_capture.py / correction_fit.py
(+ capture pickle, 100 nets).
