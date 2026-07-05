---
name: whest-experiment-design
description: "Use when: planning WHest estimator experiments, comparing variants, designing fixed-seed sweeps, choosing sample counts, tuning hyperparameters, or deciding whether a score change is meaningful. Structures hypotheses before running estimator benchmarks."
argument-hint: "estimator variant, score hypothesis, or experiment plan"
---

# WHest Experiment Design

Use this skill before editing `estimator.py` for a scoring idea or before running a sweep whose result should guide the next implementation step.

## Workflow

1. Load `whest-competition-context` and, for performance ideas, `whest-flop-optimization`.
2. State one falsifiable hypothesis in WHest scoring terms.
3. Classify parameters as scientific, nuisance, or fixed.
4. Choose a comparison protocol before seeing results.
5. Run the smallest benchmark that can reject an obviously bad idea.
6. Record the outcome in `bench_logs/` when the result changes future decisions.

## Hypothesis Template

Fill this before running a meaningful comparison:

```markdown
Hypothesis: <one sentence, falsifiable>
Prediction: <expected change in adjusted_final_layer_score and diagnostics>
Baseline: <command, seed/dataset, score fields, git state>
Variant: <implementation or config being tested>
Scientific parameters: <the idea being studied>
Nuisance parameters: <sample counts, thresholds, pruning knobs, etc.>
Fixed parameters: <seeds, n-mlps, data file, runner mode, timeout assumptions>
Success threshold: <minimum improvement beyond run variance and complexity cost>
Kill criteria: <budget exhaustion, slower effective compute, local-only gain, etc.>
```

## WHest-Specific Checks

- Compare `adjusted_final_layer_score` first, then inspect raw `final_layer_mse`, `mean_score_multiplier`, `flops_used`, `effective_compute`, and `residual_wall_time_s`.
- Keep the runner mode fixed during a comparison. Do not mix standalone and subprocess results unless the experiment is specifically about parity.
- Treat budget exhaustion as a failed experiment unless the hypothesis explicitly targets fallback behavior.
- Prefer fixed seeds or fixed evaluation datasets for close calls; random mini-runs are useful for smoke tests but weak evidence for small deltas.
- If a variant only improves all-layer diagnostics while final-layer adjusted score regresses, record that as a negative result for leaderboard purposes.
- If best results sit at a tuning boundary, widen the range or call the result inconclusive rather than overfitting the boundary.

## Recording Results

Use the current dated file in `bench_logs/` when the result should be remembered. Include:

- Command and runner mode.
- Git diff summary or variant name.
- Score fields needed to reproduce the conclusion.
- Decision: adopt, reject, rerun with more evidence, or submission-only validation.
- Failed approaches that should not be retried without new evidence.