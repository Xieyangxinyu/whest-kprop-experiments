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

## Micro-Optimization Isolation Protocol

Use this protocol for exact-output cleanup ideas, FLOP accounting tricks, backend-call reductions, or residual-wall-time hypotheses.

1. Start from the current public-confirmed baseline, not a broad local cleanup stack.
2. Patch one scientific change per temp estimator folder; avoid bundling changes until each one has isolated evidence.
3. Before scoring, smoke-test shallow depths such as 1, 2, 3, 4, 8 plus depth 32; grader smoke MLPs are not guaranteed to match public depth.
4. Run a fixed-seed subprocess n=1 screen. A candidate must keep `n_failed_mlps = 0`, keep `final_layer_mse` unchanged for exact-output ideas, lower `flops_used`, and not worsen `effective_compute`.
5. Confirm promising n=1 candidates with subprocess n=3 or n=5 before promotion. Reject n=1-only wins that reverse on n=3, even when tracked FLOPs improve.
6. Inspect both `flops_used` and `residual_wall_time_s`. A 0-FLOP or lower-FLOP change can still regress if backend/residual attribution moves unfavorably.
7. Submit only public-confirmed or locally isolated candidates that reduce score-relevant effective compute without raw-MSE drift. Record rejected exact-output ideas so they are not re-bundled later.

Current cautionary examples: `fnp.put` scatter reduced tracked FLOPs but regressed publicly; broad list-shape cleanups were smoke-risky; `fnp.var` to mean-of-square looked good at n=1 but worsened effective compute at n=3.

## Experiment Run Hygiene

Use these mechanics to avoid invalid comparisons and wasted reruns.

- For estimators that read `sobol_points.npz`, run variants from folders containing both `estimator.py` and `sobol_points.npz`; do not pass a bare `/tmp/foo.py` copy unless its directory also has the data file. A bare temp file can fail setup by looking for `/tmp/sobol_points.npz`.
- Give every baseline and variant output a unique, labeled JSON path such as `/tmp/whest_<experiment>_<label>_n3.json`; never overwrite a result you still need for attribution.
- After a long run, first check each JSON file exists and has nonzero size. Parse the top-level object and handle `{"ok": false, "error": ...}` before reading `results`; setup/smoke failures are not comparable score results.
- Prefer running baseline and variants under the same command/session when practical, but still save each output separately. If one variant fails, keep the completed outputs and rerun only the missing label.
- When comparing temp variants, print deltas from saved JSON files rather than re-running immediately. Include `flops_used`, `wall_time_s`, `residual_wall_time_s`, `effective_compute`, `final_layer_mse`, and `n_failed_mlps`.
- Record the exact baseline source path used for comparison, e.g. public-confirmed `/tmp/whest_37/estimator.py` or current active `estimator.py`; do not mix a failed broad cleanup baseline with a public-confirmed baseline.
- If a run moved to the background or output was truncated, inspect the saved JSON files directly before rerunning. Terminal tail text can omit the final metrics even when the JSON files are complete.

## Recording Results

Use the current dated file in `bench_logs/` when the result should be remembered. Include:

- Command and runner mode.
- Git diff summary or variant name.
- Score fields needed to reproduce the conclusion.
- Decision: adopt, reject, rerun with more evidence, or submission-only validation.
- Failed approaches that should not be retried without new evidence.