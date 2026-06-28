---
name: whest-estimator-debugging
description: "Use when: WHest estimator validation fails, whest run fails, score regressed, PREDICT_ERROR, SETUP_ERROR, wrong shape, NaN, Inf, budget_exhausted, combined_budget_exhausted, subprocess mismatch, local vs remote mismatch, or the user asks to correct mistakes in estimator.py."
argument-hint: "error message, score report field, or estimator file"
---

# WHest Estimator Debugging

## When to Use

Use this skill for broken or suspicious estimator behavior:

- `whest validate` or `whest run` failures.
- Wrong output shape, non-finite values, import errors, signature mismatch.
- `PREDICT_ERROR`, `SETUP_ERROR`, setup/predict timeout, budget exhaustion.
- Good local score but bad subprocess or submission score.
- A change that worsened `adjusted_final_layer_score`.

## Debugging Ladder

1. Load [debugging-playbook.md](./references/debugging-playbook.md).
2. Reproduce with the narrowest command that shows the issue.
3. Use `--debug` and `--fail-fast` for tracebacks before guessing.
4. Read the report fields: `error_code`, `budget_exhausted`, `combined_budget_exhausted`, `time_exhausted`, `residual_wall_time_s`, `flops_used`, `effective_compute`.
5. Fix the root cause, then run validation and a small local score check.
6. For submission-like issues, compare local and subprocess runner results with the same seed and `--n-mlps`.

## High-Priority Mistake Corrections

- Returned array must be `(mlp.depth, mlp.width)`, post-ReLU means for every layer.
- Use `flopscope.numpy as fnp` for numerical work; raw NumPy is not FLOP-tracked and may fail grader assumptions.
- Budget failure zeroes predictions and forces multiplier `1.0`; it is worse than a cheap valid baseline.
- `setup()` is outside the FLOP budget but has a tight setup timeout and must not depend on laptop-only paths.
- Direct `print()` inside `predict()` can inflate residual wall time and pollute worker output.

## Source Docs

- [docs/how-to/debugging-checklist.md](../../../docs/how-to/debugging-checklist.md)
- [docs/troubleshooting/common-participant-errors.md](../../../docs/troubleshooting/common-participant-errors.md)
- [docs/how-to/validate-run-package.md](../../../docs/how-to/validate-run-package.md)
- [docs/reference/estimator-contract.md](../../../docs/reference/estimator-contract.md)
