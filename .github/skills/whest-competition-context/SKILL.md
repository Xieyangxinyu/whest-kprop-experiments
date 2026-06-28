---
name: whest-competition-context
description: "Use when: working on ARC WhiteBox Estimation, WHest, whest-starterkit, estimator.py, MLP mean activation prediction, flopscope, scoring model, leaderboard metrics, or challenge context. Loads the core competition contract so answers correct misunderstandings about what is scored and what predict() must return."
argument-hint: "WHest question, estimator idea, or score report"
---

# WHest Competition Context

## When to Use

Use this skill before answering or editing anything related to:

- ARC WhiteBox Estimation / WHest / whestbench / flopscope.
- [estimator.py](../../../estimator.py), examples, or custom estimator code.
- Score reports, leaderboard metrics, FLOP budgets, or phase-1 public mini runs.
- Questions where the user may be mixing up raw MSE, adjusted score, final-layer-only scoring, or all-layer diagnostics.

## Core Mental Model

The task is to predict post-ReLU per-neuron mean activations for every layer of a random ReLU MLP under Gaussian input. `predict(self, mlp, budget)` must return a flopscope array with shape `(mlp.depth, mlp.width)`.

The leaderboard ranks `adjusted_final_layer_score`, not raw `final_layer_mse` and not `all_layers_mse`. `final_layer_mse` is multiplied by `max(0.1, effective_compute / flop_budget)`, where effective compute includes flopscope FLOPs plus residual Python wall-time converted to FLOP equivalents. If a call exceeds budget, predictions for that MLP are replaced with zeros and the multiplier is forced to `1.0`.

## Procedure

1. Load [competition-summary.md](./references/competition-summary.md).
2. If code changes are involved, confirm the estimator contract before editing.
3. If interpreting a score, distinguish ranking metric from diagnostics.
4. If proposing algorithms, reason in terms of accuracy versus effective compute and the 0.1 multiplier floor.
5. If the user asks for commands, prefer `uv run ...` commands from the repo docs.

## Source Docs

- [docs/README.md](../../../docs/README.md)
- [docs/reference/estimator-contract.md](../../../docs/reference/estimator-contract.md)
- [docs/concepts/scoring-model.md](../../../docs/concepts/scoring-model.md)
- [docs/reference/code-patterns.md](../../../docs/reference/code-patterns.md)
