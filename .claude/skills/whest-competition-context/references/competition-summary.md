# Competition Summary

## Task

Predict expected post-ReLU neuron activations for every layer of a randomly initialized ReLU MLP under standard normal input.

Required estimator method:

```python
def predict(self, mlp, budget):
    ...
```

Return a `flopscope.numpy` array of shape `(mlp.depth, mlp.width)`. Rows correspond to layers after each affine + ReLU block. Do not return input-layer values, final layer only, or pre-ReLU activations.

## MLP Object

- `mlp.width`: neurons per layer.
- `mlp.depth`: number of weight matrices/layers.
- `mlp.weights`: ordered `(width, width)` matrices.
- `mlp.seed`: per-MLP seed for reproducible predict-time randomness.

## Scoring

Leaderboard metric: `adjusted_final_layer_score`.

It is based only on final-layer MSE and a compute multiplier:

```text
adjusted_m = final_layer_mse_m * max(0.1, effective_compute_m / flop_budget)
```

The suite score is the mean of `adjusted_m` across MLPs.

Diagnostics:

- `final_layer_mse`: raw final-layer MSE, no compute multiplier.
- `all_layers_mse`: raw all-layer MSE, no compute multiplier.
- `mean_score_multiplier`: mean compute multiplier.

Important correction: improving `all_layers_mse` is useful only if it helps final-layer behavior or debugging. The leaderboard does not rank on it.

## Budget Semantics

Effective compute is analytical flopscope FLOPs plus residual Python wall time converted to FLOP equivalents. If effective compute exceeds the budget, the affected MLP's predictions are replaced by zeros and the multiplier is forced to `1.0`.

The multiplier bottoms out at `0.1`; below 10% budget use, cheaper compute does not improve score further.

## Common Context Mistakes

- Treating `final_layer_mse` as the leaderboard metric. It is a raw diagnostic.
- Optimizing only `all_layers_mse` and ignoring final layer score.
- Returning a shape `(width,)`, `(depth + 1, width)`, or final-layer-only vector.
- Using raw NumPy/Torch in submitted estimator math instead of `flopscope.numpy`.
- Assuming over-budget predictions partially count. They are zeroed for that MLP.
