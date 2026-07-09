# Packed Row-Sparse Active Propagation Probe - 2026-07-09

## Change

Added an exact chunked row-sparse ReLU matmul path in `estimator.py` for ordinary active-set propagation layers. The packed path uses flopscope-native `fnp.argsort`, `fnp.take_along_axis`, `fnp.take`, and `fnp.sum` to move row nonzeros to a fixed chunk-local prefix, then multiplies only that prefix. It is guarded by `k <= 0.75 * prev_width` and falls back to dense matmul otherwise.

## Rationale

WHest scoring charges analytical FLOPs plus `lambda * residual_wall_time_s`, not total wall time. A flopscope-native packing kernel can be wall-clock slower while still improving `effective_compute` if most extra time lands in `flopscope_backend_time_s` rather than residual time.

## Results

Baseline command before edit:

```bash
uv run whest run --estimator estimator.py --runner local --seed 0 --n-mlps 1 --profile
```

- adjusted final-layer score: `1.08e-7`
- raw final-layer MSE: `2.67e-7`
- flops used: `1.08e11`
- effective compute: `1.10e11`
- mean score multiplier: `0.40382515`

Packed candidate, same generated MLP:

```bash
uv run whest run --estimator estimator.py --runner subprocess --seed 0 --n-mlps 1 --profile
```

- adjusted final-layer score: `8.47e-8`
- raw final-layer MSE: `2.67e-7`
- flops used: `8.05e10`
- effective compute: `8.63e10`
- mean score multiplier: `0.31726375`
- failed MLPs: `0 of 1`

Public-mini subset with packed candidate:

```bash
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess --n-mlps 3 --profile
```

- adjusted final-layer score: `2.37e-7`
- raw final-layer MSE: `5.45e-7`
- flops used: `3.46e11` total, `1.15e11` mean / MLP
- effective compute: `3.66e11` total
- mean score multiplier: `0.44897192`
- failed MLPs: `0 of 3`

Same public-mini subset with `_PACKED_ROWSPARSE = False`:

- adjusted final-layer score: `2.97e-7`
- raw final-layer MSE: `5.45e-7`
- flops used: `4.51e11` total, `1.50e11` mean / MLP
- effective compute: `4.57e11` total
- mean score multiplier: `0.56002734`
- failed MLPs: `0 of 3`

## Decision

Promising submission candidate. The packed path preserved raw MSE and improved adjusted score on the fixed checks by lowering the compute multiplier. Main risk is very high `flopscope_backend_time_s` (`~60s` total for 3 public-mini MLPs), which is not part of the score multiplier but could matter if the grader enforces a wall-clock limit.

Next checks before submit: run a larger public-mini subset if time allows, and run final package/subprocess review.