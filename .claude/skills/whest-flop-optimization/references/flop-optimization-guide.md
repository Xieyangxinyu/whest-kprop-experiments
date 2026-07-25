# FLOP Optimization Guide

## Goal

Optimize `adjusted_final_layer_score`, not only raw MSE. Once compute is at or below the 0.1 multiplier floor, further FLOP savings do not improve score unless accuracy is preserved or improved.

## Baseline Commands

```bash
uv run whest run --estimator estimator.py --runner local --seed 42 --n-mlps 3 --profile
uv run whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner local
```

Use fixed seeds or fixed datasets for comparisons.

## The flopscope 0.9.x Billing Model

Authoritative reference: flopscope `docs/reference/cost-model.md` (v0.9.1). Every
operation is charged:

```
charged = int(flop_cost × dtype_rate × complex_factor × weight)
```

- `flop_cost` — shape/algorithm operation count (all size-dependent terms live here).
- `dtype_rate` — width factor of the dtype the call actually computes in.
- `complex_factor` — 1 for real dtypes; complex expands to its real-op count.
- `weight` — hardware tier, one of `{0, 1, 4, 16}`.

Unifying rule: **every byte written is metered**. Views are free; any op that writes
a new buffer bills ≥ 1 per element written; non-sequential access (sorts, computed-index
gathers, per-element selects) bills 4 per element; transcendentals bill 16.

The old 0.8.x framing "data logistics are 0-FLOP movement" is **retired**: `take`,
`take_along_axis`, fancy indexing, `concatenate`, `stack`, `reshape`, `ravel`, `copy`,
`ones`, and 3-arg `where` are all billed now.

## Dtype Rates — check your dtypes first

| class | rate |
|---|---|
| ≤ 32-bit (float32, int32, float16, int8, bool, …) | 1.0 |
| 64-bit (float64, **int64**, uint64) | 2.0 |
| longdouble (float96/float128) | 3.0 / 4.0 |
| complex64 / complex128 | 1.0 / 2.0 × a structure factor (multiply 6, add 2, contraction exact ≈ 4.13) |

- The billing dtype is `np.result_type` over the operands — **mixed `f32 @ f64`
  promotes and bills at the float64 rate**. Python scalars do not promote (NEP 50):
  `f32_array * 2.0` stays float32.
- **Local trap:** the local mini-split loader materializes `mlp.weights` as float64.
  Cast once per layer (`fnp.asarray(w).astype(fnp.float32)`, bills `numel` one time)
  or every downstream matmul bills 2×. Grader evidence (submission 318752, no cast,
  N=61,440, fit budget) indicates the hidden-suite weights are already float32 — so
  the cast is cheap insurance, not a remote lever. Keep it anyway.
- Reduction accumulators follow numpy: `sum` over int/bool inputs accumulates in
  int64 → bills 2× (measured: `fnp.sum(bool_mask, axis=1)` ≈ 2/elem). Pass
  `dtype=fnp.int32`/`float32` when the range allows to halve it.
- Integer inputs to float-only ufuncs, `linalg.*`, and mean-family composites promote
  to float64 and bill 2×. Keep everything float32 end-to-end.
- fp16 has **no discount** (rate 1.0). Complex/width packing is priced to lose or
  break even — do not build on it.

## Weight Tiers (estimator-relevant ops)

**Free (0):** basic slicing/`transpose`/views, `zeros`/`empty`, `asarray` with no
dtype change, `astype(dtype, copy=False)` when the dtype already matches.

**Weight 1 (per element written / tested):** pointwise math and comparisons,
`maximum` (use it for ReLU), `concatenate`/`stack` (numel of output), `reshape`/
`ravel`/`copy` (numel, always — even when numpy would return a view), `ones`/`full`,
`eye` (only the diagonal is billed — the `eye @ values` scatter idiom stays cheap),
`put`/`put_along_axis` (elements scattered), `nonzero`/1-arg `where` (numel scan),
reductions (`numel(in) − numel(out)`; `var`/`std` ≈ 4×numel).

**Weight 4 (non-sequential access):** `take`, `take_along_axis`, `choose`, fancy
`arr[idx]` (all 4 × numel(output); boolean-mask indexing adds numel(mask)); 3-arg
`where` (4 × numel — prefer `maximum`/mask-multiply; measured ≈ 8/elem on f32 calls);
`sort`/`argsort` (4 × n⌈log₂n⌉ per slice — ≈ 32/elem at n=256); `partition`/
`argpartition` (4 × n × len(kth)); `searchsorted` (4 × m⌈log₂n⌉); `unique`/set ops;
random reorders (`shuffle`, `permutation`, `choice`).

**Weight 16 (transcendental):** `exp`, `log`, `sin`, `sqrt` is weight 1 but `power`,
`mod`/`floor_divide`, `arctan2`, `hypot`, `logaddexp` are 16. `flops.stats.norm.pdf/cdf`
ride on these — fine per-neuron (width-sized), expensive per-sample.

## Contraction (matmul / einsum)

- One shared engine bills `(2K − 1) × M` (K = contracted dim, M = output cells),
  weight 1. `matmul` = `2mkn − mn`.
- **Symmetry-aware:** `outer(v, v)`, `inner(A, A)`, or `as_symmetric`-tagged operands
  bill the unique-orbit count `n(n+1)/2` — keep using symmetry-aware `fnp.einsum`
  (`"ij,ia,jb->ab"`) for covariance updates.
- Strassen/Winograd rearrangements: each sub-matmul is billed by its own shape, so
  the 7-vs-8-multiply saving still accrues — but the extra adds AND the quadrant
  assembly `concatenate`s now bill 1/elem (they were free pre-0.9, when this repo's
  Strassen percentages were measured). Net savings are therefore smaller than any
  pre-0.9 number in old logs; graded evidence that it still pays: submission 318752
  (2-level, guarded). Re-measure on/off under 0.9.x before citing a figure, and
  prefer the unassembled 2-block form that skips the axis=0 output concat.
- Integer matmuls keep the integer rate; `matrix_power` with a negative exponent
  routes through float64 LAPACK.

## Sparse-vs-dense routing (2026-07-25 measurements, v0.9.1)

Packed gather-einsum row-sparse propagation (argsort + take + einsum) beats the
guarded dense/Strassen path only for rows with **nnz < ~75 at width 256** (~56 at 192,
~37 at 128). Fleet fire rates put almost all rows above that, so **dense-by-default**:
the recalibrated packed configs at best tied dense-only; the pre-0.9 fitted routing
map lost ~34%. Activation-sparsity exploitation is dead under this pricing unless a
gather-free mechanism is found.

## Non-Exploitability (don't spend time here)

Aliases bill like their canonicals; equivalent contractions share one engine; casts
bill like `copy` (the `astype` free-copy loophole is closed); complex packing loses
(multiply factor 6; matmul ≈ 4.13×); 64-bit width packing breaks even before overhead.
Sub-32-bit lane tricks are officially in-bounds but bounded and small.

## flopscope Patterns

Use:

```python
import flopscope as flops
import flopscope.numpy as fnp
```

Operators on `fnp.ndarray` are tracked, so `+`, `*`, `/`, and `@` are fine.

For covariance updates, prefer symmetry-aware einsum:

```python
cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
```

Avoid chained matmuls like `w.T @ cov @ w`; they can lose symmetry information and
inflate downstream cost/warnings.

Audit any surprising bill by reading `op_log[-1].resolved_dtype` and the per-op
`flop_cost` from `flops.budget_summary_dict(ctx)` inside a `BudgetContext`.

## ReLU Gaussian Moment Pattern

For `z ~ N(mu, sigma^2)`:

```python
alpha = mu_pre / sigma_pre
mean = mu_pre * flops.stats.norm.cdf(alpha) + sigma_pre * flops.stats.norm.pdf(alpha)
```

This approximation is exact at the first layer and approximate later. Errors tend to
compound with depth; compare `all_layers_mse` and final-layer MSE to diagnose where
propagation drifts. Implement ReLU on samples with `fnp.maximum(x, 0.0)` (1/elem),
never 3-arg `where` (4–8/elem).

## Residual Wall Time

Python control flow and untracked external calls can raise `residual_wall_time_s`,
which feeds `effective_compute`. Avoid heavy Python loops, print spam inside
`predict()`, and expensive non-flopscope libraries unless they are truly worth the
residual penalty and packaging surface. Gather/sort-heavy paths cost wall time on top
of their now-nonzero billed FLOPs — dropping them helps both terms.
