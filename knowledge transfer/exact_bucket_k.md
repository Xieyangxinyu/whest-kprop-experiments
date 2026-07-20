# Exact bucket-k gathers (knowledge transfer)

2026-07-19. Reference implementation was a local scratch copy, not the durable
`examples/37_where_threading_only.py` milestone. This is a one-line change on the
Algorithm 34+ row-sparse surface: it composes with the later 317459
where-threading surface, but did not transfer cleanly in the local n=3 check on
top of array-only sample-block cleanup.

## The defect

`_packed_matmul` groups rows by firing count (nnz) into buckets
`(0, 8, 16, 32, 48, ...)`, then for each bucket gathers `k` columns per
row and contracts with `einsum('nk,nko->no')`. The shipped gather width is

```python
k = _ceil_bucket(limit, _PACKED_ROWSPARSE_BUCKET, prev_width)   # rounds UP to multiple of 16
```

flopscope bills that einsum at exactly `2*n*k*o - n*o` — **linear in the
padded k, with no concept of alignment**. Rounding `limit=8` up to `k=16`
makes the lowest-nnz bucket pay exactly 2x its necessary contraction cost.
The padding bought nothing: flopscope is an accounting model, not
hardware — there is no SIMD/cache-line benefit to 16-alignment.

## The fix

```python
k = min(limit, prev_width)
```

Rows in a bucket have `nnz <= limit` by construction (they were selected
by `searchsorted` on the sorted nnz), so gathering exactly `limit` columns
still captures every firing column. The extra columns the padding used to
gather were exact post-ReLU zeros contributing nothing.

`_ceil_bucket` and `_PACKED_ROWSPARSE_BUCKET` become dead code — delete
them when folding this in permanently.

## Exactness — fp-noise class, NOT bit-identical

First intuition ("bit-identical because padded columns are zero") is
WRONG, in an instructive way: `argpartition(mask, prev_width - k)` with a
different `k` returns the same firing-column SET in a different ORDER,
and float32 summation is order-dependent. Measured final-row drift:
2.4e-7 / 4.0e-7 (seeds 7 / 11) — the same fp-noise class as previously
accepted routing changes (chunk-boundary shifts, put-unsort). Raw MSE is
unchanged at displayed precision in every check.

## Measured evidence (all real runs, flopscope 0.8.0rc5)

Census (net seed 7): the padding affects ONLY the `limit=8` bucket — 18 of
552 einsum groups; every other bucket limit is already a 16-multiple.
That bounds the win up front: small.

Paired `predict()` vs Algorithm 34 (same net, same budget context):

| seed | flops a34 | flops a37 | delta | final-row drift |
|---|---|---|---|---|
| 7 | 107,688,138,548 | 107,617,403,316 | −0.066% | 2.4e-7 |
| 11 | 105,480,400,437 | 105,375,161,125 | −0.100% | 4.0e-7 |

Scored `whest run` seed 42 vs Algorithm 34 baselines:

| | a34 n=3 | a37 n=3 | a34 n=5 | a37 n=5 |
|---|---|---|---|---|
| Adjusted | 1.61e-7 | 1.47e-7 | 1.39e-7 | 1.34e-7 |
| Raw final MSE | 3.38e-7 | 3.38e-7 | 3.11e-7 | 3.11e-7 |
| Estimator flops | 3.24e11 | 3.23e11 | 5.32e11 | 5.32e11 |
| Multiplier | 0.478 | 0.436 | 0.444 | 0.428 |
| Failures | 0/3 | 0/3 | 0/5 | 0/5 |

## READ THE SCORED NUMBERS HONESTLY

The deterministic flop cut is −0.066%/−0.100%. The adjusted-score moves in
the scored runs (−8.7% n=3, −3.6% n=5) are far larger than that — the
difference is the RESIDUAL-WALL term of effective compute, i.e.
machine-contention luck between runs, NOT the bucket fix. This is the
standard trap (see submission_learnings_2026-07-18/19: residual-wall
contamination and the local-multiplier trust break). Honest expected
grader-side effect: ~−0.1%-class on the multiplier, raw unchanged.

Do not use this variant's scored delta as evidence for anything except
"no regression". The paired-flops measurement is the truth signal.

## Why keep it despite the small size

- Strictly fewer billed flops, no accuracy cost, no new failure modes.
- One line; composes with any surface (the einsum branch is shared).
- Same class as the Algorithm 21 lesson: flopscope prices the exact k you
  contract, so never carry hardware-alignment habits into an accounting
  model. Audit any other "round up for alignment" constants the same way
  (`_PACKED_ROWSPARSE_ROW_BUCKETS` spacing itself is a separate, already
  fitted tradeoff — the buckets are fine; only the ceil-to-16 was waste).

## Fold-in checklist

1. Replace the `_ceil_bucket(...)` call with `k = min(limit, prev_width)`
   in `_packed_matmul`.
2. Delete `_ceil_bucket` and `_PACKED_ROWSPARSE_BUCKET`.
3. Expect fp-noise drift (order-of-summation); pair-check flops strictly
   <= baseline; raw MSE unchanged at displayed precision.
4. When folding into Algorithm 36 (int64), note its einsum branch also
   uses `k` — the fix applies identically and slightly improves the
   adaptive-Q bound (smaller k -> smaller row sums; no code change needed).
