# Algorithm 20: residual-trimmed packed matmul - 2026-07-11

## Idea

Cut grader-charged residual wall time in the 315824 packed row-sparse matmul
with bit-identical math: sort + bucket-split rows ONCE per call instead of
once per chunk (bucket membership depends only on per-row nnz, so the
partition is chunk-independent), replace the fourteen per-bucket
searchsorted+int() host syncs per chunk with ONE vectorized searchsorted,
keep memory-safe chunking only inside large groups, restore row order once
per call. Motivated by the algo19 lesson that Python-side dispatch is
grader-priced while the FLOP ledger isn't the binding constraint.

Implementation: `examples/20_residual_trim.py` (surgical patch of
17_finer_antithetic_row_buckets.py). Root A/B copies: `.ab_algo17.py`,
`.ab_algo20.py` (subprocess setup resolves sobol_points.npz relative to the
estimator's directory, so examples/ paths SETUP_ERROR — run A/B arms from
repo root).

## Identity check (4 mini nets, in-process, natural predict)

- Predictions BIT-IDENTICAL on all 4 nets (np.array_equal on the full
  returned array).
- FLOPs +0.002% (deterministic; global argsort/searchsorted over full arrays
  vs per-chunk).

Exact rewrite confirmed.

## Subprocess residual A/B (13 mini nets each, sequential, same machine)

- raw final MSE IDENTICAL (4.6349e-7 both arms).
- flops flat (+0.00%).
- residual 11.23s -> 11.26s (+0.25%) - FLAT, well inside per-net noise
  (~+/-5% per net, mixed signs across the 13 nets).
- adjusted -0.57% - residual-noise-driven (mult moved +0.13%), not real.

## Root cause of the null result

On this surface the chunk loop only runs ~2 iterations per call
(n_rows ~30720, chunk 16384; ~4 at the 61440 cap) - so "once per call
instead of once per chunk" halves a dispatch count that was already
microseconds. The searchsorted+int() calls are plain synchronous numpy on
CPU, not device syncs; fourteen of them cost nothing measurable. And the
put_along_axis restore touches the same total elements whether done per
chunk or once globally, so memory traffic is unchanged. The restructure
removes Python ceremony, not wall time. Contrast with algo19 pilot-reuse,
which ADDED real memory traffic (+8.9%); this one adds and removes nothing.

## Status: NULL RESULT - do not promote

Not a regression (bit-identical, flops flat, residual flat), but zero
benefit means submission churn with no upside. Residual wall time on this
surface is dominated by the actual gather/einsum/matmul memory traffic, not
by per-chunk Python dispatch. Together with the algo19 falsification this
closes the "restructure the same math for residual" line: the remaining
residual is real compute, not overhead. Any future residual attack must
REMOVE memory traffic (fewer/smaller intermediates), not reorganize
dispatch.
