# Packing Residual-Wall-Time Cleanup - 2026-07-09

## Motivation

The Strassen A/B (strassen_vs_packing_profile_2026-07-09.md) revealed that the
shipped packing (315416) pays a big residual-wall-time tax: 3.14s residual (seed-0
n=3 subprocess), ~half of effective_compute, vs Strassen's 0.597s. Packing has
LOWER analytical FLOPs than Strassen, so removing the residual tax should let
packing beat everything - a fully legitimate win (faster own Python, exact math,
no flopscope touch). Goal: find and remove the residual source.

## Attempts (whest run --profile, seed 0, n-mlps 3, subprocess)

| variant | residual | adjusted | mult | flops | failed |
|---|---|---|---|---|---|
| packing 315416 (per-chunk k, chunk 2048) | 3.14s | 1.98e-7 | 0.788 | 3.29e11 | 0/3 |
| global-k refactor, chunk 2048 | 3.04s | 1.96e-7 | 0.782 | 3.34e11 | 0/3 |
| global-k, chunk 8192 | 2.69s | 1.86e-7 | 0.739 | 3.34e11 | 0/3 |
| global-k, chunk 16384 | - | FAIL | 1.0 | - | 3/3 (OOM) |

All packing variants exact (raw MSE 2.46e-7 identical).

## Findings

1. **Per-chunk `int(fnp.max())` syncs are NOT the residual driver.** Hoisting to
   one global `k` per call (removes ~1140 host syncs) cut residual only 3.14->3.04s.
2. **Chunk/dispatch count is a MINOR driver.** 4x fewer chunks (2048->8192) cut
   residual only 3.04->2.69s (~12%).
3. **The residual is INTRINSIC to the `fnp.take` gather.** flopscope charges the
   per-row weight gather `take(weights, order) -> (chunk, k, out)` as free indexing
   (0 backend FLOPs), so its multi-GB wall-time lands entirely in RESIDUAL. Total
   gather volume is chunk-independent -> can't be reduced without removing packing.
4. **Hard memory ceiling:** chunk 16384 (3.2GB gather) OOMs 3/3 in-subprocess
   (works in-process - subprocess has tighter memory). chunk 8192 (1.6GB) is the
   safe ceiling; footprint is chunk_rows*k*out, independent of n_rows.

## Conclusion / Decision

The packing-residual cleanup has a LOW CEILING because the residual is the gather,
which is packing itself. Best safe config (global-k + chunk 8192): ~6% better
adjusted here (1.98->1.86e-7), and less on the real leaderboard where residual is a
smaller fraction (315416 leaderboard mult ~0.40 vs ~0.79 here). This is DOMINATED
by Strassen (1.29e-7, -35%), which has no gather and structurally avoids the cost.

estimator.py REVERTED to pristine 315416 (no keepable win: cleanup capped,
Strassen legitimacy-gated). Staged Strassen+cleanup code saved at
scratchpad/estimator_strassen_and_cleanup.py. If pursuing the modest legitimate
win: global-k + chunk 8192 is safe and exact, but validate memory on max-active
public-mini MLPs first and expect a small (~2-3%) leaderboard gain.
