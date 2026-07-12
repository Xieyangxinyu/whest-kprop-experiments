# Strassen vs Packing - Real whest-run Profile - 2026-07-09

## Setup

Local test authorized (no flopscope modification, no submission). Added an
off-by-default `_USE_STRASSEN` flag + flopscope-native one-level `_strassen_matmul`
to estimator.py, routing the two dominant propagation matmuls (layer-0 and the
per-layer dense path) through it. `_dense_matmul` = plain `x@w` when the flag is
off, so the 315416 surface is byte-identical by default.

A/B via `whest run --estimator estimator.py --runner subprocess --seed 0
--n-mlps 3 --profile` (same 3 generated MLPs both runs):
- baseline = current estimator (`_PACKED_ROWSPARSE=True`, `_USE_STRASSEN=False`)
- candidate = Strassen standalone (`_PACKED_ROWSPARSE=False`, `_USE_STRASSEN=True`)

## Results

| metric | packing [315416] | strassen1 |
|---|---|---|
| adjusted_final_layer_score | 1.98e-7 | **1.29e-7** (-35%) |
| raw final_layer_mse | 2.46e-7 | 2.46e-7 (identical) |
| flops_used | 3.29e11 | 3.51e11 |
| residual_wall_time_s | 3.14 | 0.597 |
| effective_compute | 6.43e11 | 4.11e11 |
| mean_score_multiplier | 0.788 | 0.503 |
| failed MLPs | 0/3 | 0/3 |

## Key finding

Strassen has HIGHER analytical FLOPs than packing (packing skips zeros -> removes
more real multiplies), yet WINS because packing's `_packed_matmul` runs a Python
`for`-loop over row-chunks that inflates RESIDUAL wall-time (3.14s vs Strassen's
0.597s straight-line). Residual dominates effective_compute in this seed-0/n=3
regime, so Strassen's lower residual -> lower multiplier (0.503 vs 0.788) ->
-35% adjusted score, at IDENTICAL raw MSE (exact, confirmed 2.46e-7 both).

The previously-open Strassen caveats are resolved: wall-time SAFE (0 failed,
low residual), precision exact. Strassen is a strictly better compute vehicle
than the shipped packing here - and notably the shipped packing pays a
self-inflicted residual tax from its chunk loop.

## Caveats / not yet done

- seed-0/n=3 synthetic regime weights residual heavily (leaderboard 315416
  multiplier was ~0.40, lower residual fraction). WIN DIRECTION (strassen >
  packing, exact) is robust; MAGNITUDE is regime-dependent. Needs both-seed-group
  + public-mini validation before any leaderboard claim.
- Dense not profiled (estimated ~1.38e-7); Strassen has lower flops than dense
  with similar residual, so Strassen >= dense too.

## Decision

DO NOT SUBMIT. Legitimacy/prize-eligibility (Strassen decomposition lowers the
flopscope-counted FLOP metric via an algebraic matmul substitution) is an
organizer judgment - unchanged from 2026-07-08. Technical case is now very strong;
the blocker is a rules ruling, not engineering. estimator.py reverted to 315416
surface (Strassen code dormant, flag off).
