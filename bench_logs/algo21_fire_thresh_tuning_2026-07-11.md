# Algorithm 21: block-split fire-threshold tuning - 2026-07-11

## Scope

User-directed exploration of the 0.75 "dense threshold" on the algo17/315824
surface, per-layer thresholds allowed. Two distinct 0.75 knobs exist; both
covered.

## Knob 1: packed-vs-dense row k-gate (_PACKED_ROWSPARSE_MAX_K 3/4) - VACUOUS

Census of all 248 _packed_matmul calls on 4 mini nets (5.9M rows, natural
predict): ZERO rows ever exceed 0.75*prev_width nnz. The upstream column
split routes every high-fire column into the dense block first, so packed
rows top out near 0.6 normalized nnz. The gate never fires; tuning it (any
direction, any layer) has no surface to act on. Together with the 07-09
gate sweep (tightening below 3/4 deletes the packing saving) this knob is
fully bracketed. Do not revisit.

## Knob 2: _BLOCK_SPLIT_FIRE_THRESH (column split, the real "Strassen dense
threshold") - FLOP WIN, RESIDUAL LOSS, NET FALSIFIED

Cost physics: column at fire rate f costs ~1.94*f per row-output packed
(einsum) vs ~1.76 flat in the Strassen dense block -> FLOP crossover
f* ~= 0.91, not 0.75. Layer-dependent via guards: a dense block under
MIN_IN=64 cols misses the Strassen discount (plain 1.98 rate), pushing that
layer's optimum toward 1.0 (pack everything).

- Global sweep t in {0.60..0.90}, 2 nets, in-process: raw MSE EXACTLY
  invariant (routing is exact). Lowering costs flops (+2.2% at 0.60);
  raising is FLAT (-0.06% at best). Global tuning dead; 0.75 sits on a
  plateau extending right.
- Fire-rate census (240 split calls, 4 nets) + per-layer oracle (exact
  guard rules, normal approx for sparse nnz): per-layer t* map spans
  0.70-1.0 (layer 2: 0.825, layer 31: 1.0, mids 0.85-0.95). NOTE: the
  model overpredicts global savings (says -0.45% total at t=0.925;
  measured flat) - normal-approx/bucket smoothing artifact - but its
  per-layer MAP is empirically real:
- Monkeypatch A/B, fit nets 0-3: flops -0.90%, raw bit-identical.
- HOLDOUT nets 4-12: flops -0.72%, raw bit-identical. Not overfit.
- Implemented as examples/21_layerwise_fire_thresh.py (+ .ab_algo21.py).
  File reproduces monkeypatch to 28 ppm flops, exact MSE.

## Subprocess residual A/B (13 mini nets each, sequential, same machine)

- raw 4.6349e-7 -> 4.6347e-7 (-0.01%), 0 failed both arms.
- flops -0.78% (deterministic, matches holdout).
- residual 11.52s -> 13.56s (+17.7%) - the moved columns' gather traffic.
  Local mult 0.618 -> 0.674; local adjusted +7.9%.
- Grader translation (residual ~19% of effective compute):
  0.81*(-0.78%) + 0.19*(+17.7%) ~= +2.7% WORSE. Breakeven needs residual
  share <= ~4%. A trimmed map (only layers 2+31, ~-0.25% flops, ~+4%
  residual) still models negative.

## LEADERBOARD ADDENDUM (same night): local verdict OVERTURNED

Submitted as 315892 by user direction to test the grader-side exchange rate:
public adjusted 1.33775e-7 = NEW FRONTIER, -0.83% vs 315824 (1.3489e-7),
raw unchanged. The grader priced the +17.7% local gather residual at ~zero;
multiplier delta tracked the -0.78% flops delta 1:1. The section below is
kept as written for the record of WHY the local A/B said no - its
grader-translation used the ~19% residual share, which turns out not to
apply to gather-type residual. See submission_learnings_2026-07-11.md.

## Status (superseded): FALSIFIED for promotion - do not submit

Same failure mode as algo19 and the packing lessons: the packed path's
per-column FLOP price (~f) undercuts Strassen's (~0.88) on the ledger, but
every column moved to packed pays the fnp.take gather tax that flopscope
prices at zero and the grader prices as residual. The 4x gap between the
FLOP saving and the grader-priced gather cost is intrinsic, not tunable.

Durable positives if pricing ever changes (faster grader residual pricing,
or flopscope charging gathers): the per-layer t* map is real, holdout-
validated FLOP physics, kept in examples/21_layerwise_fire_thresh.py with
the census/oracle scripts' methodology described in this log.
