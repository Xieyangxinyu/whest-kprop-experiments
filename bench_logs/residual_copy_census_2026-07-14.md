# Residual copy-traffic census + gather-restore fix - 2026-07-14

## Motivation

At 315892 the multiplier (0.359) implies ~1.2e10 FLOP-eq/net of
grader-priced residual on top of 8.55e10 analytical flops (~12% of
effective compute). Grader pricing rules (315892/315898 calibration):
fnp-op backend wall (gathers) ~ FREE; copy/allocation traffic PRICED
(~2.4 FLOP-eq/byte measured on 315898's concats). Target: find and REMOVE
copy traffic (algo20 lesson: removal, not dispatch reorganization).

## Census (residual_census.py, 3 mini nets, per-site byte accounting)

Copy-class traffic on the algo21 surface: **8.09 GB/net** (~1.9e10 FLOP-eq
at 2.4/B; implied grader residual 1.2e10 -> effective ~1.5/B). Ranked:

| cluster | GB/net | sites |
|---|---|---|
| packed row-restore | 5.26 | per-chunk groups concat 1.34 + empty_like 1.34 + put_along_axis 1.34 + final chunk concat 1.24 |
| Strassen assembly | 2.72 | quadrant concats 2.30 (L180/181/182) + odd-out edge concat 0.42 (L209) |
| sample blocks | 0.11 | L397, L574 |
| everything else | ~0.00 | index concats, eye-scatter, stack |

Confirmations: the einsum weight-gather moves 78 GB/net in the grader-FREE
take class (why packing survives grading at all). flopscope arrays are
IMMUTABLE (no slice-assign / in-place add; probed) - functional assembly
is forced, so Strassen's concats are structural without a row-blocked
pipeline.

## Fix implemented (estimator.py, on the 315892 surface)

Packed row-order restore -> ONE global inverse-permutation gather:
per chunk keep the SORTED group output + (row_order + chunk offset);
after the chunk loop, concat once and `fnp.take(full_sorted,
fnp.argsort(global_order), axis=0)`. Removes the per-chunk empty_like +
put_along_axis scatter entirely (-2.68 GB/net, copy census re-measured:
8.09 -> **5.42 GB/net**). Zero-activation chunks get identity arange
orders.

## Gates

- Bit-identity: np.array_equal TRUE vs pristine algo21 on 3 mini nets,
  full (32,256) outputs (pure permutation-gather, exact by construction).
- Deterministic flops: +0.028% (the added global argsort; guaranteed cost
  traded against the residual hypothesis).
- whest validate PASS; subprocess seed-42 n=3: raw 4.26853e-7, flops
  259,048,255,088 (= algo21's 258,975,663,194 +0.028% exactly), local
  residual 3.48s vs 3.81s (local residual does not transfer; not
  evidence either way).

## Pre-registered leaderboard expectation (if/when submitted)

- raw: IDENTICAL to 315892 (3.72500e-7).
- multiplier: if scatter+alloc copies are priced like concat copies
  (~1.5-2.4 FLOP-eq/B), expect -4% to -6.5% adjusted (1.3377e-7 ->
  ~1.25-1.28e-7) minus the +0.028% flops. If ONLY concats are priced and
  put_along/empty_like are backend-free, expect ~FLAT (+0.03%) - that
  outcome would itself calibrate the scatter class for free.
- Failure mode watch: fnp.arange / int-array add on RemoteArray
  (316255-class API mismatch risk) - the ops used are all in the
  estimator's existing vocabulary except fnp.arange(start, stop); local
  validate cannot rule out RemoteArray quirks.
- Package the FOLDER (whest-packaging-file-vs-folder-trap).

## Row-blocked pipeline BUILT AND GATED (estimator_rowblock.py, late 07-14)

Post-316368 recalibration (scatters free, concats = the whole residual)
made this the live lever; built as a variant file (root stays pristine
algo21). Design keys:
- The pipeline threads x as (top, bottom) = the antithetic (+half, -half)
  blocks, and the block boundary IS the one-level Strassen row split:
  `_dense_matmul_2blk` consumes both blocks and returns (top, bottom)
  UNASSEMBLED — same quadrant slices, same 7 products, so the dense path
  is arithmetically the baseline's; only the axis=0 assembly concat
  disappears. Row-blocking alone would NOT have saved anything (each
  block's internal Strassen would still concat its own output).
- Packed path: single GLOBAL group concat + one inverse-permutation
  gather (algo27 machinery) replaces per-chunk group concat + chunk
  concat (+ the known-free scatter).
- Probes read x[:rows] with rows <= 0.2n -> served bit-identically by the
  top block. Sample blocks stay unconcatenated ((half,-half) from
  _sample_block; layer-0 relu pair kept as blocks).

Gates:
- Copy census (priced concat class): 5.40 -> 2.94 GB/net (-46%).
  Remaining: one packed global concat ~1.29 GB (minimal: one concat is
  structurally required), Strassen axis=1 quadrant concats ~1.15 GB
  (removable only by column-blocking — invasive, not attempted), odd-out
  edge ~0.42 GB, tinies.
- Fleet raw, 16 mini nets: -0.0063% (fp-noise class as designed:
  fire-rate tie flips + chunk-boundary Strassen-eligibility shifts in
  packed fallbacks; per-net range +/-0.07%).
- Fleet flops: +0.0256% (the argsort-for-scatter trade, algo27-class).
- whest validate PASS (incl. tiny 2x4 shapes); subprocess seed-42 n=3
  CLEAN 3/3, raw 4.26914e-7 (baseline 4.26878e-7, +0.009% = noise class).

## Assessment: can it improve?

YES, conditionally. Expected adjusted: -2.46 GB/net of priced concat
bytes x ~2.2 FLOP-eq/B (recalibrated) = ~5.4e9 FLOP-eq = -5.5% best case;
-3% if the effective concat price is nearer 1.5/B; floor ~flat+0.03% if
the 316368-flat result generalizes to ALL copy classes (i.e. concats are
also cheaper than 315898 suggested — possible but then 315898's +1.72%
regression needs another explanation, so unlikely). Risks: (1) uses only
grader-proven op vocabulary; (2) raw moves are fp-noise, fleet-neutral;
(3) NOT bit-identical — a knife-edge N flip is possible on some net
(same accepted class as algo26); (4) DO NOT submit until the evaluator
is healthy (scan last ~10 IDs first) — today it is down.

## Remaining copy targets (scoped, not built)

- Strassen assembly 2.3 GB/net: needs a row-blocked sample pipeline (keep
  x as top/bottom halves through matmul/relu/mean, which are all row-wise;
  kills the axis=0 assembly concat everywhere). Invasive; separate
  experiment AFTER this change is leaderboard-priced.
- Odd-out-width edge concat 0.42 GB/net: pad w with one zero column
  pre-Strassen (KB-scale concat), slice result free; ~wash on flops.
- One-change-per-submission discipline: ship the gather-restore alone
  first; its result prices the scatter class and decides if the Strassen
  restructure is worth building.
