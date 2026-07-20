# Submission learnings — 2026-07-19

## Submission Log

### Submission 317412 - Algorithm 32 power-of-2 (N=65,536, fresh seed-1005 artifact)

- Result: regressed
- Change: algo31 surface with `_TOTAL_SAMPLES = 65,536` (2^16 = 32,768 = 2^15
  Sobol half-samples x 2 antithetic) and a FRESH single-scramble artifact
  (scipy `qmc.Sobol(d=256, scramble=True, seed=1005)`, ndtri-Gaussianized,
  bit-verified pre-submit). Seed 1005 was the best of the 8 realizations in
  the same-day offline probe (`power_of_two_sobol_probe_2026-07-19.md`,
  fleet MSE 3.72e-7 vs 5.13e-7 scramble-mean at 32,768 halves). Deliberate
  leaderboard test AGAINST the probe's falsification (user direction).
- Local expectation: probe said no power-of-2 balance dip (<=2%, t<1.2) and
  a +/-30-50% realization-luck band on any fresh artifact; local paired
  seed-42 3-MLP run vs algo31: raw -1.07%, multiplier +3.64%, adjusted
  +0.69% (worse). Validate/local/subprocess all clean, raw bit-identical
  across runners.
- Leaderboard/public evidence: adjusted 1.5723e-7 (+20.2% vs 317197's
  1.3086e-7), raw 3.3814e-7 (+13.6% vs the 317172/317197-class raw
  2.9772e-7), implied multiplier 0.46498 (+5.4% vs 0.44127, predicted +6.7%
  from the sample raise; cross-day pricing caveat applies to the adjusted
  comparison but NOT to raw). Graded cleanly — no evaluator failure at
  65,536 total samples (the 316258/9 failure at 122,880 does not extend
  down to 65,536).
- Decision: do not retry. Ship surface stays Algorithm 31 / 317197
  (repo estimator.py unchanged).
- Lesson: leaderboard confirms the probe on both axes — no power-of-2
  balance gain, and realization luck dominates: the best-of-8-offline
  scramble landed +13.6% raw WORSE on the grader fleet than the shipped
  realization despite 6.7% MORE samples. Offline best-of-K realization
  selection (He-init proxy nets) does NOT transfer to the grader fleet;
  a fresh artifact is a raw-score lottery ticket with ~1x odds. Only
  incidental value: 65,536-sample cap headroom is now leaderboard-tested.

### Submission 317415 - Algorithm 33 power-of-2 prefix truncation (N=32,768)

- Result: regressed (expected); balance-dip instrument came back NEGATIVE
- Change: algo31 surface + original shipped artifact, `_TOTAL_SAMPLES =
  32,768` (2^15 total = 16,384 = 2^14 half-sample PREFIX of the
  leaderboard-validated 30,720-half realization; artifact bytes identical,
  truncation at read time). Clean power-of-2 instrument with no
  realization-lottery confound, complementing 317412.
- Local expectation: no balance dip (offline probe), adjusted worse via the
  adjusted(N) bowl; raw ~2x-class up, multiplier ~-47%-class. 2-MLP smoke:
  raw +61%, flops -46%, no failures.
- Leaderboard/public evidence: adjusted 1.6273e-7 (+24.4% vs 317197), raw
  6.5810e-7, multiplier 0.2473 (-44% vs 0.44127). KEY: raw is +18-20%
  ABOVE the smooth 1/n extrapolation from the ship point (reducible
  2.837e-7 x 1.875 + bias floor 0.14e-7 = 5.46e-7 predicted vs 6.58e-7
  measured) and above the offline probe's 1.67x fleet-mean ratio. Not
  only is there no (t,m,d)-net balance dip at 2^14 halves on the grader
  fleet -- the 16,384-prefix of the shipped scramble is an UNLUCKY draw
  at that length (prefix-luck varies WITHIN a scramble across lengths).
- Decision: do not retry; power-of-2 family closed in both directions.
  Same-day pair 317412 (1.5723e-7) vs 317415 (1.6273e-7): raw ratio 1.95
  at N-ratio 2.0 — plain 1/n, no net-balance structure anywhere.
- Lesson: power-of-2 hypothesis is now leaderboard-falsified in both
  directions (fresh-seed 2^15-half re-roll AND same-scramble 2^14-half
  prefix). Two submissions confirm the offline probe exactly; stop
  spending submissions on prefix-length/balance ideas.

## Pilot-probe variant sweep (offline, seed 42, 10 MLPs, paired vs algo31)

Variants on the algo31 surface; classification deterministic at fixed
seed so paired deltas are exact. Files: scratchpad pilot_*.json (session
f6395794); variants in pilot_variants/.

- P1 antithetic (both halves, same 512/2048 rows): raw +0.06% (wash,
  per-MLP ±0.7%, all 10 MLPs' decisions changed), flops +0.32%, local
  adjusted -2.4% carried by residual wall (contention-contaminated).
  SHIPPED as 317421 (see below).
- P2 full-block (single-stage, all 5,120 halves): raw +0.31% with a
  +5.6% outlier (trevor-johnston), flops +1.04%. DEAD — more probe
  rows never pay.
- P3 strided even-spread (same 512/2048 counts, every-10th-row): raw
  +0.32%, scatter to +5.2% (eric-sullivan), flops +0.01%. DEAD —
  confirms Sobol-prefix balance is why prefix probes work; even-spread
  destroys it at identical cost. Directly falsifies the "distribute
  pilot rows evenly across the sequence" idea.
- P4 halved (2.5%/10% -> 256/1,024 rows): raw +0.99% with outliers
  +7.4% (trevor-johnston) / +3.3% (shawna-wilson), flops only -0.18%.
  DEAD - the raw cost is ~5x the flop saving; the current 512/2,048
  sizes sit at (or below) the minimum safe probe size, not above it.

Structural takeaway: pilot classification decisions are near-optimal at
current sizes — sharper probes change decisions without net raw gain,
reshaped probes hurt. Pilot samples ARE fully reused by the mean
estimate (prefix reads, nothing consumed); the only found slack was the
positive-half-only asymmetry (P1).

### Submission 317421 - Algorithm 34 antithetic pilot probes

- Result: improved (nominal new best), with a cross-window pricing caveat
- Change: algo31 surface + `_sample_alpha` reads BOTH antithetic halves
  (first 512/2048 rows of x[0] AND matching x[1] rows, concatenated);
  4 call sites pass the 2-tuple. Same artifact, N=61,440, routing.
- Local expectation: raw wash (+0.06%), flops +0.32%, grader tie to
  +0.3% unless the local residual-wall gain survived machine-speed flip.
- Leaderboard/public evidence: adjusted 1.3028e-7 (-0.44% vs 317197's
  1.3086e-7), raw 2.9758e-7 (-0.05% vs 2.9772e-7 — wash, exactly as
  predicted), multiplier 0.43780 (-0.79% vs 0.44127). The multiplier
  drop despite +0.32% flops means lower grader-side residual wall OR
  a global repricing window (the observed 07-18 drift was ±1.55% on
  identical bytes — cross-window adjusted comparisons are invalid per
  grader-pricing-divergence). Raw comparison is pricing-free and
  confirms the local wash.
- Decision: nominal leaderboard best. To CONFIRM the multiplier gain is
  the change and not repricing, a same-day pair (re-grade of algo31
  bytes today) would be needed — one submission slot. Ship-surface
  promotion deferred to that call.
- Lesson: probe residual-wall effects DO transfer to the grader with
  sign preserved this time (unlike 315844/315892 where the trade
  flipped); antithetic pilot is free-to-slightly-positive. The 07-19
  power-of-2 line (317412/317415) plus this show: artifact/N changes
  move raw through luck; probe-side changes move only the multiplier.

### Submissions 317455/317456/317459 - ReLU/Scatter FLOP accounting probes

- Result: mixed; where-threading confirmed, put-scatter falsified.
- Change: 317455 used `fnp.put` for `_scatter` only; 317456 used
  `fnp.put` plus ReLU masks threaded through `fnp.where`; 317459 used
  where-threading only and kept the original eye-matmul `_scatter`.
- Local expectation: `fnp.put` and `fnp.where(mask, ...)` are 0-FLOP in
  flopscope, but backend/residual overhead was noisy locally.
- Leaderboard/public evidence: 317455 regressed to adjusted 1.319e-7,
  raw 2.98e-7, budget 44.43%; 317456 regressed to 1.310e-7, raw
  2.98e-7, budget 44.07%; 317459 improved to 1.285e-7, raw 2.98e-7,
  budget 43.28%, effective compute 1.18e11 vs 317421's 1.303e-7,
  budget 43.91%, effective 1.19e11.
- Decision: keep where-threading without `fnp.put`; reject put-scatter
  despite 0 tracked FLOPs because its backend/residual overhead dominates.
- Lesson: zero-FLOP data movement is not automatically score-positive;
  use it only when it also removes a charged operation without adding a
  costly backend path. The original eye-matmul scatter is faster on the
  grader than `zeros + put`.

### Submission 317460 - where-threading plus redundancy cleanups

- Result: rejected before scoring (smoke test ESTIMATOR_EXCEPTION)
- Change: 317459 where-threading surface plus exact-output cleanup of unused
  negative sample-block half, unused final variance/`final_var_mean`, unused
  intermediate dead-correction/`mc_rows` zeros, redundant intermediate-row
  combine, rows-only extra block, and closed-form layer-0 analytical moments.
- Local expectation: analytically strict compute improvement over 317459 with
  identical final-layer MSE, but tiny tracked-FLOP savings (~7.9M/MLP) relative
  to residual timing noise.
- Leaderboard/public evidence: smoke failed with `IndexError` in
  `alpha_rows[layer_idx]` because the cleanup dropped final-layer alpha and
  assumed final sampled rows were only `layer_idx >= 30`; smoke MLPs can have
  shallower depth.
- Decision: fixed in 317462; keep 317459 as baseline until fixed retry grades.
- Lesson: even depth-32-specialized leaderboard code must preserve the general
  `predict()` contract for grader smoke MLPs; keep layer-indexed lists length
  `mlp.depth` unless every read is guarded, and always materialize
  `layer_idx == mlp.depth - 1`.

### Submission 317462 - smoke-fixed where-threading cleanup stack

- Result: regressed
- Change: same as 317460, with smoke fixes: keep `alpha_rows` length equal to
  `mlp.depth` and materialize the actual final layer row for shallow MLPs.
- Local expectation: same exact-output public-depth behavior as 317460, but now
  smoke-safe; validated locally on synthetic depths 1, 2, 3, 4, 8, 32 and
  subprocess seed42/n1.
- Leaderboard/public evidence: adjusted 1.309e-7, raw 2.98e-7, budget 44.11%,
  effective 1.20e11. This regressed vs 317459 (1.285e-7, raw 2.98e-7,
  budget 43.28%, effective 1.18e11).
- Decision: reject the broad cleanup stack; keep 317459 as active public-confirmed
  baseline unless a later one-change public probe improves.
- Lesson: low-magnitude exact-output cleanup confirmation; public score may be
  dominated by residual noise despite analytically lower tracked FLOPs.

### Submission 317468 - conservative salvage cleanups

- Result: regressed
- Change: 317459 where-threading surface plus only the smoke-safe salvage set:
  array-only `_sample_block` (no unused `-half` tuple), `sorted_nnz[-1]` instead
  of `fnp.max(nnz_per_row)` after the existing argsort, and removal of unused
  `final_var_mean`. Original scatter, `fnp.var`, list shapes, and row
  bookkeeping are unchanged.
- Local expectation: exact-output; n=3 subprocess vs 317459 saved ~9.77M tracked
  FLOPs/MLP, ~0.073s residual/MLP, and ~7.32B effective compute/MLP; shallow
  smoke depths 1/2/3/4/8/32 passed.
- Leaderboard/public evidence: adjusted 1.311e-7, raw 2.98e-7, budget 44.17%,
  mean effective 1.20e11. This regressed vs 317459 (1.285e-7, raw 2.98e-7,
  budget 43.28%, effective 1.18e11) and roughly tied/regressed with 317462.
- Decision: reject the salvage bundle; keep 317459 as active baseline unless a
  new isolated public test proves otherwise.
- Lesson: even locally isolated lower-FLOP/lower-residual cleanup can fail to
  transfer publicly. For now, submit only one-change isolation artifacts if the
  hypothesis is strong enough; do not bundle exact-output cleanups.
- Component attribution after grading: do not fold this into "array-only is
  bad." Local n=3 array-only had lower FLOPs/wall/residual/effective. The
  unconfirmed bundled pieces are suspect: isolated n=1 `sorted_nnz[-1]` lowered
  tracked FLOPs (~1.9M) but worsened wall/residual/effective, and isolated
  `final_var_mean` removal saved only ~256 FLOPs while worsening residual and
  effective compute. Treat future tests as one-change public probes only.

### Submission 317472 - isolated array-only sample block

- Result: regressed
- Change: 317459 where-threading surface plus only `_sample_block` returning the
  Sobol half array instead of `(half, -half)`; layer 0 indexes `x` directly and
  downstream layers still use the normal `(top, bottom)` tuple after layer 0.
- Local expectation: exact-output and smoke-safe; shallow depths 1/2/3/4/8/32
  passed. Isolated n=3 subprocess vs 317459 had same final MSE with tracked
  FLOPs -7.86M/MLP, wall -1.82s/MLP, residual -0.141s/MLP, and effective
  compute -14.08B/MLP. This isolates the most plausible piece from failed 317468.
- Leaderboard/public evidence: adjusted 1.313e-7, raw 2.98e-7, budget 44.21%,
  effective 1.20e11. This regressed vs 317459 (1.285e-7, raw 2.98e-7,
  budget 43.28%, effective 1.18e11) despite local n=3 improvement.
- Decision: reject array-only as a public-scoring variant for now; do not infer
  from local residual wins alone.
- Lesson: use one-change public probes for micro-optimizations; even the cleanest
  unused-work removal can fail to transfer if the public residual/effective
  accounting window moves differently.

### Local exact bucket-k fold-in check - no submission yet

- Result: inconclusive/hold
- Change: on top of array-only, use `k = min(limit, prev_width)` in packed
  `einsum` groups instead of `_ceil_bucket(limit, 16, prev_width)`.
- Local expectation: strict billed contraction FLOP reduction for the `limit=8`
  bucket; expected fp-order drift but raw MSE unchanged at displayed precision.
- Local evidence: subprocess seed42/n3 vs array-only saved ~55.4M tracked
  FLOPs/MLP and wall time ~0.016s/MLP, but residual rose ~0.0053s/MLP and
  effective compute worsened by ~474M/MLP; final MSE drift was ~4.7e-12.
  Operation profile showed no new calls and `einsum` FLOPs -166M total across
  n=3, so the local loss is backend/residual attribution rather than an added
  code path.
- Decision: do not submit immediately from this machine's n=3 result; consider
  only as an isolated public probe if a second machine or larger fixed-seed run
  confirms effective-compute improvement.
- Lesson: exact bucket-k is analytically clean, but local backend/residual timing
  can flip. Treat it as a contraction-FLOP idea, not a residual-cleanup idea.
