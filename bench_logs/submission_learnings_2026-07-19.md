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
