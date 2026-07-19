# Power-of-2 Sobol prefix probe — 2026-07-19

## Question

Scrambled-Sobol prefixes are balanced (t,m,d)-nets only at power-of-2
lengths; the shipped Algorithm 31 prefix is 30,720 half-samples = 2^11*15
(x2 antithetic = 61,440). Hypothesis: a 2^14 = 16,384 or 2^15 = 32,768
half-sample prefix improves RAW final-layer MSE (accuracy, not just
MSE-per-FLOP) via the digital-net balance property.

## Method

`scripts/power_of_two_sobol_probe.py`; raw tensor saved to
`bench_logs/power_of_two_sobol_probe_2026-07-19.npz` (mse[scramble, net,
checkpoint]).

- 12 He-init 256x32 nets; fleet-mean final-layer MSE (matches scoring).
- 8 independent scrambled-Sobol realizations (scipy qmc.Sobol, seeds
  1000-1007), Gaussianized via ndtri, SAME 8 point sets across nets
  (mirrors the fleet sharing one artifact). Antithetic (x, -x) exactly
  like the estimator; the ship merge is bit-equivalent to a plain prefix
  mean, so pure forwards are a faithful proxy (same proxy as the archive
  QMC probes).
- Cumulative-prefix evaluation: one forward pair per (net, scramble)
  yields MSE at all 17 checkpoints, including +/-256 brackets around
  2^14 and 2^15.
- GT: 2^21 = 2,097,152 MC samples/net, float64 accumulation; GT noise
  floor measured (fleet mean 2.9e-8, ~5% of MSE at n=30,720 — additive
  constant across n, does not affect dip tests).

## Result — FALSIFIED

Fleet MSE (mean over 8 scrambles, SEM over scrambles):

| n_half | total | MSE | SEM |
|---|---|---|---|
| 16,128 | 32,256 | 9.324e-7 | 6.6e-8 |
| 16,384 (2^14) | 32,768 | 8.879e-7 | 6.2e-8 |
| 16,640 | 33,280 | 8.750e-7 | 6.4e-8 |
| 30,720 (SHIP) | 61,440 | 5.309e-7 | 2.4e-8 |
| 32,512 | 65,024 | 5.223e-7 | 2.8e-8 |
| 32,768 (2^15) | 65,536 | 5.133e-7 | 3.0e-8 |

- **No local minimum at either power of 2**: MSE at 16,640 is LOWER than
  at 16,384; the 30,976→32,768 region decreases smoothly ~1/n. The
  bracket checkpoints rule out a sharp balance dip directly.
- Paired dip tests (per-scramble, neighbors 1/n-rescaled):
  2^14 -1.78% (t = -0.89), 2^15 -0.99% (t = -1.19). Both sub-noise.
- Efficiency n*(MSE-floor) is flat-ish (0.0141-0.0167) across
  8k-33k with no power-of-2 structure; ~1/n scaling holds, consistent
  with the known rough/high-effective-dimension integrand
  (variance-reduction-tapped: rho^2=0.06 with smooth surrogates). The
  Sobol/antithetic gain (~0.54x vs MC) comes from low-order projections
  any prefix captures; full-net equidistribution adds nothing measurable.
- **Realization luck dominates**: fleet-mean MSE spread across the 8
  scrambles at fixed n is 33% (n=30,720) to 53% (n=32,768) of the mean.
  Any <=2% balance effect is ~20x smaller than the artifact-realization
  band. Re-confirms the 07-18 lesson (317172/317185): N-rule and
  artifact changes move raw through prefix realization luck first.

## Ship implications

- Do NOT rebuild the artifact to a power-of-2 length. Expected gain
  <=1-2% (not significant), and a fresh artifact re-rolls a +/-30-50%
  fleet realization band, abandoning the leaderboard-validated 30,720
  realization (317197, adjusted 1.3086e-7). The original scramble seed
  is unrecoverable, so extending the shipped realization to 32,768
  same-scramble halves is impossible anyway (the fp16 extension tail is
  a different scramble — a mixed-scramble 2^15 prefix has no net
  property at all).
- Raising total to 65,536 also fails economics independently: reducible
  raw -6% at multiplier +6.7% — the known flat-bowl wash, plus the
  cap->61,440 durable kill.
- Dropping to 2^14 halves (32,768 total) is priced by the same bowl:
  raw +67% reducible for multiplier -47%-class — the 317172 same-day
  pair already showed fixed-61,440 beats shorter prefixes adjusted.

Bottom line: prefix LENGTH balance is not a lever for this integrand.
The 30,720-half shipped realization stays.
