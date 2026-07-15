# Full-cov anchor re-evaluation under the 0.1 multiplier floor - 2026-07-14

## Scope

User-directed: re-open the full-cov anchor (killed 2026-07-08 on +30%
multiplier) under floor economics — "extra analytic compute can be partially
or fully free near the 0.1 floor" — on today's efficiency surface
(algo21/315892 constants; complex packing excluded). Motivated by the
leaderboard frontier (adjusted 3.8e-8 at mult 0.13, see
submission_learnings_2026-07-14.md "Competitive landscape").

Script: session scratchpad `fullcov_floor_sweep.py` (results JSON alongside).
40 mini nets, GT = dataset `final_means`. Antithetic plain-MC emulation of
the sampled final row; linearized-gate full-cov anchor rebuilt from
[[full-cov-anchor-result]] spec (sources had been deleted; only .pyc remained).

Pre-registered bars (in script docstring before running): KILL if the
oracle-weighted floor-aware optimum improves adjusted <5% vs 1.3377e-7;
SHIP-CANDIDATE if a deployable weighting improves >=15%.

## Sanity

Rebuilt anchor reproduces the 07-08 numbers on fresh nets: full-cov-lin
final-mean MSE 7.39e-5, diagonal 9.10e-4, ratio 0.081 (memory: 8.37e-5,
ratio 0.083). The rebuild is faithful.

## Result 1 - the skew hypothesis is FALSE

The floor case needed per-neuron anchor error to be heavily skewed (most
final neurons anchor-exact) so shrinkage could replace samples at low N.
Measured (mean final-layer MSE, 40 nets, cross-fitted weights):

| N | pure MC | oracle tau2 | global tau2 | alpha-binned tau2 |
|---|---------|-------------|-------------|-------------------|
| 1920 | 1.88e-5 | 1.22e-5 (g=0.65) | 3.05e-5 (WORSE) | 2.02e-5 (WORSE) |
| 3840 | 1.21e-5 | 9.96e-6 (g=0.82) | 2.16e-5 | 1.44e-5 |
| 7680 | 5.94e-6 | 5.73e-6 (g=0.97) | 1.40e-5 | 8.38e-6 |
| 30720 | 1.48e-6 | 1.34e-6 (g=0.91) | 3.50e-6 | 1.82e-6 |
| 61440 | 7.66e-7 | 7.16e-7 (g=0.93) | 1.35e-6 | 7.84e-7 |

- ORACLE per-neuron weights (unachievable upper bound): 35% variance cut at
  N=1920, decaying to ~4-9% for N>=7680. Consistent with the 07-08
  integration's +7% at full N — the whole chain reconfirms.
- DEPLOYABLE weightings (global, alpha-binned) are WORSE than pure MC at
  every N: per-neuron anchor error is too heterogeneous, and misweighted
  neurons cost more than well-shrunk neurons save.
- Mean per-neuron anchor error (7.4e-5) exceeds even 1920-sample MC noise
  (1.9e-5): the anchor is on average worse than 2k samples.

## Result 2 - floor-aware optimum (oracle g(N), measured cost constants)

score(N) = max(0.1, (F0 + dC + F1*N)/B) * (a + g(N)*b/N), with B=2.72e11,
F1=1.745e6, resid 1.9e10 at N_ref=45k, a=2.527e-8, b from raw 3.725e-7.

| config | opt N | mult | adjusted | vs 1.3377e-7 |
|---|---|---|---|---|
| baseline, resid fixed | 61440 | 0.468 | 1.308e-7 | -2.2% |
| baseline, resid~N | 17113 | 0.140 | 1.315e-7 | -1.7% |
| anchor @ nominal 2.1e9, resid fixed | 61440 | 0.476 | 1.249e-7 | -6.6% |
| anchor @ nominal, resid~N | 30899 | 0.258 | 1.248e-7 | -6.7% |
| anchor @ MEASURED 3.5e10 cost | 61440 | 0.622 | 1.634e-7 | +22.2% |

- The knee (mult 0.14) is only ~2% better than the shipped point — the
  score surface is nearly FLAT in N (variance regime), so "free compute at
  the floor" buys almost nothing on this surface.
- Best case for the anchor: -6.6%, requiring ORACLE per-neuron weights
  (deployable is negative), the NOMINAL cost (17x below the once-measured
  integration cost), and plain-MC transfer (our Sobol b is ~2x lower, which
  shrinks the anchor's relative value further).

## Verdict: KILL confirmed, now including the floor variant

Oracle ceiling 6.6% < any shippable margin; deployable weighting < 0.
The mechanism is sharper than in 07-08: (1) anchor value concentrates at
N<4k (g=0.65 at 1920), but our F0 (residual ~1.9e10 = 7% of B) puts the
floor knee at N~17k where anchor value is ~4%; the knee and the anchor's
sweet spot DO NOT OVERLAP. (2) The reopening condition is now explicit:
anchor-shrinkage becomes relevant only if per-sample cost F1 drops ~10x
(knee at N~2k) — i.e., a structurally cheaper sampler, which is the same
thing the leaderboard frontier implies. The anchor is a follower bet, not
a lever.

## Durable notes

- The public dataset (`prepared/mini`, 100 nets) carries `final_means` GT —
  offline sweeps need no GT regeneration.
- scipy must be `uv pip install`ed per-session for offline scripts (not in
  requirements; estimator itself is scipy-free).
- estimator.py reverted to algo21/315892 surface this session (byte-match
  vs the 315892 tarball); algo26 shelved as dead end (user call).
