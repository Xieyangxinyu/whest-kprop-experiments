# scipy Sobol `optimization` argument probe — 2026-07-19

## Question

Does `scipy.stats.qmc.Sobol(..., optimization=...)` ("random-cd" centered-
discrepancy minimization; "lloyd" centroidal-Voronoi relaxation) improve
fleet final-layer MSE over plain scrambled Sobol for the WHest artifact?

## Feasibility (d=256)

- `lloyd`: INFEASIBLE, structurally. scipy's implementation runs qhull
  Voronoi: needs >= d+2 = 258 points to initialize (QH6214 at n=256) and
  int-overflows at n=1024 ("QH6235 ... Did int overflow due to high-D?").
  Exponential-in-dimension geometry; no artifact-sized run can complete.
- `random-cd`: works; ~88 s at n=4,096 x d=256 (roughly linear in n;
  ~10-15 min for an artifact-sized 16k-30k set). Moves ~91% of rows.

## MSE probe — paired, seeds 1000-1003, n_half=4,096 (+antithetic), 12 nets, GT 2^20

| seed | base fleet MSE | random-cd | delta |
|---|---|---|---|
| 1000 | 3.4868e-6 | 4.6111e-6 | +32.2% |
| 1001 | 4.1275e-6 | 3.8795e-6 | -6.0% |
| 1002 | 4.8517e-6 | 4.3949e-6 | -9.4% |
| 1003 | 3.0426e-6 | 2.5532e-6 | -16.1% |

Paired delta: **+0.18% mean, SEM 10.9%, t = +0.02** — perfect null.
Base-arm realization band: 46.7% of mean.

## Read

random-cd moves ~91% of the points, so "optimizing" a scrambled set is
effectively a fresh realization draw: per-seed deltas (-16%..+32%) are as
wide as the realization-luck band itself, with zero systematic component.
Centered discrepancy targets uniformity criteria that pay off for smooth
integrands; this integrand is rough/high-effective-dimension (rho^2=0.06,
variance-reduction-tapped), the same reason power-of-2 net balance bought
nothing (317412/317415). Raw npz: sobol_randomcd_probe_2026-07-19.npz.

## Decision

DEAD. Do not build artifacts with `optimization=` args: lloyd cannot run
at d=256; random-cd is a ~10-min-per-artifact re-roll of the +/-30-50%
lottery with no expected gain. The shipped 30,720-half realization stays.

## Addendum: unscrambled Sobol (same day)

Same protocol (n_half=4,096 +antithetic, 12 nets, GT 2^20), one
deterministic unscrambled set (origin point dropped) vs the four
scrambled realizations:

- Unscrambled fleet MSE 8.908e-6 vs scrambled mean 3.877e-6
  (**+130%**), vs best scramble 3.043e-6 (**+193%**).
- Per-net it is catastrophic where it is bad: nets 2/5/7 at
  2.2-2.4e-5 (~5x the scrambled mean) — classic unscrambled-Sobol
  pathology (correlated dyadic structure, poor low-dim projections in
  high d), which Owen scrambling exists to fix.
- Raw npz: sobol_unscrambled_probe_2026-07-19.npz.

DEAD. Scrambling is load-bearing for this artifact. Sobol-construction
family fully closed 2026-07-19: prefix length (offline + 317412/317415),
optimization args (lloyd/random-cd), and scramble on/off are all
resolved in favor of the shipped 30,720-half scrambled realization.
