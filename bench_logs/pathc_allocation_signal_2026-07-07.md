# Path C (measured-variance allocation) Signal Test - 2026-07-07

## Idea

Algorithm 15 sets N up front from the analytic final variance:
`N_i = clip(49152*sqrt(V_anal_i/0.02143), 30720, 61440)`. Path C runs the 30720
base block first, MEASURES the empirical final variance, and sets the
continuation length from that. Premise: the measured signal beats the analytic
proxy, so allocation improves at equal compute.

## Command

`uv run python scripts/pathc_allocation_signal.py` — pure NumPy, 40 He-init
256x32 nets, full forward. Per net: `Q_true` = mean_j Var(relu_final_j) from
~393k MC (truth / reducible coefficient), `V_anal` (moment propagation, current
signal), `V_emp30` (30720 antithetic Sobol block, path C signal). Allocation
objective = mean_i(Q_true_i / N_i) at EQUAL mean compute; oracle = allocate on
Q_true.

## Result

Signal quality vs Q_true:
- `V_anal`  : Pearson(sqrt) 0.917, Spearman 0.896; biased LOW `V_anal/Q` mean
  0.491 +/- 0.104 (analytic var ~= half true reducible var, absorbed by anchor).
- `V_emp30` : Pearson(sqrt) 0.9999, Spearman 0.999; `V_emp30/Q` mean 1.001 +/-
  0.005. Essentially IS Q_true.

Allocation objective at mean N = 52216:
| rule | mean(Q_true/N) | vs current |
|------|----------------|-----------|
| equal-N (const)  | 1.12665e-6 | +4.2% |
| V_anal (current) | 1.08084e-6 | baseline |
| V_emp30 (path C) | 1.07290e-6 | **0.9927x (-0.73%)** |
| Q_true (oracle)  | 1.07290e-6 | 0.9927x (-0.73%) |

## Interpretation

Path C's premise is CONFIRMED (measured var is a near-perfect signal, analytic
is biased/noisier) but the WIN IS IMMATERIAL. The variance-allocation ceiling
(equal-N -> oracle) is only ~4.2%, and the current sqrt(V_anal) rule already
captures ~85% of it; path C reaches the full oracle but that is just -0.73% on
the REDUCIBLE part. On total adjusted score (bias floor unaffected, reducible
is a fraction of raw MSE) this is <~0.5%, i.e. ~<7e-10 on a 2.04e-7 score --
below the meaningful-delta threshold -- and it RE-ADDS the final-variance
measurement compute that submission 315105 removed for a gain. Net negative.

Spillover: this also bounds paths A/B's VARIANCE lever and any ML-on-variance
allocation -- perfect variance knowledge is worth <1%. The current sqrt(V) rule
is near-optimal for variance-driven allocation.

## Decision

REJECT path C: confirmed signal, immaterial payoff, re-adds removed compute.
Do NOT pursue better variance signals (measured or learned) for allocation --
ceiling < 1%. The ONLY allocation lever this test does NOT bound is the
BIAS-FLOOR term (optimal N ~ sqrt(b/a); paths A/B divide by the irreducible
per-net bias a_i, which N cannot reduce). Test a bias-aware oracle before
implementing any allocation change. Kept `scripts/pathc_allocation_signal.py`.

---

# Bias-Aware Allocation Oracle - 2026-07-07 (follow-up)

## Idea

Path C bounded the VARIANCE signal (raw MSE, equal compute). The bias-aware
question is different: optimal N ~ sqrt(b*f0/(a*g)) optimizes the ADJUSTED
score (bias floor a_i x compute multiplier). Sample bias-limited nets LESS.

## Command

`uv run python scripts/bias_aware_oracle.py` — REAL Estimator, 10 He-init nets,
per net fit MSE(N)=a+b/N (rep-averaged R=10 at N=30720/61440 vs 300k-MC GT) and
eff(N)=f0+gN (tracked flops). Then mean adjusted under current sqrt(V) vs
per-net oracle, swept over a fixed-cost f0 band (residual wall time is not
measurable offline).

## Result

Per-net structure (clean): Pearson(a,b) = -0.47, Spearman -0.59 (NEGATIVE:
high-variance nets have LOW bias floor). Bias-floor share at N=61440 = 0.30
(30% of final MSE is irreducible). a in [0, 1.1e-6], b in [0.009, 0.072].

Raw oracle looked large (x0.86 = 14% at tracked f0) BUT decomposition shows it
is a compute-LEVEL effect, not allocation:
| f0_extra | current | best-LEVEL sqrtV (rescaled anchor) | full oracle | SPREAD-only |
|----------|---------|-----------------------------------|-------------|-------------|
| 0        | 6.10e-7 (N55k) | 5.26e-7 x0.862 (N32k) | 5.23e-7 x0.857 (N37k) | **x0.9942** |
| 1e10     | 6.49e-7 | 5.84e-7 x0.899 | 5.68e-7 x0.876 | x0.9736 |
| 3e10     | 7.26e-7 | 6.96e-7 x0.958 | 6.56e-7 x0.903 | x0.9419 |

## Interpretation

The bias-aware SPREAD gain (oracle vs best-LEVEL sqrt(V), compute-matched) is
only **0.58%** at the realistic f0 (tracked ~= real, since synthetic tracked
g*meanN 1.55e11 ~= real effective 1.47e11). Because a and b are NEGATIVELY
correlated, the current sqrt(V) rule already aligns with the bias-aware optimum
(high-b=low-a nets get more samples, correctly) -- nothing to redistribute.

The large apparent gain is entirely a compute-LEVEL change (sample LESS), which
(1) directly CONTRADICTS leaderboard evidence -- 315123 raised the anchor
40960->49152 and improved on public, 315074 found more compute helped -- the
classic synthetic-does-not-transfer gap, and (2) hinges on offline-unmeasurable
residual/f0. Not actionable.

## Decision

REJECT bias-aware allocation. Allocation is TAPPED: variance signal <1% (path C)
and bias-aware spread 0.58%, both sub-noise and compute-matched. The current
sqrt(V) allocation SHAPE is essentially optimal. Do NOT pursue ML/adaptive
allocation further. The real headroom is the BIAS FLOOR ITSELF (30% of final
MSE) -- reduce a_i via better final-row estimation (shrinkage / on-neuron
moment correction), which is NOT an allocation problem. Kept
`scripts/bias_aware_oracle.py`.

---

# Bias-Floor Correction Tests - 2026-07-07 (closes the thread)

The path-C / bias-aware-oracle sections above concluded the ONLY untapped lever
is the BIAS FLOOR itself (reduce a_i via better final-row estimation:
shrinkage / on-neuron moment correction). Both concrete forms were tested
directly against the 100 public mini MLPs (exact GT). Both REJECTED.

## Idea B - Learned cross-net bias prior (shrinkage_prior_prototype.py)

Fit g(alpha)=E[delta/sigma_z | alpha] on TRAIN nets, apply dhat=g(alpha)*sigma_z
to HELD-OUT nets, 2-fold disjoint validation (the guard 315076/315118 skipped).
Command: `uv run python scripts/shrinkage_prior_prototype.py` (real Estimator,
100 nets, 6 reps).

Result:
- realized final MSE = 8.95e-7; systematic bias floor ||delta_sys||^2/total =
  **0.186** (with on-moment correction on). Bias^2 lives on the on-block: dead
  1.3% / kink 16.2% / **on 82.5%** (matches notebook section 4). On-block signed
  delta = -2.13e-6 (under-prediction, matches section 10).
- in-sample alpha-curve R^2 on delta_sys = **0.004** -> NO cross-net alpha
  structure.
- 2-fold held-out corrected/current MSE ratio = **1.0000 / 1.0000** at every
  lambda -> the prior does literally nothing out-of-sample.
- Verdict: the bias floor is real (18.6%) and on-block-concentrated, but it is
  PER-NET IDIOSYNCRATIC, not a learnable function of alpha. Exactly what
  notebook section 6 predicted. Do NOT retry alpha/sigma-keyed learned priors.

## Idea A - On-moment fold correction (_ON_MOMENT_CORRECTION)

Replace the linear on-block fold E[relu(z)]~=E[z] with the exact rectified
Gaussian E[relu(z)]=mu*Phi(a)+sigma*phi(a), a=mu/sigma, sigma=analytic diagonal
sigma_pre. Analytic, ~0 FLOPs, pointwise over final on-set. A/B via env toggle
`WHEST_ON_MOMENT` on the 100 public mini MLPs, subprocess runner, back-to-back
(WHEST_ON_MOMENT=0 == committed 315204).

Result (predict() is deterministic given the Sobol file, so delta is exact):
| metric | base (315204) | variant | delta |
|--------|---------------|---------|-------|
| raw final_layer_mse       | 4.39193e-7 | 4.39299e-7 | **+0.024% (WORSE)** |
| adjusted_final_layer_score| 2.39308e-7 | 2.39564e-7 | +0.107% (worse) |
| flops_used                | 1.3919166e13 | 1.3919167e13 | +1187 (~0, as predicted) |
| n_failed                  | 0 | 0 | - |

- The correction is ~free in compute but moves raw MSE the WRONG way. The
  diagonal-analytic sigma miscalibrates the truncation term; the sampled linear
  fold is already the better estimate (notebook section 7: "the fold is
  well-placed", clipping is not the driver). Echoes 315118's lesson: nearly-free
  signed on-block corrections do not transfer.
- Verdict: REJECT. Reverted estimator.py to the clean 315204 surface. Do NOT
  submit. Kept scripts/shrinkage_prior_prototype.py and relu_moment_propagation.py.

## Combined decision

The bias floor is now TAPPED alongside allocation and QMC geometry. Both direct
correction forms (learned prior; analytic moment fold) fail on the strongest
signal (public mini, exact GT, deterministic). The 315204 estimator sits at/near
a genuine local optimum for this active-set-fold family. Stop spending
submissions on bias/variance micro-tweaks. The only remaining structural bet is a
fundamentally different anchor (e.g. full-covariance moment propagation,
relu_moment_propagation.py) -- a larger, compute-costly effort that needs its own
scoping, not a micro-variant.
