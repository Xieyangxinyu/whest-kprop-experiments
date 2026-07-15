# Cheaper-per-sample forward pass: scoping - 2026-07-14

## Motivation

Leaderboard frontier implies ~3.5x better variance-per-effective-compute
(submission_learnings_2026-07-14.md "Competitive landscape"); the full-cov
floor re-eval (fullcov_floor_reeval_2026-07-14.md) reduced the anchor bet to
a follower on per-sample cost F1 dropping ~10x (knee at N~2k). This doc
scopes F1 reduction candidates. Constraint: no dtype tricks (user).

## Candidate inventory vs prior evidence

| candidate | status | source |
|---|---|---|
| complex/dtype packing | EXCLUDED (user; retired w/ flopscope fix) | dtype-packing memory |
| Strassen 1/2-level (12%/22%) | blocked on organizer ruling; too small anyway | variance-reduction memory |
| layer fusing / on-neuron fold | KILLED (dense fan-out costs a matmul/layer) | matmul-efficiency memory |
| column sparsity, fp16 | KILLED | matmul-efficiency memory |
| better QMC / variance-per-sample | TAPPED (rho^2=0.06 family) | variance-reduction memory |
| uncorrected active-set truncation | leaderboard-REGRESSED 2x (315004/315071) | variance-reduction memory |
| low-rank early layers + analytic bias-difference correction | UNTESTED | this doc |

MLMC kill does NOT pre-falsify the last row: r_V=1.00 killed per-sample
COUPLING; this candidate samples ONLY the cheap net and corrects the MEAN:
  est = MC_mean(cheap net) + [anchor(true) - anchor(cheap)]
Correction is a model-level control variate; anchor systematic error may
cancel in the difference.

## Hypothesis (pre-registered)

Hypothesis: for SVD rank-r truncation of layers 1..30 (layer 0 and final
layer exact), the linearized full-cov anchor difference tracks the true
final-mean difference well enough that the corrected estimator's added
error is small vs the current raw floor, while per-sample priced FLOPs drop
1.5-3.5x.

Prediction: correction residual MSE_r = mean((diff_true - diff_anchor)^2)
<= 1e-7 for some r with F1_r <= 0.6*F1; then floor-aware optimum beats
1.3377e-7 by >=20%.

Baseline: algo21/315892 constants (B=2.72e11, F1=1.745e6, resid 1.9e10 at
N_ref=45k, a=2.527e-8, raw 3.725e-7).

Scientific parameter: rank r in {16, 32, 64}.
Nuisance: which layers truncated (fixed 1..30 for probe), GT-MC size.
Fixed: 8 mini nets (ids 0-7), antithetic MC N=262144 for cheap-net GT
(GT noise ~8e-11/neuron, negligible vs bars), dataset final_means as true GT.

Error budget: current raw 3.725e-7; corrected estimator raw ~= a +
MSE_corr + b_cheap/N. Need MSE_corr <~ 1e-7 to be in play; anchor ABSOLUTE
error is 7.4e-5, so the difference must cancel ~30x (RMS 8.6e-3 -> 3e-4).

F1 accounting honesty: the CURRENT surface already prices ~54.5k
FLOPs/sample/layer (packed/split paths), not dense 131k. Rank-r prices
~1023r+256; r=32 ~ 0.6x current avg, r=16 ~ 0.3x. The 3.5x total needs
r~16, or r=32 plus row-sparse packing composing onto the U-projection.

Success threshold (continue to integration): some r with floor-aware
projected optimum <= 1.07e-7 (-20%).
Kill criteria: MSE_corr > 1e-7 at r=64 already (cancellation fails), or
projected optimum improvement < 10% at every r.

## Probe results

Script: session scratchpad `cheap_forward_probe.py` (+results JSON).
8 mini nets, cheap-net GT = 262,144-sample antithetic MC.

Run 1 (plain truncation): the cheap net DIES — 30 truncated near-critical
layers collapse activations to zero (var_ratio 0.000, MSE_uncorr 0.62 =
GT mean-square). Spectral mass loss compounds exponentially with depth.

Run 2 (norm-preserving truncation, Frobenius-rescaled so activations
stay alive):

| r | spectral mass kept/layer | MSE_uncorr | MSE_corr | cancellation | var_ratio |
|---|---|---|---|---|---|
| 16 | 0.21 | 1.00 | 4.31e-3 | 232x | 7.58 |
| 32 | 0.38 | 1.58 | 5.66e-4 | 2793x | 3.13 |
| 64 | 0.62 | 0.56 | 1.88e-4 | 2966x | 2.04 |

- The anchor-difference correction IS remarkably good in relative terms
  (~3000x cancellation) — the mechanism works.
- But the perturbation is enormous because the weights have a near-flat
  (Marchenko-Pastur-like) spectrum: rank-64 of 256 keeps only 62% of
  spectral mass per layer. 3000x cancellation of an O(0.5) bias still
  leaves 1.9e-4 — 1,900x over the 1e-7 bar, and WORSE than the bare
  anchor (6.8e-5).
- Rescaling also INFLATES the cheap net's final variance (var_ratio
  2-7.6x), so b_cheap worsens on top of the bias.
- Tradeoff curve is dead everywhere: truncating fewer layers shrinks the
  error roughly with layer count but shrinks the F1 saving faster (e.g.
  layers 1..4 ~ extrapolated residual ~3e-6, still 30x over bar, for a
  ~9% F1 cut).

## Verdict: KILL (pre-registered criterion met at r=64)

Low-rank surrogate sampling + analytic difference correction is dead for
the SAME structural reason as every prior surrogate: these nets have no
low-rank structure to exploit — the spectrum is flat, so any
spectrum-truncating surrogate is a large perturbation, and no correction
bridges 1,900x. This closes the LAST untested member of the
cheap-structured-surrogate family (linear CV rho^2=0.06, MLMC r_V=1.0,
anchor shrinkage g>=0.65, now mean-corrected low-rank).

## Strategic conclusion

Within legitimate dtype-free levers on the active-set-fold architecture
there is NO identified path to a 3.5x cheaper-per-sample forward pass:
Strassen (blocked + only 1.28x) and residual-wall trims (<1.1x) are the
only survivors, an order of magnitude short. The leaderboard frontier's
~3.5x variance-per-compute edge is therefore almost certainly a DIFFERENT
ESTIMATOR FAMILY, not a cheaper matmul under ours. Next structural bets
worth scoping (none yet designed): non-Gaussian / higher-order moment
propagation for a fundamentally better analytic base; exact piecewise
region analysis for dominant activation patterns; public-GT-calibrated
learned bias correction (same legitimacy class as _VAR_REF calibration).
