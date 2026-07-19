# Merge-and-multiply probe (conditional/sequential sampling) — KILLED - 2026-07-17

## Idea

Exploit piecewise linearity: LSH-bucket the N samples at layer l, propagate
only K bucket centroids downstream, reinvest the saved flops in more
samples. Within a cell where no downstream ReLU flips, centroid
propagation is EXACTLY unbiased; bias comes only from kink-crossing cells.
This was the untested "attack where variance is generated" candidate after
tonight's read-out-side kills (RB/EB), and is distinct from every prior
kill (not MLMC, not moment closure, not low-rank, not learned correction).

## Pre-registered bars

CONTINUE: some (l, ratio) projects fleet raw <= 3.0e-7 (-20%).
KILL: paired merge err^2 > 1e-7 at every l>=8, ratio>=8.

## Method

8 mini nets x 3 reps, N=16,384 antithetic Gaussian samples, offline dense
numpy. At l in {0,4,...,28}, sign-LSH buckets at target ratios
{4,8,16,32}, weighted centroids, downstream propagation; PAIRED contrast
(merged mean vs full-sample mean, same samples => pure merge error).
Script/data: .tmp/router_opt/merge_probe.{py,pkl}.

## Result: KILLED by 3-6 orders of magnitude

Best config (l=28, i.e. only 4 layers merged): merge err^2 = 6.35e-4 vs
the 1e-7 bar. Monotone worse at shallower l (l=8: 4.6e-3; l=0: 3.7e-2).
No config is within a factor of 1,000 of viability.

## Two structural findings (the real value of the probe)

1. **Deep activation geometry is EXTREMELY low-dimensional**: requesting
   up to 16,384 LSH cells yields only K~330 occupied at l=8, K~50 at
   l=24-28 — 14 random hyperplanes cut the deep activation cloud into
   ~50 occupied cells (16,384 samples). The ensemble collapses to a
   few-dimensional cone with depth. Variance profile confirms strong
   contraction then a floor: mean per-neuron activation variance 0.68
   (l=0) -> 0.10 (l=12) -> 0.042 plateau (l>=24).
2. **But the map on that manifold is locally ROUGH**: even within deep
   cells, 5-12% of downstream masks flip per layer between members and
   centroid. Kink boundaries cross the low-dim manifold densely. This
   single mechanism now coherently explains EVERY variance-reduction
   kill: rho^2=0.06 surrogates, stratification/rotation, RB Gaussian
   closure, and merge bias — low-dim geometry, high local roughness.

Note the circularity trap for "exact pattern merging" (group by realized
downstream activation pattern, which IS exactly linear per group): the
pattern is only known AFTER propagating, so it cannot fund upstream
savings. Do not revisit merging/clustering/conditional-restart estimators
on this architecture without a mechanism that PREDICTS downstream
patterns cheaply — none is known, and learned predictors are in the
learned-correction kill class.

## Campaign state after 2026-07-17 (one-line ledger)

Read-out side: RB/moment closure +74..160%, EB nil, no output
concentration. Sampler side: merge/conditional-restart killed 1000x over
bar. Cheap-forward side (07-14): low-rank 1900x over bar, learned
corrections +1%. QMC: 11% marginal. The leaders' ~4x raw-per-compute
remains unexplained by ANY tested family; the only unfalsified direction
on the board is exact/analytic piecewise-region integration exploiting
the ~50-cell deep geometry from finding 1 — a fundamentally different
estimator, not a modification of algo28.
