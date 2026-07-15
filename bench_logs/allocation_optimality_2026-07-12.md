# Allocation optimality study — 2026-07-12/13

Source: `allocation_optimality.ipynb` (41 cells, fully executed; caches `.alloc_sweep_v2.npz`,
`.alloc_gt4m_cache.npz`, `.alloc_ml_gt_cache.npz`). Surface: 315998 estimator (Algorithm 21/24).
Protocol everywhere: A/B scrambled-Sobol replicates, GT noise-floor subtraction, fleet-level
paired stats (single-realization per-net MSE noise: median 53% |A−B|/mean — never decision-grade).

## Question

How optimal is the production rule `N_i = clip(49152·sqrt(V_i/0.02143), 30720, 61440)`, and can
any per-net signal (NN/ML, pilot variance, alternative shapes, higher cap) beat it?

## Results by section

- **§1–8 honest oracle (10 He-init nets, 4.2M GT):** rule regret vs cross-rep honest oracle
  **5.3%** (naive same-rep oracle fabricates 61% — winner's curse). Oracle picks reproduce
  across replicates on only 10% of nets; no cheap signal correlates with the pick (ρ ≤ 0.14).
- **§9 ML kill test (70 nets, 512 weight/ADF/pilot features, 6 regressors + 4 classifiers):**
  the per-net label is UNMEASURABLE — split-rep reliability r ≈ 0 (b: 0.02, a: −0.01), prefer
  decision agreement 46%. Positive control failed for every model incl. the V-baseline
  ⇒ void-by-noise ⇒ **NN/ML allocation closed for good.**
- **§9d fleet paired constant test (n=70):** 30720 vs 61440 flops-only WASH (−0.8σ); with fixed
  grader residual (+1.9e10) **61440 wins +2.3σ** — the rule's high-N side is correct.
- **§10 pilot-b̂ plug-in: KILLED.** Closed-form N* degenerates to clamp constants under both
  pricings; the binary cross-rep oracle (ceiling for ANY 30720/61440 per-net signal) loses to
  the best constant (+4.3% flops / +7.8%, +2.1σ grader).
- **§10c rule-signal swap (b̂ for V, compute-matched calibration): WASH** (+1σ worse).
  Calibration independently rediscovered ref ≈ 0.0214 ≈ V_REF; spearman(V, b̂) = 0.88.
- **§11 cap raise, extended all-Sobol artifact (N=92,160, 70 nets):** curve extrapolates across
  the artifact edge (ratio 0.95, no SOBOL FLIP at fleet level); vs 61440 flops-only +0.1σ,
  grader-priced −1.1σ — flat.
- **§13 closed form + cap 122,880, calibrated at the source (100 mini nets = V_REF provenance
  data; eval on 10 disjoint full-split nets):**
  - Provenance confirmed: V_REF = 0.02143 = fleet-mean *analytic* V; baked avg_variance mean
    0.0495, analytic biased low ×0.49 (matches 2026-07-07 V_anal/Q), corr 0.90.
  - Mini bias-floor share at 61,440 is only **6%** (a = 2.53e-8) vs ~40% inferred for graded
    from 315824 ⇒ closed form picks all exceed the old cap (mean N ≈ 105k, 2.1× the rule).
  - **Flops-only: wash (−0.2σ). Grader-residual-priced: closed form 1.405e-7 vs rule 1.619e-7
    = −13%, −1.7σ (n=10).** Constant 61,440 (1.582e-7) again beats the rule under grader
    pricing. Raw MSE halves (2.47e-7 vs 5.22e-7) at ~2× samples.

## §13 rebuild on the Algorithm 21 surface + Algorithm 25 submission prep (2026-07-13/14)

The organizers' correction of the flopscope complex-cost bug retires Algorithm 24's c64
packing, so §13 was rebuilt with packing-free costs (`estimator_algo21.py` = git 41a8964).
Predictions carry over (algo21/24 are BLAS-rounding identical: max |diff| 7e-7, asserted on
every cache reuse); only the cost side was re-measured (cache keys `*_flops21`, 200 rep-A
calibration runs + 56 eval runs).

- Packing-free costs: **f1 = 1.745e6 FLOPs/sample** (1.36x algo24), f0 = 1.03e9,
  f0_grader = 2.00e10. Picks shrink to 62k–122.9k (20% at cap, 62% fleet utilization).
- Result unchanged, slightly stronger: **flops-only wash (−0.7σ); grader-priced closed form
  1.791e-7 vs rule 2.073e-7 = −13.6%, −1.8σ (n=10)**; constant 61,440 = 2.047e-7.
- **07-05 higher-anchor regression re-priced and resolved**: under 07-05 economics
  (~2.96e6 FLOPs/sample, flops-only) the same closed form puts the fleet optimum at the bottom
  clamp — anchor-65536 overspent exactly as predicted. Today's cheaper samples + the residual
  charge move the fleet optimum to ~101k. The precedent argued about era costs, not sampling.

**Algorithm 25 shipped** (`submission-algo25-closedform-alloc.tar.gz`, manifest sha
61846d7d…): Algorithm 21 + closed-form allocation `N_i = clip(sqrt(b_i·F0g/(a·F1)), 30720,
122880)` with b_i = 3.139 × base-block antithetic pair-mean variance (free, spearman 0.80 vs
analytic V), budget guard 0.8×budget, fp16 extended artifact (61,440 half-points = 122,880
all-Sobol samples; bit-exact fp32 prefix + seed-20260711 scrambled tail; 27.7 MiB).
End-to-end validation on the 10 full-split nets (own allocation, rep-A lineage): grader-priced
**1.688e-7 vs rule 2.045e-7 (−17%, −1.2σ single-rep)**, flops-only also ahead (−0.7σ), picks
68k–122.9k, max utilization 79%, `whest validate` PASS. Note the deployed base-block b̂
disagrees with §13's pilot b̂ on floor-dominated nets (full0: ranked lowest vs highest) and
scored better there — it measures the estimator's own reducible variance, the correct quantity.

## Durable conclusions

1. Per-net allocation labels do not exist at measurable cost; all future allocation claims must
   be fleet-paired and cross-realization. NN/ML allocation is permanently closed.
2. The production rule is within 5.3% of perfect foresight; every cheap per-net improvement
   tested (pilot variance, signal swap, binary switching) loses or washes.
3. §13 is the ONE open suggestive lead: closed form + extended artifact + cap 122,880 wins
   −13.6%/−1.8σ under grader-residual pricing on the packing-free algo21 surface (see the
   rebuild section above). The gain hinges on the fixed-residual assumption (if residual
   wall-time scales with N it shrinks/flips — the 315844 failure mode). The 2026-07-05
   higher-anchor regression (2.633e-7 → 2.896e-7) is *explained* by that era's per-sample
   costs under flops-only pricing, not evidence against this bet. Adjudication = one
   grader-side probe (Algorithm 25, packaged and validated).
4. Methodology: GT noise-floor subtraction is mandatory (fabricated ~half the apparent oracle
   gap in a 524k-GT first pass); winner's curse on per-net oracles is severe (61% → 5.3%).
