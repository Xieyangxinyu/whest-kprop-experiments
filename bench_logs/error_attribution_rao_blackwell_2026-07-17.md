# Algo28 error attribution + last-layer Rao-Blackwell test - 2026-07-17

## Attribution (8 mini nets; 5 scrambled-Sobol, 3 iid, 3 no-class replicates)

- Fleet raw decomposition on the replicate class: sampling VARIANCE 88%
  (4.93e-7) / structural BIAS 12% (0.66e-7) — matches the graded model's
  ~93/7. Part of the bias reading is GT MC noise (organizers' final_means
  are sampled), so true estimator bias is <12%.
- NO hot-output concentration: top-8 outputs carry 9% of variance (18% of
  bias); 50% of variance needs the top 70/256 outputs. Per-output
  targeting (importance tilts, hot-output extra treatment) is DEAD ON
  ARRIVAL; only broadband levers can move raw.
- QMC: scrambled-Sobol/iid variance ratio 0.89 (thin marginal headroom);
  the SHIPPED unscrambled artifact beats the scrambled-replicate class by
  ~18% (4.57e-7 vs 5.59e-7) — original prefix quality is real; replicate
  variance overstates shipped variance ~x1.2.
- No-classification ablation: apparent bias RISES without dead/on
  machinery (1.36e-7 vs 0.66e-7; R=3, noisy) — classification is not the
  bias source; thresholds not worth touching.
- Data: .tmp/router_opt/error_attribution.{py,pkl}.

## Last-layer Rao-Blackwell / Gaussian smoothing — FALSIFIED

Test: capture final-layer pre-activations z (logging-only spies + predict
replica, BIT-IDENTICAL to stock, checked); per output replace shipped
mean(ReLU(z_j)) with plug-in Gaussian m*Phi(m/s)+s*phi(m/s) from the same
samples; alpha-gated hybrids; plus per-net James-Stein shrink toward the
analytic row. 40 reps x 8 nets vs mini GT:

| estimator | fleet raw | vs shipped |
|---|---|---|
| shipped | 5.183e-7 | — |
| RB only where alpha<0 | 9.031e-7 | +74% |
| RB alpha<1 | 1.062e-6 | +105% |
| RB everywhere | 1.349e-6 | +160% |
| EB shrink to anchor | 5.177e-7 | -0.11% |

VERDICT: the final-layer pre-activation is NOT near-Gaussian (penultimate
ReLU activations are sparse/skewed and strongly dependent; CLT does not
rescue the weighted sum). Any moment-closure smoothing of the last layer
is bias-dominated — consistent with the full-cov anchor result (12x
better than diagonal, still worse than sampling) and the deep-band
"anchor error ~ all penultimate-mean". EB shrinkage is correctly
self-limiting and worthless (~-0.1%).

Kills (do not revisit): last-layer Gaussian/moment-closure substitution
at ANY alpha gate; per-output shrinkage toward analytic anchors;
per-output targeting of "high-magnitude" outputs (no concentration).
Data: .tmp/router_opt/rao_blackwell_test.{py,pkl}.

## Where this leaves the per-sample-value campaign

All cheap broadband levers on the algo21/28 architecture are now
measured: QMC (thin), classification (fine), last-layer smoothing (bias
bomb), shrinkage (nil), output targeting (no concentration). The
leaders' ~4x raw-per-compute advantage (floor riders at raw ~7.7e-7
inside 0.1*B) is NOT reachable by post-processing this estimator; it
requires a structurally different propagation/sampling scheme. Next
probes should attack the SAMPLER (e.g., conditional/sequential sampling
of the active subnetwork, path-space or layer-wise variance splitting)
rather than the final-layer read-out.
