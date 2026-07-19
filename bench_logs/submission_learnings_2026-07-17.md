# Submission learnings - 2026-07-17

## Submission Log

### Submission 316894 - Algorithm 29b sync-batch

- Result: regressed (+2.06% adjusted vs 316405)
- Change: packed-matmul group boundaries via ONE vectorized searchsorted +
  one tolist()/chunk (host syncs 950->124/net, ~826 backend ops removed);
  Winograd reassociation of Strassen combines; redundant 8-bucket dropped.
  Raw designed bit-identical to 316405; local flops -0.0089%.
- Local expectation: raw identical; multiplier flat to ~-1% if op dispatch
  is priced (316005 ~7e5 FLOP-eq/op hypothesis). Fully gated, packaged
  07-14, validate-package clean.
- Leaderboard evidence: adjusted 1.3518369824728245e-7, raw
  3.7246087714493114e-7 — raw BIT-IDENTICAL to 316405 on the grader
  (design confirmed), multiplier 0.36295 vs 0.35562 = +0.73pp (+2.06%)
  despite -0.0089% local flops. Graded in a NIGHT window (~02:20Z submit,
  terminal within ~35 min) right after a 3-of-4 failure cluster.
- Decision: revert — ship surface stays 316405 bytes (estimator.py already
  is). Do not retry sync/op-count reduction for score.
- Lesson: op-dispatch pricing value is FALSIFIED on this surface; the
  multiplier moved +2% AGAINST a locally flop-neutral diff, which means
  grader-side pricing has DIVERGED from our local flopscope: either the
  pending complex-cost correction landed with broader re-pricing (gathers/
  concats/array ops?), or vectorized-searchsorted/tolist paths are priced
  remotely. LOCAL FLOPSCOPE IS NO LONGER A TRUSTED MULTIPLIER PROXY until
  recalibrated — do not gate future variants on local flop deltas alone.

## Open questions raised

1. Pricing divergence scope: is 316405 itself still priced 0.35562 today?
   Cheapest discriminator = one verbatim re-grade of the 316405 artifact
   (byte-identical pair protocol, as with 316405/316416). If its multiplier
   also comes back ~0.363, the re-pricing is global (correction landed) and
   ALL pricing-based conclusions (29c neutrality, unpack economics, router
   -1% class) need a recalibration pass; if it stays 0.3556, the 29b diff
   itself is priced +2% and sync-batching is score-negative per se.
2. Night-window model: two overnight grades (316424, 316894) vs the 07-14/15
   deadline deaths — night is HIGH-VARIANCE, not uniformly fatal. Keep
   morning preference for frontier slots, but night probes are not wasted
   by default.
