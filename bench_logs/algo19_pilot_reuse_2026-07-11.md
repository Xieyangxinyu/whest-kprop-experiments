# Algorithm 19: pilot-reuse classification - 2026-07-11

## Idea (from the 315844 postmortem)

Replace the staged mid-layer kink->dead demotion probes in the 315824 surface
(dedicated 5% probe matmuls + 20% recheck matmuls, outputs discarded) with a
single pilot pass that reuses the base samples: propagate the leading 20% of
block rows over ALL candidate active columns with the ordinary
packed/block-split machinery, read sampled alphas off those pre-activations,
decide demotions, keep the pilot chunk as the surviving columns' first rows,
propagate only the remaining 80% over the survivors. Attacks probe COST
without losing sampled-alpha adaptivity. Dead->kink promotion probes and the
layer-30 on/dead probes untouched (their columns are outside the computed
set; nothing to reuse). Promoted columns get a small dedicated pilot top-up.

Implementation: `examples/19_pilot_reuse.py` (surgical patch of
17_finer_antithetic_row_buckets.py). One ordering bug found and fixed during
implementation: the pilot must only DECIDE in the demotion block; propagation
happens after the mid-layer promotion probes, which read the previous layer's
x. Side benefit: every probe-eligible column now gets the full 20% row budget
for its alpha (staged gave 5% to "stable", 20% only to "uncertain").

## In-process A/B (13 mini nets, natural predict, exact GT, 2026-07-11)

- Classification parity: per-layer refined kink counts IDENTICAL to the
  staged probes on all checked nets.
- Raw MSE: identical aggregate (4.6341e-7 both arms, -0.00%).
- FLOPs: 7.9530e10 -> 7.9329e10 (-0.25%, deterministic) - matches the
  pre-registered ~0.3% probe-cost estimate.
- Adjusted (flops + 1.9e10 grader residual pricing): 1.7297e-7 -> 1.7263e-7
  (-0.20%).

An exact-rewrite win: identical predictions, strictly cheaper. Risk-free on
the FLOP ledger; the open question is grader-side residual (one fused pass +
concat vs three staged calls per probing layer).

## Subprocess residual A/B (13 mini nets each, sequential, same machine)

- raw final MSE IDENTICAL both arms (4.6349e-7) - exact rewrite confirmed.
- flops 1.0339e12 -> 1.0313e12 (-0.25%) - consistent with in-process.
- residual 11.26s -> 12.26s (+8.9%) - KILLS the idea. Local mult 0.6108 ->
  0.6384; grader translation (residual ~19% of effective compute, relative
  residual deltas transfer): net ~+1.5% WORSE adjusted.

Root cause: reusing the pilot chunk forces concatenate([x_pilot, x_rest])
every probing layer - copying the whole ~30k x kink activation matrix
(~20MB x ~29 layers ~= 600MB memory traffic per net) that the original
single full-block matmul never pays. The eliminated probe matmuls were
FLOP-counted but wall-cheap; the added copies are FLOP-free but
wall-expensive. Same residual-vs-FLOPs trap as the packing-cleanup and
315844 lessons.

## Addendum: layer-targeted probe deletion ALSO falsified (same day)

Re-validated probe deletion with the layer-targeted dead band (-3.0 layers
0-29, -4.0 layers 30-31; `examples/19_lateband_probe_free.py`) at proper
scale: 32 mini nets, natural predict, exact GT, grader-priced.

- adjusted +4.65%, raw +4.58%, flops -0.63% (deterministic).
- Bias-case census: 2/32 raw blowups (net 2 +90%, net 17 +59%) with ZERO
  mirror-image improvements - asymmetric, systematic, not noise. The probes'
  mid-layer sampled-alpha protection is load-bearing at ~6% net frequency;
  no static analytic threshold replicates it (any-layer misclassification
  propagates to the scored row).
- Phase-2 residual A/B skipped: no residual saving rescues +4.6% raw.

PROBE-DELETION LINE CLOSED (three variants falsified: global -4.0 =
submission 315844 on the leaderboard; global -3.0 and lateband -3/-4 = local
bias blowups). The staged probes earn their keep; only their COST was ever
on the table, and pilot-reuse (below/above) showed the cost cannot be
restructured away either.

## Status: FALSIFIED - do not promote

The one durable positive: single-pass 20%-row pilot decisions are IDENTICAL
to the staged 5%+20% probe decisions (verified per layer on 3 nets). If probe
cost ever matters again, the cheap variant is to simplify the *dedicated*
probe to one 20% pass (drop the recheck stage, one less Python pass, no
concat) - expected value tiny (~0.1%), measure residual before believing it.
Do not retry pilot-reuse-with-concat without a copy-free composition
mechanism (e.g. preallocated writes), which flopscope's functional API does
not currently offer.
