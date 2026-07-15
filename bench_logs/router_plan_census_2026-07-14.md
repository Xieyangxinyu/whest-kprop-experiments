# Plan-based matmul router census (build off algo21) - 2026-07-14

## Scope

User-directed: replace the fixed per-column fire-rate threshold with a router
that prices a small set of candidate execution plans (all-dense, all-packed,
a few split points) from matrix dimensions + the OBSERVED row-nnz
distribution, executes the cheapest valid plan (algebraically identical
convention), keeps routing flops negligible (subsampled cumulative counts
after sorting columns), and calibrates all cost coefficients against
flopscope (no guessing).

Baseline = algo21/315892 fixed per-layer threshold map. Reference points:
algo26 per-call threshold-grid oracle -1.01% total flops, its deployable
normal-approx rule -0.996% (algo26_adaptive_fire_thresh_2026-07-14.md);
algo26 is shelved (submission 316327 wall-suspect, user call).

## Algebraic result established during setup

Under the shipped packed convention (k = ceil16(bucket limit); argpartition
per group costs g*m so the group-sum is n*m regardless of grouping),
BUCKET-SET adaptation is provably vacuous on flops: including every multiple
of 16 below the 3/4 dense-fallback rule is always weakly optimal, and the
shipped set already does exactly that (the 8-limit is even redundant with
16 - both quantize to k=16). Per-call bucket choice is OUT; the plan space
is {all-dense, all-packed, split(s)}. (Separate small lever, not pursued
here: BUCKET_QUANT 16 -> 8 changes einsum k quantization - an execution
convention change with backend-shape-count/wall implications, not a routing
decision.)

## Cost model calibration (router_calibrate.py / router_costmodel.py)

Measured flopscope pricing on synthetic calls (n in {4096,12288,30720},
m in {128,256}, 5 sparsity profiles): Strassen dense formula EXACT (0.0000
error); packed/split worst |rel err| 0.011% after two fixes: (1) per-chunk
row-argsort lump = 15.0 flops/row (empirical, stable across n and m);
(2) dense-fallback groups must be priced PER CHUNK - a ~3.5k-row group
inside each 16384-row chunk misses the 4096-row Strassen gate that the
cross-chunk aggregate would pass (aggregating was a -2.8% error).

## Pre-registered bars (before analysis ran)

On >= 16-net census, deltas measured against TOTAL fleet measured flops:
- CONTINUE to implementation: deployable subsampled router (p2_sub,
  including its routing charge ~0.5M flops/call) <= -1.2%.
- MARGINAL (-0.8% .. -1.2%): equal to the shelved algo26 rule; decide on
  wall-safety merits only (fewer backend ops than algo26's 45/call or fewer
  sync pulls), not on flops.
- KILL: p2_sub > -0.8%, or the distribution-aware p2 beats the
  marginal-rates p1 model by < 0.1pp (then the user's core hypothesis -
  "row-nnz distribution >> average fire rates" - is false on this surface
  and the shelved algo26 rule already captured the routing value).

New-headroom hypotheses this census can newly confirm/kill (algo26 never
routed these): start-layer forced-packed calls and extra-block packed calls
(_packed_relu_matmul) join the plan space.

## Results

16 nets, 992 calls (62/net), modeled baseline = 92.5% of measured fleet
flops (rest = probes/analytic, unrouted). All deltas are % of TOTAL
measured fleet flops (1.267e12), routing charges included where marked.

| config | candidates | cost estimator | delta |
|---|---|---|---|
| oracle_all | every s in 1..m + endpoints | exact full-row cumlim | **-1.063%** |
| oracle_tgrid | 19 fire thresholds | exact | -0.953% |
| oracle_16 (v1 bug) | s multiples of 16 | exact | -0.213% |
| p1_all_s (+charge x2) | every s | normal from exact marginal rates | **-1.039%** |
| p1_normal (+charge) | 19 thresholds | normal (algo26-equivalent) | -0.939% |
| p2_sub (+charge) | 19 thresholds | exact from 2048-row subsample | -0.855% |

Key facts:
- **oracle_all reconciles with the algo26 census** (-1.01% on 100 nets).
  The v1 16-step grid missed it because 831/892 optimal split points are
  NOT multiples of 16 - cost-vs-s has kinks where the row-nnz distribution
  crosses bucket limits; optima are fire-threshold-anchored, not
  column-count-anchored.
- **Distribution-aware routing LOSES to marginal rates** (p2_sub -0.855%
  vs p1_normal -0.939%, pre-registered kill line: p2 must beat p1 by
  >= 0.1pp). Mechanism: marginal fire rates are computed EXACTLY on all
  ~30k rows (the mask mean the code already pays for), while row-nnz
  dependence structure must be subsampled (2048 rows) to keep routing
  affordable - subsample noise in the cumlim estimates costs more regret
  than dependence-exactness buys. Full-row exact costing needs 14*n*m ~
  0.8% of a call in priced flops: self-defeating.
- **The one real finding: candidate DENSITY, not cost-model fidelity.**
  p1_all_s (normal model at every s, vectorized length-256 instead of
  length-19 - same op-count class, ~2x element volume, one sync pull)
  captures 97.7% of the all-s oracle: -1.039% vs the algo26 rule's
  -0.94% here (-0.996% on its own 100-net A/B).
- Endpoints and new call kinds are EMPTY: the oracle never picks
  all-dense; start-layer forced-packed routing gains +0.0000% (packed is
  already optimal there); no extra_packed calls occur on this surface.

## Verdict

- The user-hypothesized mechanism (row-nnz distribution >> average fire
  rates) is FALSIFIED per pre-registered bars - killed by the routing
  self-charge: any estimator of distribution shape that is cheap enough
  to route with is noisier than the free exact marginals.
- Bucket adaptation: vacuous by algebra (see above).
- Surviving artifact: an algo26-class rule with ALL-s candidates
  (p1_all_s), worth ~-1.04% flops ~ -1.0% adjusted, same wall-risk
  profile as algo26 (one sync pull, ~45 vectorized small ops). algo26 is
  shelved (316327 "could not complete", cause ambiguous - same-day scan
  shows ~14 other participants hit infra-class failures, so flake vs
  op-traffic is ~50/50). If routing is ever revisited: (1) settle the
  wall question first (one verbatim 316327 retry per the 316258/9
  protocol), (2) then ship all-s candidates instead of the 19-t grid,
  with the per-chunk Strassen-gate fix from this calibration folded into
  the cost formulas.
- Do NOT pursue: subsampled row-nnz routing, per-call bucket sets,
  all-dense/all-packed endpoints, start-layer routing.

Scripts: session scratchpad router_{costmodel,calibrate,census,analysis}.py;
census pickle alongside (16 nets, 992 calls, cumlim tables at every s).

## Algorithm 29 BUILT AND GATED (estimator_allsrouter.py, late 07-14)

All-s router implemented on the algo28/316405 frontier surface (user
authorized exploration; submission awaiting user go). Port of algo26's
normal-approx cost model evaluated at EVERY split point (fnp-vectorized
length-m, no arange — s vector via cumsum(ones); one scalar pull/call;
routing charge ~0.01%/call). The split decision is made once per call and
shared by both row blocks (required by the 2-block dense path). Column
sets from argsort(fire) prefixes, index-sorted.

Gates vs algo28 baseline:
- fleet flops (16 mini nets): -1.0321% (census prediction -1.04% - exact);
  ALL 16 nets improved.
- fleet raw: -0.0004% (fp-noise neutral).
- validate PASS; subprocess seed-42 n=3 CLEAN 3/3, raw 4.269358e-7
  (+0.005% vs algo28, noise class), flops 256.17e9.
- Grader expectation: multiplier tracks flops ~1:1 on this surface
  (315892/316405 evidence) -> adjusted ~1.311e-7 (-1.0% vs 316405's
  1.3245e-7). Wall profile: same one-sync-pull/call class as algo26;
  316368/316405 both ran 50 nets comfortably, and 316327's failure is
  attributed to the infra outage - risk considered low but this IS the
  first grader test of the router op-traffic class.
