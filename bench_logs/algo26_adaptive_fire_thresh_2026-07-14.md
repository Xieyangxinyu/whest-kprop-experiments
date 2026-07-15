# Algorithm 26: per-call adaptive fire threshold - 2026-07-14

## Scope

User-directed: "improve the layerwise firing rate by tuning on 100 mini nets
not only 4." Census of all 6,000 split calls (100 mini nets, natural predict)
on the reverted algo21/315892 surface. Scripts in session scratchpad
(census_fire_thresh.py, fire_oracle.py, ab_adaptive.py); methodology below.

## Step 1: fixed-map refit on 100 nets - KILLED (pre-registered)

Success bar was split-half CV <= -0.10% flops; kill < 0.05%.

- Exact per-call cost curves from a census-calibrated flopscope model
  (pricing probed exactly: matmul m*o*(2n-1), einsum n*o*(2k-1),
  argpartition n*m, bool-sum n*(m-1), add/greater/mean n*m; gathers and
  np.asarray FREE). Calibration vs measured ctx.flops_used deltas:
  p50 |err| 0.006%, p99 0.03%, max 0.8%.
- 100-net refit map: in-sample -0.154% split-layer, but CV -0.058%/-0.036%.
  The 4-net map was already near-optimal AS A FIXED MAP; in-sample gain is
  noise-fitting. Do not re-tune fixed per-layer maps again.

## Step 2: the real finding - per-net adaptivity

- Per-CALL oracle (exact curves): -1.143% split-layer = -1.012% TOTAL
  estimator flops (split layers are 89.3% of all flops).
- Per-net-per-layer oracle: -1.136% - adaptivity needed is per net, not
  per call within a net.
- Per-net t* dispersion is huge (p10-p90 ~ +/-0.1 per layer): nets genuinely
  disagree; NO fixed map can capture this (ceiling ~-0.15% in-sample).

## Step 3: runtime rule - Algorithm 26

Per split call, pick t from a 19-point grid (0.55..1.0 step 0.025) by argmin
of a normal-approx cost model (independent-Bernoulli row-nnz, exact guard
rules, exact one-level Strassen formula for the dense block, plain rate for
bucket dense-fallback/tail). Parameter-free: computed from the call's own
fire vector, nothing fitted, no overfit channel. The model's cost LEVELS are
biased (the known 07-11 normal-approx artifact) but the bias is shared
across t and cancels in the argmin: offline it captures 98.4% of the exact
oracle (-0.996% total flops).

Implementation: pure fnp ops, fully vectorized (~45 backend ops, ~60k priced
flops per call = 0.005% overhead; ONE scalar pull for the argmin index). All
cost arithmetic float32 (int32 would overflow at ~4e10). Python scalars, not
fnp.float32 scalars, as operands - the scalar-on-left fnp.float32 pattern
triggers numpy auto-routing (grader RemoteArray API risk, cf. 316255).
No numpy side-channel: rule math is flopscope-priced.

## A/B, 100 mini nets, natural in-process runs (vs census baseline)

- fleet flops: -0.998% (offline predicted -0.996%); ALL 100 nets improved
  (range -0.30%..-1.65%, zero regressions).
- raw MSE: +0.0017% fleet (BLAS rounding from rerouting; max per-net 0.043%).
  NOTE: threshold changes are NOT bit-identical (the 07-11 "exactly
  invariant" claim doesn't survive routing changes; deltas are ~1e-11 class).
- adjusted proxy (raw_i * flops_i): -1.04%.
- seed-42 n=3 gates: local/subprocess parity EXACT (raw 4.268777e-7, flops
  256,397,338,852 identical); flops -0.996% vs algo21's 258,975,663,194;
  residual wall DOWN 3.81s -> 3.33s, backend wall DOWN 52.4s -> 43.9s (the
  smarter routing saves more wall than the rule's small ops add).
- whest validate PASS; packaged submission-algo26-adaptive-fire.tar.gz
  (estimator.py + fp32 sobol_points.npz + requirements[scipy] + LICENSE).

## Grader expectation

Multiplier tracks deterministic flops ~1:1 on this surface (315892, 315998
evidence) and gather residual is priced ~zero, so expect adjusted ~-1.0% vs
315892's 1.337748e-7 -> ~1.324e-7, raw ~unchanged 3.7250e-7. Residual risk:
the rule's extra small-op traffic is new backend-op-count territory
(315898 lesson: layout-restructuring needs its own grader test) - locally
it is wall-NEGATIVE, so the risk is bounded.

## Durable methodology notes

- flopscope pricing is exactly reverse-engineerable with micro-probes;
  np.asarray on fnp arrays is free (census instrumentation trick).
- Layer split costs are separable (routing exact, no downstream coupling):
  per-layer/per-call argmin is globally optimal.
- Biased cost models rank correctly when the bias is shared across the
  decision axis - the same normal-approx that failed for LEVEL prediction
  (07-11) is near-oracle for per-call argmin.
