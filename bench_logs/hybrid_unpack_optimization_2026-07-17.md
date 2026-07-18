# Hybrid unpack + N-raise optimization — FALSIFIED - 2026-07-17

## Scope

Handoff step 2 (user-endorsed, 07-15): minimize mult(F)*raw(N) over
(unpacked-layer-set U, N <= ~84k) subject to per-net grader wall <= ~43s
(60s S5.5 cap / 1.4 slow-worker margin). Idea: packed gather/einsum
machinery is ~90% of wall at 10-20 GFLOP/s effective; unpacking layers to
dense (GEMM speed) frees wall to spend on more samples.

Decision bar (handoff): build Algorithm 31 only if the optimum clears ~-2%
net vs 316405 (adjusted 1.324541e-7).

## Method

Wall benchmark (.tmp/router_opt/wall_bench.py): for every
`_block_split_matmul_2blk` call on 8 mini-split nets (1 warmup), time the
shipped plan AND a forced `_dense_matmul_2blk` on the same inputs
(order-alternated to cancel cache bias; flopscope is eager, verified;
per-call flops from session-counter deltas). Shipped result propagates;
dense result discarded. 496 calls, 62/net (31 layers x base+extra blocks).

Wall model VALIDATED: local per-net walls 12.7-20.9s x 2.3 = 29-48s grader
matches the user-read panel 32.0-44.4s. Spied calls = 93% of local wall.
Score model VALIDATED: with F0 corrected to 1.832e10 (= 0.35562*B - F1*N0),
the model reproduces graded 316405 adjusted to 4 decimals
(N0 = 44,930 sample-units, F1 = 1.745e6, a = 2.527e-8, b = 1.56e-2).

## Per-layer wall/flop exchange (mean over 8 nets, full-dense unpack)

Best layers (s saved per Gflop paid): layer 1 forced-packed 0.73, layer 31
0.67, layer 2 0.60, layers 3-4 ~0.5; mid layers 0.23-0.44. Absolute:
layer 1 saves 1.46 s/net for +2.0 Gflop/net (+45.4k flops/sample); deep
layers save 0.15-0.35 s/net for +0.3-1.0 Gflop each.

## Result: dominated EVERYWHERE on the grid

Greedy frontier (unpack best-ratio layers until wall-feasible), r = N scale
vs today, corrected model:

- r=1.00 needs U={1,31} (today's tail net 20.9s local = 48s grader is
  ALREADY over the 43s line): adjusted +2.5..4% WORSE.
- r=1.35 (N~61k model units) needs 7 layers unpacked: +8.8% worse.
- r=1.79 (wall-free optimum N*=80.5k, max possible gain -3.4%): needs
  18.8s local wall bought on the worst net = >=25.7 Gflop/net at the BEST
  measured rate, vs 5.7 Gflop/net headroom before the entire N-gain is
  spent. Off by 4.5x; no combinatorial refinement (partial-column
  threshold moves, per-net sets) closes a gap that size, and all 31
  layers' rates sit in a narrow 0.23-0.73 s/Gflop band.

VERDICT: hybrid unpack + N-raise is DEAD as a score lever. Do NOT build
Algorithm 31. The wall->flops exchange rate of dense unpacking is ~4-5x
too expensive for sample economics that only offer -3.4% wall-free.
Consistent with 07-11 ("cap raise is a wash") and 07-14 ("slowness is
earned"): the packed flop discount is worth more than the wall it costs,
even under the 60s cap.

## Side findings (reusable)

- Wall calibration: grader/local = 2.3x confirmed on an 8-net panel match;
  per-net wall is ~linear in N; spied matmul calls carry 93% of wall.
- 43s-cap feasibility TODAY: worst bench net (20.9s local, N=47.5k) sits
  at ~48s grader daytime — over the 1.4x-slow-night safety line. The
  cap-only 49152 insurance (29c, score-neutral measured) remains the only
  non-score-negative wall knob; unpacking as pure insurance is dominated
  by 29c (score-negative vs neutral).
- Per-call bench data: .tmp/router_opt/wall_bench.pkl (496 calls, walls +
  exact flops for shipped and dense plans); optimizer:
  .tmp/router_opt/unpack_optimize.py.

## Campaign status after this result

Allocation/N axis now closed from THREE ends: signal replacement (07-11),
cap raise at current economics (07-11), wall-financed N-raise (this).
Remaining live route per the 07-14 competitive analysis: per-sample value
(structurally better estimator), not FLOP price or N.
