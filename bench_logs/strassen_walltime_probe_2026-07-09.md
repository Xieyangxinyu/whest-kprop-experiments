# Strassen Wall-Time Probe - 2026-07-09

## Question

Resolve the open caveat from the 2026-07-08 Strassen finding (12% flopscope-FLOP
cut, precision proven harmless): does the ~25-ops-per-matmul overhead blow up
wall-time past the remote grading cliff, as it did for full-block packing
(315417/315420)?

## Method

flopscope-native one-level and two-level Strassen (all-3-dims split, even shapes),
measured FLOPs via `flops.BudgetContext`, median wall over 3 reps, and max abs
error vs standard `@`, on the estimator's dominant sample-propagation shapes.
Script: scratchpad/strassen_probe.py.

## Results

| shape (m,k,n) | method | FLOP% | wall% | max err |
|---|---|---|---|---|
| (61440,212,212) | strassen1 | 88.1 | 537 | 8.2e-7 |
| | strassen2 | 78.1 | 1208 | 1.2e-6 |
| (30720,168,168) | strassen1 | 88.2 | 447 | 8.1e-7 |
| | strassen2 | 78.6 | 1069 | 8.4e-7 |
| (61440,256,256) | strassen1 | 88.0 | 387 | 1.0e-6 |
| | strassen2 | 77.9 | 992 | 1.2e-6 |

## Conclusion

- FLOP cut confirmed: 12% (1-level) / 22% (2-level). Precision harmless (~1e-6,
  ~600x below the ~6e-4 sampling floor).
- Per-matmul wall-time rises ~5x (1-level) / ~10x (2-level): 7 small BLAS calls
  are less cache/vector-efficient than 1 big call, plus 18 bandwidth-bound adds.
- BUT absolute wall-time is SAFE. Dense predict = 0.74s (gate-sweep baseline);
  Strassen-dense ~= 0.74 x 5 ~= 3.9s, vs the already-passing packing (315416)
  predict = 16s. Strassen delivers the SAME 12% FLOP cut as packing at ~4x LESS
  wall-time. The open wall-time caveat resolves in Strassen's favor.
- Strassen DOMINATES the packing lever: same saving, exact, less wall-time, no
  per-row gather.

## Not yet measured

Isolated-matmul wall-clock only; NOT a full `whest run --profile` on an
integrated Strassen estimator, so the exact residual_wall_time / multiplier split
is inferred, not confirmed. Would need integration + profile to nail the realized
adjusted-score effect.

## Decision

UNCHANGED: do NOT submit. This is a legitimacy/prize-eligibility judgment (metric
substitution vs estimate improvement) only the organizers can make - see
variance-reduction-tapped.md Strassen section. Technical viability is now
established (works, wall-time-safe, beats packing); the blocker is rules
confirmation, not engineering. Note the tension: 315416 (packing) is the same
class of "same-estimate cheaper-compute" optimization and was already accepted.
