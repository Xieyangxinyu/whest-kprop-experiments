# Packing k-gate x Continuation-Block Sweep - 2026-07-09

## Question

Is there a k-gate ratio + continuation-coverage config that lowers the compute
multiplier below Algorithm 16 / submission `315416` (base-block-only, `k<=0.75`)
while keeping max predict wall-time at or below it (the remote grading cliff that
killed `315417`/`315420`)?

## Protocol

In-process sweep, 5 continuation-triggering stress MLPs (`build_mlp` seeds 0-4,
all target 50k-61k samples). Per config: analytical FLOPs via
`flops.BudgetContext(flop_budget=272e9)`, `predict()` wall-time, and a
prediction-identity check vs dense (packing must be exact). Knobs:
`_PACKED_ROWSPARSE_MAX_K_NUM/_DEN` and `_PACKED_ROWSPARSE_EXTRA_BLOCKS`.

## Results

| config | mean Gflop | mult (anal) | max wall | pred-id vs dense |
|---|---|---|---|---|
| dense (no pack) | 162.2 | 0.596 | 0.74s | 0 |
| 0.75 base [=315416] | 143.2 | 0.526 | 16.18s | 8.8e-7 |
| 0.75 both [=315417] | 126.4 | 0.465 | 29.80s | 1.1e-6 |
| 0.50 base | 162.6 | 0.598 | 0.94s | 0 |
| 0.50 both | 163.0 | 0.599 | 1.13s | 0 |
| 0.40 both | 163.0 | 0.599 | 1.12s | 0 |

(Local wall-times run higher than the learnings' remote numbers - different
machine - but the RATIO holds: `both` is ~2x `base`, matching `315417` failing
while `315416` passes.)

## Conclusion

No wall-time-safe sweet spot exists. The packing FLOP saving and its wall-time
cost come from the SAME layers (`k in (0.5, 0.75]`): tighten the gate to cut
wall-time and the saving vanishes (0.50/0.40 configs = dense FLOPs, mult ~0.60).
The only real saving is at `k<=0.75`, and extending it to the continuation block
(`both`) doubles wall-time into the `315417` failure regime. `315416` (0.75 base)
is the wall-time-safe frontier.

## Decision

Reject the gate-tuning / continuation-packing family. Do NOT submit a new
matmul-efficiency variant; ship/keep `315416`. Matmul lever confirmed tapped.
`pred-id <= 1e-6` confirms all packed paths are exact (raw MSE never traded).

Marginal-only option, NOT recommended: a `0.60 both` point would sit partway
(mult ~0.49, wall ~22s local) right AT the remote cliff - low EV (~7% mult), high
remote-failure risk, cannot be validated locally. Burned twice already
(`315417`, `315420`).
