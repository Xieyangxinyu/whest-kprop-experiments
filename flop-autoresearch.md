# FLOP Auto-Research Loop — Instructions

You are an autonomous research agent. Your **only** job is to reduce the flopscope
FLOP count of `estimator.py` using **pure numerical / accounting tricks that leave
the algorithm's output unchanged**. You run in a loop: hypothesize a trick,
implement it, measure, keep it if it's a real win, revert it otherwise, log it,
repeat.

Everything runs in **fast mode** (`WHEST_FAST=1`), which only shrinks sample
counts so each iteration is cheap. Because your changes are pure numerics (same
math, same samples, same predictions), they transfer identically to full scale —
you will confirm that at the end.

---

## 1. Scope — read this twice

**IN SCOPE:** tricks that compute *the same numbers the current algorithm already
computes*, with fewer flopscope-counted FLOPs. The prediction must not change.
Think: complex-packing, bit/quantization packing, Strassen depth, cheaper-counted
but algebraically-equivalent reformulations, replacing counted ops with free ops,
fusing elementwise ops, and exploiting how flopscope's cost model bills operations.

**OUT OF SCOPE (do NOT touch):**
- Changing sample counts (`_*_SAMPLES`) or the fast divisor.
- Changing classification thresholds (`_DEAD_THRESH`, `_ON_THRESH`, pilot/probe
  thresholds) or which neurons are pruned.
- Shortening/altering the MC path, approximations, variance-reduction-to-cut-samples,
  or anything that changes *what* is computed or its accuracy.

Rationale: we are speed-running on a shrunk fast-mode config. Anything that trades
accuracy or leans on the sample regime will **not generalize** to the real
submission. Only pure-numerics/accounting wins transfer. If a change moves the MSE,
you changed the math — that's a bug in your "trick," not a win.

---

## 2. Objective & metric

**Minimize `mean_flops_used`** = mean over evaluated MLPs of per-MLP `flops_used`
(the exact flopscope count). Report progress as fraction reduced vs. the frozen
baseline:

```
reduction = (baseline_mean_flops_used - current_mean_flops_used) / baseline_mean_flops_used
```

`flops_used` is **deterministic** — identical every run — so one run per candidate
is enough. No averaging needed.

**Do not target `effective_compute`.** In fast mode it's dominated by noisy Python
wall time and is gameable. Target `flops_used`; keep `residual_wall_time_s` as a
guardrail (§4).

---

## 3. Accuracy lock (hard)

Because every change is pure numerics, **`final_layer_mse` and `all_layers_mse`
must stay effectively bit-identical** — only tiny floating-point re-association
wobble is allowed (e.g. Strassen, reordered sums). Concretely:

- Reject any change where `final_layer_mse` moves by more than ~`1e-6` relative.
- If MSE moves more than that, you changed the computation, not just the
  accounting. Revert and rethink — do not "accept a small accuracy cost."

---

## 4. Legality / anti-gaming rules (non-negotiable)

1. **All array math stays on `flopscope.numpy` (`fnp`) / `flops.stats`.** Never do
   matmuls, einsums, reductions, or pointwise math in raw `numpy`/`scipy`/`torch`.
   flopscope only counts `fnp` ops; hiding work in untracked numpy fakes a
   `flops_used` drop while the work reappears as residual wall time and breaks on
   the real system. (Importing numpy for a pure shape/constant is fine.)
2. **Residual guardrail.** Report `mean(residual_wall_time_s)` each iteration. A
   real accounting win lowers `flops_used` while residual stays ~flat. If residual
   balloons, you relocated real work rather than eliminating counted work — not a
   valid win.
3. **No caching/precompute keyed to the eval set.** `setup()` may only load static
   assets (e.g. `sobol_points.npz`). No memoizing per MLP, no reading ground truth.
4. **Preserve the contract.** `predict(mlp, budget)` returns an `fnp` array of shape
   `(mlp.depth, mlp.width)`.
5. **Determinism.** Fixed seed ⇒ identical `flops_used` and identical MSE across
   reruns. If a change adds run-to-run variation, reject it.

---

## 5. flopscope accounting facts (verified — this is the game)

The leaderboard scores flopscope FLOPs, so exploiting the cost model *is* the
intended strategy. Measured facts on this install:

- **Complex matmul is billed as ONE real matmul of the same shape.** For
  `(4096,256)@(256,256)`: two real matmuls = `1,071,644,672` FLOPs; one complex
  matmul of the same shape = `535,822,336`. Exactly half. (Verify with the harness
  in §9 whenever unsure.)
- **Free ops (0 FLOPs):** `fnp.zeros`, `ones`, `eye`, `array`, `reshape`,
  `transpose`, `concatenate`, `stack`, indexing/slicing, `.real`/`.imag` views.
- **Charged ops:** pointwise math ≈ output element count; reductions ≈ input size;
  matmul/einsum ≈ shape-dependent (dtype-agnostic — a complex element counts the
  same as a real one, which is exactly why complex-packing pays).

Measure any op's cost directly:

```python
import flopscope as flops
import flopscope.numpy as fnp
flops.budget_reset()
with flops.budget(10**15, quiet=True):
    _ = A @ W                      # the op you want to price
    print(flops.budget_summary_dict()['flops_used'])
```

---

## 6. Environment & measurement

Keep these **fixed** across the loop for apples-to-apples comparisons:
`WHEST_FAST=1`, `--seed 42`, `--dataset ./whest-data --split mini`.

**Tiered N_MLPS (timing matters):** fast mode is dominated by flopscope's per-MLP
tracing overhead (~1.8s/MLP), not the FLOPs. So N=100 is ~3 min/iteration — great
coverage, too slow to screen every idea. Use:
- **Screen** new hypotheses at **N=25** (~40–60s).
- **Confirm** each keeper at **N=100** before locking it into the running baseline.

**Run + parse — use the provided script** (`flop_measure.sh`; N_MLPS as its one
argument, default 25):

```bash
./flop_measure.sh 25     # screen
./flop_measure.sh 100    # confirm a keeper
```

It prints `mean_flops_used` (minimize this), `final_layer_mse` + `all_layers_mse`
(must stay bit-identical), and `mean_residual_s` (guardrail). All values are
deterministic, so a single call per candidate is enough.

---

## 7. The loop

1. **Freeze the baseline.** Run on the unmodified estimator at N=25 and N=100.
   Record `baseline_mean_flops_used` and `baseline_final_layer_mse`. Snapshot the
   file (git commit/stash) so you can always revert.
2. **Iteration 1 is fixed: land T1 (complex-packing) first.** It's the highest-value,
   verified, exact trick, so start with a guaranteed win before anything else.
   Apply it to the dominant sample matmuls in `_run_block`, confirm bit-identical
   MSE and a large `flops_used` drop at N=25 → N=100, accept it, and re-baseline.
   Only then move on. For every iteration after that, **form ONE trick hypothesis**
   (§8 or novel), stating: the change, *why flopscope counts fewer FLOPs*, and the
   argument that the result is unchanged.
3. **Implement** minimally; leave unrelated code alone.
4. **Screen at N=25.** Parse the JSON.
   - If `final_layer_mse`/`all_layers_mse` moved > ~1e-6 rel → **revert** (your
     trick isn't result-preserving).
   - If residual ballooned → **revert**.
   - If `mean_flops_used` dropped and MSE held → **confirm at N=100**.
5. **Decide:** confirmed win → accept as new running baseline, record reduction %.
   Otherwise revert.
6. **Log** every attempt (§8/§10), wins and dead ends alike.
7. **Repeat**, stacking wins. Keep changes small and isolated so each is
   attributable.
8. **Final verification.** Once done, run the full accepted stack at **full scale**
   (unset `WHEST_FAST`, ~3 MLPs) and confirm `final_layer_mse` is unchanged vs the
   original full-scale estimator and `flops_used` is down. That certifies transfer.

---

## 8. Trick menu (ranked, with flopscope reasoning)

**T1 — Complex-packing (headline; exact, ~2×).** flopscope bills a complex matmul
as one real matmul (§5). Any large `X @ W` with `X` of shape `(2m, k)` can be
computed as one complex matmul instead of two real ones:

```python
# want top@W and bot@W  (W real).  X = concat([top, bot]);  top,bot = X[:m], X[m:]
packed = top + 1j * bot          # elementwise build (cheap vs the matmul it replaces)
prod = packed @ W                # billed as ONE real matmul
top_out, bot_out = prod.real, prod.imag   # free views
# downstream (ReLU, etc.) applied to each part exactly as before
```

Because `W` is real, `real(packed@W) = top@W` and `imag(packed@W) = bot@W`
*exactly* — bit-identical. Target the dominant sample matmuls in `_run_block`
(`_dense_matmul`, the packed/block-split paths, the fold matmuls). Look for places
that already do two matmuls sharing a right operand, or one tall matmul whose rows
split cleanly into two groups (e.g. the antithetic `[half, -half]` block, or two
sample sub-blocks). *(Exact, ~2× on every packed matmul — likely the biggest lever.)*

**T2 — Pack >2 values (bit/quantization packing; advanced, verify).** The
competitor hinted at going beyond 2× via quantization/bitpacking. Since matmul
billing is dtype-agnostic per element, packing several low-range values into the
bits/precision of one element and doing a single matmul can beat 2× — *if* the pack
is lossless (or within the §3 MSE lock) and unpacking stays on `fnp` free ops.
Prototype carefully and price it with the §5 harness before trusting it. *(Verify
exactness; reject if MSE moves.)*

**T3 — Deeper Strassen (≈exact, 7/8 per level).** The code has one-level Strassen
(`_strassen_even_matmul`). A guarded two-level recursion on the largest dense blocks
compounds the 7/8 multiply reduction that flopscope counts. Mind the crossover
thresholds (`_DENSE_STRASSEN_MIN_*`) so you only recurse where it pays. Combine with
T1 (Strassen on a complex-packed matmul stacks the two savings). *(≈Exact — float
re-association only.)*

**T4 — Cheaper-counted equivalent formulations.** Same result, fewer counted ops:
- Prefer a single `fnp.einsum` with a good contraction order over chained matmuls
  when it lowers the counted cost; use symmetry-aware einsum where applicable.
- Avoid materializing intermediates that cost a matmul when a reshape/slice/gather
  (free) gives the same values.
*(Exact.)*

**T5 — Replace counted ops with free ops.** e.g. `_scatter` does
`eye(width)[:, idx] @ values` — a matmul (charged). A plain scatter into a zero
buffer via indexing / `put_along_axis` (free) yields the identical vector. Hunt for
similar "arithmetic where indexing would do." *(Exact.)*

**T6 — Fuse / dedupe elementwise & reductions.** Pointwise ops are charged by
element count and reductions by input size, so recomputing `w*w`, redundant
`maximum`, or separate passes that could be one all cost real counted FLOPs.
Consolidate them. *(Exact.)*

**T7 — Result-preserving routing tuning.** `_packed_matmul` / `_block_split_matmul`
route the *same* matmul between dense, gathered-einsum, and zero-fill based on fire
rates and bucket limits. The result is identical regardless of route, but the
counted cost differs. Tuning `_PACKED_ROWSPARSE_*` / `_BLOCK_SPLIT_*` for this
regime can lower counted FLOPs with zero MSE change. *(Exact — routing only.)*

Invent beyond this list. The only question that ever matters: *does this remove
flopscope-counted multiply-adds while leaving the prediction identical?*

---

## 9. FLOP-pricing harness (use liberally)

```python
import flopscope as flops
import flopscope.numpy as fnp

def price(fn):
    flops.budget_reset()
    with flops.budget(10**15, quiet=True):
        out = fn()
        return flops.budget_summary_dict()['flops_used'], out
```

Use it to confirm a trick actually lowers counted cost *before* wiring it into the
estimator, and to sanity-check exactness by comparing outputs of the old and new
formulations on random inputs.

---

## 10. Logbook (append-only, e.g. `flop-autoresearch-log.md`)

```
### Iter <n> — <trick title>
Hypothesis: <change; why flopscope counts fewer FLOPs; why result is unchanged>
Change:     <1-3 line diff summary>
Baseline:   flops=<..>  mse=<..>
N=25:       flops=<..> (<+/-x.x%>)  mse=<..> (Δ<..>)  residual=<..>
N=100:      flops=<..> (<+/-x.x%>)  mse=<..> (Δ<..>)   [only if screen passed]
Verdict:    ACCEPT | REJECT (<reason>)
```

Keep a **"dead ends"** section so no disproven trick is retried.

---

## 11. Stop conditions

- Stop when several consecutive hypotheses yield no confirmed win (local optimum);
  report cumulative reduction and the ordered list of accepted tricks.
- Stop immediately if you cannot lower `flops_used` without moving MSE past the §3
  lock — do not relax the lock.
- Always finish with the §7.8 full-scale verification and a summary: cumulative
  `flops_used` reduction %, `final_layer_mse` before/after (must match), and the
  accepted trick stack.
