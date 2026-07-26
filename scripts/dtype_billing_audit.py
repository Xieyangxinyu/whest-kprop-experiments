"""Dtype vetting of the shipped estimator: profile a real predict() and
break billed FLOPs down by resolved billing dtype, op by op.

flopscope 0.9.x bills `flop_cost x dtype_rate x complex_factor x weight`
(f64/i64 rate 2.0, f32 rate 1.0) and records every op's resolved dtype in
BudgetContext.op_log. This audit runs the estimator on a mini-split MLP
(float64 weights, the loader-realistic worst case) and reports:

  1. billed FLOPs by resolved dtype (+ the premium paid vs a rate-1.0 world)
  2. every (op, dtype) pair billing above f32 rate, ranked by cost
  3. the shipped sobol_points.npz storage dtype

Usage: uv run python scripts/dtype_billing_audit.py
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import flopscope as flops
from flopscope._dtype_billing import rate_for
import whestbench as wb

spec = importlib.util.spec_from_file_location("est_audit", REPO / "estimator.py")
est_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est_mod)

d = np.load(REPO / "sobol_points.npz")
print("sobol_points.npz storage:", {k: str(d[k].dtype) for k in d.files})

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
mlp = wb.mlp_at(ds, 0)
print("mlp.weights dtype as loaded:", np.asarray(mlp.weights[0]).dtype)

est = est_mod.Estimator()
with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as ctx:
    pred = est.predict(mlp, int(272e9))
print(f"predict() returned dtype={pred.dtype}; total billed {ctx.flops_used:,}\n")

by_dtype = defaultdict(int)
by_op_dtype = defaultdict(lambda: [0, 0])  # billed, count
for op in ctx.op_log:
    dt = op.resolved_dtype or "<untyped>"
    by_dtype[dt] += op.flop_cost
    rec = by_op_dtype[(op.op_name, dt)]
    rec[0] += op.flop_cost
    rec[1] += 1

total = ctx.flops_used
print(f"{'resolved dtype':<14} {'billed':>16} {'share':>8} {'rate':>5} {'premium vs rate-1':>18}")
premium_total = 0
for dt, billed in sorted(by_dtype.items(), key=lambda kv: -kv[1]):
    try:
        rate = rate_for(np.dtype(dt)) if dt != "<untyped>" else 1.0
    except Exception:
        rate = 1.0
    prem = billed - int(billed / rate)
    premium_total += prem
    print(f"{dt:<14} {billed:>16,} {billed / total:>7.2%} {rate:>5.1f} {prem:>18,}")
print(f"\nTOTAL dtype premium (would vanish if everything billed at rate 1.0): "
      f"{premium_total:,} = {premium_total / total:.3%} of billed\n")

print(f"{'op':<22} {'dtype':<10} {'billed':>14} {'share':>9} {'calls':>7}")
rows = [(k, v) for k, v in by_op_dtype.items()]
shown = 0
for (opn, dt), (billed, cnt) in sorted(rows, key=lambda kv: -kv[1][0]):
    is_f32ish = dt in ("float32", "int32", "int16", "int8", "uint8", "bool", "float16", "<untyped>")
    if is_f32ish:
        continue
    print(f"{opn:<22} {dt:<10} {billed:>14,} {billed / total:>8.3%} {cnt:>7}")
    shown += 1
    if shown >= 20:
        break
if shown == 0:
    print("(no op billed above f32 rate)")

print(f"\n--- top 10 f32 ops for scale ---")
for (opn, dt), (billed, cnt) in sorted(rows, key=lambda kv: -kv[1][0])[:10]:
    print(f"{opn:<22} {dt:<10} {billed:>14,} {billed / total:>8.3%} {cnt:>7}")
