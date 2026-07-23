"""Profile current estimator: FLOP cost + wall time per op (1 mini MLP)."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from datasets import load_dataset
from whestbench import SetupContext
from whestbench.domain import MLP

from estimator import Estimator

FLOP_BUDGET = 272_000_000_000

ds = load_dataset("aicrowd/arc-whestbench-public-2026", split="mini", streaming=True)
mlp_dict = next(iter(ds))
weights = [np.array(w, dtype=np.float32) for w in mlp_dict["weights"]]
mlp = MLP(width=256, depth=32, weights=weights, seed=mlp_dict["mlp_seed"])

estimator = Estimator()
ctx = SetupContext(seed=0, width=256, depth=32, flop_budget=FLOP_BUDGET,
                   api_version="v1", submission_dir=str(Path.cwd()))
estimator.setup(ctx)
estimator._load_sobol_points()

t0 = time.perf_counter()
with flops.BudgetContext(flop_budget=FLOP_BUDGET):
    result = estimator.predict(mlp, FLOP_BUDGET)
wall_total = time.perf_counter() - t0

summary = flops.budget_summary_dict(by_namespace=False)
ops = summary["operations"]

categories = {
    "matmul": ["matmul", "matvec", "vecmat", "dot", "inner", "vdot",
               "einsum", "tensordot", "vecdot"],
    "sort/argsort/partition": ["sort", "argsort", "partition", "searchsorted"],
    "take/gather/index": ["take", "take_along_axis", "choose", "__getitem__"],
    "concatenate/stack": ["concatenate", "stack", "hstack", "vstack", "dstack"],
    "where/maximum": ["where", "maximum", "minimum", "clip"],
    "reductions": ["sum", "mean", "var", "std", "max", "min", "prod",
                   "count_nonzero", "nonzero", "searchsorted"],
    "elementwise": ["add", "multiply", "subtract", "divide", "negative",
                   "greater", "less", "equal", "not_equal", "abs", "sqrt",
                   "square", "exp", "log", "sin", "cos", "power",
                   "logical_and", "logical_or", "logical_not"],
    "creation": ["random.normal", "zeros", "ones", "eye", "full", "empty",
                "array", "asarray", "arange"],
    "scatter": ["put", "put_along_axis", "place", "fill_diagonal"],
    "reshape/broadcast": ["reshape", "broadcast_to", "expand_dims", "ravel",
                          "flatten", "squeeze"],
    "astype": ["astype"],
    "other": [],
}

cat_data = defaultdict(lambda: {"flops": 0, "calls": 0, "wall": 0.0, "ops": []})
for op_name, op_info in ops.items():
    flop_cost = op_info["flop_cost"]
    wall = op_info.get("flopscope_backend_time_s", 0)
    calls = op_info["calls"]
    matched = False
    for cat, names in categories.items():
        for name in names:
            if op_name == name or op_name.startswith(name + "."):
                cat_data[cat]["flops"] += flop_cost
                cat_data[cat]["calls"] += calls
                cat_data[cat]["wall"] += wall
                cat_data[cat]["ops"].append(op_name)
                matched = True
                break
        if matched:
            break
    if not matched:
        cat_data["other"]["flops"] += flop_cost
        cat_data["other"]["calls"] += calls
        cat_data["other"]["wall"] += wall
        cat_data["other"]["ops"].append(op_name)

total_flops = summary["flops_used"]
total_wall = summary["flopscope_backend_time_s"]

print("=" * 90)
print(f"Estimator profile on 1 mini MLP")
print(f"Total FLOPs:       {total_flops:>17,.0f}")
print(f"  Backend:         {summary['flopscope_backend_time_s']:>17.3f}s")
print(f"  Overhead:        {summary['flopscope_overhead_time_s']:>17.3f}s")
print(f"  Residual:        {summary['residual_wall_time_s']:>17.3f}s")
print(f"  Total wall:      {wall_total:>17.3f}s")
print()
print(f"{'Category':<30} {'FLOPs':>14} {'%FLOP':>7} {'Wall(s)':>9} {'%Wall':>7} {'Calls':>7}")
print("-" * 90)
for cat, data in sorted(cat_data.items(), key=lambda x: -x[1]["flops"]):
    pct_flops = data["flops"] / total_flops * 100 if total_flops else 0
    pct_wall = data["wall"] / total_wall * 100 if total_wall else 0
    print(f"{cat:<30} {data['flops']:>14,.0f} {pct_flops:>6.1f}% {data['wall']:>9.3f} {pct_wall:>6.1f}% {data['calls']:>7}")

print("-" * 90)
print("\nTop 20 individual ops by FLOP cost:")
print(f"{'Op name':<40} {'FLOPs':>14} {'%FLOP':>7} {'Wall(s)':>9} {'Calls':>7}")
print("-" * 80)
by_flops = sorted(ops.items(), key=lambda x: -x[1]["flop_cost"])
for op_name, op_info in by_flops[:20]:
    pct = op_info["flop_cost"] / total_flops * 100 if total_flops else 0
    print(f"{op_name:<40} {op_info['flop_cost']:>14,.0f} {pct:>6.1f}% {op_info.get('flopscope_backend_time_s', 0):>9.3f} {op_info['calls']:>7}")

print("\nTop 20 individual ops by wall time:")
print(f"{'Op name':<40} {'Wall(s)':>9} {'FLOPs':>14} {'Calls':>7}")
print("-" * 80)
by_wall = sorted(ops.items(), key=lambda x: -x[1].get("flopscope_backend_time_s", 0))
for op_name, op_info in by_wall[:20]:
    print(f"{op_name:<40} {op_info.get('flopscope_backend_time_s', 0):>9.3f} {op_info['flop_cost']:>14,.0f} {op_info['calls']:>7}")
