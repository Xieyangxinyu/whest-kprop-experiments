"""Hyperparameter sweep of the 318873+dtype-hygiene surface under flopscope
0.9.1 billing.

Scope: EXACT-REROUTING knobs only (Strassen guards, cold-slice thresholds and
caps, pilot-cold alpha). These change how the same numbers are computed, so
predictions move only at fp-reorder level and variants rank by billed FLOPs;
raw MSE vs the mini split's stored final_means is tracked as a guard.
Statistical knobs (classification thresholds, pilot sizing) are NOT swept —
those axes were closed by earlier accuracy campaigns.

Usage:
  uv run python scripts/hp_sweep_09x.py            # stage A: one-at-a-time
  N_NETS=10 CONFIGS=combo uv run python ...        # stage B: combos on 10 nets
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import flopscope as flops
import whestbench as wb

EST_PATH = REPO / "examples" / "318937.py"   # dtype-hygiene bytes (ship base)
N_NETS = int(os.environ.get("N_NETS", "3"))
BUDGET = 272_000_000_000
CONFIG_SET = os.environ.get("CONFIGS", "stageA")

spec = importlib.util.spec_from_file_location("est_sweep", EST_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

points = np.load(REPO / "sobol_points.npz")["points"]
ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
nets = [(wb.mlp_at(ds, i), np.asarray(ds[i]["final_means"], dtype=np.float64)) for i in range(N_NETS)]

BASE = {}  # filled from module on first use

STAGE_A = [
    ("base", {}),
    # Strassen guards (concat 1/elem changed recursion economics)
    ("strassen_min_dim=64", {"_STRASSEN_MIN_DIM": 64}),
    ("strassen_min_dim=128", {"_STRASSEN_MIN_DIM": 128}),
    ("strassen_depth=2", {"_STRASSEN_MAX_RECURSE_DEPTH": 2}),
    ("strassen_depth=2+dim64", {"_STRASSEN_MAX_RECURSE_DEPTH": 2, "_STRASSEN_MIN_DIM": 64}),
    ("strassen_depth=0", {"_STRASSEN_MAX_RECURSE_DEPTH": 0}),
    ("strassen_min_rows=2048", {"_STRASSEN_MIN_ROWS": 2048}),
    ("strassen_min_rows=8192", {"_STRASSEN_MIN_ROWS": 8192}),
    # Cold-slice continuation knobs
    ("cold_fire=0.02", {"_COLD_FIRE_THRESH": 0.02}),
    ("cold_fire=0.05", {"_COLD_FIRE_THRESH": 0.05}),
    ("cold_max_k=48", {"_COLD_MAX_K": 48}),
    ("cold_max_k=96", {"_COLD_MAX_K": 96}),
    ("cold_fire2=0.005", {"_COLD_FIRE_THRESH2": 0.005}),
    ("cold_fire2=0.02", {"_COLD_FIRE_THRESH2": 0.02}),
    ("cold_min_hot=64", {"_COLD_MIN_HOT_DIM": 64}),
    # Pilot-block cold knobs
    ("pilot_alpha=-2.17", {"_PILOT_ALPHA_COLD": -2.17}),   # Phi ~ 1.5%
    ("pilot_alpha=-1.645", {"_PILOT_ALPHA_COLD": -1.645}), # Phi ~ 5%
    ("pilot_max_k=96", {"_PILOT_MAX_K": 96}),
]

COMBO = [("combo", json.loads(os.environ.get("COMBO_JSON", "{}")))]

# Statistical knobs: staged-probe sizing (5%/20%/0.35 chosen pre-antithetic,
# pre-0.9.x). Raw MSE moves are REAL here — run at N_NETS=10, judge raw
# before F.
PROBES = [
    ("base", {}),
    ("pilot_frac=0.025", {"_PILOT_FRACTION": 0.025}),
    ("pilot_frac=0.10", {"_PILOT_FRACTION": 0.10}),
    ("recheck_frac=0.10", {"_PILOT_RECHECK_FRACTION": 0.10}),
    ("recheck_frac=0.30", {"_PILOT_RECHECK_FRACTION": 0.30}),
    ("margin=0.25", {"_PILOT_RECHECK_MARGIN": 0.25}),
    ("margin=0.50", {"_PILOT_RECHECK_MARGIN": 0.50}),
    ("frac.025+re.10", {"_PILOT_FRACTION": 0.025, "_PILOT_RECHECK_FRACTION": 0.10}),
]

configs = {"stageA": STAGE_A, "combo": COMBO, "probes": PROBES}[CONFIG_SET]


def run_config(name, overrides):
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(mod, k)
        setattr(mod, k, v)
    try:
        fs, mses = [], []
        for mlp, gt in nets:
            est = mod.Estimator()
            est._sobol_points = points
            with flops.BudgetContext(flop_budget=int(1e12), quiet=True) as ctx:
                pred = est.predict(mlp, BUDGET)
            f = ctx.flops_used
            last = np.asarray(pred[-1], dtype=np.float64)
            mses.append(float(np.mean((last - gt) ** 2)))
            fs.append(f)
        return float(np.mean(fs)), float(np.mean(mses))
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)


base_f = base_mse = None
print(f"nets={N_NETS} budget={BUDGET/1e9:.0f}G  (FLOP-only multiplier; raw = guard)")
print(f"{'config':<28} {'F (G)':>9} {'dF':>8} {'raw MSE':>11} {'draw':>8} {'adj (F-only)':>13} {'dadj':>8}")
for name, overrides in configs:
    f, mse = run_config(name, overrides)
    adj = mse * max(0.1, f / BUDGET)
    if base_f is None:
        base_f, base_mse, base_adj = f, mse, adj
    print(f"{name:<28} {f/1e9:>9.2f} {f/base_f-1:>+7.2%} {mse:>11.4e} "
          f"{mse/base_mse-1:>+7.2%} {adj:>13.4e} {adj/(base_adj)-1:>+7.2%}")
