"""Offline probe: can external k-prop act as a prior for low-sample active-set estimates?

Run from the external k-prop environment, for example:

    cd .tmp/mlp_cumulant_propagation
    uv run --with whestbench --with flopscope python ../../scripts/kprop_shrinkage_probe.py

The script compares public-mini baked ground truth against:
- external mlp_kprop k=3 final post-ReLU means
- our active-set estimator via eval_variants at several fixed sample counts
- scalar shrinkage blends active + lambda * (kprop - active)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
KPROP_ROOT = REPO_ROOT / ".tmp" / "mlp_cumulant_propagation"
for path in (REPO_ROOT, KPROP_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_variants as ev
import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from mlp_kprop.kprop_harmonic import mlp_kprop
from mlp_kprop.mlp import MLP as KMLP
from whestbench.domain import MLP as WMLP

WIDTH = 256
DEPTH = 32
BUDGET = ev.BUDGET


def make_wmlp(row) -> WMLP:
    weights = [fnp.array(np.asarray(weight, dtype=np.float32)) for weight in row["weights"]]
    return WMLP(width=WIDTH, depth=DEPTH, weights=weights, seed=int(row["mlp_seed"]))


def make_kmlp(row) -> KMLP:
    weights = [np.asarray(weight, dtype=np.float32) for weight in row["weights"]]
    mlp = KMLP(
        input_dim=WIDTH,
        hidden_dim=WIDTH,
        output_dim=WIDTH,
        num_layers=DEPTH + 1,
        nonlin="relu",
        init_kind="manual",
        w_var=1.0,
        b_var=0.0,
    ).to("cpu")
    with torch.no_grad():
        for layer_idx, weight in enumerate(weights):
            mlp.Ws[layer_idx].weight.copy_(torch.from_numpy(weight.T))
            if mlp.Ws[layer_idx].bias is not None:
                mlp.Ws[layer_idx].bias.zero_()
        mlp.Ws[DEPTH].weight.copy_(torch.eye(WIDTH))
        if mlp.Ws[DEPTH].bias is not None:
            mlp.Ws[DEPTH].bias.zero_()
    return mlp


def kprop_final(row, k_max: int) -> tuple[np.ndarray, float]:
    mlp = make_kmlp(row)
    k_in = {1: torch.zeros(WIDTH), 2: torch.eye(WIDTH)}
    start = time.perf_counter()
    tower = mlp_kprop(
        mlp,
        k_in,
        k_max=k_max,
        output_all=True,
        up_to_layer="act31",
        output_d_max=1,
        factor=k_max in (3, 4),
    )
    elapsed = time.perf_counter() - start
    return tower["act31"][1].to_tensor().detach().numpy().astype(np.float64), elapsed


def active_final(row, sobol_points: np.ndarray, sample_count: int) -> tuple[np.ndarray, int]:
    mlp = make_wmlp(row)
    variant = next(v for v in ev.VARIANTS if v.name == "pilot l29+30 borderline 4/-4")
    old_n_samples = ev.N_SAMPLES
    try:
        ev.N_SAMPLES = sample_count
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            pred = np.asarray(ev.predict_variant(mlp, sobol_points, variant), dtype=np.float64)[-1]
    finally:
        ev.N_SAMPLES = old_n_samples
    return pred, int(ctx.flops_used)


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((pred - truth) ** 2))


def fit_lambda(active: list[np.ndarray], kprop: list[np.ndarray], truth: list[np.ndarray]) -> float:
    residual = np.concatenate([target - base for base, target in zip(active, truth)])
    direction = np.concatenate([prior - base for base, prior in zip(active, kprop)])
    denom = float(direction @ direction)
    if denom <= 1e-30:
        return 0.0
    return float((direction @ residual) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--sample-counts", default="8192,16384,30720")
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    sample_counts = [int(item) for item in args.sample_counts.split(",") if item]
    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[index] for index in indices]
    truths = [np.asarray(row["final_means"], dtype=np.float64) for row in rows]
    train_slice = slice(0, args.train_count)
    test_slice = slice(args.train_count, None)
    sobol_points = np.load(args.sobol)["points"]

    print(f"kprop shrinkage probe split={args.split} indices={indices} train={indices[:args.train_count]} test={indices[args.train_count:]}")
    kprop_preds = []
    kprop_times = []
    for index, row in zip(indices, rows):
        pred, elapsed = kprop_final(row, args.k_max)
        kprop_preds.append(pred)
        kprop_times.append(elapsed)
        print(f"k{args.k_max} {index:4d} {row['mlp_name']:<24} mse={mse(pred, truths[len(kprop_preds)-1]):.3e} time={elapsed:.2f}s", flush=True)

    kprop_train = float(np.mean([mse(pred, truth) for pred, truth in zip(kprop_preds[train_slice], truths[train_slice])]))
    kprop_test = float(np.mean([mse(pred, truth) for pred, truth in zip(kprop_preds[test_slice], truths[test_slice])]))
    print(f"k{args.k_max} summary train_mse={kprop_train:.3e} test_mse={kprop_test:.3e} mean_time={np.mean(kprop_times):.2f}s")

    print(f"\n{'samples':>8} {'lambda':>9} {'train_active':>13} {'train_blend':>13} {'test_active':>13} {'test_blend':>13} {'test_oracle':>13} {'flops':>12}")
    for sample_count in sample_counts:
        active_preds = []
        flops_used = []
        for row in rows:
            pred, flops_count = active_final(row, sobol_points, sample_count)
            active_preds.append(pred)
            flops_used.append(flops_count)
        fitted_lambda = fit_lambda(active_preds[train_slice], kprop_preds[train_slice], truths[train_slice])
        fitted_lambda = float(np.clip(fitted_lambda, -2.0, 2.0))
        oracle_lambda = fit_lambda(active_preds[test_slice], kprop_preds[test_slice], truths[test_slice])
        oracle_lambda = float(np.clip(oracle_lambda, -2.0, 2.0))
        train_active = float(np.mean([mse(pred, truth) for pred, truth in zip(active_preds[train_slice], truths[train_slice])]))
        test_active = float(np.mean([mse(pred, truth) for pred, truth in zip(active_preds[test_slice], truths[test_slice])]))
        train_blend = float(
            np.mean([
                mse(base + fitted_lambda * (prior - base), truth)
                for base, prior, truth in zip(active_preds[train_slice], kprop_preds[train_slice], truths[train_slice])
            ])
        )
        test_blend = float(
            np.mean([
                mse(base + fitted_lambda * (prior - base), truth)
                for base, prior, truth in zip(active_preds[test_slice], kprop_preds[test_slice], truths[test_slice])
            ])
        )
        test_oracle = float(
            np.mean([
                mse(base + oracle_lambda * (prior - base), truth)
                for base, prior, truth in zip(active_preds[test_slice], kprop_preds[test_slice], truths[test_slice])
            ])
        )
        print(
            f"{sample_count:8d} {fitted_lambda:+9.4f} {train_active:13.3e} {train_blend:13.3e} "
            f"{test_active:13.3e} {test_blend:13.3e} {test_oracle:13.3e} {np.mean(flops_used):12.2e}"
        )


if __name__ == "__main__":
    main()
