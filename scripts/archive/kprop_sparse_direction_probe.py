"""Probe sparsity/selectability of the external k=3 correction direction.

This offline diagnostic asks whether the useful shrinkage direction
    d = kprop_final - active_final
is concentrated in a small subset of final neurons and whether deployable masks
(final class, alpha, analytical disagreement, final weight norms) can select it.

Run from the external k-prop repo environment:
    cd .tmp/mlp_cumulant_propagation
    uv run --with whestbench --with flopscope python ../../scripts/kprop_sparse_direction_probe.py
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
from estimator import Estimator
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


def active_prediction(row, sobol_points: np.ndarray, sample_count: int) -> tuple[np.ndarray, int]:
    mlp = make_wmlp(row)
    variant = next(v for v in ev.VARIANTS if v.name == "pilot l29+30 borderline 4/-4")
    old_n_samples = ev.N_SAMPLES
    try:
        ev.N_SAMPLES = sample_count
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            pred = np.asarray(ev.predict_variant(mlp, sobol_points, variant), dtype=np.float64)
    finally:
        ev.N_SAMPLES = old_n_samples
    return pred, int(ctx.flops_used)


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


def class_mask(indices) -> np.ndarray:
    mask = np.zeros(WIDTH, dtype=bool)
    if len(indices):
        mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def top_mask(score: np.ndarray, keep: float) -> np.ndarray:
    count = max(1, int(round(keep * score.size)))
    selected = np.argpartition(score, -count)[-count:]
    mask = np.zeros(score.size, dtype=bool)
    mask[selected] = True
    return mask


def fit_lambda(active: list[np.ndarray], direction: list[np.ndarray], truth: list[np.ndarray]) -> float:
    residual = np.concatenate([target - base for base, target in zip(active, truth)])
    flat_direction = np.concatenate(direction)
    denom = float(flat_direction @ flat_direction)
    if denom <= 1e-30:
        return 0.0
    return float((flat_direction @ residual) / denom)


def mean_mse(preds: list[np.ndarray], truths: list[np.ndarray]) -> float:
    return float(np.mean([np.mean((pred - truth) ** 2) for pred, truth in zip(preds, truths)]))


def masked_direction(direction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(direction)
    out[mask] = direction[mask]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=30720)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--keep", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    keep_values = [float(item) for item in args.keep.split(",") if item]
    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[index] for index in indices]
    sobol_points = np.load(args.sobol)["points"]

    active_preds = []
    kprop_preds = []
    truths = []
    structures = []
    active_full = []
    print(f"sparse kprop direction probe indices={indices} train={indices[:args.train_count]} test={indices[args.train_count:]} sample_count={args.sample_count}")
    for index, row in zip(indices, rows):
        full_pred, _ = active_prediction(row, sobol_points, args.sample_count)
        kpred, elapsed = kprop_final(row, args.k_max)
        truth = np.asarray(row["final_means"], dtype=np.float64)
        mlp = make_wmlp(row)
        structure = Estimator()._initial_structure(mlp, mlp.width)
        active_full.append(full_pred)
        active_preds.append(full_pred[-1])
        kprop_preds.append(kpred)
        truths.append(truth)
        structures.append(structure)
        print(
            f"{index:4d} {row['mlp_name']:<24} active={mean_mse([full_pred[-1]], [truth]):.3e} "
            f"k{args.k_max}={mean_mse([kpred], [truth]):.3e} time={elapsed:.2f}s",
            flush=True,
        )

    directions = [k - a for a, k in zip(active_preds, kprop_preds)]
    train_n = args.train_count
    train_slice = slice(0, train_n)
    test_slice = slice(train_n, None)
    print(f"\nbase train={mean_mse(active_preds[train_slice], truths[train_slice]):.3e} test={mean_mse(active_preds[test_slice], truths[test_slice]):.3e}")
    print(f"{'mask':<22} {'keep':>6} {'lambda':>9} {'train_mse':>12} {'test_mse':>12} {'test_oracle':>12} {'avg_active':>10}")

    mask_families = []
    for keep in keep_values:
        mask_families.append(("oracle_abs_d", keep, [top_mask(np.abs(direction), keep) for direction in directions]))
        mask_families.append(("abs_base_anal", keep, [top_mask(np.abs(active[-1] - np.asarray(structure["analytical_rows"][-1], dtype=np.float64)), keep) for active, structure in zip(active_full, structures)]))
        mask_families.append(("abs_active", keep, [top_mask(np.abs(active[-1]), keep) for active in active_full]))
        mask_families.append(("final_alpha", keep, [top_mask(np.asarray(structure["alpha_rows"][-1], dtype=np.float64), keep) for structure in structures]))
        mask_families.append(("final_on_alpha", keep, [class_mask(structure["on_indices"][-1]) & top_mask(np.asarray(structure["alpha_rows"][-1], dtype=np.float64), keep) for structure in structures]))
    mask_families.extend(
        [
            ("final_on", 1.0, [class_mask(structure["on_indices"][-1]) for structure in structures]),
            ("final_kink", 1.0, [class_mask(structure["kink_indices"][-1]) for structure in structures]),
        ]
    )

    for name, keep, masks in mask_families:
        dirs = [masked_direction(direction, mask) for direction, mask in zip(directions, masks)]
        lam = fit_lambda(active_preds[train_slice], dirs[train_slice], truths[train_slice])
        lam = float(np.clip(lam, -2.0, 2.0))
        oracle_lam = fit_lambda(active_preds[test_slice], dirs[test_slice], truths[test_slice])
        oracle_lam = float(np.clip(oracle_lam, -2.0, 2.0))
        train_pred = [base + lam * direction for base, direction in zip(active_preds[train_slice], dirs[train_slice])]
        test_pred = [base + lam * direction for base, direction in zip(active_preds[test_slice], dirs[test_slice])]
        oracle_pred = [base + oracle_lam * direction for base, direction in zip(active_preds[test_slice], dirs[test_slice])]
        print(
            f"{name:<22} {keep:6.2f} {lam:+9.4f} {mean_mse(train_pred, truths[train_slice]):12.3e} "
            f"{mean_mse(test_pred, truths[test_slice]):12.3e} {mean_mse(oracle_pred, truths[test_slice]):12.3e} "
            f"{np.mean([mask.mean() for mask in masks]):10.3f}"
        )


if __name__ == "__main__":
    main()
