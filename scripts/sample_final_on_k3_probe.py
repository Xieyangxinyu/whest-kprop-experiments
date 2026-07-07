"""Probe sample-estimated final-on K3 corrections around the active-set estimator.

This is an offline diagnostic for a deployable-ish idea:
- keep the current active-set estimator as baseline
- use an extra sample pass to estimate final preactivation moments for selected
  final-on neurons
- convert sample mean/variance/K3 into an Edgeworth ReLU mean
- fit one damping coefficient on train MLPs and test on held-out MLPs

The full-network sample pass is intentionally explicit here to test the signal;
it is not yet a submission-ready implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_variants as ev
import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from estimator import Estimator
from learn_k4_downstream_gate import _edgeworth_mean
from whestbench.domain import MLP

WIDTH = 256
DEPTH = 32
BUDGET = ev.BUDGET


def make_mlp(row) -> MLP:
    weights = [fnp.array(np.asarray(weight, dtype=np.float32)) for weight in row["weights"]]
    return MLP(width=WIDTH, depth=DEPTH, weights=weights, seed=int(row["mlp_seed"]))


def active_prediction(row, sobol_points: np.ndarray, sample_count: int) -> tuple[np.ndarray, int]:
    mlp = make_mlp(row)
    variant = next(v for v in ev.VARIANTS if v.name == "pilot l29+30 borderline 4/-4")
    old_n_samples = ev.N_SAMPLES
    try:
        ev.N_SAMPLES = sample_count
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            pred = np.asarray(ev.predict_variant(mlp, sobol_points, variant), dtype=np.float64)
    finally:
        ev.N_SAMPLES = old_n_samples
    return pred, int(ctx.flops_used)


def load_samples(sobol_points: np.ndarray, sample_count: int) -> np.ndarray:
    half_count = sample_count // 2
    half = np.asarray(sobol_points[:half_count, :WIDTH], dtype=np.float32)
    return np.concatenate([half, -half], axis=0)


def final_pre_samples(row, samples: np.ndarray, selected: np.ndarray) -> np.ndarray:
    weights = [np.asarray(weight, dtype=np.float32) for weight in row["weights"]]
    x = samples
    for layer_idx in range(DEPTH - 1):
        x = np.maximum(x @ weights[layer_idx], 0.0)
    return x @ weights[-1][:, selected]


def central3(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return np.mean(centered * centered * centered, axis=0)


def top_mask(score: np.ndarray, keep: float) -> np.ndarray:
    count = max(1, int(round(keep * score.size)))
    selected = np.argpartition(score, -count)[-count:]
    mask = np.zeros(score.size, dtype=bool)
    mask[selected] = True
    return mask


def selected_mask(row, mode: str, keep: float, active_pred: np.ndarray) -> np.ndarray:
    mlp = make_mlp(row)
    structure = Estimator()._initial_structure(mlp, mlp.width)
    mask = np.zeros(WIDTH, dtype=bool)
    final_on = np.asarray(structure["on_indices"][-1], dtype=np.int64)
    final_kink = np.asarray(structure["kink_indices"][-1], dtype=np.int64)
    alpha = np.asarray(structure["alpha_rows"][-1], dtype=np.float64)
    analytical = np.asarray(structure["analytical_rows"][-1], dtype=np.float64)
    if mode == "final_on":
        mask[final_on] = True
        return mask
    if mode == "final_kink":
        mask[final_kink] = True
        return mask
    if mode == "top_on_alpha":
        if final_on.size == 0:
            return mask
        local_score = alpha[final_on]
        count = max(1, int(round(keep * final_on.size)))
        chosen = final_on[np.argpartition(local_score, -count)[-count:]]
        mask[chosen] = True
        return mask
    if mode == "top_on_anal_diff":
        if final_on.size == 0:
            return mask
        local_score = np.abs(active_pred[-1, final_on] - analytical[final_on])
        count = max(1, int(round(keep * final_on.size)))
        chosen = final_on[np.argpartition(local_score, -count)[-count:]]
        mask[chosen] = True
        return mask
    if mode == "top_active":
        mask = top_mask(np.abs(active_pred[-1]), keep)
        return mask
    raise ValueError(mode)


def fit_lambda(base: list[np.ndarray], directions: list[np.ndarray], truths: list[np.ndarray]) -> float:
    residual = np.concatenate([truth - pred for pred, truth in zip(base, truths)])
    direction = np.concatenate(directions)
    denom = float(direction @ direction)
    if denom <= 1e-30:
        return 0.0
    return float((direction @ residual) / denom)


def mean_mse(preds: list[np.ndarray], truths: list[np.ndarray]) -> float:
    return float(np.mean([np.mean((pred - truth) ** 2) for pred, truth in zip(preds, truths)]))


def correction_directions(row, active_pred: np.ndarray, samples: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    directions = {"sample_relu": np.zeros(WIDTH), "edge_sample": np.zeros(WIDTH), "gaussian_sample": np.zeros(WIDTH)}
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return directions
    pre = final_pre_samples(row, samples, selected)
    sample_relu = np.maximum(pre, 0.0).mean(axis=0)
    pre_mean = pre.mean(axis=0)
    centered = pre - pre_mean[None, :]
    pre_var = np.maximum(np.mean(centered * centered, axis=0), 1e-12)
    pre_k3 = central3(pre)
    zeros = np.zeros(selected.size)
    gaussian = _edgeworth_mean(pre_mean, pre_var, zeros, zeros)
    edge = _edgeworth_mean(pre_mean, pre_var, pre_k3, zeros)
    directions["sample_relu"][selected] = sample_relu - active_pred[-1, selected]
    directions["gaussian_sample"][selected] = gaussian - active_pred[-1, selected]
    directions["edge_sample"][selected] = edge - active_pred[-1, selected]
    return directions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--active-samples", type=int, default=30720)
    parser.add_argument("--correction-samples", default="4096,8192,16384")
    parser.add_argument("--modes", default="final_on,top_on_alpha,top_on_anal_diff,top_active")
    parser.add_argument("--keep", default="0.10,0.20,0.40")
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    correction_samples = [int(item) for item in args.correction_samples.split(",") if item]
    modes = [item for item in args.modes.split(",") if item]
    keep_values = [float(item) for item in args.keep.split(",") if item]
    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[index] for index in indices]
    sobol_points = np.load(args.sobol)["points"]
    truths = [np.asarray(row["final_means"], dtype=np.float64) for row in rows]
    active_preds = []
    flops_used = []
    for index, row in zip(indices, rows):
        pred, flops_count = active_prediction(row, sobol_points, args.active_samples)
        active_preds.append(pred)
        flops_used.append(flops_count)
        print(f"active {index:4d} {row['mlp_name']:<24} mse={mean_mse([pred[-1]], [truths[len(active_preds)-1]]):.3e}", flush=True)

    train_n = args.train_count
    train_slice = slice(0, train_n)
    test_slice = slice(train_n, None)
    base_final = [pred[-1] for pred in active_preds]
    print(f"\nbase train={mean_mse(base_final[train_slice], truths[train_slice]):.3e} test={mean_mse(base_final[test_slice], truths[test_slice]):.3e} mean_active_flops={np.mean(flops_used):.2e}")
    print(f"{'corrN':>6} {'mode':<18} {'keep':>6} {'kind':<16} {'lambda':>9} {'train_mse':>12} {'test_mse':>12} {'test_oracle':>12} {'active_frac':>11}")
    for corr_n in correction_samples:
        samples = load_samples(sobol_points, corr_n)
        for mode in modes:
            mode_keeps = [1.0] if mode in {"final_on", "final_kink"} else keep_values
            for keep in mode_keeps:
                masks = [selected_mask(row, pred, mode, keep) if False else None for row, pred in []]
                masks = [selected_mask(row, mode, keep, pred) for row, pred in zip(rows, active_preds)]
                all_dirs = []
                for row, pred, mask in zip(rows, active_preds, masks):
                    all_dirs.append(correction_directions(row, pred, samples, mask))
                for kind in ("sample_relu", "gaussian_sample", "edge_sample"):
                    dirs = [entry[kind] for entry in all_dirs]
                    lam = fit_lambda(base_final[train_slice], dirs[train_slice], truths[train_slice])
                    lam = float(np.clip(lam, -2.0, 2.0))
                    oracle_lam = fit_lambda(base_final[test_slice], dirs[test_slice], truths[test_slice])
                    oracle_lam = float(np.clip(oracle_lam, -2.0, 2.0))
                    train_pred = [base + lam * direction for base, direction in zip(base_final[train_slice], dirs[train_slice])]
                    test_pred = [base + lam * direction for base, direction in zip(base_final[test_slice], dirs[test_slice])]
                    oracle_pred = [base + oracle_lam * direction for base, direction in zip(base_final[test_slice], dirs[test_slice])]
                    print(
                        f"{corr_n:6d} {mode:<18} {keep:6.2f} {kind:<16} {lam:+9.4f} "
                        f"{mean_mse(train_pred, truths[train_slice]):12.3e} {mean_mse(test_pred, truths[test_slice]):12.3e} "
                        f"{mean_mse(oracle_pred, truths[test_slice]):12.3e} {np.mean([mask.mean() for mask in masks]):11.3f}"
                    )


if __name__ == "__main__":
    main()
