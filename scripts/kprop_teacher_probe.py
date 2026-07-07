"""Probe whether cheap deployable features can distill external k-prop directions.

This is an offline diagnostic. It uses external mlp_kprop as a teacher and asks:
can per-final-neuron features available to our estimator predict the direction
(kprop_final - active_final) well enough to improve held-out final MSE?

Run from the external k-prop repo environment, e.g.:

    cd .tmp/mlp_cumulant_propagation
    uv run --with whestbench --with flopscope python ../../scripts/kprop_teacher_probe.py
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


def class_one_hot(indices, width: int) -> np.ndarray:
    out = np.zeros(width, dtype=np.float64)
    if len(indices):
        out[np.asarray(indices, dtype=np.int64)] = 1.0
    return out


def features_for(row, active_pred: np.ndarray) -> np.ndarray:
    mlp = make_wmlp(row)
    structure = Estimator()._initial_structure(mlp, mlp.width)
    weights = [np.asarray(weight, dtype=np.float64) for weight in row["weights"]]
    final_w = weights[-1]
    abs_w = np.abs(final_w)
    active_final = active_pred[-1]
    active_l30 = active_pred[-2]
    analytical_final = np.asarray(structure["analytical_rows"][-1], dtype=np.float64)
    alpha_final = np.asarray(structure["alpha_rows"][-1], dtype=np.float64)
    alpha_l30 = np.asarray(structure["alpha_rows"][-2], dtype=np.float64)
    final_on = class_one_hot(structure["on_indices"][-1], WIDTH)
    final_kink = class_one_hot(structure["kink_indices"][-1], WIDTH)
    final_dead = class_one_hot(structure["dead_indices"][-1], WIDTH)
    l30_on = class_one_hot(structure["on_indices"][-2], WIDTH)
    l30_kink = class_one_hot(structure["kink_indices"][-2], WIDTH)
    l30_dead = class_one_hot(structure["dead_indices"][-2], WIDTH)

    global_values = np.array(
        [
            np.mean(active_final),
            np.std(active_final),
            np.mean(active_l30),
            np.std(active_l30),
            np.mean(final_on),
            np.mean(final_kink),
            np.mean(l30_on),
            np.mean(l30_kink),
        ],
        dtype=np.float64,
    )
    global_tiled = np.repeat(global_values[None, :], WIDTH, axis=0)

    per_neuron = np.stack(
        [
            np.ones(WIDTH),
            active_final,
            np.log1p(np.abs(active_final)),
            analytical_final,
            active_final - analytical_final,
            alpha_final,
            np.clip(alpha_final, -8.0, 8.0),
            final_on,
            final_kink,
            final_dead,
            final_w.T @ active_l30,
            abs_w.T @ active_l30,
            np.sqrt(np.maximum((final_w * final_w).T @ (active_l30 * active_l30), 1e-12)),
            np.sum(abs_w, axis=0),
            np.sqrt(np.sum(final_w * final_w, axis=0)),
            np.max(abs_w, axis=0),
            final_w.T @ l30_on,
            final_w.T @ l30_kink,
            final_w.T @ l30_dead,
            abs_w.T @ l30_on,
            abs_w.T @ l30_kink,
            abs_w.T @ l30_dead,
            alpha_l30.mean() * np.ones(WIDTH),
            alpha_l30.std() * np.ones(WIDTH),
        ],
        axis=1,
    )
    return np.concatenate([per_neuron, global_tiled], axis=1)


def expand_features(features: np.ndarray) -> np.ndarray:
    signed_sqrt = np.sign(features) * np.sqrt(np.abs(features) + 1e-12)
    logs = np.log1p(np.abs(features))
    selected = np.stack(
        [
            features[:, 1] * features[:, 5],
            features[:, 4] * features[:, 7],
            features[:, 4] * features[:, 8],
            features[:, 10] * features[:, 16],
            features[:, 13] * features[:, 7],
            features[:, 14] * features[:, 8],
        ],
        axis=1,
    )
    return np.concatenate([features, signed_sqrt, logs, selected], axis=1)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train_x - center) / scale, (test_x - center) / scale


def fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    xtx = features.T @ features
    rhs = features.T @ target
    return np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), rhs)


def fit_lambda(active: list[np.ndarray], direction: list[np.ndarray], truth: list[np.ndarray]) -> float:
    residual = np.concatenate([target - base for base, target in zip(active, truth)])
    flat_direction = np.concatenate(direction)
    denom = float(flat_direction @ flat_direction)
    if denom <= 1e-30:
        return 0.0
    return float((flat_direction @ residual) / denom)


def mean_mse(preds: list[np.ndarray], truths: list[np.ndarray]) -> float:
    return float(np.mean([np.mean((pred - truth) ** 2) for pred, truth in zip(preds, truths)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=30720)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--ridges", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    ridges = [float(item) for item in args.ridges.split(",") if item]
    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[index] for index in indices]
    sobol_points = np.load(args.sobol)["points"]

    active_preds = []
    kprop_preds = []
    truths = []
    feature_rows = []
    kprop_times = []
    flops_used = []
    print(f"teacher probe indices={indices} train={indices[:args.train_count]} test={indices[args.train_count:]} sample_count={args.sample_count}")
    for index, row in zip(indices, rows):
        active, flops_count = active_prediction(row, sobol_points, args.sample_count)
        kpred, elapsed = kprop_final(row, args.k_max)
        truth = np.asarray(row["final_means"], dtype=np.float64)
        active_preds.append(active[-1])
        kprop_preds.append(kpred)
        truths.append(truth)
        feature_rows.append(features_for(row, active))
        kprop_times.append(elapsed)
        flops_used.append(flops_count)
        print(
            f"{index:4d} {row['mlp_name']:<24} active={mean_mse([active[-1]], [truth]):.3e} "
            f"k{args.k_max}={mean_mse([kpred], [truth]):.3e} time={elapsed:.2f}s",
            flush=True,
        )

    train_n = args.train_count
    train_slice = slice(0, train_n)
    test_slice = slice(train_n, None)
    train_x = np.concatenate(feature_rows[train_slice], axis=0)
    test_x = np.concatenate(feature_rows[test_slice], axis=0)
    train_x, test_x = standardize(expand_features(train_x), expand_features(test_x))
    train_teacher = np.concatenate([k - a for a, k in zip(active_preds[train_slice], kprop_preds[train_slice])])
    test_teacher = np.concatenate([k - a for a, k in zip(active_preds[test_slice], kprop_preds[test_slice])])
    train_residual = np.concatenate([t - a for a, t in zip(active_preds[train_slice], truths[train_slice])])
    test_residual = np.concatenate([t - a for a, t in zip(active_preds[test_slice], truths[test_slice])])

    train_active = mean_mse(active_preds[train_slice], truths[train_slice])
    test_active = mean_mse(active_preds[test_slice], truths[test_slice])
    train_kprop = mean_mse(kprop_preds[train_slice], truths[train_slice])
    test_kprop = mean_mse(kprop_preds[test_slice], truths[test_slice])
    teacher_lambda = fit_lambda(active_preds[train_slice], [k - a for a, k in zip(active_preds[train_slice], kprop_preds[train_slice])], truths[train_slice])
    teacher_lambda = float(np.clip(teacher_lambda, -2.0, 2.0))
    test_teacher_blend = mean_mse(
        [a + teacher_lambda * (k - a) for a, k in zip(active_preds[test_slice], kprop_preds[test_slice])],
        truths[test_slice],
    )
    print(
        f"\nsummary active train={train_active:.3e} test={test_active:.3e} "
        f"kprop train={train_kprop:.3e} test={test_kprop:.3e} "
        f"teacher_lambda={teacher_lambda:+.4f} teacher_test={test_teacher_blend:.3e} "
        f"mean_kprop_time={np.mean(kprop_times):.2f}s mean_active_flops={np.mean(flops_used):.2e}"
    )
    print(f"{'target':<12} {'ridge':>9} {'lambda':>9} {'train_mse':>12} {'test_mse':>12} {'corr_test':>10}")
    for target_name, train_target, test_target in (
        ("teacher", train_teacher, test_teacher),
        ("residual", train_residual, test_residual),
    ):
        for ridge in ridges:
            coef = fit_ridge(train_x, train_target, ridge)
            train_dir_flat = train_x @ coef
            test_dir_flat = test_x @ coef
            train_dirs = list(train_dir_flat.reshape(train_n, WIDTH))
            test_dirs = list(test_dir_flat.reshape(len(indices) - train_n, WIDTH))
            fitted_lambda = fit_lambda(active_preds[train_slice], train_dirs, truths[train_slice])
            fitted_lambda = float(np.clip(fitted_lambda, -2.0, 2.0))
            train_blend = mean_mse(
                [base + fitted_lambda * direction for base, direction in zip(active_preds[train_slice], train_dirs)],
                truths[train_slice],
            )
            test_blend = mean_mse(
                [base + fitted_lambda * direction for base, direction in zip(active_preds[test_slice], test_dirs)],
                truths[test_slice],
            )
            corr = float(np.corrcoef(test_dir_flat, test_target)[0, 1]) if np.std(test_dir_flat) > 0 and np.std(test_target) > 0 else 0.0
            print(f"{target_name:<12} {ridge:9.3g} {fitted_lambda:+9.4f} {train_blend:12.3e} {test_blend:12.3e} {corr:+10.3f}")


if __name__ == "__main__":
    main()
