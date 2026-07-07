"""Analyze folded final-on projection stop layers.

For final-on neurons, the scored output is approximately the final pre-activation
mean. This script folds the final-on projection matrix backward through confident
on blocks down to a stop layer L, then compares:

  oracle: E[h^L]^T A_L using higher-moment true layer means
  sample: sample mean of h^L @ A_L using Sobol forward samples

It also reports the numerical rank of A_L. A useful top-submission-style solver
would have a stop layer where oracle error stays low and A_L is low-rank.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP

from estimator import Estimator
from learn_cumulant_closure import _moment_path, _weights
from sample_k3_projection_probe import _load_samples


BUDGET = 272_000_000_000
WIDTH = 256
DEPTH = 32


def _forward_layers(weights: np.ndarray, samples: np.ndarray) -> list[np.ndarray]:
    out = []
    x = samples
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        out.append(x)
    return out


def _fold_to_stop(weights: np.ndarray, alpha_rows: list[np.ndarray], final_on: np.ndarray, stop_layer: int) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = weights[-1][:, final_on].astype(np.float64)
    constant = np.zeros(a_matrix.shape[1], dtype=np.float64)
    for layer_idx in range(DEPTH - 2, stop_layer, -1):
        alpha = alpha_rows[layer_idx]
        dead = alpha < -3.0
        on = alpha > 3.0
        kink = (~dead) & (~on)
        # Contributions from kink/dead above stop are dropped into the constant
        # only if true means are supplied by the caller. Here we return A_stop;
        # caller accounts for dropped kink terms with true means for oracle.
        if np.any(on):
            a_matrix = weights[layer_idx][:, on].astype(np.float64) @ a_matrix[on, :]
        else:
            a_matrix = np.zeros((WIDTH, a_matrix.shape[1]), dtype=np.float64)
        a_matrix[dead, :] = 0.0
        _ = kink
    return a_matrix, constant


def _fold_oracle_projection(
    weights: np.ndarray, means: np.ndarray, alpha_rows: list[np.ndarray], final_on: np.ndarray, stop_layer: int
) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = weights[-1][:, final_on].astype(np.float64)
    total = np.zeros(a_matrix.shape[1], dtype=np.float64)
    for layer_idx in range(DEPTH - 2, stop_layer, -1):
        alpha = alpha_rows[layer_idx]
        dead = alpha < -3.0
        on = alpha > 3.0
        kink = (~dead) & (~on)
        if np.any(kink):
            total += means[layer_idx, kink] @ a_matrix[kink, :]
        if np.any(on):
            a_matrix = weights[layer_idx][:, on].astype(np.float64) @ a_matrix[on, :]
        else:
            a_matrix = np.zeros((WIDTH, a_matrix.shape[1]), dtype=np.float64)
        a_matrix[dead, :] = 0.0
    total += means[stop_layer] @ a_matrix
    return total, a_matrix


def _energy_ranks(a_matrix: np.ndarray) -> tuple[int, int, int]:
    if a_matrix.size == 0 or min(a_matrix.shape) == 0:
        return 0, 0, 0
    singular = np.linalg.svd(a_matrix, compute_uv=False)
    energy = np.cumsum(singular * singular)
    if energy[-1] <= 0:
        return 0, 0, 0
    energy /= energy[-1]
    return tuple(int(np.searchsorted(energy, q) + 1) for q in (0.90, 0.95, 0.99))


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="8,9,10,11")
    parser.add_argument("--sample-counts", default="2048,4096,8192")
    parser.add_argument("--stops", default="30,29,28,27,26,25,24,22,20,16")
    parser.add_argument("--sobol", default=str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    sample_counts = [int(item) for item in args.sample_counts.split(",") if item]
    stop_layers = [int(item) for item in args.stops.split(",") if item]
    print(f"indices={indices} sample_counts={sample_counts} stops={stop_layers}")

    for sample_count in sample_counts:
        samples = _load_samples(args.sobol, sample_count).astype(np.float32)
        accum = {stop: [] for stop in stop_layers}
        for index in indices:
            weights = _weights(index).astype(np.float32)
            mlp = MLP(width=WIDTH, depth=DEPTH, weights=[fnp.array(weights[layer]) for layer in range(DEPTH)])
            estimator = Estimator()
            structure = estimator._initial_structure(mlp, WIDTH)
            alpha_rows = [np.asarray(alpha, dtype=np.float64) for alpha in structure["alpha_rows"]]
            final_on = alpha_rows[-1] > 3.0
            if not np.any(final_on):
                continue
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True):
                baseline = np.asarray(estimator.predict(mlp, BUDGET)[-1], dtype=np.float64)
            data = np.load(_moment_path(index))
            means = np.asarray(data["mean"], dtype=np.float64)
            true_final = means[-1]
            true_pre_final = np.asarray(data["pre_mean"][-1], dtype=np.float64)
            h_layers = _forward_layers(weights, samples)
            for stop in stop_layers:
                oracle_proj, a_matrix = _fold_oracle_projection(weights, means, alpha_rows, final_on, stop)
                sample_proj = h_layers[stop][:, :] @ a_matrix
                sample_proj = sample_proj.mean(axis=0)
                pred_oracle = baseline.copy()
                pred_sample = baseline.copy()
                pred_true_pre = baseline.copy()
                pred_oracle[final_on] = oracle_proj
                pred_sample[final_on] = sample_proj
                pred_true_pre[final_on] = true_pre_final[final_on]
                r90, r95, r99 = _energy_ranks(a_matrix)
                accum[stop].append(
                    (
                        _mse(baseline, true_final),
                        _mse(pred_true_pre, true_final),
                        _mse(pred_oracle, true_final),
                        _mse(pred_sample, true_final),
                        r90,
                        r95,
                        r99,
                        int(final_on.sum()),
                    )
                )
        print(f"\nN={sample_count}")
        print("stop base true_pre oracle_stop sample_stop r90 r95 r99 final_on")
        for stop in stop_layers:
            arr = np.asarray(accum[stop], dtype=np.float64)
            if arr.size == 0:
                continue
            means = arr.mean(axis=0)
            print(
                f"{stop:>4} {means[0]:.3e} {means[1]:.3e} {means[2]:.3e} {means[3]:.3e} "
                f"{means[4]:4.1f} {means[5]:4.1f} {means[6]:4.1f} {means[7]:6.1f}"
            )


if __name__ == "__main__":
    main()