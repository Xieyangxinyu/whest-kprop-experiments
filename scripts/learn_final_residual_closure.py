"""Learn a final-layer-only residual closure from cheap deployable features.

This is an offline experiment aimed at the top-submission fingerprint: poor or
irrelevant intermediate rows, but decent final-layer means at low compute.  The
model starts from diagonal Gaussian moment propagation, builds per-final-neuron
features from weights, gates, and effective sensitivity, and fits a tiny ridge
model for the final residual.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import special

from learn_cumulant_closure import _moment_path, _weights


WIDTH = 256
DEPTH = 32


def _relu_moments(pre_mean: np.ndarray, pre_var: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre_var = np.maximum(pre_var, 1e-12)
    sigma = np.sqrt(pre_var)
    alpha = pre_mean / sigma
    phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2.0 * np.pi)
    Phi = special.ndtr(alpha)
    post_mean = pre_mean * Phi + sigma * phi
    ez2 = (pre_mean * pre_mean + pre_var) * Phi + pre_mean * sigma * phi
    post_var = np.maximum(ez2 - post_mean * post_mean, 1e-12)
    return post_mean, post_var, alpha, Phi


def _forward_diagonal(weights: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.zeros(WIDTH)
    var = np.ones(WIDTH)
    rows = []
    vars_ = []
    alphas = []
    gates = []
    pre_means = []
    pre_vars = []
    for layer_idx, w in enumerate(weights):
        pre_mean = w.T @ mean
        pre_var = np.maximum((w * w).T @ var, 1e-12)
        mean, var, alpha, gate = _relu_moments(pre_mean, pre_var)
        pre_means.append(pre_mean)
        pre_vars.append(pre_var)
        rows.append(mean)
        vars_.append(var)
        alphas.append(alpha)
        gates.append(gate)
    return {
        "mean": np.asarray(rows),
        "var": np.asarray(vars_),
        "alpha": np.asarray(alphas),
        "gate": np.asarray(gates),
        "pre_mean": np.asarray(pre_means),
        "pre_var": np.asarray(pre_vars),
    }


def _effective_sensitivity(weights: np.ndarray, gates: np.ndarray) -> np.ndarray:
    effective = weights[0] * gates[0][None, :]
    for layer_idx in range(1, DEPTH - 1):
        effective = effective @ weights[layer_idx]
        effective = effective * gates[layer_idx][None, :]
    return effective @ weights[-1]


def _forward_covariance(weights: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.zeros(WIDTH)
    cov = np.eye(WIDTH)
    rows = []
    vars_ = []
    alphas = []
    gates = []
    pre_means = []
    pre_vars = []
    for w in weights:
        pre_mean = w.T @ mean
        pre_cov = w.T @ cov @ w
        pre_var = np.maximum(np.diag(pre_cov), 1e-12)
        mean, var, alpha, gate = _relu_moments(pre_mean, pre_var)
        cov = pre_cov * np.outer(gate, gate)
        np.fill_diagonal(cov, var)
        pre_means.append(pre_mean)
        pre_vars.append(pre_var)
        rows.append(mean)
        vars_.append(var)
        alphas.append(alpha)
        gates.append(gate)
    return {
        "mean": np.asarray(rows),
        "var": np.asarray(vars_),
        "alpha": np.asarray(alphas),
        "gate": np.asarray(gates),
        "pre_mean": np.asarray(pre_means),
        "pre_var": np.asarray(pre_vars),
    }


def _case_features(index: int, baseline_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = _weights(index)
    data = np.load(_moment_path(index))
    true_final = np.asarray(data["mean"][-1], dtype=np.float64)
    diag = _forward_diagonal(weights)
    base_stats = _forward_covariance(weights) if baseline_mode == "covariance" else diag
    final_pre_mean = diag["pre_mean"][-1]
    final_pre_var = diag["pre_var"][-1]
    final_sigma = np.sqrt(np.maximum(final_pre_var, 1e-12))
    final_alpha = diag["alpha"][-1]
    final_gate = diag["gate"][-1]
    baseline = base_stats["mean"][-1]
    final_var = base_stats["var"][-1]
    layer30_mean = base_stats["mean"][-2]
    layer30_var = base_stats["var"][-2]
    layer30_alpha = diag["alpha"][-2]
    layer30_gate = diag["gate"][-2]
    final_w = weights[-1]
    abs_w = np.abs(final_w)
    eff = _effective_sensitivity(weights, diag["gate"])
    abs_eff = np.abs(eff)

    global_features = np.array(
        [
            baseline.mean(),
            baseline.std(),
            np.mean(final_var),
            np.mean(final_alpha > 3.0),
            np.mean(np.abs(final_alpha) <= 3.0),
            np.mean(layer30_alpha > 3.0),
            np.mean(np.abs(layer30_alpha) <= 3.0),
            layer30_mean.mean(),
            layer30_mean.std(),
            layer30_var.mean(),
        ],
        dtype=np.float64,
    )
    global_tiled = np.repeat(global_features[None, :], WIDTH, axis=0)

    per_neuron = np.stack(
        [
            baseline,
            np.log1p(np.abs(baseline)),
            final_pre_mean,
            final_sigma,
            final_alpha,
            final_gate,
            base_stats["pre_mean"][-1],
            np.sqrt(np.maximum(base_stats["pre_var"][-1], 1e-12)),
            base_stats["alpha"][-1],
            base_stats["gate"][-1],
            final_var,
            np.sum(abs_w, axis=0),
            np.sqrt(np.sum(final_w * final_w, axis=0)),
            np.max(abs_w, axis=0),
            np.sum(final_w, axis=0),
            np.mean(final_w > 0.0, axis=0),
            abs_w.T @ layer30_mean,
            np.sqrt((final_w * final_w).T @ layer30_var),
            final_w.T @ layer30_gate,
            abs_w.T @ layer30_gate,
            np.sqrt(np.sum(eff * eff, axis=0)),
            np.sum(abs_eff, axis=0),
            np.max(abs_eff, axis=0),
        ],
        axis=1,
    )
    features = np.concatenate([per_neuron, global_tiled], axis=1)
    return features, baseline, true_final


def _expand_features(features: np.ndarray) -> np.ndarray:
    base = features
    signed_sqrt = np.sign(base) * np.sqrt(np.abs(base) + 1e-12)
    logs = np.log1p(np.abs(base))
    selected = np.stack(
        [
            base[:, 0] * base[:, 4],
            base[:, 0] * base[:, 16],
            base[:, 4] * base[:, 16],
            base[:, 8] * base[:, 16],
            base[:, 13] * base[:, 16],
            base[:, 5] * base[:, 28],
        ],
        axis=1,
    )
    return np.concatenate([base, signed_sqrt, logs, selected], axis=1)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - center) / scale, (test - center) / scale


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    xtx = features.T @ features
    rhs = features.T @ target
    return np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), rhs)


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((pred - truth) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--baseline", choices=["diagonal", "covariance"], default="covariance")
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    cases = [_case_features(index, args.baseline) for index in indices]
    train_cases = cases[: args.train_count]
    test_cases = cases[args.train_count :]
    train_x = np.concatenate([case[0] for case in train_cases])
    test_x = np.concatenate([case[0] for case in test_cases])
    train_x = _expand_features(train_x)
    test_x = _expand_features(test_x)
    train_x, test_x = _standardize(train_x, test_x)

    train_baseline = np.concatenate([case[1] for case in train_cases])
    train_truth = np.concatenate([case[2] for case in train_cases])
    test_baseline = np.concatenate([case[1] for case in test_cases])
    test_truth = np.concatenate([case[2] for case in test_cases])
    train_residual = train_truth - train_baseline

    coef = _fit_ridge(train_x, train_residual, args.ridge)
    train_pred = train_baseline + train_x @ coef
    test_pred = test_baseline + test_x @ coef
    print(f"train={indices[:args.train_count]} test={indices[args.train_count:]} ridge={args.ridge} baseline={args.baseline}")
    print(f"train baseline={_mse(train_baseline, train_truth):.3e} learned={_mse(train_pred, train_truth):.3e}")
    print(f"test  baseline={_mse(test_baseline, test_truth):.3e} learned={_mse(test_pred, test_truth):.3e}")
    for scale in (0.25, 0.50, 0.75, 1.00):
        pred = test_baseline + scale * (test_x @ coef)
        print(f"test scale={scale:.2f} mse={_mse(pred, test_truth):.3e}")


if __name__ == "__main__":
    main()