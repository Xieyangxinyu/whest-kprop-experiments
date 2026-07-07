"""Learn sample allocation from convex per-MLP curve-fit targets.

This script consumes a cached Pareto table containing per-MLP scores at several
sample counts. For each MLP it fits a simple curve

    final_mse(N) ~= b + a / N
    flops(N)     ~= d + c * N

then computes the continuous optimum for the convex surrogate and learns a
portable policy that predicts that optimum from cheap cached features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.learn_allocation_policy import fit_ridge, fit_tree, predict_one, print_tree, standardize_train_valid


BUDGET = 272_000_000_000
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def nnls_mse_curve(sample_counts: np.ndarray, mse: np.ndarray) -> tuple[float, float, float]:
    """Fit mse ~= b + a / N with b,a >= 0 by a small grid over b."""
    inv_n = 1.0 / sample_counts
    best: tuple[float, float, float] | None = None
    y_max = max(0.0, float(np.max(mse)))
    for b_value in np.linspace(0.0, y_max, 401):
        residual = mse - b_value
        a_value = float(np.dot(inv_n, residual) / max(np.dot(inv_n, inv_n), 1e-30))
        a_value = max(0.0, a_value)
        pred = b_value + a_value * inv_n
        loss = float(np.mean((pred - mse) ** 2))
        if best is None or loss < best[0]:
            best = (loss, b_value, a_value)
    assert best is not None
    return best[1], best[2], best[0]


def fit_linear_compute(sample_counts: np.ndarray, flops: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(sample_counts), sample_counts])
    fixed, per_sample = np.linalg.lstsq(design, flops, rcond=None)[0]
    return max(0.0, float(fixed)), max(0.0, float(per_sample))


def nearest_sample(sample_counts: np.ndarray, value: float) -> int:
    return int(sample_counts[np.argmin(np.abs(sample_counts - value))])


def load_feature_records(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {int(record["mlp_id"]): record.get("features", {}) for record in payload["records"]}


def is_leaky_feature(name: str) -> bool:
    return name.startswith("curve_") or name in {"compute_fixed", "compute_per_sample"}


def augment_log_features(features: dict[str, float]) -> dict[str, float]:
    augmented = dict(features)
    for name, value in features.items():
        if not np.isfinite(value):
            continue
        if value >= 0.0 and not name.startswith("log1p_"):
            augmented[f"log1p_{name}"] = float(np.log1p(value))
    return augmented


def add_ratio(features: dict[str, float], numerator: str, denominator: str, name: str) -> None:
    if numerator in features and denominator in features:
        features[name] = float(features[numerator] / (abs(features[denominator]) + 1e-12))


def add_product(features: dict[str, float], left: str, right: str, name: str) -> None:
    if left in features and right in features:
        features[name] = float(features[left] * features[right])


def augment_interaction_features(features: dict[str, float]) -> dict[str, float]:
    augmented = dict(features)
    # Sampling-noise proxy divided by bias/disagreement proxy.
    add_ratio(augmented, "stat_base_final_vom_mean", "stat_base_final_anal_diff_mean", "ratio_final_vom_to_anal_diff")
    add_ratio(augmented, "stat_base_l30_vom_mean", "stat_base_l30_anal_diff_mean", "ratio_l30_vom_to_anal_diff")
    add_ratio(augmented, "prefix_final_drift_20_30_mse", "stat_base_final_anal_diff_mean", "ratio_final_drift_to_anal_diff")
    add_ratio(augmented, "prefix_l30_drift_20_30_mse", "stat_base_l30_anal_diff_mean", "ratio_l30_drift_to_anal_diff")
    add_ratio(augmented, "stat_base_final_on_vom_mean", "stat_base_final_on_anal_diff_mean", "ratio_final_on_vom_to_anal_diff")
    add_ratio(augmented, "stat_base_final_kink_vom_mean", "stat_base_final_kink_anal_diff_mean", "ratio_final_kink_vom_to_anal_diff")

    # Scale/collapse interactions that should flag bias-limited MLPs.
    add_product(augmented, "anal_final_var_mean", "final_on_frac", "prod_var_final_on_frac")
    add_product(augmented, "anal_final_var_mean", "final_kink_frac", "prod_var_final_kink_frac")
    add_product(augmented, "base_pred_final_mean", "final_on_frac", "prod_pred_final_mean_on_frac")
    add_product(augmented, "base_pred_l30_mean", "l30_on_frac", "prod_pred_l30_mean_on_frac")
    add_product(augmented, "prefix_final_drift_20_30_mse", "final_on_frac", "prod_final_drift_on_frac")
    add_product(augmented, "prefix_l30_drift_20_30_mse", "l30_on_frac", "prod_l30_drift_on_frac")
    add_product(augmented, "stat_base_l30_anal_diff_mean", "l30_on_frac", "prod_l30_diff_on_frac")
    add_product(augmented, "stat_base_final_anal_diff_mean", "final_on_frac", "prod_final_diff_on_frac")

    # High-alpha on-neuron interactions.
    add_product(augmented, "final_on_alpha_ge6", "stat_base_final_on_anal_diff_mean", "prod_high_on_final_on_diff")
    add_product(augmented, "l30_alpha_ge6", "stat_base_l30_anal_diff_mean", "prod_l30_high_on_l30_diff")
    return augmented


def build_records(
    pareto_path: Path,
    feature_path: Path,
    include_curve_features: bool,
    log_features: bool,
    interaction_features: bool,
) -> tuple[list[dict], np.ndarray]:
    payload = json.loads(pareto_path.read_text())
    sample_counts = np.asarray(payload["samples"], dtype=np.float64)
    feature_records = load_feature_records(feature_path)
    rows = []
    for record in payload["records"]:
        mse = np.asarray([record["final_mses"][str(int(n))] for n in sample_counts], dtype=np.float64)
        flops = np.asarray([record["flops_used"][str(int(n))] for n in sample_counts], dtype=np.float64)
        costs = np.asarray([record["costs"][str(int(n))] for n in sample_counts], dtype=np.float64)
        b_value, a_value, fit_loss = nnls_mse_curve(sample_counts, mse)
        d_value, c_value = fit_linear_compute(sample_counts, flops)
        if b_value <= 1e-30 or c_value <= 1e-30:
            n_star = 40960.0
        else:
            n_star = math.sqrt(max(a_value * d_value, 0.0) / max(b_value * c_value, 1e-30))
        n_clip = max(float(np.min(sample_counts)), min(float(np.max(sample_counts)), n_star))
        n_nearest = nearest_sample(sample_counts, n_clip)

        features = dict(feature_records.get(int(record["id"]), {}))
        for key, value in record.get("mom", {}).items():
            features[f"mom_{key}"] = float(value)
        if include_curve_features:
            features.update(
                {
                    "curve_a": a_value,
                    "curve_b": b_value,
                    "curve_fit_loss": fit_loss,
                    "compute_fixed": d_value,
                    "compute_per_sample": c_value,
                    "curve_log_a_over_b": math.log(max(a_value, 1e-30)) - math.log(max(b_value, 1e-30)),
                }
            )
        if log_features:
            features = augment_log_features(features)
        if interaction_features:
            features = augment_interaction_features(features)
        rows.append(
            {
                "id": int(record["id"]),
                "name": record["name"],
                "features": features,
                "costs": costs,
                "mse": mse,
                "flops": flops,
                "oracle_index": int(np.argmin(costs)),
                "convex_n_star": n_star,
                "convex_n_clip": n_clip,
                "convex_index": int(np.where(sample_counts == n_nearest)[0][0]),
            }
        )
    return rows, sample_counts


def mean_policy_cost(rows: list[dict], actions: np.ndarray) -> float:
    return float(np.mean([row["costs"][action] for row, action in zip(rows, actions)]))


def add_intercept(x_values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((x_values.shape[0], 1), dtype=x_values.dtype), x_values], axis=1)


def predict_ridge(x_values: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return add_intercept(x_values) @ coef


def run_experiment(rows: list[dict], sample_counts: np.ndarray, folds: int, min_leaf: int) -> None:
    feature_names = sorted(name for name in rows[0]["features"].keys() if not is_leaky_feature(name))
    x_values = np.asarray([[row["features"].get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float64)
    costs = np.asarray([row["costs"] for row in rows], dtype=np.float64)
    oracle_actions = np.asarray([row["oracle_index"] for row in rows], dtype=np.int64)
    convex_actions = np.asarray([row["convex_index"] for row in rows], dtype=np.int64)
    convex_targets = np.asarray([math.log(row["convex_n_clip"]) for row in rows], dtype=np.float64)
    n_rows = len(rows)

    print("BASELINES")
    for idx, sample_count in enumerate(sample_counts.astype(int)):
        print(f"fixed {sample_count}: score={mean_policy_cost(rows, np.full(n_rows, idx, dtype=np.int64)):.9e}")
    print(f"oracle discrete: score={float(np.mean(np.min(costs, axis=1))):.9e}")
    print(f"convex fit nearest: score={mean_policy_cost(rows, convex_actions):.9e} counts={dict(zip(*np.unique(convex_actions, return_counts=True)))}")
    print("oracle action counts", {int(sample_counts[k]): int(v) for k, v in zip(*np.unique(oracle_actions, return_counts=True))})

    fold_ids = np.arange(n_rows) % folds
    rng = np.random.default_rng(86420)
    rng.shuffle(fold_ids)

    print("\nRIDGE LOG-N* CV")
    best = None
    for alpha in RIDGE_ALPHAS:
        pred_log_n = np.zeros(n_rows, dtype=np.float64)
        for fold in range(folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            x_train, x_valid = standardize_train_valid(x_values[train_idx], x_values[valid_idx])
            coef = fit_ridge(x_train, convex_targets[train_idx], alpha)
            pred_log_n[valid_idx] = predict_ridge(x_valid, coef)
        pred_n = np.exp(pred_log_n)
        actions = np.asarray([nearest_sample(sample_counts, n_value) for n_value in pred_n], dtype=np.int64)
        action_indices = np.asarray([int(np.where(sample_counts == action)[0][0]) for action in actions], dtype=np.int64)
        score = mean_policy_cost(rows, action_indices)
        counts = {int(sample_counts[k]): int(v) for k, v in zip(*np.unique(action_indices, return_counts=True))}
        corr = float(np.corrcoef(pred_n, [row["convex_n_clip"] for row in rows])[0, 1])
        print(f"alpha={alpha:<7g} score={score:.9e} corr_target={corr:+.4f} counts={counts}")
        if best is None or score < best[0]:
            best = (score, alpha, counts, pred_n, action_indices)
    if best is not None:
        score, alpha, counts, pred_n, action_indices = best
        print(f"BEST ridge logN: score={score:.9e} alpha={alpha:g} counts={counts}")
        print(f"corr predicted N vs oracle discrete N={float(np.corrcoef(pred_n, sample_counts[oracle_actions])[0,1]):+.4f}")

    print("\nDIRECT COST TREE CV")
    for depth in (1, 2, 3):
        actions = np.zeros(n_rows, dtype=np.int64)
        for fold in range(folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            tree = fit_tree(x_values, costs, feature_names, train_idx, depth=depth, min_leaf=min_leaf)
            actions[valid_idx] = [predict_tree_action(tree, row) for row in x_values[valid_idx]]
        score = mean_policy_cost(rows, actions)
        counts = {int(sample_counts[k]): int(v) for k, v in zip(*np.unique(actions, return_counts=True))}
        print(f"depth={depth} score={score:.9e} counts={counts}")

    print("\nTOP FEATURE SPEARMAN WITH CONVEX TARGET")
    target_rank = np.argsort(np.argsort(convex_targets))
    rankings = []
    for idx, name in enumerate(feature_names):
        feature = x_values[:, idx]
        if np.std(feature) <= 1e-12:
            continue
        corr = float(np.corrcoef(np.argsort(np.argsort(feature)), target_rank)[0, 1])
        rankings.append((abs(corr), corr, name))
    for _abs_corr, corr, name in sorted(rankings, reverse=True)[:25]:
        print(f"{name:<36} spearman={corr:+.4f}")


def predict_tree_action(tree: dict, row: np.ndarray) -> int:
    while "action" not in tree:
        if row[tree["feature_index"]] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return int(tree["action"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-cache", default=".tmp/pareto_allocation_mini.json")
    parser.add_argument("--feature-cache", default=".tmp/allocation_policy_mini_prefix.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument("--include-curve-features", action="store_true")
    parser.add_argument("--log-features", action="store_true")
    parser.add_argument("--interaction-features", action="store_true")
    args = parser.parse_args()
    rows, sample_counts = build_records(
        Path(args.pareto_cache),
        Path(args.feature_cache),
        args.include_curve_features,
        args.log_features,
        args.interaction_features,
    )
    run_experiment(rows, sample_counts, folds=args.folds, min_leaf=args.min_leaf)


if __name__ == "__main__":
    main()