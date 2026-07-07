"""Learn a tiny sample-allocation policy for algorithm 14/15.

This is an offline experiment. It evaluates each public MLP at several Sobol
sample counts, computes features available from the analytical pass plus a
30,720-sample base run, and fits small greedy decision trees whose leaves choose
the sample count with the best training adjusted score.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy import special

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from whestbench import MLP

import eval_variants as ev


SAMPLES = (30720, 40960, 49152)
PREFIX_SAMPLES = (10240, 20480, 30720)
ACTION_NAMES = {30720: "easy30", 40960: "mid40", 49152: "hard49"}
BASE_VARIANT_NAME = "pilot l29+30 borderline 4/-4"
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
POLICY_MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15)


def _scatterless_hhi(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=np.float64))
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    shares = values / total
    return float(np.sum(shares * shares))


def _top_share(values: np.ndarray, k: int) -> float:
    values = np.abs(np.asarray(values, dtype=np.float64))
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    return float(np.sum(np.sort(values)[-min(k, values.size) :]) / total)


def analytical_features(weights_np: np.ndarray) -> dict[str, float]:
    depth, width = weights_np.shape[0], weights_np.shape[1]
    mu_post = np.zeros(width, dtype=np.float64)
    var_post = np.zeros(width, dtype=np.float64)
    alpha_rows = []
    mu_rows = []
    var_rows = []
    type_rows = []
    for layer_idx in range(depth):
        w = weights_np[layer_idx].astype(np.float64, copy=False)
        if layer_idx == 0:
            mu_pre = np.zeros(width, dtype=np.float64)
            var_pre = np.sum(w * w, axis=0)
        else:
            mu_pre = w.T @ mu_post
            var_pre = np.sum(w * w * var_post[:, None], axis=0)

        var_pre = np.maximum(var_pre, 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi = np.exp(-0.5 * alpha * alpha) * np.float64(0.3989422804014327)
        Phi = special.ndtr(alpha)
        mu_post = mu_pre * Phi + sigma_pre * phi
        var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - mu_post * mu_post
        var_post = np.maximum(var_post, 1e-12)

        typ = np.full(width, 1, dtype=np.int8)
        typ[alpha < -3.0] = 0
        typ[alpha > 3.0] = 2
        alpha_rows.append(alpha)
        mu_rows.append(mu_post.copy())
        var_rows.append(var_post.copy())
        type_rows.append(typ)

    alpha_rows = np.asarray(alpha_rows)
    mu_rows = np.asarray(mu_rows)
    var_rows = np.asarray(var_rows)
    type_rows = np.asarray(type_rows)
    features: dict[str, float] = {}

    for layer in (27, 28, 29, 30, 31):
        alpha = alpha_rows[layer]
        typ = type_rows[layer]
        prefix = f"l{layer}"
        features[f"{prefix}_dead"] = float(np.sum(typ == 0))
        features[f"{prefix}_kink"] = float(np.sum(typ == 1))
        features[f"{prefix}_on"] = float(np.sum(typ == 2))
        features[f"{prefix}_active"] = float(np.sum(typ != 0))
        features[f"{prefix}_on_frac"] = float(np.mean(typ == 2))
        features[f"{prefix}_kink_frac"] = float(np.mean(typ == 1))
        features[f"{prefix}_border_on"] = float(np.sum((alpha > 3.0) & (alpha <= 4.0)))
        features[f"{prefix}_border_dead"] = float(np.sum((alpha >= -4.0) & (alpha < -3.0)))
        features[f"{prefix}_alpha_std"] = float(np.std(alpha))
        features[f"{prefix}_alpha_abs_mean"] = float(np.mean(np.abs(alpha)))
        features[f"{prefix}_alpha_ge6"] = float(np.sum(alpha >= 6.0))

    final_mu = mu_rows[-1]
    final_var = var_rows[-1]
    final_alpha = alpha_rows[-1]
    final_type = type_rows[-1]
    features.update(
        {
            "anal_final_mu_mean": float(np.mean(final_mu)),
            "anal_final_mu_std": float(np.std(final_mu)),
            "anal_final_mu_max": float(np.max(final_mu)),
            "anal_final_var_mean": float(np.mean(final_var)),
            "anal_final_var_std": float(np.std(final_var)),
            "anal_final_var_max": float(np.max(final_var)),
            "final_on_frac": float(np.mean(final_type == 2)),
            "final_kink_frac": float(np.mean(final_type == 1)),
            "final_dead_frac": float(np.mean(final_type == 0)),
            "final_on_alpha_ge6": float(np.sum(final_alpha >= 6.0)),
            "final_border_on": float(np.sum((final_alpha > 3.0) & (final_alpha <= 4.0))),
            "final_border_dead": float(np.sum((final_alpha >= -4.0) & (final_alpha < -3.0))),
        }
    )
    return features


def fixed_sample_variant(sample_count: int):
    base = next(v for v in ev.VARIANTS if v.name == BASE_VARIANT_NAME)
    return replace(
        base,
        name=f"fixed {sample_count}",
        dynamic_sample_mode="anal_var_q",
        dynamic_easy_samples=sample_count,
        dynamic_mid_samples=sample_count,
        dynamic_hard_samples=sample_count,
    )


def prefix_drift_features(prefix_rows: dict[int, dict[str, np.ndarray]]) -> dict[str, float]:
    features: dict[str, float] = {}
    for layer_name in ("final", "l30"):
        row10 = prefix_rows[10240][layer_name]
        row20 = prefix_rows[20480][layer_name]
        row30 = prefix_rows[30720][layer_name]
        diff10_20 = row20 - row10
        diff20_30 = row30 - row20
        mse10_20 = float(np.mean(diff10_20 * diff10_20))
        mse20_30 = float(np.mean(diff20_30 * diff20_30))
        abs20_30 = np.abs(diff20_30)
        prefix = f"prefix_{layer_name}"
        features[f"{prefix}_drift_10_20_mse"] = mse10_20
        features[f"{prefix}_drift_20_30_mse"] = mse20_30
        features[f"{prefix}_drift_ratio"] = float(mse20_30 / max(mse10_20, 1e-30))
        features[f"{prefix}_drift_20_30_mean_abs"] = float(np.mean(abs20_30))
        features[f"{prefix}_drift_20_30_max_abs"] = float(np.max(abs20_30))
        features[f"{prefix}_drift_20_30_top10_share"] = _top_share(abs20_30, 10)
        features[f"{prefix}_drift_20_30_hhi"] = _scatterless_hhi(abs20_30)
    return features


def evaluate_dataset(split: str, sobol_points_path: Path, limit: int | None, force: bool, cache_path: Path) -> dict:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    sobol_points = np.load(sobol_points_path)["points"]
    if sobol_points.shape[0] < max(SAMPLES) // 2:
        raise SystemExit(f"{sobol_points_path} has {sobol_points.shape[0]} half-points; need {max(SAMPLES) // 2}")

    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    run_sample_counts = tuple(sorted(set(SAMPLES) | set(PREFIX_SAMPLES)))
    variants = {sample_count: fixed_sample_variant(sample_count) for sample_count in run_sample_counts}
    records = []
    for row_index, row in enumerate(dataset):
        weights_np = np.asarray(row["weights"], dtype=np.float32)
        weights = [fnp.array(weights_np[layer]) for layer in range(weights_np.shape[0])]
        mlp = MLP(width=weights_np.shape[1], depth=weights_np.shape[0], weights=weights)
        gt_final = np.asarray(row["final_means"], dtype=np.float32)
        gt_all = np.asarray(row["all_layer_means"], dtype=np.float32)

        features = analytical_features(weights_np)
        costs = {}
        final_mses = {}
        flops_used = {}
        all_mses = {}
        pred_features = {}
        stats_30720 = {}
        prefix_rows = {}
        for sample_count, variant in variants.items():
            with flops.BudgetContext(flop_budget=ev.BUDGET, quiet=True) as ctx:
                pred = np.asarray(ev.predict_variant(mlp, sobol_points, variant))
            stats = ev.predict_variant.last_stats
            err_final = pred[-1] - gt_final
            err_all = pred - gt_all
            final_mse = float(np.mean(err_final * err_final))
            if sample_count in SAMPLES:
                final_mses[str(sample_count)] = final_mse
                all_mses[str(sample_count)] = float(np.mean(err_all * err_all))
                flops_used[str(sample_count)] = float(ctx.flops_used)
                costs[str(sample_count)] = final_mse * max(0.1, ctx.flops_used / ev.BUDGET)
            if sample_count in PREFIX_SAMPLES:
                prefix_rows[sample_count] = {
                    "final": pred[-1].astype(np.float64, copy=True),
                    "l30": pred[30].astype(np.float64, copy=True),
                }

            if sample_count == 30720:
                pred_features = {
                    "base_pred_final_mean": float(np.mean(pred[-1])),
                    "base_pred_final_std": float(np.std(pred[-1])),
                    "base_pred_final_max": float(np.max(pred[-1])),
                    "base_pred_final_top10_share": _top_share(pred[-1], 10),
                    "base_pred_final_hhi": _scatterless_hhi(pred[-1]),
                    "base_pred_l30_mean": float(np.mean(pred[30])),
                    "base_pred_l30_std": float(np.std(pred[30])),
                    "base_pred_l30_top10_share": _top_share(pred[30], 10),
                    "base_pred_l30_hhi": _scatterless_hhi(pred[30]),
                }
                stats_30720 = {
                    f"stat_{key}": float(value)
                    for key, value in stats.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }

        features.update(pred_features)
        features.update(prefix_drift_features(prefix_rows))
        features.update(stats_30720)
        best_sample = min(SAMPLES, key=lambda sample_count: costs[str(sample_count)])
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": row["mlp_name"],
                "features": features,
                "costs": costs,
                "final_mses": final_mses,
                "all_mses": all_mses,
                "flops_used": flops_used,
                "best_sample": best_sample,
            }
        )
        if (row_index + 1) % 20 == 0:
            print(f"evaluated {row_index + 1}/{len(dataset)}", flush=True)

    payload = {"split": split, "samples": SAMPLES, "records": records}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def mean_cost(records: list[dict], actions: np.ndarray) -> float:
    return float(np.mean([record["costs"][str(SAMPLES[action])] for record, action in zip(records, actions)]))


def leaf_action(indices: np.ndarray, costs: np.ndarray) -> int:
    return int(np.argmin(np.mean(costs[indices], axis=0)))


def leaf_objective(indices: np.ndarray, costs: np.ndarray) -> float:
    return float(len(indices) * np.min(np.mean(costs[indices], axis=0)))


def fit_tree(X: np.ndarray, costs: np.ndarray, feature_names: list[str], indices: np.ndarray, depth: int, min_leaf: int):
    if depth == 0 or len(indices) < 2 * min_leaf:
        return {"action": leaf_action(indices, costs), "n": int(len(indices))}

    base_obj = leaf_objective(indices, costs)
    best = None
    for feature_index in range(X.shape[1]):
        values = X[indices, feature_index]
        unique = np.unique(values[np.isfinite(values)])
        if unique.size <= 1:
            continue
        if unique.size > 32:
            thresholds = np.unique(np.quantile(unique, np.linspace(0.1, 0.9, 17)))
        else:
            thresholds = (unique[:-1] + unique[1:]) / 2.0
        for threshold in thresholds:
            left = indices[values <= threshold]
            right = indices[values > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            obj = leaf_objective(left, costs) + leaf_objective(right, costs)
            if best is None or obj < best[0]:
                best = (obj, feature_index, float(threshold), left, right)

    if best is None or best[0] >= base_obj:
        return {"action": leaf_action(indices, costs), "n": int(len(indices))}

    _, feature_index, threshold, left, right = best
    return {
        "feature": feature_names[feature_index],
        "feature_index": int(feature_index),
        "threshold": threshold,
        "n": int(len(indices)),
        "left": fit_tree(X, costs, feature_names, left, depth - 1, min_leaf),
        "right": fit_tree(X, costs, feature_names, right, depth - 1, min_leaf),
    }


def predict_one(tree: dict, row: np.ndarray) -> int:
    while "action" not in tree:
        if row[tree["feature_index"]] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return int(tree["action"])


def print_tree(tree: dict, indent: str = "") -> None:
    if "action" in tree:
        print(f"{indent}-> {ACTION_NAMES[SAMPLES[tree['action']]]} (n={tree['n']})")
        return
    print(f"{indent}if {tree['feature']} <= {tree['threshold']:.6g}:  # n={tree['n']}")
    print_tree(tree["left"], indent + "  ")
    print(f"{indent}else:")
    print_tree(tree["right"], indent + "  ")


def standardize_train_valid(X_train: np.ndarray, X_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return (X_train - mu) / sigma, (X_valid - mu) / sigma


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X_design = add_intercept(X)
    penalty = np.eye(X_design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(X_design.T @ X_design + penalty, X_design.T @ y)


def predict_ridge(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ coef


def delta_policy_actions(pred_easy: np.ndarray, pred_hard: np.ndarray, margin: float) -> np.ndarray:
    actions = np.ones(pred_easy.shape[0], dtype=np.int64)
    choose_easy = (pred_easy < -margin) & (pred_easy <= pred_hard)
    choose_hard = (pred_hard < -margin) & (~choose_easy)
    actions[choose_easy] = 0
    actions[choose_hard] = 2
    return actions


def run_ridge_delta_cv(
    X: np.ndarray,
    costs: np.ndarray,
    records: list[dict],
    fold_ids: np.ndarray,
    alphas: tuple[float, ...],
    margins: tuple[float, ...],
) -> None:
    log_costs = np.log(np.maximum(costs, 1e-30))
    y_easy = log_costs[:, 0] - log_costs[:, 1]
    y_hard = log_costs[:, 2] - log_costs[:, 1]
    print("\nRIDGE DELTA BANDIT CV")
    best = None
    for alpha in alphas:
        pred_easy_all = np.zeros(X.shape[0], dtype=np.float64)
        pred_hard_all = np.zeros(X.shape[0], dtype=np.float64)
        for fold in np.unique(fold_ids):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            X_train, X_valid = standardize_train_valid(X[train_idx], X[valid_idx])
            easy_coef = fit_ridge(X_train, y_easy[train_idx], alpha)
            hard_coef = fit_ridge(X_train, y_hard[train_idx], alpha)
            pred_easy_all[valid_idx] = predict_ridge(X_valid, easy_coef)
            pred_hard_all[valid_idx] = predict_ridge(X_valid, hard_coef)

        for margin in margins:
            actions = delta_policy_actions(pred_easy_all, pred_hard_all, margin)
            score = mean_cost(records, actions)
            counts = dict(zip(*np.unique(actions, return_counts=True)))
            print(f"alpha={alpha:<6g} margin={margin:<4g} score={score:.9e} counts={counts}")
            if best is None or score < best[0]:
                best = (score, alpha, margin, counts, pred_easy_all, pred_hard_all)

    if best is None:
        return
    score, alpha, margin, counts, pred_easy_all, pred_hard_all = best
    print(f"BEST ridge-delta: score={score:.9e} alpha={alpha:g} margin={margin:g} counts={counts}")
    true_best = np.argmin(costs, axis=1)
    actions = delta_policy_actions(pred_easy_all, pred_hard_all, margin)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, pred in zip(true_best, actions):
        confusion[truth, pred] += 1
    print("truth rows x predicted cols (easy, mid, hard):")
    print(confusion)


def run_policy_experiment(payload: dict, depth_values: tuple[int, ...], min_leaf: int, folds: int) -> None:
    records = payload["records"]
    feature_names = sorted(records[0]["features"].keys())
    X = np.asarray([[record["features"].get(name, 0.0) for name in feature_names] for record in records], dtype=np.float64)
    costs = np.asarray([[record["costs"][str(sample_count)] for sample_count in SAMPLES] for record in records], dtype=np.float64)
    labels = np.argmin(costs, axis=1)
    n = len(records)
    fixed_actions = {sample_count: np.full(n, i, dtype=np.int64) for i, sample_count in enumerate(SAMPLES)}

    print("\nBASELINES")
    for sample_count, actions in fixed_actions.items():
        print(f"fixed {sample_count}: score={mean_cost(records, actions):.9e}")
    print(f"oracle: score={float(np.mean(np.min(costs, axis=1))):.9e}")
    values, counts = np.unique(labels, return_counts=True)
    print("oracle action counts", {SAMPLES[int(value)]: int(count) for value, count in zip(values, counts)})

    # Existing analytical-variance quartile rule from algorithm 15.
    var = X[:, feature_names.index("anal_final_var_mean")]
    q_actions = np.full(n, 1, dtype=np.int64)
    q_actions[var < 0.0158787] = 0
    q_actions[var >= 0.0299710] = 2
    print(f"anal_var quartile rule: score={mean_cost(records, q_actions):.9e} counts={dict(zip(*np.unique(q_actions, return_counts=True)))}")

    fold_ids = np.arange(n) % folds
    rng = np.random.default_rng(12345)
    rng.shuffle(fold_ids)
    run_ridge_delta_cv(X, costs, records, fold_ids, RIDGE_ALPHAS, POLICY_MARGINS)

    print("\nCROSS-VALIDATED TREES")
    for depth in depth_values:
        pred_actions = np.zeros(n, dtype=np.int64)
        for fold in range(folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            tree = fit_tree(X, costs, feature_names, train_idx, depth=depth, min_leaf=min_leaf)
            pred_actions[valid_idx] = [predict_one(tree, row) for row in X[valid_idx]]
        print(
            f"depth {depth}: score={mean_cost(records, pred_actions):.9e} "
            f"counts={dict(zip(*np.unique(pred_actions, return_counts=True)))}"
        )

    print("\nFULL-MINI FIT TREE")
    tree = fit_tree(X, costs, feature_names, np.arange(n), depth=max(depth_values), min_leaf=min_leaf)
    print_tree(tree)
    train_actions = np.asarray([predict_one(tree, row) for row in X], dtype=np.int64)
    print(f"train score={mean_cost(records, train_actions):.9e} counts={dict(zip(*np.unique(train_actions, return_counts=True)))}")

    print("\nTOP FEATURE SPEARMAN WITH ORACLE LABEL")
    label_float = labels.astype(np.float64)
    rows = []
    for index, name in enumerate(feature_names):
        x = X[:, index]
        if np.std(x) == 0:
            continue
        corr = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(label_float)))[0, 1])
        rows.append((abs(corr), corr, name))
    for _, corr, name in sorted(rows, reverse=True)[:20]:
        print(f"{name:<32} spearman={corr:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mini")
    parser.add_argument("--sobol-points", default="/tmp/whest_sobol_points_24576.npz")
    parser.add_argument("--cache", default=".tmp/allocation_policy_mini.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    payload = evaluate_dataset(
        split=args.split,
        sobol_points_path=Path(args.sobol_points),
        limit=args.limit,
        force=args.force,
        cache_path=Path(args.cache),
    )
    run_policy_experiment(payload, depth_values=(1, 2, 3), min_leaf=args.min_leaf, folds=args.folds)


if __name__ == "__main__":
    main()