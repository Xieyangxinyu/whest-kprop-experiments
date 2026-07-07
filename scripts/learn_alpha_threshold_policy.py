"""Learn an adaptive always-on alpha threshold policy.

This offline experiment treats the Stage-1 always-on threshold as the action.
For each public MLP it evaluates the algorithm-14 family under several
thresholds, then fits small policies from cheap analytical/base features to the
best threshold by per-MLP adjusted score.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from whestbench import MLP

import eval_variants as ev
from scripts.learn_allocation_policy import (
    analytical_features,
    fit_tree,
    predict_one,
    print_tree,
    standardize_train_valid,
    fit_ridge,
    predict_ridge,
)


THRESHOLDS = (3.0, 3.25, 3.5, 4.0)
BASE_VARIANT_NAME = "pilot l29+30 borderline 4/-4"


def threshold_variant(threshold: float):
    base = next(v for v in ev.VARIANTS if v.name == BASE_VARIANT_NAME)
    return replace(base, name=f"on_thresh {threshold:.2f}", on_thresh=threshold)


def evaluate_dataset(split: str, sobol_points_path: Path, limit: int | None, force: bool, cache_path: Path) -> dict:
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    sobol_points = np.load(sobol_points_path)["points"]
    if sobol_points.shape[0] < ev.N_SAMPLES // 2:
        raise SystemExit(f"{sobol_points_path} has {sobol_points.shape[0]} half-points; need {ev.N_SAMPLES // 2}")

    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    variants = {threshold: threshold_variant(threshold) for threshold in THRESHOLDS}
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
        all_mses = {}
        flops_used = {}
        stats_by_threshold = {}
        for threshold, variant in variants.items():
            with flops.BudgetContext(flop_budget=ev.BUDGET, quiet=True) as ctx:
                pred = np.asarray(ev.predict_variant(mlp, sobol_points, variant))
            stats = dict(ev.predict_variant.last_stats)
            err_final = pred[-1] - gt_final
            err_all = pred - gt_all
            final_mse = float(np.mean(err_final * err_final))
            key = f"{threshold:.2f}"
            final_mses[key] = final_mse
            all_mses[key] = float(np.mean(err_all * err_all))
            flops_used[key] = float(ctx.flops_used)
            costs[key] = final_mse * max(0.1, ctx.flops_used / ev.BUDGET)
            stats_by_threshold[key] = {
                stat_key: float(value)
                for stat_key, value in stats.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }
            if threshold == 3.0:
                features.update(
                    {
                        "base_pred_final_mean": float(np.mean(pred[-1])),
                        "base_pred_final_std": float(np.std(pred[-1])),
                        "base_pred_final_max": float(np.max(pred[-1])),
                        "base_pred_l30_mean": float(np.mean(pred[30])),
                        "base_pred_l30_std": float(np.std(pred[30])),
                    }
                )
                features.update({f"stat_{k}": v for k, v in stats_by_threshold[key].items()})

        best_threshold = min(THRESHOLDS, key=lambda threshold: costs[f"{threshold:.2f}"])
        records.append(
            {
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": row["mlp_name"],
                "features": features,
                "costs": costs,
                "final_mses": final_mses,
                "all_mses": all_mses,
                "flops_used": flops_used,
                "stats": stats_by_threshold,
                "best_threshold": best_threshold,
            }
        )
        if (row_index + 1) % 20 == 0:
            print(f"evaluated {row_index + 1}/{len(dataset)}", flush=True)

    payload = {"split": split, "thresholds": THRESHOLDS, "records": records}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def mean_cost(records: list[dict], actions: np.ndarray) -> float:
    return float(np.mean([record["costs"][f"{THRESHOLDS[action]:.2f}"] for record, action in zip(records, actions)]))


def run_experiment(payload: dict, folds: int, min_leaf: int) -> None:
    records = payload["records"]
    feature_names = sorted(records[0]["features"].keys())
    X = np.asarray([[record["features"].get(name, 0.0) for name in feature_names] for record in records], dtype=np.float64)
    costs = np.asarray([[record["costs"][f"{threshold:.2f}"] for threshold in THRESHOLDS] for record in records], dtype=np.float64)
    labels = np.argmin(costs, axis=1)
    n = len(records)
    print("\nBASELINES")
    for idx, threshold in enumerate(THRESHOLDS):
        actions = np.full(n, idx, dtype=np.int64)
        print(f"fixed threshold {threshold:.2f}: score={mean_cost(records, actions):.9e}")
    print(f"oracle: score={float(np.mean(np.min(costs, axis=1))):.9e}")
    values, counts = np.unique(labels, return_counts=True)
    print("oracle threshold counts", {THRESHOLDS[int(value)]: int(count) for value, count in zip(values, counts)})

    fold_ids = np.arange(n) % folds
    rng = np.random.default_rng(24680)
    rng.shuffle(fold_ids)

    print("\nRIDGE COST MODEL CV")
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    best = None
    log_costs = np.log(np.maximum(costs, 1e-30))
    for alpha in alphas:
        pred_costs = np.zeros_like(costs)
        for fold in range(folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            X_train, X_valid = standardize_train_valid(X[train_idx], X[valid_idx])
            for action_idx in range(len(THRESHOLDS)):
                coef = fit_ridge(X_train, log_costs[train_idx, action_idx], alpha)
                pred_costs[valid_idx, action_idx] = predict_ridge(X_valid, coef)
        actions = np.argmin(pred_costs, axis=1)
        score = mean_cost(records, actions)
        counts = dict(zip(*np.unique(actions, return_counts=True)))
        print(f"alpha={alpha:<6g} score={score:.9e} counts={counts}")
        if best is None or score < best[0]:
            best = (score, alpha, counts, actions)
    if best is not None:
        score, alpha, counts, actions = best
        print(f"BEST ridge cost: score={score:.9e} alpha={alpha:g} counts={counts}")
        confusion = np.zeros((len(THRESHOLDS), len(THRESHOLDS)), dtype=np.int64)
        for truth, pred in zip(labels, actions):
            confusion[truth, pred] += 1
        print("truth rows x predicted cols:")
        print(confusion)

    print("\nTREE CV")
    for depth in (1, 2, 3):
        pred_actions = np.zeros(n, dtype=np.int64)
        for fold in range(folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            tree = fit_tree(X, costs, feature_names, train_idx, depth=depth, min_leaf=min_leaf)
            pred_actions[valid_idx] = [predict_one(tree, row) for row in X[valid_idx]]
        print(f"depth {depth}: score={mean_cost(records, pred_actions):.9e} counts={dict(zip(*np.unique(pred_actions, return_counts=True)))}")

    print("\nFULL-MINI FIT TREE")
    tree = fit_tree(X, costs, feature_names, np.arange(n), depth=3, min_leaf=min_leaf)
    print_tree(tree)
    actions = np.asarray([predict_one(tree, row) for row in X], dtype=np.int64)
    print(f"train score={mean_cost(records, actions):.9e} counts={dict(zip(*np.unique(actions, return_counts=True)))}")

    print("\nTOP FEATURE SPEARMAN WITH ORACLE THRESHOLD LABEL")
    label_ranks = np.argsort(np.argsort(labels.astype(np.float64)))
    rows = []
    for index, name in enumerate(feature_names):
        x = X[:, index]
        if np.std(x) == 0:
            continue
        corr = float(np.corrcoef(np.argsort(np.argsort(x)), label_ranks)[0, 1])
        rows.append((abs(corr), corr, name))
    for _, corr, name in sorted(rows, reverse=True)[:20]:
        print(f"{name:<36} spearman={corr:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mini")
    parser.add_argument("--sobol-points", default="/tmp/whest_sobol_points_full.npz")
    parser.add_argument("--cache", default=".tmp/alpha_threshold_policy_mini.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-leaf", type=int, default=8)
    args = parser.parse_args()
    payload = evaluate_dataset(
        split=args.split,
        sobol_points_path=Path(args.sobol_points),
        limit=args.limit,
        force=args.force,
        cache_path=Path(args.cache),
    )
    run_experiment(payload, folds=args.folds, min_leaf=args.min_leaf)


if __name__ == "__main__":
    main()