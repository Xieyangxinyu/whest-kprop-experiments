"""Offline stopping-bandit diagnostic for dynamic sample allocation.

Given cached per-MLP losses at several sample counts, learn whether to stop at
an intermediate checkpoint or continue to a later checkpoint. This is a first
step toward a sequential adaptive estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.learn_allocation_policy import fit_ridge, standardize_train_valid


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MARGINS = (0.0, 1e-9, 2.5e-9, 5e-9, 1e-8, 2e-8, 5e-8)


def load_features(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {int(record["mlp_id"]): record.get("features", {}) for record in payload["records"]}


def load_records(pareto_path: Path, feature_path: Path) -> tuple[list[dict], list[int]]:
    payload = json.loads(pareto_path.read_text())
    sample_counts = [int(sample) for sample in payload["samples"]]
    feature_by_id = load_features(feature_path)
    records = []
    for record in payload["records"]:
        features = dict(feature_by_id.get(int(record["id"]), {}))
        for key, value in record.get("mom", {}).items():
            features[f"mom_{key}"] = float(value)
        costs = {int(sample): float(record["costs"][str(sample)]) for sample in sample_counts}
        records.append(
            {
                "id": int(record["id"]),
                "name": record["name"],
                "features": features,
                "costs": costs,
            }
        )
    return records, sample_counts


def add_intercept(x_values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((x_values.shape[0], 1), dtype=x_values.dtype), x_values], axis=1)


def predict_ridge(x_values: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return add_intercept(x_values) @ coef


def fit_gain_ridge(x_values: np.ndarray, y_values: np.ndarray, alpha: float) -> np.ndarray:
    return fit_ridge(x_values, y_values, alpha)


def evaluate_one_step(records: list[dict], sample_counts: list[int], stop_sample: int, folds: int) -> None:
    future_samples = [sample for sample in sample_counts if sample > stop_sample]
    if not future_samples:
        raise ValueError(f"no future samples after {stop_sample}")

    feature_names = sorted(records[0]["features"].keys())
    x_values = np.asarray([[record["features"].get(name, 0.0) for name in feature_names] for record in records], dtype=np.float64)
    stop_loss = np.asarray([record["costs"][stop_sample] for record in records], dtype=np.float64)
    future_loss = np.asarray([min(record["costs"][sample] for sample in future_samples) for record in records], dtype=np.float64)
    best_future_sample = np.asarray(
        [min(future_samples, key=lambda sample: record["costs"][sample]) for record in records], dtype=np.int64
    )
    gain = stop_loss - future_loss
    n_rows = len(records)

    print(f"\nONE-STEP BANDIT AT {stop_sample}")
    print(f"stop score={float(np.mean(stop_loss)):.9e}")
    print(f"oracle continue/stop score={float(np.mean(np.minimum(stop_loss, future_loss))):.9e}")
    print(f"always continue-to-best-future score={float(np.mean(future_loss)):.9e}")
    print(f"continue helpful count={(gain > 0).sum()} / {n_rows}")
    values, counts = np.unique(best_future_sample[gain > 0], return_counts=True)
    print("helpful future sample counts", {int(value): int(count) for value, count in zip(values, counts)})

    fold_ids = np.arange(n_rows) % folds
    rng = np.random.default_rng(13579 + stop_sample)
    rng.shuffle(fold_ids)

    best = None
    for alpha in RIDGE_ALPHAS:
        pred_gain = np.zeros(n_rows, dtype=np.float64)
        for fold in range(folds):
            train_idx = np.where(fold_ids != fold)[0]
            valid_idx = np.where(fold_ids == fold)[0]
            x_train, x_valid = standardize_train_valid(x_values[train_idx], x_values[valid_idx])
            coef = fit_gain_ridge(x_train, gain[train_idx], alpha)
            pred_gain[valid_idx] = predict_ridge(x_valid, coef)
        for margin in MARGINS:
            continue_mask = pred_gain > margin
            chosen_loss = np.where(continue_mask, future_loss, stop_loss)
            score = float(np.mean(chosen_loss))
            if best is None or score < best[0]:
                best = (score, alpha, margin, int(continue_mask.sum()), pred_gain)
            print(
                f"alpha={alpha:<7g} margin={margin:<8g} score={score:.9e} "
                f"continue={int(continue_mask.sum())}"
            )

    assert best is not None
    score, alpha, margin, count, pred_gain = best
    print(f"BEST stop bandit: score={score:.9e} alpha={alpha:g} margin={margin:g} continue={count}")
    print(f"corr(pred_gain,true_gain)={float(np.corrcoef(pred_gain, gain)[0,1]):+.4f}")
    rankings = []
    gain_rank = np.argsort(np.argsort(gain))
    for index, name in enumerate(feature_names):
        feature = x_values[:, index]
        if np.std(feature) <= 1e-12:
            continue
        corr = float(np.corrcoef(np.argsort(np.argsort(feature)), gain_rank)[0, 1])
        rankings.append((abs(corr), corr, name))
    print("top feature Spearman with continue gain")
    for _abs_corr, corr, name in sorted(rankings, reverse=True)[:20]:
        print(f"  {name:<36} {corr:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-cache", default=".tmp/pareto_allocation_mini.json")
    parser.add_argument("--feature-cache", default=".tmp/allocation_policy_mini_prefix.json")
    parser.add_argument("--stop-samples", type=int, nargs="+", default=[30720, 40960])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    records, sample_counts = load_records(Path(args.pareto_cache), Path(args.feature_cache))
    print("fixed sample scores")
    for sample in sample_counts:
        print(f"  {sample}: {np.mean([record['costs'][sample] for record in records]):.9e}")
    print(f"discrete oracle: {np.mean([min(record['costs'].values()) for record in records]):.9e}")
    for stop_sample in args.stop_samples:
        evaluate_one_step(records, sample_counts, stop_sample, args.folds)


if __name__ == "__main__":
    main()