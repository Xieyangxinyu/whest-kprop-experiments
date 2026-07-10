"""Learn gates for K3 corrections that matter to ReLU means.

The retained O(W^2) K3 approximation can be harmful, so this script tests a
different screening question: can we identify positions where it is worth
computing the full third-cumulant contraction, while zeroing K3 elsewhere?
"""

from __future__ import annotations

import argparse

import numpy as np

from learn_cumulant_closure import _expand_features, _extract_blocks, _fit_ridge, _moment_path, _standardize
from learn_k4_downstream_gate import DEPTH, WIDTH, _edgeworth_mean


def _load_arrays(indices: list[int], feature_mode: str) -> dict[str, np.ndarray]:
    features = []
    pre_mean = []
    pre_var = []
    true_post_mean = []
    full3 = []
    stored3 = []
    for index in indices:
        block = _extract_blocks(index)
        data = np.load(_moment_path(index))
        mean = data["pre_mean"].astype(np.float64)[1:].reshape(-1)
        m2 = data["pre_m2"].astype(np.float64)[1:].reshape(-1)
        features.append(block.features)
        pre_mean.append(mean)
        pre_var.append(np.maximum(m2 - mean * mean, 1e-12))
        true_post_mean.append(data["mean"].astype(np.float64)[1:].reshape(-1))
        full3.append(block.full3)
        stored3.append(block.stored3)
    x = np.concatenate(features)
    if feature_mode == "expanded":
        x = _expand_features(x)
    return {
        "features": x,
        "pre_mean": np.concatenate(pre_mean),
        "pre_var": np.concatenate(pre_var),
        "true_post_mean": np.concatenate(true_post_mean),
        "full3": np.concatenate(full3),
        "stored3": np.concatenate(stored3),
    }


def _region_mask(num_cases: int, mode: str) -> np.ndarray:
    layer_ids = np.arange(DEPTH - 1)[:, None].repeat(WIDTH, axis=1)
    if mode == "all":
        one_case = np.ones((DEPTH - 1, WIDTH), dtype=bool)
    elif mode == "late":
        one_case = layer_ids >= 24
    elif mode == "final":
        one_case = layer_ids == DEPTH - 2
    else:
        raise ValueError(mode)
    return np.tile(one_case.reshape(-1), num_cases)


def _evaluate(name: str, test: dict[str, np.ndarray], scores: dict[str, np.ndarray], region: np.ndarray) -> None:
    pre_mean = test["pre_mean"][region]
    pre_var = test["pre_var"][region]
    true_mean = test["true_post_mean"][region]
    full3 = test["full3"][region]
    stored3 = test["stored3"][region]
    zero = np.zeros_like(full3)
    gaussian = _edgeworth_mean(pre_mean, pre_var, zero, zero)
    stored = _edgeworth_mean(pre_mean, pre_var, stored3, zero)
    full = _edgeworth_mean(pre_mean, pre_var, full3, zero)
    print(
        f"\n{name}: gaussian={np.mean((gaussian - true_mean) ** 2):.3e} "
        f"stored3={np.mean((stored - true_mean) ** 2):.3e} full3={np.mean((full - true_mean) ** 2):.3e}"
    )
    delta = full - gaussian
    for score_name, full_score in scores.items():
        score = full_score[region]
        print(f"  score={score_name}")
        for keep in (0.02, 0.05, 0.10, 0.20, 0.40):
            keep_count = max(1, int(round(keep * score.size)))
            selected = np.argpartition(score, -keep_count)[-keep_count:]
            mask = np.zeros(score.size, dtype=bool)
            mask[selected] = True
            gated_k3 = np.where(mask, full3, 0.0)
            gated = _edgeworth_mean(pre_mean, pre_var, gated_k3, zero)
            mse = float(np.mean((gated - true_mean) ** 2))
            true_threshold = np.quantile(np.abs(delta), 1.0 - keep)
            true_top = np.abs(delta) >= true_threshold
            recall = float(np.mean(mask[true_top])) if np.any(true_top) else 0.0
            print(f"    keep={keep:0.2f} mse={mse:.3e} recall_top_delta={recall:.3f} active={np.mean(mask):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-1)
    parser.add_argument("--feature-mode", choices=["simple", "expanded"], default="expanded")
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    train_indices = indices[: args.train_count]
    test_indices = indices[args.train_count :]
    train = _load_arrays(train_indices, args.feature_mode)
    test = _load_arrays(test_indices, args.feature_mode)
    train_x, test_x = _standardize(train["features"], test["features"])
    zero_train = np.zeros_like(train["full3"])
    train_gaussian = _edgeworth_mean(train["pre_mean"], train["pre_var"], zero_train, zero_train)
    train_full = _edgeworth_mean(train["pre_mean"], train["pre_var"], train["full3"], zero_train)
    train_delta = train_full - train_gaussian
    signed_coef = _fit_ridge(train_x, train_delta, args.ridge)
    abs_coef = _fit_ridge(train_x, np.abs(train_delta), args.ridge)
    signed_pred = test_x @ signed_coef
    abs_pred = np.maximum(test_x @ abs_coef, 0.0)
    test_zero = np.zeros_like(test["full3"])
    oracle_delta = np.abs(_edgeworth_mean(test["pre_mean"], test["pre_var"], test["full3"], test_zero) - _edgeworth_mean(test["pre_mean"], test["pre_var"], test_zero, test_zero))
    scores = {
        "learned_abs_delta": abs_pred,
        "learned_abs_signed": np.abs(signed_pred),
        "stored3_mag": np.abs(test["stored3"]),
        "full3_oracle_delta": oracle_delta,
    }
    print(f"loaded train={train_indices} test={test_indices} features={args.feature_mode}")
    for mode, label in (("all", "all transitions"), ("late", "late transitions 24..30"), ("final", "final transition 30->31")):
        _evaluate(label, test, scores, _region_mask(len(test_indices), mode))


if __name__ == "__main__":
    main()