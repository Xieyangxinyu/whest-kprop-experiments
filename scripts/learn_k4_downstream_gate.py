"""Learn gates for K4 corrections that matter to ReLU means.

This script shifts the target from "predict the omitted fourth cumulant" to
"rank positions whose omitted fourth cumulant changes the next post-ReLU mean".
The evaluation reports an oracle-value mask: if the learned gate selects a
position, we use the true omitted K4 value there; otherwise it is zeroed. That
answers whether we can learn which interactions should be computed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import special

from learn_cumulant_closure import _expand_features, _extract_blocks, _fit_ridge, _moment_path, _standardize


WIDTH = 256
DEPTH = 32


@dataclass
class Case:
    index: int
    features: np.ndarray
    pre_mean: np.ndarray
    pre_var: np.ndarray
    true_post_mean: np.ndarray
    stored3: np.ndarray
    stored4: np.ndarray
    full4: np.ndarray


def _tail_moments(alpha: np.ndarray, max_power: int = 7) -> list[np.ndarray]:
    threshold = -alpha
    phi = np.exp(-0.5 * threshold * threshold) / np.sqrt(2.0 * np.pi)
    moments = [special.ndtr(alpha), phi]
    for power in range(2, max_power + 1):
        moments.append((threshold ** (power - 1)) * phi + (power - 1) * moments[power - 2])
    return moments


def _hermite_relu_integral(alpha: np.ndarray, order: int) -> np.ndarray:
    moments = _tail_moments(alpha, 7)
    if order == 3:
        h = moments[3] - 3.0 * moments[1]
        xh = moments[4] - 3.0 * moments[2]
    elif order == 4:
        h = moments[4] - 6.0 * moments[2] + 3.0 * moments[0]
        xh = moments[5] - 6.0 * moments[3] + 3.0 * moments[1]
    elif order == 6:
        h = moments[6] - 15.0 * moments[4] + 45.0 * moments[2] - 15.0 * moments[0]
        xh = moments[7] - 15.0 * moments[5] + 45.0 * moments[3] - 15.0 * moments[1]
    else:
        raise ValueError(order)
    return alpha * h + xh


def _edgeworth_mean(pre_mean: np.ndarray, pre_var: np.ndarray, k3: np.ndarray, k4: np.ndarray) -> np.ndarray:
    pre_var = np.maximum(pre_var, 1e-12)
    sigma = np.sqrt(pre_var)
    alpha = np.clip(pre_mean / sigma, -10.0, 10.0)
    phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2.0 * np.pi)
    gaussian = pre_mean * special.ndtr(alpha) + sigma * phi
    skew = np.clip(k3 / np.maximum(sigma**3, 1e-12), -8.0, 8.0)
    excess = np.clip(k4 / np.maximum(pre_var * pre_var, 1e-12), -16.0, 16.0)
    correction = (skew / 6.0) * _hermite_relu_integral(alpha, 3)
    correction += (excess / 24.0) * _hermite_relu_integral(alpha, 4)
    correction += (skew * skew / 72.0) * _hermite_relu_integral(alpha, 6)
    return np.maximum(gaussian + sigma * correction, 0.0)


def _load_case(index: int) -> Case:
    block = _extract_blocks(index)
    data = np.load(_moment_path(index))
    pre_mean = np.asarray(data["pre_mean"], dtype=np.float64)[1:].reshape(-1)
    pre_m2 = np.asarray(data["pre_m2"], dtype=np.float64)[1:].reshape(-1)
    true_post_mean = np.asarray(data["mean"], dtype=np.float64)[1:].reshape(-1)
    pre_var = np.maximum(pre_m2 - pre_mean * pre_mean, 1e-12)
    return Case(index, block.features, pre_mean, pre_var, true_post_mean, block.stored3, block.stored4, block.full4)


def _split_arrays(cases: list[Case], feature_mode: str):
    features = np.concatenate([case.features for case in cases])
    if feature_mode == "expanded":
        features = _expand_features(features)
    return {
        "features": features,
        "pre_mean": np.concatenate([case.pre_mean for case in cases]),
        "pre_var": np.concatenate([case.pre_var for case in cases]),
        "true_post_mean": np.concatenate([case.true_post_mean for case in cases]),
        "stored3": np.concatenate([case.stored3 for case in cases]),
        "stored4": np.concatenate([case.stored4 for case in cases]),
        "full4": np.concatenate([case.full4 for case in cases]),
    }


def _final_mask(num_cases: int) -> np.ndarray:
    one_case = np.zeros((DEPTH - 1, WIDTH), dtype=bool)
    one_case[-1, :] = True
    return np.tile(one_case.reshape(-1), num_cases)


def _evaluate_region(name: str, arrays: dict[str, np.ndarray], scores: dict[str, np.ndarray], region: np.ndarray) -> None:
    pre_mean = arrays["pre_mean"][region]
    pre_var = arrays["pre_var"][region]
    true_mean = arrays["true_post_mean"][region]
    stored3 = arrays["stored3"][region]
    stored4 = arrays["stored4"][region]
    full4 = arrays["full4"][region]

    gaussian = _edgeworth_mean(pre_mean, pre_var, np.zeros_like(stored3), np.zeros_like(stored4))
    stored = _edgeworth_mean(pre_mean, pre_var, stored3, stored4)
    full = _edgeworth_mean(pre_mean, pre_var, stored3, full4)
    base_mse = float(np.mean((stored - true_mean) ** 2))
    full_mse = float(np.mean((full - true_mean) ** 2))
    gauss_mse = float(np.mean((gaussian - true_mean) ** 2))
    print(f"\n{name}: gaussian={gauss_mse:.3e} stored_k4={base_mse:.3e} full_k4={full_mse:.3e}")

    delta = full - stored
    for score_name, full_score in scores.items():
        score = full_score[region]
        print(f"  score={score_name}")
        for keep in (0.02, 0.05, 0.10, 0.20, 0.40):
            keep_count = max(1, int(round(keep * score.size)))
            selected = np.argpartition(score, -keep_count)[-keep_count:]
            mask = np.zeros(score.size, dtype=bool)
            mask[selected] = True
            gated = stored + np.where(mask, delta, 0.0)
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
    cases = [_load_case(index) for index in indices]
    train_cases = cases[: args.train_count]
    test_cases = cases[args.train_count :]
    if not train_cases or not test_cases:
        raise SystemExit("need non-empty train and test splits")

    train = _split_arrays(train_cases, args.feature_mode)
    test = _split_arrays(test_cases, args.feature_mode)
    train_x, test_x = _standardize(train["features"], test["features"])

    train_stored = _edgeworth_mean(train["pre_mean"], train["pre_var"], train["stored3"], train["stored4"])
    train_full = _edgeworth_mean(train["pre_mean"], train["pre_var"], train["stored3"], train["full4"])
    train_delta = train_full - train_stored

    signed_coef = _fit_ridge(train_x, train_delta, args.ridge)
    abs_coef = _fit_ridge(train_x, np.abs(train_delta), args.ridge)
    signed_pred = test_x @ signed_coef
    abs_pred = np.maximum(test_x @ abs_coef, 0.0)

    scores = {
        "learned_abs_delta": abs_pred,
        "learned_abs_signed": np.abs(signed_pred),
        "stored4_mag": np.abs(test["stored4"]),
        "full4_oracle_delta": np.abs(
            _edgeworth_mean(test["pre_mean"], test["pre_var"], test["stored3"], test["full4"])
            - _edgeworth_mean(test["pre_mean"], test["pre_var"], test["stored3"], test["stored4"])
        ),
    }

    print(f"loaded train={[c.index for c in train_cases]} test={[c.index for c in test_cases]} features={args.feature_mode}")
    all_region = np.ones_like(test["pre_mean"], dtype=bool)
    final_region = _final_mask(len(test_cases))
    late_region = np.tile((np.arange(DEPTH - 1)[:, None] >= 24).repeat(WIDTH, axis=1).reshape(-1), len(test_cases))
    _evaluate_region("all transitions", test, scores, all_region)
    _evaluate_region("late transitions 24..30", test, scores, late_region)
    _evaluate_region("final transition 30->31", test, scores, final_region)


if __name__ == "__main__":
    main()