"""Probe higher-moment ReLU mean corrections on Keenan's moment dataset.

This is an offline research script. It uses the public higher-moment files as
oracle supervision to test whether pre-activation skew/kurtosis can explain the
residual of the Gaussian ReLU mean formula on held-out MLPs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from huggingface_hub import hf_hub_download
from scipy import special


REPO = "keenanpepper/arc-whestbench-higher-moments-2026"


@dataclass
class MomentCase:
    index: int
    alpha: np.ndarray
    sigma: np.ndarray
    skew: np.ndarray
    excess: np.ndarray
    true_mean: np.ndarray
    gaussian_mean: np.ndarray


def _download(index: int) -> str:
    return hf_hub_download(REPO, f"full/mlp_{index:05d}.npz", repo_type="dataset")


def _load_case(index: int) -> MomentCase:
    data = np.load(_download(index))
    pre_mean = data["pre_mean"].astype(np.float64)
    pre_m2 = data["pre_m2"].astype(np.float64)
    pre_m3 = data["pre_m3"].astype(np.float64)
    pre_m4 = data["pre_m4"].astype(np.float64)
    true_mean = data["mean"].astype(np.float64)

    variance = np.maximum(pre_m2 - pre_mean * pre_mean, 1e-12)
    sigma = np.sqrt(variance)
    alpha = pre_mean / sigma
    phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2.0 * np.pi)
    gaussian_mean = pre_mean * special.ndtr(alpha) + sigma * phi

    centered3 = pre_m3 - 3.0 * pre_mean * pre_m2 + 2.0 * pre_mean**3
    centered4 = pre_m4 - 4.0 * pre_mean * pre_m3 + 6.0 * pre_mean * pre_mean * pre_m2 - 3.0 * pre_mean**4
    skew = centered3 / np.maximum(sigma**3, 1e-12)
    excess = centered4 / np.maximum(variance * variance, 1e-12) - 3.0

    skew = np.nan_to_num(np.clip(skew, -8.0, 8.0))
    excess = np.nan_to_num(np.clip(excess, -16.0, 16.0))
    return MomentCase(index, alpha, sigma, skew, excess, true_mean, gaussian_mean)


def _tail_moments(alpha: np.ndarray, max_power: int = 7) -> list[np.ndarray]:
    """Return I_k(t)=int_t^inf x^k phi(x) dx for t=-alpha."""
    threshold = -alpha
    phi_t = np.exp(-0.5 * threshold * threshold) / np.sqrt(2.0 * np.pi)
    moments = [special.ndtr(alpha), phi_t]
    for power in range(2, max_power + 1):
        moments.append((threshold ** (power - 1)) * phi_t + (power - 1) * moments[power - 2])
    return moments


def _poly_integral(coeffs: dict[int, float], moments: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(moments[0])
    for power, coeff in coeffs.items():
        out = out + coeff * moments[power]
    return out


def _edgeworth_basis(case: MomentCase) -> np.ndarray:
    """Feature basis for residual/sigma using truncated Hermite integrals."""
    moments = _tail_moments(case.alpha, 7)

    # Probabilists' Hermite polynomials.
    hermite = {
        3: {3: 1.0, 1: -3.0},
        4: {4: 1.0, 2: -6.0, 0: 3.0},
        5: {5: 1.0, 3: -10.0, 1: 15.0},
        6: {6: 1.0, 4: -15.0, 2: 45.0, 0: -15.0},
    }

    basis = []
    for order in (3, 4, 5, 6):
        h = _poly_integral(hermite[order], moments)
        xh = _poly_integral({power + 1: coeff for power, coeff in hermite[order].items()}, moments)
        basis.append(case.alpha * h + xh)

    b3, b4, b5, b6 = basis
    features = [
        case.skew * b3,
        case.excess * b4,
        case.skew * case.skew * b6,
        case.skew * case.excess * b5,
        case.skew * b4,
        case.excess * b3,
    ]
    return np.stack(features, axis=-1)


def _poly_basis(case: MomentCase) -> np.ndarray:
    alpha = np.clip(case.alpha, -8.0, 8.0)
    skew = case.skew
    excess = case.excess
    features = [
        skew,
        skew * alpha,
        skew * alpha * alpha,
        excess,
        excess * alpha,
        excess * alpha * alpha,
        skew * skew,
        skew * excess,
    ]
    return np.stack(features, axis=-1)


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    xtx = features.T @ features
    rhs = features.T @ target
    return np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), rhs)


def _flatten(cases: list[MomentCase], feature_fn, layer_slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    scales = []
    for case in cases:
        residual = (case.true_mean - case.gaussian_mean) / np.maximum(case.sigma, 1e-12)
        xs.append(feature_fn(case)[layer_slice].reshape(-1, feature_fn(case).shape[-1]))
        ys.append(residual[layer_slice].reshape(-1))
        scales.append(case.sigma[layer_slice].reshape(-1))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(scales)


def _score(cases: list[MomentCase], coef: np.ndarray, feature_fn, label: str) -> None:
    final_gauss_errors = []
    final_corr_errors = []
    all_gauss_errors = []
    all_corr_errors = []
    for case in cases:
        correction = case.sigma * (feature_fn(case) @ coef)
        corrected = case.gaussian_mean + correction
        all_gauss_errors.append(np.mean((case.gaussian_mean - case.true_mean) ** 2))
        all_corr_errors.append(np.mean((corrected - case.true_mean) ** 2))
        final_gauss_errors.append(np.mean((case.gaussian_mean[-1] - case.true_mean[-1]) ** 2))
        final_corr_errors.append(np.mean((corrected[-1] - case.true_mean[-1]) ** 2))
    print(
        f"{label:<12} all {np.mean(all_gauss_errors):.3e} -> {np.mean(all_corr_errors):.3e} "
        f"final {np.mean(final_gauss_errors):.3e} -> {np.mean(final_corr_errors):.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    cases = [_load_case(index) for index in indices]
    train = cases[: args.train_count]
    test = cases[args.train_count :]
    if not train or not test:
        raise SystemExit("need at least one train and one test MLP")

    print(f"loaded train={[c.index for c in train]} test={[c.index for c in test]}")

    for name, feature_fn in (("edge", _edgeworth_basis), ("poly", _poly_basis)):
        x_train, y_train, _ = _flatten(train, feature_fn, slice(0, 31))
        coef = _fit_ridge(x_train, y_train, args.ridge)
        print(f"{name} coef", np.array2string(coef, precision=4, suppress_small=False))
        _score(train, coef, feature_fn, f"{name} train")
        _score(test, coef, feature_fn, f"{name} test")


if __name__ == "__main__":
    main()