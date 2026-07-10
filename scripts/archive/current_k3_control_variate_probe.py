"""Test K3 corrections as a control variate around the current estimator.

This script is an offline diagnostic. It uses the current submission estimator as
the baseline final prediction, estimates final scalar K3 from Sobol-propagated
full-network pre-activations, converts that K3 into an Edgeworth mean delta using
oracle pre mean/variance, then fits a single damping coefficient on train MLPs.

The purpose is to answer: if we had a decent gate and scalar K3 estimate, does a
K3-derived correction move the current active-set estimator in a useful direction?
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
from learn_k4_downstream_gate import _edgeworth_mean
from sample_k3_projection_probe import _central3, _final_pre_samples, _load_samples, _top_mask


WIDTH = 256
BUDGET = 272_000_000_000


def _mlp_from_weights(weights_np: np.ndarray) -> MLP:
    weights = [fnp.array(weights_np[layer].astype(np.float32)) for layer in range(weights_np.shape[0])]
    return MLP(width=weights_np.shape[1], depth=weights_np.shape[0], weights=weights)


def _current_final_prediction(weights_np: np.ndarray) -> np.ndarray:
    estimator = Estimator()
    mlp = _mlp_from_weights(weights_np)
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True):
        pred = estimator.predict(mlp, BUDGET)
    return np.asarray(pred[-1], dtype=np.float64)


def _case(index: int, samples: np.ndarray, keep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights_np = _weights(index)
    data = np.load(_moment_path(index))
    true_final = np.asarray(data["mean"][-1], dtype=np.float64)
    pre_mean = np.asarray(data["pre_mean"][-1], dtype=np.float64)
    pre_m2 = np.asarray(data["pre_m2"][-1], dtype=np.float64)
    pre_m3 = np.asarray(data["pre_m3"][-1], dtype=np.float64)
    pre_var = np.maximum(pre_m2 - pre_mean * pre_mean, 1e-12)
    true_k3 = pre_m3 - 3.0 * pre_mean * pre_m2 + 2.0 * pre_mean**3

    baseline = _current_final_prediction(weights_np)
    gaussian = _edgeworth_mean(pre_mean, pre_var, np.zeros(WIDTH), np.zeros(WIDTH))
    true_delta = _edgeworth_mean(pre_mean, pre_var, true_k3, np.zeros(WIDTH)) - gaussian
    mask = _top_mask(np.abs(true_delta), keep)

    pre_samples = _final_pre_samples(weights_np, samples)
    sample_k3 = _central3(pre_samples)
    selected_k3 = np.zeros(WIDTH)
    selected_k3[mask] = sample_k3[mask]
    correction = _edgeworth_mean(pre_mean, pre_var, selected_k3, np.zeros(WIDTH)) - gaussian
    return baseline, correction, true_final


def _fit_lambda(baselines: list[np.ndarray], corrections: list[np.ndarray], truths: list[np.ndarray]) -> float:
    residual = np.concatenate([truth - baseline for baseline, truth in zip(baselines, truths)])
    correction = np.concatenate(corrections)
    denom = float(correction @ correction)
    if denom <= 1e-30:
        return 0.0
    return float((correction @ residual) / denom)


def _mean_mse(preds: list[np.ndarray], truths: list[np.ndarray]) -> float:
    return float(np.mean([np.mean((pred - truth) ** 2) for pred, truth in zip(preds, truths)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--sample-counts", default="8192,16384,32768")
    parser.add_argument("--keep", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--sobol", default=str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    train_indices = indices[: args.train_count]
    test_indices = indices[args.train_count :]
    sample_counts = [int(item) for item in args.sample_counts.split(",") if item]
    keep_fracs = [float(item) for item in args.keep.split(",") if item]
    print(f"train={train_indices} test={test_indices}")

    for sample_count in sample_counts:
        samples = _load_samples(args.sobol, sample_count)
        print(f"\nN={sample_count}")
        for keep in keep_fracs:
            train_cases = [_case(index, samples, keep) for index in train_indices]
            test_cases = [_case(index, samples, keep) for index in test_indices]
            train_baselines, train_corrections, train_truths = map(list, zip(*train_cases))
            test_baselines, test_corrections, test_truths = map(list, zip(*test_cases))
            fitted_lambda = _fit_lambda(train_baselines, train_corrections, train_truths)
            fitted_lambda = float(np.clip(fitted_lambda, -2.0, 2.0))
            train_base = _mean_mse(train_baselines, train_truths)
            test_base = _mean_mse(test_baselines, test_truths)
            train_fit = _mean_mse(
                [base + fitted_lambda * corr for base, corr in zip(train_baselines, train_corrections)], train_truths
            )
            test_fit = _mean_mse(
                [base + fitted_lambda * corr for base, corr in zip(test_baselines, test_corrections)], test_truths
            )
            test_oracle_lambda = _fit_lambda(test_baselines, test_corrections, test_truths)
            test_oracle_lambda = float(np.clip(test_oracle_lambda, -2.0, 2.0))
            test_oracle = _mean_mse(
                [base + test_oracle_lambda * corr for base, corr in zip(test_baselines, test_corrections)], test_truths
            )
            print(
                f"  keep={keep:0.2f} lambda={fitted_lambda:+.3f} "
                f"train {train_base:.3e}->{train_fit:.3e} "
                f"test {test_base:.3e}->{test_fit:.3e} "
                f"test_oracle_lambda={test_oracle_lambda:+.3f} test_oracle={test_oracle:.3e}"
            )


if __name__ == "__main__":
    main()