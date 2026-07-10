"""Probe sample-estimated scalar K3 projections for selected final neurons.

The block experiments suggest full K3 is valuable, but stored O(W^2) K3 is
harmful. This script asks a narrower deployability question: if a gate selected
the important final neurons, can a moderate number of propagated samples estimate
the scalar final pre-activation K3 well enough to improve the final ReLU mean?

To isolate sample noise, the default evaluation uses oracle pre mean/variance
from the higher-moment dataset and only substitutes sampled K3 on selected
neurons. That is not deployable by itself; it is an upper-bound diagnostic for
the selected scalar-K3 idea.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from learn_cumulant_closure import _moment_path, _weights
from learn_k4_downstream_gate import _edgeworth_mean


WIDTH = 256
DEPTH = 32


def _load_samples(path: str, sample_count: int) -> np.ndarray:
    data = np.load(path)
    points = data["points"]
    half_count = sample_count // 2
    if points.shape[0] < half_count:
        raise ValueError(f"{path} has {points.shape[0]} half-points, need {half_count}")
    half = np.asarray(points[:half_count, :WIDTH], dtype=np.float32)
    return np.concatenate([half, -half], axis=0)


def _final_pre_samples(weights: np.ndarray, samples: np.ndarray) -> np.ndarray:
    x = samples
    for layer_idx in range(DEPTH - 1):
        x = np.maximum(x @ weights[layer_idx], 0.0)
    return x @ weights[-1]


def _central3(values: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    if center is None:
        center = values.mean(axis=0)
    centered = values - center
    return np.mean(centered * centered * centered, axis=0)


def _true_final_arrays(index: int):
    data = np.load(_moment_path(index))
    pre_mean = np.asarray(data["pre_mean"][-1], dtype=np.float64)
    pre_m2 = np.asarray(data["pre_m2"][-1], dtype=np.float64)
    pre_m3 = np.asarray(data["pre_m3"][-1], dtype=np.float64)
    true_mean = np.asarray(data["mean"][-1], dtype=np.float64)
    pre_var = np.maximum(pre_m2 - pre_mean * pre_mean, 1e-12)
    true_k3 = pre_m3 - 3.0 * pre_mean * pre_m2 + 2.0 * pre_mean**3
    return pre_mean, pre_var, true_k3, true_mean


def _top_mask(score: np.ndarray, keep: float) -> np.ndarray:
    count = max(1, int(round(keep * score.size)))
    selected = np.argpartition(score, -count)[-count:]
    mask = np.zeros(score.size, dtype=bool)
    mask[selected] = True
    return mask


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="8,9,10,11")
    parser.add_argument("--sample-counts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--keep", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--sobol", default=str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    sample_counts = [int(item) for item in args.sample_counts.split(",") if item]
    keep_fracs = [float(item) for item in args.keep.split(",") if item]

    print(f"indices={indices} sample_counts={sample_counts} sobol={args.sobol}")
    for sample_count in sample_counts:
        samples = _load_samples(args.sobol, sample_count)
        rows = []
        for index in indices:
            weights = _weights(index)
            pre_mean, pre_var, true_k3, true_final_mean = _true_final_arrays(index)
            gaussian = _edgeworth_mean(pre_mean, pre_var, np.zeros(WIDTH), np.zeros(WIDTH))
            true_k3_pred = _edgeworth_mean(pre_mean, pre_var, true_k3, np.zeros(WIDTH))
            oracle_delta = np.abs(true_k3_pred - gaussian)

            pre_samples = _final_pre_samples(weights, samples)
            sample_pre_mean = pre_samples.mean(axis=0)
            sample_centered = pre_samples - sample_pre_mean
            sample_pre_var = np.maximum(np.mean(sample_centered * sample_centered, axis=0), 1e-12)
            sample_k3_centered_sample = _central3(pre_samples)
            sample_k3_centered_true = _central3(pre_samples, center=pre_mean)
            sample_relu_mean = np.maximum(pre_samples, 0.0).mean(axis=0)

            base = {
                "gaussian": _mse(gaussian, true_final_mean),
                "true_k3_all": _mse(true_k3_pred, true_final_mean),
                "sample_relu_mean": _mse(sample_relu_mean, true_final_mean),
                "sample_gaussian_moments": _mse(
                    _edgeworth_mean(sample_pre_mean, sample_pre_var, np.zeros(WIDTH), np.zeros(WIDTH)), true_final_mean
                ),
                "sample_k3_all_sample_moments": _mse(
                    _edgeworth_mean(sample_pre_mean, sample_pre_var, sample_k3_centered_sample, np.zeros(WIDTH)),
                    true_final_mean,
                ),
                "sample_k3_all_sample_center": _mse(
                    _edgeworth_mean(pre_mean, pre_var, sample_k3_centered_sample, np.zeros(WIDTH)), true_final_mean
                ),
                "sample_k3_all_true_center": _mse(
                    _edgeworth_mean(pre_mean, pre_var, sample_k3_centered_true, np.zeros(WIDTH)), true_final_mean
                ),
            }
            for keep in keep_fracs:
                mask = _top_mask(oracle_delta, keep)
                true_selected = np.zeros(WIDTH)
                sample_selected = np.zeros(WIDTH)
                sample_true_center_selected = np.zeros(WIDTH)
                true_selected[mask] = true_k3[mask]
                sample_selected[mask] = sample_k3_centered_sample[mask]
                sample_true_center_selected[mask] = sample_k3_centered_true[mask]
                base[f"true_k3_top{keep:g}"] = _mse(
                    _edgeworth_mean(pre_mean, pre_var, true_selected, np.zeros(WIDTH)), true_final_mean
                )
                base[f"sample_k3_top{keep:g}_sample_moments"] = _mse(
                    _edgeworth_mean(sample_pre_mean, sample_pre_var, sample_selected, np.zeros(WIDTH)), true_final_mean
                )
                base[f"sample_k3_top{keep:g}_sample_center"] = _mse(
                    _edgeworth_mean(pre_mean, pre_var, sample_selected, np.zeros(WIDTH)), true_final_mean
                )
                base[f"sample_k3_top{keep:g}_true_center"] = _mse(
                    _edgeworth_mean(pre_mean, pre_var, sample_true_center_selected, np.zeros(WIDTH)), true_final_mean
                )
            rows.append(base)

        keys = list(rows[0])
        print(f"\nN={sample_count}")
        for key in keys:
            print(f"  {key:<34} {np.mean([row[key] for row in rows]):.3e}")


if __name__ == "__main__":
    main()