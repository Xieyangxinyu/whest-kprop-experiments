"""Active-set Monte Carlo sampling estimator for ReLU MLPs.

This sampler exploits the sparse/dead structure visible in the activation
analysis notebooks:

1. Run a small full-network pilot sample.
2. Estimate each layer neuron's firing rate and mean activation.
3. Keep only neurons whose pilot firing rate exceeds a threshold.
4. Spend the main sampling budget on the resulting active subnetwork.

Dropped neurons keep their pilot mean in the returned prediction and are treated
as zero downstream in the main pass. This is biased, but can reduce the cost per
main sample if deep layers have many reliably dead or near-dead units.

Environment knobs:

* ``WHEST_ACTIVE_PILOT_SAMPLES``: full-network pilot samples, default 512.
* ``WHEST_ACTIVE_MAIN_SAMPLES``: active-subnetwork main samples, default 12288.
* ``WHEST_ACTIVE_FIRE_THRESHOLD``: keep units with pilot firing rate at least
    this value, default 0.02.
* ``WHEST_ACTIVE_USE_ANTITHETIC``: use x/-x input pairs, default 1.
"""

from __future__ import annotations

import os

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_PILOT_SAMPLES = int(os.environ.get("WHEST_ACTIVE_PILOT_SAMPLES", "512"))
_MAIN_SAMPLES = int(os.environ.get("WHEST_ACTIVE_MAIN_SAMPLES", "12288"))
_FIRE_THRESHOLD = float(os.environ.get("WHEST_ACTIVE_FIRE_THRESHOLD", "0.02"))
_USE_ANTITHETIC = os.environ.get("WHEST_ACTIVE_USE_ANTITHETIC", "1") != "0"


def _normal_samples(rng, n_samples: int, width: int) -> fnp.ndarray:
    if _USE_ANTITHETIC:
        n_pairs = max(1, (n_samples + 1) // 2)
        half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
        return fnp.concatenate([half, -half], axis=0)
    return fnp.array(rng.standard_normal((n_samples, width)).astype(fnp.float32))


class Estimator(BaseEstimator):
    """Pilot-pruned active-subnetwork sampler."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Estimate all layer means with active-set sampling."""
        _ = budget
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width

        # Pilot pass through the full network.
        x = _normal_samples(rng, _PILOT_SAMPLES, width)
        pilot_rows = []
        active_indices = []
        active_counts = []
        for w in mlp.weights:
            x = fnp.maximum(x @ w, 0.0)
            pilot_mean = fnp.mean(x, axis=0)
            firing_rate = fnp.mean(x > 0.0, axis=0)
            keep = firing_rate >= _FIRE_THRESHOLD
            idx = fnp.nonzero(keep)[0]
            active_indices.append(idx)
            active_counts.append(int(fnp.sum(keep)))
            pilot_rows.append(pilot_mean)

        # Main pass through only the retained active subnetwork.
        x_main = _normal_samples(rng, _MAIN_SAMPLES, width)
        rows = []
        prev_idx = None
        for layer_idx, w in enumerate(mlp.weights):
            idx = active_indices[layer_idx]
            active_count = active_counts[layer_idx]

            if active_count == 0:
                rows.append(pilot_rows[layer_idx])
                x_main = fnp.zeros((x_main.shape[0], 0))
                prev_idx = idx
                continue

            if prev_idx is None:
                w_active = w[:, idx]
            else:
                w_active = w[prev_idx, :][:, idx]

            x_main = fnp.maximum(x_main @ w_active, 0.0)
            active_mean = fnp.mean(x_main, axis=0)

            row = pilot_rows[layer_idx]
            row = row.copy()
            row[idx] = active_mean
            rows.append(row)
            prev_idx = idx

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)