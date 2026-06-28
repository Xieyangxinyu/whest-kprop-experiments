"""Antithetic Monte Carlo sampling estimator for ReLU MLPs.

This is direct Gaussian-input sampling with paired inputs ``x`` and ``-x``.
The input distribution is symmetric, and the first-layer ReLU response for a
pair has less variance than two unrelated samples. The deeper-network effect is
empirical, so this file exists to benchmark whether the variance reduction
survives through the random ReLU stack.

``WHEST_MC_SAMPLES`` controls the total number of inputs after pairing. If it is
odd, it is rounded up to the next even number.
"""

from __future__ import annotations

import os

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_DEFAULT_SAMPLES = int(os.environ.get("WHEST_MC_SAMPLES", "8192"))


class Estimator(BaseEstimator):
    """Estimate activation means with antithetic Gaussian input pairs."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Return antithetic Monte Carlo estimates of all layer means."""
        _ = budget
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width
        n_pairs = max(1, (_DEFAULT_SAMPLES + 1) // 2)

        x_half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
        x = fnp.concatenate([x_half, -x_half], axis=0)

        rows = []
        for w in mlp.weights:
            x = fnp.maximum(x @ w, 0.0)
            rows.append(fnp.mean(x, axis=0))

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)