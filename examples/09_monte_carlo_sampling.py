"""Direct Monte Carlo sampling estimator for ReLU MLPs.

This estimator spends FLOPs on the thing the benchmark asks for directly:
sample Gaussian inputs, run them through the MLP, and average post-ReLU
activations at every layer.

The sample count is controlled by ``WHEST_MC_SAMPLES``. The default 8192 samples
costs roughly 34.5B FLOPs at width=256/depth=32. This crosses the phase-1 score
multiplier floor (about 27.2B effective FLOPs), but the variance reduction was
worth it in local scored comparisons.
"""

from __future__ import annotations

import os

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_DEFAULT_SAMPLES = int(os.environ.get("WHEST_MC_SAMPLES", "8192"))


class Estimator(BaseEstimator):
    """Estimate activation means by direct forward sampling."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Return Monte Carlo estimates of all post-ReLU layer means."""
        _ = budget
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width
        n_samples = _DEFAULT_SAMPLES

        x = fnp.array(rng.standard_normal((n_samples, width)).astype(fnp.float32))
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