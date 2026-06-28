"""Normal-lognormal mean propagation estimator for ReLU MLPs.

This extends ``02_mean_propagation.py`` with the normal-lognormal prior
predictive correction suggested by Noci et al., "Precise characterization of
the prior predictive distribution of deep ReLU networks".

The diagonal baseline assumes each pre-activation is Gaussian:

    pre ~ N(mu_pre, sigma_pre^2)

This example instead uses a one-parameter normal-lognormal scale mixture:

    pre = mu_pre + sigma_pre * Y * X
    X ~ N(0, 1)
    Y = exp(S),  S ~ N(-a_l, a_l)

so E[Y^2] = 1 and ``sigma_pre^2`` remains the pre-activation variance. The
layerwise parameter follows the paper's kurtosis formula for equal width m:

    kurtosis_l = 3 * ((m + 5) / m) ** (l - 1)
    a_l = log(kurtosis_l / 3) / 4 = (l - 1) * log(1 + 5 / m) / 4

For ``a_l = 0`` this reduces to the usual Gaussian ReLU moment formula. For
``a_l > 0`` we average the exact Gaussian ReLU moments over the lognormal scale
with 5-point Gauss-Hermite quadrature. This is still O(width^2) per layer, like
the diagonal mean-propagation baseline, but pays a small pointwise constant.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# 5-point Gauss-Hermite rule for E[f(N(0, 1))]. We store normalized weights,
# i.e. weights divided by sqrt(pi).
_GH_NODES = (
    -2.0201828704560856,
    -0.9585724646138185,
    0.0,
    0.9585724646138185,
    2.0201828704560856,
)
_GH_WEIGHTS = (
    0.01125741132772069,
    0.22207592200561263,
    0.5333333333333333,
    0.22207592200561263,
    0.01125741132772069,
)


class Estimator(BaseEstimator):
    """Diagonal propagation with a normal-lognormal ReLU moment correction."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _relu_moments(self, mu_pre: fnp.ndarray, var_pre: fnp.ndarray, layer_idx: int, width: int) -> tuple[fnp.ndarray, fnp.ndarray]:
        sigma_pre = fnp.sqrt(fnp.maximum(var_pre, 1e-12))

        # Layer 1 is exactly Gaussian. Deeper layers get the paper's finite-width
        # depth correction, using the exact kurtosis expression rather than the
        # first-order 5*l/(4m) approximation.
        a = 0.25 * layer_idx * float(fnp.log(1.0 + 5.0 / width))
        if a <= 0.0:
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)
            mean = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            return mean, fnp.maximum(ez2 - mean * mean, 0.0)

        sqrt_2a = float(fnp.sqrt(2.0 * a))
        mean = fnp.zeros(width)
        ez2 = fnp.zeros(width)
        for node, weight in zip(_GH_NODES, _GH_WEIGHTS):
            scale = float(fnp.exp(-a + sqrt_2a * node))
            sigma_scaled = sigma_pre * scale
            alpha = mu_pre / fnp.maximum(sigma_scaled, 1e-12)
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            mean_part = mu_pre * Phi_alpha + sigma_scaled * phi_alpha
            ez2_part = (mu_pre * mu_pre + sigma_scaled * sigma_scaled) * Phi_alpha + mu_pre * sigma_scaled * phi_alpha
            mean = mean + weight * mean_part
            ez2 = ez2 + weight * ez2_part

        return mean, fnp.maximum(ez2 - mean * mean, 0.0)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer post-ReLU means with diagonal NLN propagation."""
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        var = fnp.ones(width)
        rows = []

        for layer_idx, w in enumerate(mlp.weights):
            mu_pre = w.T @ mu
            var_pre = (w * w).T @ var
            mu, var = self._relu_moments(mu_pre, var_pre, layer_idx, width)
            rows.append(mu)

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)