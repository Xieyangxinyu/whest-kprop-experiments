"""Normal-lognormal K=4 covariance propagation for ReLU MLPs.

This experiment combines two ideas:

* the K=4 ReLU covariance series from ``05_paper_series_covariance.py``;
* the normal-lognormal prior-predictive correction from Noci et al.,
  "Precise characterization of the prior predictive distribution of deep ReLU
  networks".

For each layer, the marginal pre-activation is approximated as

    pre = mu_pre + sigma_pre * Y * X

where ``X ~ N(0, 1)`` and ``Y = exp(S)``, ``S ~ N(-a_l, a_l)``. We use
5-point Gauss-Hermite quadrature over ``S`` to compute the post-ReLU mean,
diagonal variance, and the first four ReLU-series coefficients.

This is an experimental approximation. The exact finite-width paper result is
a Meijer-G mixture over active subnetworks, not a drop-in replacement for the
joint covariance update. The point of this file is to make the idea executable
and benchmarkable against the shipped examples.
"""

from __future__ import annotations

import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e100
_NLN_STRENGTH = float(os.environ.get("WHEST_NLN_STRENGTH", "1.0"))

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
    """K=4 covariance propagation with normal-lognormal marginal moments."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _relu_terms(
        self,
        mu_pre: fnp.ndarray,
        var_pre: fnp.ndarray,
        layer_idx: int,
        width: int,
    ) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
        sigma_pre = fnp.sqrt(fnp.maximum(var_pre, 1e-12))
        a = _NLN_STRENGTH * 0.25 * layer_idx * float(fnp.log(1.0 + 5.0 / width))

        if a <= 0.0:
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)
            mean = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mean * mean, 0.0)
            a1 = sigma_pre * Phi_alpha
            a2 = sigma_pre * phi_alpha
            a3 = -sigma_pre * alpha * phi_alpha
            a4 = sigma_pre * (alpha * alpha - 1.0) * phi_alpha
            return mean, var_post, a1, a2, a3, a4

        sqrt_2a = float(fnp.sqrt(2.0 * a))
        mean = fnp.zeros(width)
        ez2 = fnp.zeros(width)
        a1 = fnp.zeros(width)
        a2 = fnp.zeros(width)
        a3 = fnp.zeros(width)
        a4 = fnp.zeros(width)

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

            a1 = a1 + weight * sigma_scaled * Phi_alpha
            a2 = a2 + weight * sigma_scaled * phi_alpha
            a3 = a3 + weight * (-sigma_scaled * alpha * phi_alpha)
            a4 = a4 + weight * sigma_scaled * (alpha * alpha - 1.0) * phi_alpha

        var_post = fnp.maximum(ez2 - mean * mean, 0.0)
        return mean, var_post, a1, a2, a3, a4

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict post-ReLU means with NLN-corrected K=4 covariance propagation."""
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0
        rows = []

        for layer_idx, w in enumerate(mlp.weights):
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var_np))
                mu = mu / scale
                cov = cov / (scale * scale)
                log_scale += float(fnp.log(scale))

            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            mu, var_post, a1, a2, a3, a4 = self._relu_terms(mu_pre, var_pre, layer_idx, width)

            denom = fnp.outer(sigma_pre, sigma_pre)
            rho = cov_pre / denom
            rho2 = rho * rho
            rho3 = rho2 * rho
            rho4 = rho2 * rho2

            cov = rho * fnp.outer(a1, a1)
            cov = cov + 0.5 * rho2 * fnp.outer(a2, a2)
            cov = cov + (1.0 / 6.0) * rho3 * fnp.outer(a3, a3)
            cov = cov + (1.0 / 24.0) * rho4 * fnp.outer(a4, a4)
            fnp.fill_diagonal(cov, var_post)

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)