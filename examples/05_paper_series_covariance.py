"""Paper-series covariance propagation estimator for ReLU MLPs.

This is the same full-covariance propagation structure as
``03_covariance_propagation.py``, but it replaces the starter-kit off-diagonal
"gain" approximation with the fourth-order covariance series from Wright,
Nakahira, and Moura, "An Analytic Solution to Covariance Propagation in Neural
Networks".

For a ReLU nonlinearity, the starter gain update is exactly the K=1 truncation
of the paper's covariance series. This example keeps terms K=1..4:

    cov(ReLU(x_i), ReLU(x_j)) ~= sum_k rho_ij^k / k! * a_ik * a_jk

where the ReLU-specific terms are:

    a1 = sigma * Phi(alpha)
    a2 = sigma * phi(alpha)
    a3 = -sigma * alpha * phi(alpha)
    a4 = sigma * (alpha^2 - 1) * phi(alpha)

The diagonal variance is still set to the exact univariate ReLU variance.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e100


class Estimator(BaseEstimator):
    """Full covariance propagation with the paper's fourth-order ReLU series."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict post-ReLU per-layer means with K=4 covariance propagation."""
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0
        rows = []

        for w in mlp.weights:
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
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            denom = fnp.outer(sigma_pre, sigma_pre)
            rho = cov_pre / denom
            rho2 = rho * rho
            rho3 = rho2 * rho
            rho4 = rho2 * rho2

            a1 = sigma_pre * Phi_alpha
            a2 = sigma_pre * phi_alpha
            a3 = -sigma_pre * alpha * phi_alpha
            a4 = sigma_pre * (alpha * alpha - 1.0) * phi_alpha

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
