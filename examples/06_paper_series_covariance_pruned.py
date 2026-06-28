"""Paper-series covariance propagation with active-set dead-neuron pruning.

This extends ``05_paper_series_covariance.py`` with a simple dead-neuron rule:
after each ReLU moment update, units whose pre-activation z-score
``alpha = mu_pre / sigma_pre`` is below -2 are treated as dead. Their predicted
mean is set to zero and only the surviving active covariance submatrix is carried
into the next layer.

Unlike merely zeroing rows and columns in a dense 256 x 256 covariance matrix,
this active-set form can reduce later covariance-update FLOPs because the next
linear covariance propagation uses a ``k x k`` covariance and a ``k x width``
weight slice, where ``k`` is the number of active units from the previous layer.
"""

from __future__ import annotations

import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e100
_ALPHA_PRUNE_THRESHOLD = float(os.environ.get("WHEST_ALPHA_PRUNE_THRESHOLD", "-2.0"))


class Estimator(BaseEstimator):
    """K=4 covariance propagation with alpha-based active-set pruning."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict post-ReLU per-layer means with active covariance pruning."""
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        active_idx = None
        active_count = width
        log_scale = 0.0
        rows = []

        for w in mlp.weights:
            if active_count == 0:
                rows.append(fnp.zeros(width))
                continue

            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var_np))
                mu = mu / scale
                cov = cov / (scale * scale)
                log_scale += float(fnp.log(scale))

            if active_idx is None:
                w_active = w
            else:
                w_active = w[active_idx, :]

            mu_pre = w_active.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w_active, w_active)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            mu_post = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)

            denom = flops.as_symmetric(fnp.outer(sigma_pre, sigma_pre), symmetry=(0, 1))
            rho = flops.as_symmetric(cov_pre / denom, symmetry=(0, 1))
            rho2 = rho * rho
            rho3 = rho2 * rho
            rho4 = rho2 * rho2

            a1 = sigma_pre * Phi_alpha
            a2 = sigma_pre * phi_alpha
            a3 = -sigma_pre * alpha * phi_alpha
            a4 = sigma_pre * (alpha * alpha - 1.0) * phi_alpha

            cov_full = rho * fnp.outer(a1, a1)
            cov_full = cov_full + 0.5 * rho2 * fnp.outer(a2, a2)
            cov_full = cov_full + (1.0 / 6.0) * rho3 * fnp.outer(a3, a3)
            cov_full = cov_full + (1.0 / 24.0) * rho4 * fnp.outer(a4, a4)
            cov_full = flops.as_symmetric(cov_full, symmetry=(0, 1))
            fnp.fill_diagonal(cov_full, var_post)

            keep = alpha >= _ALPHA_PRUNE_THRESHOLD
            active_idx = fnp.nonzero(keep)[0]
            active_count = int(fnp.sum(keep))

            mu_full = fnp.where(keep, mu_post, 0.0)
            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu_full * scale_factor)

            if active_count == 0:
                mu = fnp.zeros(0)
                cov = fnp.zeros((0, 0))
            else:
                mu = mu_full[active_idx]
                cov = flops.as_symmetric(cov_full[active_idx][:, active_idx], symmetry=(0, 1))

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)
