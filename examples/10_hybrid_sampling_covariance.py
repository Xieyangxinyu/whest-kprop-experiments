"""Hybrid Monte Carlo + covariance propagation estimator.

Direct sampling is unbiased but noisy. K=4 covariance propagation is biased but
cheap and low-variance. This example uses covariance propagation as a control
variate / shrinkage target:

    prediction = analytic + beta * (sample_mean - analytic)
               = (1 - beta) * analytic + beta * sample_mean

``beta=1`` recovers direct Monte Carlo. ``beta=0`` recovers the analytic K=4
covariance example. Intermediate values reduce sampling variance while keeping
some sampling correction. The normal-lognormal/heavy-tail diagnostics motivate
this direction: deep activations have noisy tails, so spend samples where they
help but shrink them toward a structured prior prediction.

Environment knobs:

* ``WHEST_HYBRID_SAMPLES``: number of Gaussian input samples, default 4096.
* ``WHEST_HYBRID_BETA``: MC weight in the convex blend, default 0.9.
* ``WHEST_HYBRID_ANTITHETIC``: use x/-x input pairs, default 1.
"""

from __future__ import annotations

import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e100
_DEFAULT_SAMPLES = int(os.environ.get("WHEST_HYBRID_SAMPLES", "4096"))
_DEFAULT_BETA = float(os.environ.get("WHEST_HYBRID_BETA", "0.9"))
_USE_ANTITHETIC = os.environ.get("WHEST_HYBRID_ANTITHETIC", "1") != "0"


class Estimator(BaseEstimator):
    """Blend K=4 covariance propagation with direct Gaussian-input sampling."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _analytic_prediction(self, mlp: MLP) -> fnp.ndarray:
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

    def _sample_prediction(self, mlp: MLP) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width
        if _USE_ANTITHETIC:
            n_pairs = max(1, (_DEFAULT_SAMPLES + 1) // 2)
            x_half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
            x = fnp.concatenate([x_half, -x_half], axis=0)
        else:
            x = fnp.array(rng.standard_normal((_DEFAULT_SAMPLES, width)).astype(fnp.float32))
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(x @ w, 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Return a convex blend of analytic and sampled activation means."""
        _ = budget
        analytic = self._analytic_prediction(mlp)
        sampled = self._sample_prediction(mlp)
        beta = _DEFAULT_BETA
        return analytic + beta * (sampled - analytic)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)