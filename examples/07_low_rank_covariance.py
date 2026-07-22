"""Low-rank covariance propagation estimator for ReLU MLPs.

This example explores a cheaper alternative to full covariance propagation. The
state stores covariance as a low-rank factor matrix

    cov ~= factors @ factors.T

instead of a dense covariance matrix. At each layer we:

1. propagate the factors through the linear map,
2. use the resulting diagonal variance to compute ReLU moments,
3. classify and drop analytically dead output neurons,
4. repair the post-ReLU diagonal variance with diagonal factors, and
5. truncate the active covariance factors back to a fixed rank with SVD.

Early layers are not yet low-rank, so this example keeps exact full covariance
until ``_LOW_RANK_START_LAYER`` and only compresses once the spectrum has begun
to concentrate. This is a progressive K1 -> K2 style approximation: mean and
variance decide the active set first, and only the active covariance subspace is
compressed for the next layer.
"""

from __future__ import annotations

import warnings

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_DEAD_ALPHA_THRESHOLD = -3.0
_LOW_RANK = 32
_LOW_RANK_START_LAYER = 24
_MIN_VARIANCE = 1e-12


def _scatter(values: fnp.ndarray, idx: fnp.ndarray, width: int) -> fnp.ndarray:
    """Functionally place values at idx into a zero vector of length width."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        selector = fnp.eye(width, dtype=fnp.float32)[:, idx]
    return selector @ values


def _compress_factors(factors: fnp.ndarray, rank: int) -> fnp.ndarray:
    """Return rank leading singular-vector factors preserving factors @ factors.T."""
    if factors.shape[0] == 0 or factors.shape[1] == 0 or rank <= 0:
        return fnp.zeros((factors.shape[0], 0))
    u, singular, _vh = fnp.linalg.svd(factors, full_matrices=False)
    keep = min(rank, singular.shape[0])
    return u[:, :keep] * singular[:keep]


def _compress_covariance(cov: fnp.ndarray, rank: int) -> fnp.ndarray:
    """Return low-rank factors approximating a positive semidefinite matrix."""
    if cov.shape[0] == 0 or rank <= 0:
        return fnp.zeros((cov.shape[0], 0))
    u, singular, _vh = fnp.linalg.svd(cov, full_matrices=False)
    keep = min(rank, singular.shape[0])
    return u[:, :keep] * fnp.sqrt(fnp.maximum(singular[:keep], 0.0))


class Estimator(BaseEstimator):
    """Low-rank covariance propagation with analytical dead-neuron pruning."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer output means with low-rank covariance factors."""
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width, dtype=fnp.float32)
        factors = None
        active_idx = None

        rows = []
        for layer_idx, w in enumerate(mlp.weights):
            if active_idx is not None and len(active_idx) == 0:
                rows.append(fnp.zeros(width))
                cov = fnp.zeros((0, 0))
                factors = fnp.zeros((0, 0)) if factors is not None else None
                continue

            w_active = w if active_idx is None else w[active_idx, :]
            if factors is None:
                mu_pre = w_active.T @ mu
                cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w_active, w_active)
                var_pre = fnp.maximum(fnp.diag(cov_pre), _MIN_VARIANCE)
                pre_factors = None
            else:
                mu_pre = w_active.T @ mu
                pre_factors = w_active.T @ factors
                var_pre = fnp.maximum(fnp.sum(pre_factors * pre_factors, axis=1), _MIN_VARIANCE)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre

            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            post_mean_full = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            post_var_full = fnp.maximum(ez2 - post_mean_full * post_mean_full, 0.0)

            active_mask = alpha >= _DEAD_ALPHA_THRESHOLD
            next_idx = fnp.nonzero(active_mask)[0]
            rows.append(_scatter(post_mean_full[next_idx], next_idx, width))

            if len(next_idx) == 0:
                mu = fnp.zeros(0)
                cov = fnp.zeros((0, 0))
                factors = fnp.zeros((0, 0)) if factors is not None else None
                active_idx = next_idx
                continue

            gain = Phi_alpha[next_idx]
            if factors is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", flops.SymmetryLossWarning)
                    cov_next_pre = cov_pre[next_idx, :][:, next_idx]
                    cov_next = fnp.multiply(fnp.outer(gain, gain), cov_next_pre)
                fnp.fill_diagonal(cov_next, post_var_full[next_idx])
                cov_next = flops.as_symmetric(cov_next, symmetry=(0, 1))
                if layer_idx + 1 >= _LOW_RANK_START_LAYER:
                    factors = _compress_covariance(cov_next, _LOW_RANK)
                    cov = fnp.zeros((0, 0))
                else:
                    cov = cov_next
            else:
                active_pre_factors = pre_factors[next_idx, :]
                gained_factors = gain[:, None] * active_pre_factors

                gained_diag = fnp.sum(gained_factors * gained_factors, axis=1)
                diag_repair = fnp.sqrt(fnp.maximum(post_var_full[next_idx] - gained_diag, 0.0))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", flops.SymmetryLossWarning)
                    diag_factors = fnp.eye(len(next_idx), dtype=fnp.float32) * diag_repair[:, None]
                factors = _compress_factors(fnp.concatenate([gained_factors, diag_factors], axis=1), _LOW_RANK)

            mu = post_mean_full[next_idx]
            active_idx = next_idx

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)