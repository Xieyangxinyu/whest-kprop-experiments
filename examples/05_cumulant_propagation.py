"""Marginal cumulant propagation estimator for ReLU MLPs.

This is a compact flopscope.numpy adaptation of the cumulant-propagation idea
from ARC's ``mlp_cumulant_propagation`` repo. The full upstream algorithm tracks
structured high-order cumulant tensors. For a starter-kit example, we keep the
same ReLU/Wick-coefficient intuition but store only:

* the mean vector,
* the full covariance matrix,
* per-neuron third cumulants, and
* per-neuron fourth cumulants.

At each layer we propagate the linear step, then estimate ReLU raw moments with
a truncated Edgeworth/Wick expansion around the Gaussian with matching mean and
variance. The returned prediction is still the post-ReLU mean for every layer,
with shape ``(depth, width)``.

This is intentionally educational rather than a full port of harmonic k-prop:
cross-neuron third/fourth cumulants are dropped, while covariance is kept full.
"""

from __future__ import annotations

import math
import warnings

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e100
_DEAD_ALPHA_THRESHOLD = -3.0
_MIN_VARIANCE = 1e-12
_SQRT_TWO_PI_INV = 0.3989422804014327

# Probabilists' Hermite polynomials, represented as {power: coefficient}.
_HERMITE = {
    0: {0: 1.0},
    3: {3: 1.0, 1: -3.0},
    4: {4: 1.0, 2: -6.0, 0: 3.0},
    6: {6: 1.0, 4: -15.0, 2: 45.0, 0: -15.0},
}


def _tail_moments(alpha: fnp.ndarray, max_power: int) -> list[fnp.ndarray]:
    """Return I_k = integral_{-alpha}^inf x^k phi(x) dx for k <= max_power."""
    threshold = -alpha
    phi = fnp.exp(-0.5 * threshold * threshold) * _SQRT_TWO_PI_INV
    moments = [flops.stats.norm.cdf(alpha), phi]
    for power in range(2, max_power + 1):
        moments.append((threshold ** (power - 1)) * phi + (power - 1) * moments[power - 2])
    return moments


def _shifted_relu_hermite_integral(
    alpha: fnp.ndarray,
    power: int,
    hermite_order: int,
    tail_moments: list[fnp.ndarray],
) -> fnp.ndarray:
    """Compute integral (alpha+x)^power He_order(x) phi(x) over x > -alpha."""
    total = tail_moments[0] * 0.0
    hermite = _HERMITE[hermite_order]
    for shifted_power in range(power + 1):
        shifted_coef = math.comb(power, shifted_power)
        alpha_power = power - shifted_power
        alpha_factor = 1.0 if alpha_power == 0 else alpha ** alpha_power
        for hermite_power, hermite_coef in hermite.items():
            total = total + shifted_coef * hermite_coef * alpha_factor * tail_moments[shifted_power + hermite_power]
    return total


def _relu_raw_moments(
    mean: fnp.ndarray,
    variance: fnp.ndarray,
    k3: fnp.ndarray,
    k4: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Approximate E[ReLU(Z)^p] for p=1..4 using marginal cumulants of Z."""
    variance = fnp.maximum(variance, _MIN_VARIANCE)
    sigma = fnp.sqrt(variance)
    alpha = fnp.clip(mean / sigma, -10.0, 10.0)
    tail_moments = _tail_moments(alpha, max_power=10)

    skew = fnp.clip(k3 / fnp.maximum(sigma * sigma * sigma, _MIN_VARIANCE), -8.0, 8.0)
    excess = fnp.clip(k4 / fnp.maximum(variance * variance, _MIN_VARIANCE), -16.0, 16.0)

    raw_moments = []
    for power in range(1, 5):
        gaussian = _shifted_relu_hermite_integral(alpha, power, 0, tail_moments)
        correction = skew * _shifted_relu_hermite_integral(alpha, power, 3, tail_moments) / 6.0
        correction = correction + excess * _shifted_relu_hermite_integral(alpha, power, 4, tail_moments) / 24.0
        correction = correction + skew * skew * _shifted_relu_hermite_integral(alpha, power, 6, tail_moments) / 72.0
        raw = (sigma ** power) * (gaussian + correction)
        raw_moments.append(fnp.maximum(raw, 0.0))
    return raw_moments[0], raw_moments[1], raw_moments[2], raw_moments[3]


def _relu_cumulants(
    mean: fnp.ndarray,
    variance: fnp.ndarray,
    k3: fnp.ndarray,
    k4: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Return post-ReLU mean, variance, third cumulant, and fourth cumulant."""
    raw1, raw2, raw3, raw4 = _relu_raw_moments(mean, variance, k3, k4)

    post_mean = raw1
    post_variance = fnp.maximum(raw2 - raw1 * raw1, 0.0)
    post_k3 = raw3 - 3.0 * raw1 * raw2 + 2.0 * raw1 * raw1 * raw1
    central4 = raw4 - 4.0 * raw1 * raw3 + 6.0 * raw1 * raw1 * raw2 - 3.0 * raw1 * raw1 * raw1 * raw1
    post_k4 = central4 - 3.0 * post_variance * post_variance

    variance_floor = fnp.maximum(post_variance, _MIN_VARIANCE)
    std_floor = fnp.sqrt(variance_floor)
    post_k3 = fnp.clip(post_k3, -8.0 * variance_floor * std_floor, 8.0 * variance_floor * std_floor)
    post_k4 = fnp.clip(
        post_k4,
        -16.0 * variance_floor * variance_floor,
        16.0 * variance_floor * variance_floor,
    )
    return post_mean, post_variance, post_k3, post_k4


class Estimator(BaseEstimator):
    """Full covariance propagation plus marginal K3/K4 ReLU corrections."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Predict per-layer output means with marginal cumulant propagation."""
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        k3 = fnp.zeros(width)
        k4 = fnp.zeros(width)
        active_idx = None
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            if active_idx is not None and len(active_idx) == 0:
                rows.append(fnp.zeros(width))
                continue

            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var_np))
                mu = mu / scale
                cov = cov / (scale * scale)
                k3 = k3 / (scale * scale * scale)
                k4 = k4 / (scale * scale * scale * scale)
                log_scale += float(fnp.log(scale))

            w_active = w if active_idx is None else w[active_idx, :]

            mu_pre = w_active.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w_active, w_active)
            var_pre = fnp.maximum(fnp.diag(cov_pre), _MIN_VARIANCE)

            k3_pre = (w_active * w_active * w_active).T @ k3
            w2 = w_active * w_active
            k4_pre = (w2 * w2).T @ k4

            alpha = mu_pre / fnp.sqrt(var_pre)
            post_mean, post_variance, post_k3, post_k4 = _relu_cumulants(mu_pre, var_pre, k3_pre, k4_pre)

            gain = flops.stats.norm.cdf(alpha)
            cov_full = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov_full, post_variance)

            active_mask = alpha >= _DEAD_ALPHA_THRESHOLD
            active_idx = fnp.nonzero(active_mask)[0]

            scale_factor = float(fnp.exp(log_scale))
            row = fnp.where(active_mask, post_mean * scale_factor, 0.0)
            rows.append(row)

            mu = post_mean[active_idx]
            k3 = post_k3[active_idx]
            k4 = post_k4[active_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", flops.SymmetryLossWarning)
                cov_active = cov_full[active_idx, :][:, active_idx]
            cov = flops.as_symmetric(cov_active, symmetry=(0, 1))

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)