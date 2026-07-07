"""Noncompetitive Algorithm 16 probe: diagonal Edgeworth moment closure.

This is a deployable probe inspired by the public higher-moment dataset. It
tracks only per-neuron cumulants up to fourth order, assumes independence across
neurons during linear contractions, and uses an Edgeworth expansion to update
post-ReLU raw moments. Local tests showed this diagonal closure is far from the
active-set Sobol estimator, so keep it as a research note rather than the next
submission path.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


_SQRT_2PI_INV = 0.3989422804014327
_K3_COEF = 1.0 / 6.0
_K4_COEF = 1.0 / 24.0
_K6_COEF = 1.0 / 72.0


def _tail_moments(alpha):
    threshold = -alpha
    phi = flops.stats.norm.pdf(alpha)
    moments = [flops.stats.norm.cdf(alpha), phi]
    for power in range(2, 11):
        moments.append((threshold ** (power - 1)) * phi + fnp.float32(power - 1) * moments[power - 2])
    return moments


def _shifted_poly_integral(alpha, moments, coeffs, moment_power: int):
    out = fnp.zeros_like(alpha)
    for hermite_power, hermite_coeff in coeffs:
        for shift_power in range(moment_power + 1):
            # binomial(moment_power, shift_power) * alpha^(moment_power-shift_power) * x^shift_power
            if moment_power == 0:
                binom = 1.0
            elif moment_power == 1:
                binom = (1.0, 1.0)[shift_power]
            elif moment_power == 2:
                binom = (1.0, 2.0, 1.0)[shift_power]
            elif moment_power == 3:
                binom = (1.0, 3.0, 3.0, 1.0)[shift_power]
            else:
                binom = (1.0, 4.0, 6.0, 4.0, 1.0)[shift_power]
            out = out + fnp.float32(hermite_coeff * binom) * (alpha ** (moment_power - shift_power)) * moments[hermite_power + shift_power]
    return out


def _relu_raw_moments(mu, var, k3, k4):
    var = fnp.maximum(var, 1e-12)
    sigma = fnp.sqrt(var)
    alpha = mu / sigma
    skew = fnp.clip(k3 / fnp.maximum(sigma**3, 1e-12), -6.0, 6.0)
    excess = fnp.clip(k4 / fnp.maximum(var * var, 1e-12), -12.0, 12.0)

    moments = _tail_moments(alpha)
    h0 = ((0, 1.0),)
    h3 = ((3, 1.0), (1, -3.0))
    h4 = ((4, 1.0), (2, -6.0), (0, 3.0))
    h6 = ((6, 1.0), (4, -15.0), (2, 45.0), (0, -15.0))

    raw = []
    for moment_power in range(1, 5):
        base = _shifted_poly_integral(alpha, moments, h0, moment_power)
        corr3 = _shifted_poly_integral(alpha, moments, h3, moment_power)
        corr4 = _shifted_poly_integral(alpha, moments, h4, moment_power)
        corr6 = _shifted_poly_integral(alpha, moments, h6, moment_power)
        standardized = base + fnp.float32(_K3_COEF) * skew * corr3 + fnp.float32(_K4_COEF) * excess * corr4
        standardized = standardized + fnp.float32(_K6_COEF) * skew * skew * corr6
        raw.append(fnp.maximum((sigma ** moment_power) * standardized, 0.0))
    return raw


class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width)
        var = fnp.ones(width)
        k3 = fnp.zeros(width)
        k4 = fnp.zeros(width)
        rows = []

        for w in mlp.weights:
            w2 = w * w
            pre_mu = w.T @ mu
            pre_var = w2.T @ var
            pre_k3 = (w2 * w).T @ k3
            pre_k4 = (w2 * w2).T @ k4

            m1, m2, m3, m4 = _relu_raw_moments(pre_mu, pre_var, pre_k3, pre_k4)
            post_var = fnp.maximum(m2 - m1 * m1, 1e-12)
            post_k3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
            central4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1**4
            post_k4 = central4 - 3.0 * post_var * post_var

            mu = m1
            var = post_var
            k3 = fnp.clip(post_k3, -1e6, 1e6)
            k4 = fnp.clip(post_k4, -1e6, 1e6)
            rows.append(mu)

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)