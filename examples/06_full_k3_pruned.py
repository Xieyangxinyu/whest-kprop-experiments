"""Factorized K3 propagation estimator for ReLU MLPs.

This is a compact flopscope.numpy port of the upstream ``k_max=3`` SIMPLE
``use_pK=True`` path from ARC's ``mlp_cumulant_propagation`` package. It is more
faithful than the marginal/dense K3 probes because it carries:

* full covariance,
* all-distinct third cumulants in factored form, and
* repeated K3 diagonal slices ``(3)`` and ``(2, 1)`` via the upstream pK-to-K
    conversion formulas, and
* the fourth-order radial state that feeds the next K3 update through its
    ``(2, 2)`` slice.

The implementation still uses the WHest active-set idea: after K1/K2 determine
which output neurons are analytically dead, the returned row zeros those neurons
and the next layer carries only the active subspace.
"""

from __future__ import annotations

import math
import warnings

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_DEAD_ALPHA_THRESHOLD = -3.0
_MIN_VARIANCE = 1e-12
_SQRT_TWO_PI_INV = 0.3989422804014327

_BASE_TERMS = (
    ((1,), (), 1),
    ((1,), ((3,),), 1),
    ((1,), ((3,), (3,)), 1),
    ((1, 1), ((0, 3), (1, 1)), 2),
    ((1, 1), ((0, 3), (1, 2)), 2),
    ((1, 1), ((0, 3), (2, 1)), 2),
    ((1, 1), ((1, 1),), 1),
    ((1, 1), ((1, 1), (1, 1)), 1),
    ((1, 1), ((1, 1), (1, 2)), 2),
    ((1, 1), ((1, 2),), 2),
    ((1, 1), ((1, 2), (1, 2)), 2),
    ((1, 1), ((1, 2), (2, 1)), 1),
    ((2,), (), 1),
    ((2,), ((3,),), 1),
    ((2,), ((3,), (3,)), 1),
    ((2, 1), ((0, 3), (1, 1)), 1),
    ((2, 1), ((0, 3), (1, 2)), 1),
    ((2, 1), ((0, 3), (2, 1)), 1),
    ((2, 1), ((1, 1),), 1),
    ((2, 1), ((1, 1), (1, 1)), 1),
    ((2, 1), ((1, 1), (1, 2)), 1),
    ((2, 1), ((1, 1), (2, 1)), 1),
    ((2, 1), ((1, 1), (3, 0)), 1),
    ((2, 1), ((1, 2),), 1),
    ((2, 1), ((1, 2), (1, 2)), 1),
    ((2, 1), ((1, 2), (2, 1)), 1),
    ((2, 1), ((1, 2), (3, 0)), 1),
    ((2, 1), ((2, 1),), 1),
    ((2, 1), ((2, 1), (2, 1)), 1),
    ((2, 1), ((2, 1), (3, 0)), 1),
    ((3,), (), 1),
    ((3,), ((3,),), 1),
    ((3,), ((3,), (3,)), 1),
)

_FOURTH_TERMS = (
    ((2, 2), ((0, 3), (1, 1)), 2),
    ((2, 2), ((0, 3), (1, 2)), 2),
    ((2, 2), ((0, 3), (2, 1)), 2),
    ((2, 2), ((0, 3), (2, 2)), 2),
    ((2, 2), ((0, 4), (1, 1)), 2),
    ((2, 2), ((0, 4), (1, 2)), 2),
    ((2, 2), ((0, 4), (2, 1)), 2),
    ((2, 2), ((0, 4), (2, 2)), 2),
    ((2, 2), ((1, 1),), 1),
    ((2, 2), ((1, 1), (1, 1)), 1),
    ((2, 2), ((1, 1), (1, 2)), 2),
    ((2, 2), ((1, 1), (2, 2)), 1),
    ((2, 2), ((1, 2),), 2),
    ((2, 2), ((1, 2), (1, 2)), 2),
    ((2, 2), ((1, 2), (2, 1)), 1),
    ((2, 2), ((1, 2), (2, 2)), 2),
    ((2, 2), ((1, 3),), 2),
    ((2, 2), ((2, 2),), 1),
    ((2, 2), ((2, 2), (2, 2)), 1),
    ((4,), (), 1),
    ((4,), ((3,),), 1),
    ((4,), ((3,), (3,)), 1),
    ((4,), ((3,), (4,)), 1),
    ((4,), ((4,),), 1),
    ((4,), ((4,), (4,)), 1),
)


def _zero_diag(matrix: fnp.ndarray) -> fnp.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        return matrix * (1.0 - fnp.eye(matrix.shape[0], dtype=fnp.float32))


def _scatter(values: fnp.ndarray, idx: fnp.ndarray, width: int) -> fnp.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        selector = fnp.eye(width, dtype=fnp.float32)[:, idx]
    return selector @ values


def _expand(array: fnp.ndarray, positions: tuple[int, ...], ndim: int) -> fnp.ndarray:
    shape = [1] * ndim
    for axis, position in enumerate(positions):
        shape[position] = array.shape[axis]
    return fnp.reshape(array, tuple(shape))


def _vec_part_coef(vec_part: tuple[tuple[int, ...], ...]) -> float:
    if len(vec_part) == 0:
        return 1.0
    total = 1.0
    for vector in set(vec_part):
        count = vec_part.count(vector)
        total *= math.factorial(count)
        for entry in vector:
            total *= math.factorial(entry) ** count
    return 1.0 / total


def _raw_gaussian_relu_moment(mean: fnp.ndarray, variance: fnp.ndarray, power: int) -> fnp.ndarray:
    variance = fnp.maximum(variance, _MIN_VARIANCE)
    sigma = fnp.sqrt(variance)
    alpha = mean / sigma
    threshold = -alpha
    phi = fnp.exp(-0.5 * threshold * threshold) * _SQRT_TWO_PI_INV
    moments = [flops.stats.norm.cdf(alpha), phi]
    for current_power in range(2, power + 1):
        moments.append((threshold ** (current_power - 1)) * phi + (current_power - 1) * moments[current_power - 2])

    total = moments[0] * 0.0
    for x_power in range(power + 1):
        coef = math.comb(power, x_power)
        alpha_power = power - x_power
        alpha_factor = 1.0 if alpha_power == 0 else alpha ** alpha_power
        total = total + coef * alpha_factor * moments[x_power]
    return (sigma ** power) * total


def _hermite(order: int, alpha: fnp.ndarray) -> fnp.ndarray:
    if order == 0:
        return fnp.ones(alpha.shape)
    if order == 1:
        return alpha
    if order == 2:
        return alpha * alpha - 1.0
    if order == 3:
        return alpha * alpha * alpha - 3.0 * alpha
    if order == 4:
        alpha2 = alpha * alpha
        return alpha2 * alpha2 - 6.0 * alpha2 + 3.0
    raise ValueError(f"unsupported Hermite order {order}")


def _relu_wick(mean: fnp.ndarray, variance: fnp.ndarray, deriv_order: int, power: int) -> fnp.ndarray:
    variance = fnp.maximum(variance, _MIN_VARIANCE)
    sigma = fnp.sqrt(variance)
    alpha = mean / sigma
    phi = fnp.exp(-0.5 * alpha * alpha) * _SQRT_TWO_PI_INV
    if deriv_order < power:
        falling = math.prod(range(power - deriv_order + 1, power + 1))
        return falling * _raw_gaussian_relu_moment(mean, variance, power - deriv_order)
    if power > 1:
        return math.factorial(power) * _relu_wick(mean, variance, deriv_order - power + 1, 1)
    if deriv_order == 0:
        return sigma * phi + mean * flops.stats.norm.cdf(alpha)
    if deriv_order == 1:
        return flops.stats.norm.cdf(alpha)
    return ((-1.0) ** (deriv_order - 2)) * (sigma ** (-(deriv_order - 1))) * _hermite(deriv_order - 2, alpha) * phi


class _K3State:
    def __init__(self, a: fnp.ndarray, b: fnp.ndarray, c: fnp.ndarray) -> None:
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def empty(cls, width: int) -> "_K3State":
        empty = fnp.zeros((width, 0))
        return cls(empty, empty, empty)

    def transform(self, weights: fnp.ndarray) -> "_K3State":
        return _K3State(weights.T @ self.a, weights.T @ self.b, weights.T @ self.c)

    def take(self, idx: fnp.ndarray) -> "_K3State":
        return _K3State(self.a[idx], self.b[idx], self.c[idx])

    def scale_axes(self, values: fnp.ndarray) -> "_K3State":
        return _K3State(values[:, None] * self.a, values[:, None] * self.b, values[:, None] * self.c)

    def add(self, a: fnp.ndarray, b: fnp.ndarray, c: fnp.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            self.a = fnp.concatenate([self.a, a], axis=1)
            self.b = fnp.concatenate([self.b, b], axis=1)
            self.c = fnp.concatenate([self.c, c], axis=1)

    def slice3(self) -> fnp.ndarray:
        if self.a.shape[1] == 0:
            return fnp.zeros(self.a.shape[0])
        return fnp.sum(self.a * self.b * self.c, axis=1)

    def slice21(self) -> fnp.ndarray:
        if self.a.shape[1] == 0:
            return fnp.zeros((self.a.shape[0], self.a.shape[0]))
        part = (self.a * self.b) @ self.c.T
        part = part + (self.a * self.c) @ self.b.T
        part = part + (self.b * self.c) @ self.a.T
        return _zero_diag(part / 3.0)


class _K4RadialState:
    def __init__(self, core: float | fnp.ndarray, metric: fnp.ndarray) -> None:
        self.core = core
        self.metric = metric

    @classmethod
    def zero(cls, width: int) -> "_K4RadialState":
        return cls(0.0, fnp.eye(width, dtype=fnp.float32))

    def transform(self, weights: fnp.ndarray) -> "_K4RadialState":
        return _K4RadialState(self.core, weights.T @ self.metric @ weights)

    def take(self, idx: fnp.ndarray) -> "_K4RadialState":
        return _K4RadialState(self.core, fnp.eye(len(idx), dtype=fnp.float32))

    def slice4(self) -> fnp.ndarray:
        diagonal = fnp.diag(self.metric)
        return self.core * diagonal * diagonal

    def slice22(self) -> fnp.ndarray:
        diagonal = fnp.diag(self.metric)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            result = self.core * (diagonal[:, None] * diagonal[None, :] / 3.0 + 2.0 * self.metric * self.metric / 3.0)
        return _zero_diag(result)

    def slice31(self) -> fnp.ndarray:
        diagonal = fnp.diag(self.metric)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            result = self.core * diagonal[:, None] * self.metric
        return _zero_diag(result)


def _from_repeated(slice3: fnp.ndarray, slice21: fnp.ndarray) -> _K3State:
    width = len(slice3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        eye = fnp.eye(width, dtype=fnp.float32)
        a = slice3[:, None] * eye + 3.0 * slice21.T
    return _K3State(a, eye, eye)


def _get_slice(
    mean: fnp.ndarray,
    cov: fnp.ndarray,
    k3: _K3State,
    k4: _K4RadialState,
    part: tuple[int, ...],
) -> fnp.ndarray:
    sorted_part = tuple(sorted(part, reverse=True))
    if sorted_part == (1,):
        return mean
    if sorted_part == (2,):
        return fnp.diag(cov)
    if sorted_part == (1, 1):
        return _zero_diag(cov)
    if sorted_part == (3,):
        return k3.slice3()
    if sorted_part == (2, 1):
        slice21 = k3.slice21()
        return slice21 if part == (2, 1) else slice21.T
    if sorted_part == (4,):
        return k4.slice4()
    if sorted_part == (3, 1):
        slice31 = k4.slice31()
        return slice31 if part == (3, 1) else slice31.T
    if sorted_part == (2, 2):
        return k4.slice22()
    raise ValueError(f"unsupported slice {part}")


def _eval_vec_part(
    mean: fnp.ndarray,
    cov: fnp.ndarray,
    k3: _K3State,
    k4: _K4RadialState,
    vec_part: tuple[tuple[int, ...], ...],
    ndim: int,
) -> fnp.ndarray:
    if len(vec_part) == 0:
        return fnp.ones((mean.shape[0],) * ndim)
    pieces = []
    for vector in vec_part:
        positions = tuple(axis for axis, entry in enumerate(vector) if entry > 0)
        slice_part = tuple(entry for entry in vector if entry > 0)
        pieces.append(_expand(_get_slice(mean, cov, k3, k4, slice_part), positions, ndim))
    result = pieces[0]
    for piece in pieces[1:]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            result = result * piece
    return _vec_part_coef(vec_part) * result


def _multiply_wicks(
    term: fnp.ndarray,
    mean: fnp.ndarray,
    variance: fnp.ndarray,
    deriv_orders: tuple[int, ...],
    powers: tuple[int, ...],
) -> fnp.ndarray:
    result = term
    ndim = len(deriv_orders)
    for axis, (deriv_order, power) in enumerate(zip(deriv_orders, powers)):
        wick = _relu_wick(mean, variance, deriv_order, power)
        shape = [1] * ndim
        shape[axis] = -1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            result = result * fnp.reshape(wick, tuple(shape))
    return result


def _pk_slices(
    mean: fnp.ndarray,
    variance: fnp.ndarray,
    cov: fnp.ndarray,
    k3: _K3State,
    k4: _K4RadialState,
) -> dict:
    width = mean.shape[0]
    out = {
        (1,): fnp.zeros(width),
        (2,): fnp.zeros(width),
        (1, 1): fnp.zeros((width, width)),
        (3,): fnp.zeros(width),
        (2, 1): fnp.zeros((width, width)),
        (4,): fnp.zeros(width),
        (2, 2): fnp.zeros((width, width)),
    }
    for int_part, vec_part, count in _BASE_TERMS + _FOURTH_TERMS:
        ndim = len(int_part)
        term = _eval_vec_part(mean, cov, k3, k4, vec_part, ndim)
        deriv_orders = tuple(sum(vector[axis] for vector in vec_part) for axis in range(ndim))
        contribution = count * _multiply_wicks(term, mean, variance, deriv_orders, int_part)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            out[int_part] = out[int_part] + contribution
    out[(1, 1)] = _zero_diag((out[(1, 1)] + out[(1, 1)].T) * 0.5)
    out[(2, 1)] = _zero_diag(out[(2, 1)])
    out[(2, 2)] = _zero_diag((out[(2, 2)] + out[(2, 2)].T) * 0.5)
    return out


def _k4_from_pk(pk: dict, next_mean: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
    pk1 = pk[(1,)]
    pk2 = pk[(2,)]
    pk3 = pk[(3,)]
    pk4 = pk[(4,)]
    pk11 = pk[(1, 1)]
    pk21 = pk[(2, 1)]
    pk12 = pk21.T
    pk22 = pk[(2, 2)]

    k4_4 = pk4 - 4.0 * pk1 * pk3 - 3.0 * pk2 * pk2 + 12.0 * pk1 * pk1 * pk2 - 6.0 * pk1 * pk1 * pk1 * pk1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        k4_22 = pk22 - 2.0 * pk1[None, :] * pk21 - 2.0 * pk1[:, None] * pk12
        k4_22 = k4_22 - 2.0 * pk11 * pk11 + 4.0 * pk1[:, None] * pk1[None, :] * pk11
    return k4_4, _zero_diag(k4_22)


def _project_k4_radial(k4_4: fnp.ndarray, k4_22: fnp.ndarray) -> _K4RadialState:
    width = len(k4_4)
    core = (3.0 / (width * (width + 2.0))) * (fnp.sum(k4_4) + fnp.sum(k4_22))
    return _K4RadialState(core, fnp.eye(width, dtype=fnp.float32))


def _nonlinear_simple_k3(
    mean: fnp.ndarray,
    cov: fnp.ndarray,
    k3: _K3State,
    k4: _K4RadialState,
) -> tuple[fnp.ndarray, fnp.ndarray, _K3State, _K4RadialState]:
    variance = fnp.maximum(fnp.diag(cov), _MIN_VARIANCE)
    pk = _pk_slices(mean, variance, cov, k3, k4)

    next_mean = pk[(1,)]
    next_var = fnp.maximum(pk[(2,)] - next_mean * next_mean, 0.0)
    next_cov = pk[(1, 1)]
    fnp.fill_diagonal(next_cov, next_var)
    next_cov = flops.as_symmetric(next_cov, symmetry=(0, 1))

    next_k3_diag = pk[(3,)] - 3.0 * pk[(2,)] * next_mean + 2.0 * next_mean * next_mean * next_mean
    next_k3_21 = pk[(2, 1)] - 2.0 * pk[(1, 1)] * next_mean[:, None]
    next_k3_21 = _zero_diag(next_k3_21)

    gain = _relu_wick(mean, variance, 1, 1)
    wick2 = _relu_wick(mean, variance, 2, 1)
    wick3 = _relu_wick(mean, variance, 3, 1)
    wick4 = _relu_wick(mean, variance, 4, 1)
    wk11 = _zero_diag(cov)
    wk21 = k3.slice21() / 2.0
    wk12 = wk21.T
    wk22 = k4.slice22() / 4.0

    pk111 = k3.scale_axes(gain)
    eye = fnp.eye(mean.shape[0], dtype=fnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        fac1 = gain[:, None] * wk11 + wick2[:, None] * wk21
        fac2 = 3.0 * eye
        fac3 = (
            wick2[:, None] * gain[None, :] * wk11
            + wick3[:, None] * gain[None, :] * wk21
            + wick2[:, None] * wick2[None, :] * wk12
            + wick3[:, None] * wick2[None, :] * wk22
        ).T
    pk111.add(fac1, fac2, fac3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        fac1 = gain[:, None] * wk12 + wick2[:, None] * wk22
        fac2 = 3.0 * eye
        fac3 = (
            wick3[:, None] * gain[None, :] * wk11
            + wick4[:, None] * gain[None, :] * wk21
            + wick3[:, None] * wick2[None, :] * wk12
            + wick4[:, None] * wick2[None, :] * wk22
        ).T
    pk111.add(fac1, fac2, fac3)

    metric = k4.metric
    metric_diag = fnp.diag(metric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        ones = fnp.ones(metric_diag.shape)
        fac1 = wick2[:, None] * (k4.core * metric_diag)[:, None] * ones[None, :]
        fac2 = gain[:, None] * eye
        fac3 = 0.5 * gain[:, None] * metric
    pk111.add(fac1, fac2, fac3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        fac1 = gain[:, None] * metric * k4.core
        fac2 = wick2[:, None] * eye
        fac3 = gain[:, None] * metric
    pk111.add(fac1, fac2, fac3)

    residual3 = next_k3_diag - pk111.slice3()
    residual21 = next_k3_21 - pk111.slice21()
    repeated = _from_repeated(residual3, residual21)
    pk111.add(repeated.a, repeated.b, repeated.c)
    k4_4, k4_22 = _k4_from_pk(pk, next_mean)
    next_k4 = _project_k4_radial(k4_4, k4_22)
    return next_mean, next_cov, pk111, next_k4


class Estimator(BaseEstimator):
    """SIMPLE k-prop with factored K3, radial K4, and active-set pruning."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        _ = budget
        width = mlp.width

        mean = fnp.zeros(width)
        cov = fnp.eye(width, dtype=fnp.float32)
        k3 = _K3State.empty(width)
        k4 = _K4RadialState.zero(width)
        active_idx = None
        rows = []

        for weights in mlp.weights:
            if active_idx is not None and len(active_idx) == 0:
                rows.append(fnp.zeros(width))
                continue
            active_weights = weights if active_idx is None else weights[active_idx, :]
            mean_pre = active_weights.T @ mean
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, active_weights, active_weights)
            k3_pre = k3.transform(active_weights)
            k4_pre = k4.transform(active_weights)

            post_mean, post_cov, post_k3, post_k4 = _nonlinear_simple_k3(mean_pre, cov_pre, k3_pre, k4_pre)
            alpha = mean_pre / fnp.sqrt(fnp.maximum(fnp.diag(cov_pre), _MIN_VARIANCE))
            active_mask = alpha >= _DEAD_ALPHA_THRESHOLD
            next_idx = fnp.nonzero(active_mask)[0]
            rows.append(fnp.where(active_mask, post_mean, 0.0))

            if len(next_idx) == 0:
                mean = fnp.zeros(0)
                cov = fnp.zeros((0, 0))
                k3 = _K3State.empty(0)
                k4 = _K4RadialState.zero(0)
            else:
                mean = post_mean[next_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", flops.SymmetryLossWarning)
                    cov = post_cov[next_idx, :][:, next_idx]
                cov = flops.as_symmetric(cov, symmetry=(0, 1))
                k3 = post_k3.take(next_idx)
                k4 = post_k4.take(next_idx)
            active_idx = next_idx

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=8, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=1_500_000_000_000)