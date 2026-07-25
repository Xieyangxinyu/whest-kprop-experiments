"""Fixed 61,440-sample estimator: antithetic pilots, 2-level Strassen, recursive cold slicing.

This estimator uses a constant 61,440 antithetic samples (10,240 pilot + 51,200
continuation) for every MLP. Dead/active/on classification probes estimate alpha
from both antithetic halves (pair-balanced). Sampled active propagation uses a
guarded 2-level recursive Strassen decomposition. Cold-column slicing runs in
both blocks:

- Continuation block: the pilot fire census orders each layer's columns
  coldest-first; rows are scatter-ordered by their support PATTERN over an
  ultra-cold / warm-cold split of the cold set, so each pattern group's rows
  and its needed column slab are contiguous and every correction lands as a
  plain slice-add (two-level carve from one scatter).
- Pilot block: columns are ordered by the analytic alpha of the producing
  layer (a fire-rate predictor available before materialization) and the same
  single-level carve applies, with the row order RESTORED after each layer via
  an inverse-permutation scatter so the Sobol-prefix probe rows and the
  antithetic pairing are untouched.

Weights are cast to float32 once per layer. The final row is sampled
accurately, while intermediate returned rows are analytical fillers.
"""

from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_BASE_SAMPLES = 10240
_TOTAL_SAMPLES = 61440
_DEAD_THRESH = -3.0
_ON_THRESH = 3.0
_PILOT_FRACTION = 0.05
_PILOT_RECHECK_FRACTION = 0.20
_PILOT_RECHECK_MARGIN = 0.35
_PILOT_ON_THRESH = 3.0
_PILOT_DEAD_THRESH = -2.5
_ON_PROBE_MAX = 4.0
_DEAD_PROBE_MIN = -4.0
_REFINE_DEAD_START_LAYER = 1
_REFINE_DEAD_STOP_LAYER = 29
_DEMOTE_ACTIVE_DEAD_START_LAYER = 1
_DEMOTE_ACTIVE_DEAD_STOP_LAYER = 29
_ACTIVE_DEAD_PROBE_MAX = -2.5
_ACTIVE_DEAD_THRESH = -3.0
_MATERIALIZE_INTERMEDIATE_ROWS = False
_CAST_WEIGHTS_F32 = True
_COLD_SLICE = True
_COLD_START_LAYER = 2   # first layer whose input matmul may be cold-sliced
_COLD_STOP_LAYER = 29   # fold layers 30/31 are never sliced
_COLD_FIRE_THRESH = 0.03  # columns firing below this pilot rate are cold
_COLD_MIN_K = 8
_COLD_MAX_K = 64
_COLD_MIN_HOT_DIM = 96  # keep the hot block Strassen-eligible
_COLD_REORDER = "put"   # 'put' = scatter reorder; 'take' = gather fallback
_COLD_FIRE_THRESH2 = 0.01  # inner (ultra-cold) recursion cut
_COLD_MIN_K2 = 4
_PILOT_COLD = True
_PILOT_ALPHA_COLD = -1.88  # Phi(alpha) ~= 0.03: analytic proxy for the fire threshold
_PILOT_MIN_K = 8
_PILOT_MAX_K = 64


def _scatter(values, idx, width):
    """Functionally place values at idx into a zero vector of length width."""
    return fnp.eye(width, dtype=fnp.float32)[:, idx] @ values


def _probe_rows(n_samples: int, fraction: float) -> int:
    return max(2, min(n_samples, int(n_samples * fraction)))


def _sample_alpha(x, weights, rows: int, n_samples: int = None):
    if n_samples is not None:
        half = n_samples // 2
        pre = fnp.concatenate([x[:rows, :] @ weights, x[half:half + rows, :] @ weights], axis=0)
    else:
        pre = x[:rows, :] @ weights
    mean = fnp.mean(pre, axis=0)
    var = fnp.var(pre, axis=0)
    return mean / fnp.sqrt(fnp.maximum(var, 1e-12))


def _staged_threshold_split(source_idx, x, weights, n_samples: int, threshold: float):
    primary_rows = _probe_rows(n_samples, _PILOT_FRACTION)
    primary_alpha = _sample_alpha(x, weights, primary_rows, n_samples)
    threshold_value = fnp.float32(threshold)
    above_mask = primary_alpha > threshold_value

    recheck_rows = _probe_rows(n_samples, _PILOT_RECHECK_FRACTION)
    if recheck_rows <= primary_rows:
        return source_idx[above_mask], source_idx[~above_mask]

    uncertain_mask = fnp.abs(primary_alpha - threshold_value) <= fnp.float32(_PILOT_RECHECK_MARGIN)
    uncertain_idx = source_idx[uncertain_mask]
    if len(uncertain_idx) == 0:
        return source_idx[above_mask], source_idx[~above_mask]

    stable_mask = ~uncertain_mask
    stable_above_idx = source_idx[above_mask & stable_mask]
    stable_below_idx = source_idx[(~above_mask) & stable_mask]

    recheck_alpha = _sample_alpha(x, weights[:, uncertain_mask], recheck_rows, n_samples)
    recheck_above_mask = recheck_alpha > threshold_value
    above_idx = fnp.concatenate([stable_above_idx, uncertain_idx[recheck_above_mask]])
    below_idx = fnp.concatenate([stable_below_idx, uncertain_idx[~recheck_above_mask]])
    return fnp.sort(above_idx), fnp.sort(below_idx)


_USE_STRASSEN = True
_STRASSEN_MIN_DIM = 96
_STRASSEN_MAX_RECURSE_DEPTH = 1
_STRASSEN_MIN_ROWS = 4096


def _strassen_impl(x, w, depth=0):
    m, k = x.shape
    _, n = w.shape
    if depth > _STRASSEN_MAX_RECURSE_DEPTH:
        return x @ w
    if k < _STRASSEN_MIN_DIM or n < _STRASSEN_MIN_DIM:
        return x @ w

    if k % 2:
        x = fnp.concatenate([x, fnp.zeros((m, 1), dtype=x.dtype)], axis=1)
        w = fnp.concatenate([w, fnp.zeros((1, n), dtype=w.dtype)], axis=0)
        k += 1
    pad_m = m % 2
    if pad_m:
        x = fnp.concatenate([x, fnp.zeros((1, k), dtype=x.dtype)], axis=0)
    pad_n = n % 2
    if pad_n:
        w = fnp.concatenate([w, fnp.zeros((k, 1), dtype=w.dtype)], axis=1)

    mm, nn = m + pad_m, n + pad_n
    mh, kh, nh = mm // 2, k // 2, nn // 2

    recurse = (
        mh >= _STRASSEN_MIN_ROWS
        and kh >= _STRASSEN_MIN_DIM
        and nh >= _STRASSEN_MIN_DIM
        and (mh % 2) == 0 and (kh % 2) == 0 and (nh % 2) == 0
    )

    def _prod(a, b):
        if recurse:
            return _strassen_impl(a, b, depth + 1)
        return a @ b

    A11, A12 = x[:mh, :kh], x[:mh, kh:]
    A21, A22 = x[mh:, :kh], x[mh:, kh:]
    B11, B12 = w[:kh, :nh], w[:kh, nh:]
    B21, B22 = w[kh:, :nh], w[kh:, nh:]

    M1 = _prod(A11 + A22, B11 + B22)
    M2 = _prod(A21 + A22, B11)
    M3 = _prod(A11, B12 - B22)
    M4 = _prod(A22, B21 - B11)
    M5 = _prod(A11 + A12, B22)
    M6 = _prod(A21 - A11, B11 + B12)
    M7 = _prod(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    top = fnp.concatenate([C11, C12], axis=1)
    bot = fnp.concatenate([C21, C22], axis=1)
    out = fnp.concatenate([top, bot], axis=0)
    return out[:m, :n]


def _strassen_matmul(x, w):
    return _strassen_impl(x, w, depth=0)


def _dense_matmul(x, w):
    if _USE_STRASSEN:
        return _strassen_matmul(x, w)
    return x @ w


def _cold_slice_relu_matmul(x, w_active, kc):
    """Exact relu(x @ w) with the kc coldest input columns sliced out of the
    dense matmul for rows that have no support on them. Columns of x are
    already ordered coldest-first (imposed via the producing weight slice)."""
    n = x.shape[0]
    cold_nnz = fnp.sum(x[:, :kc] > fnp.float32(0.0), axis=1)
    hasf = (cold_nnz > 0).astype(fnp.float32)
    csum = fnp.cumsum(hasf)
    n_sup = int(csum[-1])
    if n_sup == 0:
        return fnp.maximum(_dense_matmul(x[:, kc:], w_active[kc:, :]), 0.0)
    if n_sup >= n - 1:
        return fnp.maximum(_dense_matmul(x, w_active), 0.0)

    if _COLD_REORDER == "put":
        nof = (cold_nnz == 0).astype(fnp.float32)
        csum2 = fnp.cumsum(nof)
        dest_f = hasf * (csum - fnp.float32(1.0)) + nof * (
            csum2 + fnp.float32(float(n_sup - 1))
        )
        dest = dest_f.astype(fnp.int64)
        xp = fnp.zeros(x.shape, dtype=fnp.float32)
        fnp.put_along_axis(xp, fnp.broadcast_to(dest[:, None], x.shape), x, axis=0)
    else:
        order = fnp.argsort(fnp.negative(hasf))
        xp = fnp.take(x, order, axis=0)

    pre_hot = _dense_matmul(xp[:, kc:], w_active[kc:, :])
    pre_cold = xp[:n_sup, :kc] @ w_active[:kc, :]
    top = fnp.maximum(pre_hot[:n_sup] + pre_cold, 0.0)
    bottom = fnp.maximum(pre_hot[n_sup:], 0.0)
    return fnp.concatenate([top, bottom], axis=0)


def _cold_slice_relu_matmul2(x, w_active, kc, kc2):
    """Two-level carve: rows ordered by support pattern over the ultra-cold
    [0, kc2) and warm-cold [kc2, kc) slabs — {both, ultra-only, warm-only,
    none}. One scatter makes every group's rows and its slab contiguous, so
    each correction is a plain slice-add (no cross-order accumulation)."""
    if kc2 <= 0:
        return _cold_slice_relu_matmul(x, w_active, kc)
    n = x.shape[0]
    hu = fnp.sum(x[:, :kc2] > fnp.float32(0.0), axis=1) > 0
    hw = fnp.sum(x[:, kc2:kc] > fnp.float32(0.0), axis=1) > 0
    g_both = (hu & hw).astype(fnp.float32)
    g_uo = (hu & (~hw)).astype(fnp.float32)
    g_wo = ((~hu) & hw).astype(fnp.float32)
    g_no = ((~hu) & (~hw)).astype(fnp.float32)
    c1 = fnp.cumsum(g_both)
    c2 = fnp.cumsum(g_uo)
    c3 = fnp.cumsum(g_wo)
    c4 = fnp.cumsum(g_no)
    nb = int(c1[-1])
    nu = int(c2[-1])
    nw = int(c3[-1])
    if nb + nu + nw == 0:
        return fnp.maximum(_dense_matmul(x[:, kc:], w_active[kc:, :]), 0.0)
    if nb >= n - 1:
        return fnp.maximum(_dense_matmul(x, w_active), 0.0)
    dest_f = (
        g_both * (c1 - fnp.float32(1.0))
        + g_uo * (c2 + fnp.float32(float(nb - 1)))
        + g_wo * (c3 + fnp.float32(float(nb + nu - 1)))
        + g_no * (c4 + fnp.float32(float(nb + nu + nw - 1)))
    )
    dest = dest_f.astype(fnp.int64)
    xp = fnp.zeros(x.shape, dtype=fnp.float32)
    fnp.put_along_axis(xp, fnp.broadcast_to(dest[:, None], x.shape), x, axis=0)

    pre_hot = _dense_matmul(xp[:, kc:], w_active[kc:, :])
    parts = []
    if nb:
        cb = xp[:nb, :kc] @ w_active[:kc, :]
        parts.append(fnp.maximum(pre_hot[:nb] + cb, 0.0))
    if nu:
        cu = xp[nb:nb + nu, :kc2] @ w_active[:kc2, :]
        parts.append(fnp.maximum(pre_hot[nb:nb + nu] + cu, 0.0))
    if nw:
        cw = xp[nb + nu:nb + nu + nw, kc2:kc] @ w_active[kc2:kc, :]
        parts.append(fnp.maximum(pre_hot[nb + nu:nb + nu + nw] + cw, 0.0))
    ns = nb + nu + nw
    if ns < n:
        parts.append(fnp.maximum(pre_hot[ns:], 0.0))
    if len(parts) == 1:
        return parts[0]
    return fnp.concatenate(parts, axis=0)


def _cold_slice_relu_matmul_restore(x, w_active, kc):
    """Pilot-block variant: identical single-level carve, but the row order is
    restored afterwards (inverse-permutation scatter) so the Sobol-prefix
    probe rows and the antithetic pairing stay exactly in place."""
    n = x.shape[0]
    cold_nnz = fnp.sum(x[:, :kc] > fnp.float32(0.0), axis=1)
    hasf = (cold_nnz > 0).astype(fnp.float32)
    csum = fnp.cumsum(hasf)
    n_sup = int(csum[-1])
    if n_sup == 0:
        return fnp.maximum(_dense_matmul(x[:, kc:], w_active[kc:, :]), 0.0)
    if n_sup >= n - 1:
        return fnp.maximum(_dense_matmul(x, w_active), 0.0)
    nof = (cold_nnz == 0).astype(fnp.float32)
    csum2 = fnp.cumsum(nof)
    dest_f = hasf * (csum - fnp.float32(1.0)) + nof * (
        csum2 + fnp.float32(float(n_sup - 1))
    )
    dest = dest_f.astype(fnp.int64)
    xp = fnp.zeros(x.shape, dtype=fnp.float32)
    fnp.put_along_axis(xp, fnp.broadcast_to(dest[:, None], x.shape), x, axis=0)
    pre_hot = _dense_matmul(xp[:, kc:], w_active[kc:, :])
    pre_cold = xp[:n_sup, :kc] @ w_active[:kc, :]
    top = fnp.maximum(pre_hot[:n_sup] + pre_cold, 0.0)
    bottom = fnp.maximum(pre_hot[n_sup:], 0.0)
    y = fnp.concatenate([top, bottom], axis=0)
    inv = fnp.zeros(n, dtype=fnp.int64)
    fnp.put_along_axis(inv, dest, fnp.arange(n), axis=0)
    out_buf = fnp.zeros(y.shape, dtype=fnp.float32)
    fnp.put_along_axis(out_buf, fnp.broadcast_to(inv[:, None], y.shape), y, axis=0)
    return out_buf


class Estimator(BaseEstimator):
    """Staged active-set refinement with smooth hard-network continuation."""

    def __init__(self) -> None:
        self._sobol_points = None

    def setup(self, ctx: SetupContext) -> None:
        fnp.random.default_rng(ctx.seed)
        sobol_path = Path(ctx.submission_dir) / "sobol_points.npz"
        data = fnp.load(str(sobol_path))
        self._sobol_points = data["points"]

    def _load_sobol_points(self) -> None:
        if self._sobol_points is None:
            data = fnp.load(str(Path(__file__).resolve().parent / "sobol_points.npz"))
            self._sobol_points = data["points"]

    def _initial_structure(self, mlp: MLP, width: int) -> dict:
        active_indices = []
        kink_indices = []
        on_indices = []
        dead_indices = []
        dead_corrections = []
        analytical_rows = []
        alpha_rows = []
        mu_post = fnp.zeros(width)
        var_post = fnp.zeros(width)

        for layer_idx, w in enumerate(mlp.weights):
            if layer_idx == 0:
                mu_pre = fnp.zeros(width)
                var_pre = fnp.sum(w * w, axis=0)
            else:
                mu_pre = w.T @ mu_post
                var_pre = fnp.sum(w * w * var_post[:, None], axis=0)

            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre

            dead_mask = alpha < _DEAD_THRESH
            on_mask = alpha > _ON_THRESH
            kink_mask = (~dead_mask) & (~on_mask)

            dead_idx = fnp.nonzero(dead_mask)[0]
            alpha_rows.append(alpha)
            dead_indices.append(dead_idx)
            active_indices.append(fnp.nonzero(~dead_mask)[0])
            kink_indices.append(fnp.nonzero(kink_mask)[0])
            on_indices.append(fnp.nonzero(on_mask)[0])

            phi = flops.stats.norm.pdf(alpha)
            Phi = flops.stats.norm.cdf(alpha)
            mu_post = mu_pre * Phi + sigma_pre * phi
            var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - mu_post * mu_post
            var_post = fnp.maximum(var_post, 1e-12)
            analytical_rows.append(mu_post)

            if len(dead_idx) > 0 and (_MATERIALIZE_INTERMEDIATE_ROWS or layer_idx == mlp.depth - 1):
                dead_corrections.append(_scatter(mu_post[dead_idx], dead_idx, width))
            else:
                dead_corrections.append(fnp.zeros(width))

        return {
            "active_indices": active_indices,
            "kink_indices": kink_indices,
            "on_indices": on_indices,
            "dead_indices": dead_indices,
            "dead_corrections": dead_corrections,
            "analytical_rows": analytical_rows,
            "alpha_rows": alpha_rows,
            "final_var_mean": float(fnp.mean(var_post)),
        }

    def _sample_block(self, start_half: int, half_count: int, width: int):
        self._load_sobol_points()
        end_half = start_half + half_count
        if self._sobol_points.shape[0] < end_half:
            raise ValueError(
                f"sobol_points.npz has {self._sobol_points.shape[0]} half-samples, "
                f"but this estimator needs {end_half}."
            )
        half = fnp.array(self._sobol_points[start_half:end_half, :width])
        return fnp.concatenate([half, -half], axis=0)

    def _run_block(self, mlp: MLP, structure: dict, x, n_samples: int, refine: bool, cold_plan=None) -> tuple[list, float, dict]:
        width = mlp.width
        active_indices = list(structure["active_indices"])
        kink_indices = list(structure["kink_indices"])
        on_indices = list(structure["on_indices"])
        dead_indices = list(structure["dead_indices"])
        dead_corrections = list(structure["dead_corrections"])
        analytical_rows = structure["analytical_rows"]
        alpha_rows = structure["alpha_rows"]

        mc_rows = []
        prev_idx = None
        x_before_fold = None
        fire_stats = {}

        for layer_idx, w in enumerate(mlp.weights):
            idx = active_indices[layer_idx]
            kink_idx = kink_indices[layer_idx]
            on_idx = on_indices[layer_idx]

            if len(idx) == 0:
                mc_rows.append(fnp.zeros(width))
                x = fnp.zeros((n_samples, 0))
                prev_idx = idx
                x_before_fold = None
                continue

            if (
                refine
                and prev_idx is not None
                and _DEMOTE_ACTIVE_DEAD_START_LAYER <= layer_idx <= _DEMOTE_ACTIVE_DEAD_STOP_LAYER
            ):
                active_dead_probe_mask = alpha_rows[layer_idx][kink_idx] <= fnp.float32(_ACTIVE_DEAD_PROBE_MAX)
                trusted_kink_idx = kink_idx[~active_dead_probe_mask]
                probe_kink_idx = kink_idx[active_dead_probe_mask]
                if len(probe_kink_idx) > 0:
                    w_kink_probe = w[prev_idx, :][:, probe_kink_idx]
                    kept_probe_kink_idx, demoted_kink_idx = _staged_threshold_split(
                        probe_kink_idx, x, w_kink_probe, n_samples, _ACTIVE_DEAD_THRESH
                    )
                else:
                    kept_probe_kink_idx = kink_idx[:0]
                    demoted_kink_idx = kink_idx[:0]

                if len(demoted_kink_idx) > 0 and _MATERIALIZE_INTERMEDIATE_ROWS:
                    dead_corrections[layer_idx] = dead_corrections[layer_idx] + _scatter(
                        analytical_rows[layer_idx][demoted_kink_idx], demoted_kink_idx, width
                    )
                    dead_indices[layer_idx] = fnp.sort(fnp.concatenate([dead_indices[layer_idx], demoted_kink_idx]))

                kink_idx = fnp.sort(fnp.concatenate([trusted_kink_idx, kept_probe_kink_idx]))
                idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                kink_indices[layer_idx] = kink_idx
                active_indices[layer_idx] = idx

            if layer_idx == 30 and len(on_idx) > 0 and prev_idx is not None:
                x_before_fold = x

                if refine:
                    on_probe_mask = alpha_rows[layer_idx][on_idx] <= fnp.float32(_ON_PROBE_MAX)
                    trusted_on_idx = on_idx[~on_probe_mask]
                    probe_on_idx = on_idx[on_probe_mask]
                    if len(probe_on_idx) > 0:
                        w_on_probe = w[prev_idx, :][:, probe_on_idx]
                        kept_probe_on_idx, demoted_idx = _staged_threshold_split(
                            probe_on_idx, x, w_on_probe, n_samples, _PILOT_ON_THRESH
                        )
                    else:
                        demoted_idx = on_idx[:0]
                        kept_probe_on_idx = on_idx[:0]

                    on_idx = fnp.sort(fnp.concatenate([trusted_on_idx, kept_probe_on_idx]))
                    kink_idx = fnp.sort(fnp.concatenate([kink_idx, demoted_idx]))
                    on_indices[layer_idx] = on_idx
                    kink_indices[layer_idx] = kink_idx

                    dead_idx = dead_indices[layer_idx]
                    if len(dead_idx) > 0:
                        dead_probe_mask = alpha_rows[layer_idx][dead_idx] >= fnp.float32(_DEAD_PROBE_MIN)
                        trusted_dead_idx = dead_idx[~dead_probe_mask]
                        probe_dead_idx = dead_idx[dead_probe_mask]
                        if len(probe_dead_idx) > 0:
                            w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                            promoted_idx, remaining_probe_dead_idx = _staged_threshold_split(
                                probe_dead_idx, x, w_dead_probe, n_samples, _PILOT_DEAD_THRESH
                            )
                        else:
                            promoted_idx = dead_idx[:0]
                            remaining_probe_dead_idx = dead_idx[:0]

                        remaining_dead_idx = fnp.sort(fnp.concatenate([trusted_dead_idx, remaining_probe_dead_idx]))
                        if len(promoted_idx) > 0 and _MATERIALIZE_INTERMEDIATE_ROWS:
                            dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                                analytical_rows[layer_idx][promoted_idx], promoted_idx, width
                            )
                        dead_indices[layer_idx] = remaining_dead_idx
                        kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_idx]))
                        idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                        kink_indices[layer_idx] = kink_idx
                        active_indices[layer_idx] = idx

                w_kink = w[prev_idx, :][:, kink_idx]
                x_kink = fnp.maximum(x @ w_kink, 0.0)
                kink_mean = fnp.mean(x_kink, axis=0)

                mean_prev = fnp.mean(x, axis=0)
                w_on = w[prev_idx, :][:, on_idx]
                on_mean = mean_prev @ w_on

                mc_rows.append(_scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width))

                x = x_kink
                prev_idx = kink_idx
                continue

            if layer_idx == 31 and x_before_fold is not None and len(on_indices[30]) > 0:
                fold_on_idx = on_indices[30]
                fold_prev_idx = active_indices[29]

                w_from_kink = w[prev_idx, :][:, kink_idx]
                pre_from_kink = x @ w_from_kink

                w_fold_layer = mlp.weights[30]
                w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]
                w_this_from_on = w[fold_on_idx, :][:, kink_idx]
                w_folded = w_fold_on @ w_this_from_on
                pre_from_on = x_before_fold @ w_folded

                x_kink = fnp.maximum(pre_from_kink + pre_from_on, 0.0)
                kink_mean = fnp.mean(x_kink, axis=0)

                prev_layer_mean = mc_rows[30]
                fold_active_idx = active_indices[30]
                w_to_on = w[fold_active_idx, :][:, on_idx]
                on_mean = prev_layer_mean[fold_active_idx] @ w_to_on

                mc_rows.append(_scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width))
                prev_idx = idx
                x_before_fold = None
                continue

            if (
                refine
                and prev_idx is not None
                and _REFINE_DEAD_START_LAYER <= layer_idx <= _REFINE_DEAD_STOP_LAYER
            ):
                dead_idx = dead_indices[layer_idx]
                if len(dead_idx) > 0:
                    dead_probe_mask = alpha_rows[layer_idx][dead_idx] >= fnp.float32(_DEAD_PROBE_MIN)
                    trusted_dead_idx = dead_idx[~dead_probe_mask]
                    probe_dead_idx = dead_idx[dead_probe_mask]

                    if len(probe_dead_idx) > 0:
                        w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                        promoted_idx, remaining_probe_dead_idx = _staged_threshold_split(
                            probe_dead_idx, x, w_dead_probe, n_samples, _PILOT_DEAD_THRESH
                        )
                    else:
                        promoted_idx = dead_idx[:0]
                        remaining_probe_dead_idx = dead_idx[:0]

                    remaining_dead_idx = fnp.sort(fnp.concatenate([trusted_dead_idx, remaining_probe_dead_idx]))
                    if len(promoted_idx) > 0 and _MATERIALIZE_INTERMEDIATE_ROWS:
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                            analytical_rows[layer_idx][promoted_idx], promoted_idx, width
                        )
                    dead_indices[layer_idx] = remaining_dead_idx
                    kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_idx]))
                    idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = idx

            if prev_idx is None:
                w_active = w[:, idx]
                half_rows = n_samples // 2
                pre_half = _dense_matmul(x[:half_rows, :], w_active)
                x = fnp.concatenate([fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)], axis=0)
            else:
                if refine and _PILOT_COLD and 1 <= layer_idx <= 29 and len(idx) > 1:
                    idx = idx[fnp.argsort(alpha_rows[layer_idx][idx])]
                    active_indices[layer_idx] = idx
                w_active = w[prev_idx, :][:, idx]
                if refine:
                    kp = 0
                    if (
                        _PILOT_COLD
                        and 2 <= layer_idx <= 29
                        and len(prev_idx) >= _COLD_MIN_HOT_DIM + _PILOT_MIN_K
                    ):
                        kp = int(fnp.sum(
                            alpha_rows[layer_idx - 1][prev_idx] < fnp.float32(_PILOT_ALPHA_COLD)
                        ))
                        kp = min(kp, _PILOT_MAX_K, len(prev_idx) - _COLD_MIN_HOT_DIM)
                    if kp >= _PILOT_MIN_K:
                        x = _cold_slice_relu_matmul_restore(x, w_active, kp)
                    else:
                        x = fnp.maximum(_dense_matmul(x, w_active), 0.0)
                else:
                    plan = cold_plan.get(layer_idx) if cold_plan else None
                    if plan:
                        x = _cold_slice_relu_matmul2(x, w_active, plan[0], plan[1])
                    else:
                        x = fnp.maximum(_dense_matmul(x, w_active), 0.0)
            if (
                refine
                and _COLD_SLICE
                and _COLD_START_LAYER - 1 <= layer_idx <= _COLD_STOP_LAYER - 1
                and len(idx) > 0
            ):
                fire_stats[layer_idx] = fnp.sum(x > fnp.float32(0.0), axis=0)
            if _MATERIALIZE_INTERMEDIATE_ROWS or layer_idx >= 30:
                mc_rows.append(_scatter(fnp.mean(x, axis=0), idx, width))
            else:
                mc_rows.append(fnp.zeros(width))
            prev_idx = idx

        if _MATERIALIZE_INTERMEDIATE_ROWS:
            w0 = mlp.weights[0]
            sigma_0 = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-12))
            row0 = sigma_0 * fnp.float32(0.3989422804014327)
            rows = [row0 + dead_corrections[0]] + [
                mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))
            ]
        else:
            rows = list(analytical_rows[:-1]) + [mc_rows[-1] + dead_corrections[-1]]

        refined_structure = {
            "active_indices": active_indices,
            "kink_indices": kink_indices,
            "on_indices": on_indices,
            "dead_indices": dead_indices,
            "dead_corrections": dead_corrections,
            "analytical_rows": analytical_rows,
            "alpha_rows": alpha_rows,
            "fire_stats": fire_stats,
        }
        return rows, 0.0, refined_structure

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        if _CAST_WEIGHTS_F32:
            mlp = _F32MLP(mlp)
        width = mlp.width
        structure = self._initial_structure(mlp, width)
        base_x = self._sample_block(0, _BASE_SAMPLES // 2, width)
        base_rows, _, refined_structure = self._run_block(
            mlp, structure, base_x, _BASE_SAMPLES, refine=True
        )
        extra_samples = _TOTAL_SAMPLES - _BASE_SAMPLES
        cold_plan = _plan_cold_slices(refined_structure, _BASE_SAMPLES) if _COLD_SLICE else None
        extra_x = self._sample_block(_BASE_SAMPLES // 2, extra_samples // 2, width)
        extra_rows, _, _ = self._run_block(
            mlp, refined_structure, extra_x, extra_samples, refine=False, cold_plan=cold_plan
        )
        combined_rows = [
            (base_row * _BASE_SAMPLES + extra_row * extra_samples) / _TOTAL_SAMPLES
            for base_row, extra_row in zip(base_rows, extra_rows)
        ]
        return fnp.stack(combined_rows, axis=0)

class _F32MLP:
    """Cast weights to float32 once."""

    def __init__(self, mlp: MLP) -> None:
        self.width = mlp.width
        self.depth = mlp.depth
        self.weights = [fnp.asarray(w).astype(fnp.float32) for w in mlp.weights]


def _plan_cold_slices(structure, n_pilot: int):
    """Order each producing layer's active list coldest-first (free: absolute
    neuron ids flow through the weight slices) and size the cold set from the
    pilot fire census. All work stays on flopscope arrays; only one scalar per
    planned layer is read back for control flow."""
    fire_stats = structure.get("fire_stats") or {}
    active = structure["active_indices"]
    plan = {}
    thresh_cnt = fnp.float32(_COLD_FIRE_THRESH * n_pilot)
    thresh2_cnt = fnp.float32(_COLD_FIRE_THRESH2 * n_pilot)
    for j in range(_COLD_START_LAYER, _COLD_STOP_LAYER + 1):
        fire_cnt = fire_stats.get(j - 1)
        if fire_cnt is None or j >= len(active):
            continue
        idx = active[j - 1]
        acts = len(idx)
        out_j = len(active[j])
        if acts == 0 or out_j == 0 or len(fire_cnt) != acts:
            continue
        k = int(fnp.sum(fire_cnt < thresh_cnt))
        k = min(k, _COLD_MAX_K, acts - _COLD_MIN_HOT_DIM)
        if k < _COLD_MIN_K:
            continue
        k2 = int(fnp.sum(fire_cnt < thresh2_cnt))
        k2 = min(k2, k - _COLD_MIN_K2)
        if k2 < _COLD_MIN_K2:
            k2 = 0
        order = fnp.argsort(fire_cnt)
        active[j - 1] = idx[order]
        plan[j] = (k, k2)
    return plan
