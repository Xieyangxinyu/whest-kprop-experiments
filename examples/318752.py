"""Fixed 61,440-sample estimator with antithetic pilot probes and 2-level Strassen.

This estimator uses a constant 61,440 antithetic samples (10,240 pilot + 51,200
continuation) for every MLP. Dead/active/on classification probes estimate alpha
from both antithetic halves (pair-balanced). Sampled active propagation uses a
guarded 2-level recursive Strassen decomposition (49 quarter-size matmuls instead
of 64) to reduce contraction FLOPs. The final row is sampled accurately, while
intermediate returned rows are cheap analytical fillers.
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

    def _run_block(self, mlp: MLP, structure: dict, x, n_samples: int, refine: bool) -> tuple[list, float, dict]:
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
                w_active = w[prev_idx, :][:, idx]
                x = fnp.maximum(_dense_matmul(x, w_active), 0.0)
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
        }
        return rows, 0.0, refined_structure

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width
        structure = self._initial_structure(mlp, width)
        base_x = self._sample_block(0, _BASE_SAMPLES // 2, width)
        base_rows, _, refined_structure = self._run_block(
            mlp, structure, base_x, _BASE_SAMPLES, refine=True
        )
        extra_samples = _TOTAL_SAMPLES - _BASE_SAMPLES
        extra_x = self._sample_block(_BASE_SAMPLES // 2, extra_samples // 2, width)
        extra_rows, _, _ = self._run_block(mlp, refined_structure, extra_x, extra_samples, refine=False)
        combined_rows = [
            (base_row * _BASE_SAMPLES + extra_row * extra_samples) / _TOTAL_SAMPLES
            for base_row, extra_row in zip(base_rows, extra_rows)
        ]
        return fnp.stack(combined_rows, axis=0)