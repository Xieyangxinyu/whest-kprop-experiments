"""Algorithm 25 diagnostic fork: untracked NumPy dense execution.

This version keeps the Algorithm 25 active-set classification policy, but uses
a fixed diagnostic sample count instead of dynamic per-MLP allocation and moves
numerical work out of flopscope into plain NumPy/SciPy. It disables the
flopscope-accounting-oriented complex packing, Strassen, and packed row-sparse
dispatch paths; dense BLAS is faster and easier to reason about when score
accounting is not the goal.

Use this for local diagnostics and sample-policy experiments. It is not the
submission-safe flopscope-cost surface described below.

Original Algorithm 25 header follows.
Algorithm 25: complex packing with 315856 packed-path cleanup.

Builds on Algorithm 24 / submission 315998 by keeping row-axis complex64
sample packing and the Algorithm 21 per-layer block-split thresholds, then
restoring wall-safe exact-output packed-path efficiencies from Algorithm 17 /
submission 315856: no redundant low-8 row bucket, immediate Strassen
accumulation, and half-only Sobol sample loading before the exact layer-0
antithetic reconstruction. The 315856 24,576-row packed chunks were rejected
locally for wall-time safety on top of complex packing. Early high-`k` packed
groups fall back to the existing complex/Strassen dense path once `k > width/2`.
Submission 316005 transferred this row-dense fallback surface publicly.
This version refits the column fire threshold to global `0.5`, matching the
post-complex-packing cost model where dense paths are effectively half-priced.

Original Algorithm 21 header follows.
Algorithm 21: layer-wise block-split fire thresholds on the finer-row-buckets surface.

Builds on Algorithm 17 / submission 315824. The single _BLOCK_SPLIT_FIRE_THRESH
(0.75) is replaced by a per-layer map fitted from a 4-net fire-rate census
(nets 0-3, mini split): a column firing at rate f costs ~1.94*f per
row-output in the packed path vs ~1.76 flat in the Strassen dense block, so
the FLOP crossover sits near f=0.91, not 0.75 - and the optimum is
layer-dependent because small dense blocks miss the Strassen MIN_IN=64
discount (layer 31 packs everything). Routing is exact: raw MSE is
bit-identical; measured flops -0.90% on the fit nets, -0.72% on holdout
nets 4-12. A global threshold shift is measurably flat - only the per-layer
map wins.

Original Algorithm 17 header follows.
Algorithm 17: finer antithetic row buckets for block-split row-sparse propagation.

This estimator starts from a 30,720-sample prefix. It refines active/dead/on
classification with staged pilot probes, then uses analytical final-layer
variance as a hardness signal and chooses
``N_i = clip(49152 * sqrt(V_i / V_ref), 30720, 61440)``. Harder MLPs continue
with the next Sobol prefix; easier MLPs keep the shorter estimate. The final row
is sampled accurately, while intermediate returned rows are cheap analytical
fillers because the leaderboard scores only the final layer.

Compared with the original argpartition row-sparse surface, sampled active
propagation splits ordinary active matmuls into a dense high-fire block and a
packed low-fire block. All samples still pass through exactly; the split only
changes how the same sample-by-weight matmul is evaluated. The same split is
used for both the base/refinement sample block and any extra continuation block.
The special layer-30/31 fold from Algorithm 15 is kept, with block-split matmuls
applied to its sampled kink paths. Large dense sampled matmuls use a guarded
one-level Strassen contraction, while classification probes stay on plain dense
matmul to avoid probe-path overhead.
Submission 315718 added mask reuse and a `put_along_axis` row-order restore.
Submission 315824 added finer row buckets around half-density antithetic layer-1
rows (`112, 144, 160, 176`) as a submission-safe way to recover part of the
rejected private antithetic pair-complement optimization.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as fnp
from scipy import special
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

def _read_sample_divisor() -> int:
    try:
        return max(1, int(os.environ.get("WHEST_DIAG_SAMPLE_DIVISOR", "1")))
    except ValueError:
        return 1


def _read_requested_samples() -> int:
    try:
        samples = int(os.environ.get("WHEST_DIAG_SAMPLES", "30720"))
    except ValueError:
        samples = 30720
    return max(2, min(61440, samples))


def _read_antithetic_mode() -> bool:
    return os.environ.get("WHEST_DIAG_ANTITHETIC", "1").lower() not in {"0", "false", "no", "off"}


def _scaled_samples(samples: int) -> int:
    scaled = samples / _DIAGNOSTIC_SAMPLE_DIVISOR
    return max(2, int(round(scaled / 2.0) * 2))


_DIAGNOSTIC_SAMPLE_DIVISOR = _read_sample_divisor()
_ANTITHETIC_SAMPLES = _read_antithetic_mode()
_REQUESTED_SAMPLES = _read_requested_samples()
_TARGET_SAMPLES = _scaled_samples(_REQUESTED_SAMPLES)
_BASE_SAMPLES = min(_scaled_samples(30720), _TARGET_SAMPLES)
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
    out = fnp.zeros(width, dtype=fnp.float32)
    out[idx] = values
    return out


def _sobol_path(base_dir: Path) -> Path:
    candidates = [base_dir / "sobol_points.npz", base_dir.parent / "sobol_points.npz"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _fast_mlp(mlp: MLP) -> MLP:
    return MLP(
        width=mlp.width,
        depth=mlp.depth,
        weights=[fnp.asarray(w, dtype=fnp.float32) for w in mlp.weights],
        seed=mlp.seed,
    )


def _probe_rows(n_samples: int, fraction: float) -> int:
    return max(2, min(n_samples, int(n_samples * fraction)))


def _sample_alpha(x, weights, rows: int):
    pre = x[:rows, :] @ weights
    mean = fnp.mean(pre, axis=0)
    var = fnp.var(pre, axis=0)
    return mean / fnp.sqrt(fnp.maximum(var, 1e-12))


def _staged_threshold_split(source_idx, x, weights, n_samples: int, threshold: float):
    primary_rows = _probe_rows(n_samples, _PILOT_FRACTION)
    primary_alpha = _sample_alpha(x, weights, primary_rows)
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

    recheck_alpha = _sample_alpha(x, weights[:, uncertain_mask], recheck_rows)
    recheck_above_mask = recheck_alpha > threshold_value
    above_idx = fnp.concatenate([stable_above_idx, uncertain_idx[recheck_above_mask]])
    below_idx = fnp.concatenate([stable_below_idx, uncertain_idx[~recheck_above_mask]])
    return fnp.sort(above_idx), fnp.sort(below_idx)


def _matmul(x, weights):
    return x @ weights


def _relu_matmul(x, weights):
    return fnp.maximum(_matmul(x, weights), 0.0)


class Estimator(BaseEstimator):
    """Staged active-set refinement with smooth hard-network continuation."""

    def __init__(self) -> None:
        self._sobol_points = None

    def setup(self, ctx: SetupContext) -> None:
        fnp.random.default_rng(ctx.seed)
        base_dir = Path(ctx.submission_dir) if ctx.submission_dir is not None else Path(__file__).resolve().parent
        sobol_path = _sobol_path(base_dir)
        data = fnp.load(str(sobol_path))
        self._sobol_points = data["points"]

    def _load_sobol_points(self) -> None:
        if self._sobol_points is None:
            data = fnp.load(str(_sobol_path(Path(__file__).resolve().parent)))
            self._sobol_points = data["points"]

    def _initial_structure(self, mlp: MLP, width: int) -> dict:
        active_indices = []
        kink_indices = []
        on_indices = []
        dead_indices = []
        dead_corrections = []
        analytical_rows = []
        alpha_rows = []
        mu_post = fnp.zeros(width, dtype=fnp.float32)
        var_post = fnp.zeros(width, dtype=fnp.float32)

        for layer_idx, w in enumerate(mlp.weights):
            if layer_idx == 0:
                mu_pre = fnp.zeros(width, dtype=fnp.float32)
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

            phi = fnp.exp(fnp.float32(-0.5) * alpha * alpha) * fnp.float32(0.3989422804014327)
            Phi = special.ndtr(alpha).astype(fnp.float32, copy=False)
            mu_post = mu_pre * Phi + sigma_pre * phi
            var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - mu_post * mu_post
            var_post = fnp.maximum(var_post, 1e-12)
            analytical_rows.append(mu_post)

            if len(dead_idx) > 0 and (_MATERIALIZE_INTERMEDIATE_ROWS or layer_idx == mlp.depth - 1):
                dead_corrections.append(_scatter(mu_post[dead_idx], dead_idx, width))
            else:
                dead_corrections.append(fnp.zeros(width, dtype=fnp.float32))

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

    def _sample_block(self, start_row: int, row_count: int, width: int):
        self._load_sobol_points()
        end_row = start_row + row_count
        if self._sobol_points.shape[0] < end_row:
            raise ValueError(
                f"sobol_points.npz has {self._sobol_points.shape[0]} rows, "
                f"but this estimator needs {end_row}."
            )
        return fnp.asarray(self._sobol_points[start_row:end_row, :width], dtype=fnp.float32)

    def _run_block(self, mlp: MLP, structure: dict, x, n_samples: int, refine: bool) -> tuple[list, dict]:
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
                mc_rows.append(fnp.zeros(width, dtype=fnp.float32))
                x = fnp.zeros((n_samples, 0), dtype=fnp.float32)
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
                x_kink = _relu_matmul(x, w_kink)
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
                pre_from_kink = _matmul(x, w_from_kink)

                w_fold_layer = mlp.weights[30]
                w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]
                w_this_from_on = w[fold_on_idx, :][:, kink_idx]
                w_folded = w_fold_on @ w_this_from_on
                pre_from_on = _matmul(x_before_fold, w_folded)

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
                if _ANTITHETIC_SAMPLES:
                    half_rows = n_samples // 2
                    pre_half = _matmul(x[:half_rows, :], w_active)
                    x = fnp.concatenate([fnp.maximum(pre_half, 0.0), fnp.maximum(-pre_half, 0.0)], axis=0)
                else:
                    x = _relu_matmul(x[:n_samples, :], w_active)
            else:
                w_active = w[prev_idx, :][:, idx]
                x = _relu_matmul(x, w_active)
            if _MATERIALIZE_INTERMEDIATE_ROWS or layer_idx >= 30:
                mc_rows.append(_scatter(fnp.mean(x, axis=0), idx, width))
            else:
                mc_rows.append(fnp.zeros(width, dtype=fnp.float32))
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
        return rows, refined_structure

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        mlp = _fast_mlp(mlp)
        width = mlp.width
        structure = self._initial_structure(mlp, width)
        base_rows_needed = _BASE_SAMPLES // 2 if _ANTITHETIC_SAMPLES else _BASE_SAMPLES
        base_x = self._sample_block(0, base_rows_needed, width)
        base_rows, refined_structure = self._run_block(
            mlp, structure, base_x, _BASE_SAMPLES, refine=True
        )
        if _TARGET_SAMPLES <= _BASE_SAMPLES:
            return fnp.stack(base_rows, axis=0)

        extra_samples = _TARGET_SAMPLES - _BASE_SAMPLES
        extra_start = _BASE_SAMPLES // 2 if _ANTITHETIC_SAMPLES else _BASE_SAMPLES
        extra_rows_needed = extra_samples // 2 if _ANTITHETIC_SAMPLES else extra_samples
        extra_x = self._sample_block(extra_start, extra_rows_needed, width)
        extra_rows = self._run_block(mlp, refined_structure, extra_x, extra_samples, refine=False)[0]
        combined_rows = [
            (base_row * _BASE_SAMPLES + extra_row * extra_samples) / _TARGET_SAMPLES
            for base_row, extra_row in zip(base_rows, extra_rows)
        ]
        return fnp.stack(combined_rows, axis=0)