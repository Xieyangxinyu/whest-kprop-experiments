"""Two-ended layer-30 refinement for the Sobol active-set estimator.

This is the current best submission family after submission 314954. It starts
from the Sobol active-set fold estimator and adds a small layer-30 pilot that
refines Stage 1 classification in both safe directions:

1. Demote weak always-on neurons back to kink when pilot alpha is not confident.
2. Promote weak dead neurons back to kink when pilot alpha suggests they are not
   confidently dead.

The pilot reuses the same Sobol samples used by the final estimate. It only adds
probe FLOPs for the refinement decision.
"""

from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_MAIN_SAMPLES = 40960  # 20480 Sobol half-samples x 2 (antithetic)
_DEAD_THRESH = -3.0
_ON_THRESH = 3.0
_PILOT_FRACTION = 0.05
_PILOT_ON_THRESH = 3.0
_PILOT_DEAD_THRESH = -2.5


def _scatter(values, idx, width):
    """Functionally place values at idx into a zero vector of length width."""
    return fnp.eye(width, dtype=fnp.float32)[:, idx] @ values


class Estimator(BaseEstimator):
    """Sobol QMC + analytical classification + two-ended layer-30 refinement."""

    def __init__(self) -> None:
        self._sobol_points = None

    def setup(self, ctx: SetupContext) -> None:
        fnp.random.default_rng(ctx.seed)
        sobol_path = Path(ctx.submission_dir) / "sobol_points.npz"
        data = fnp.load(str(sobol_path))
        self._sobol_points = data["points"]

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width

        if self._sobol_points is None:
            data = fnp.load(str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
            self._sobol_points = data["points"]

        active_indices = []
        kink_indices = []
        on_indices = []
        dead_indices = []
        dead_corrections = []
        analytical_rows = []
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

            if len(dead_idx) > 0:
                dead_corrections.append(_scatter(mu_post[dead_idx], dead_idx, width))
            else:
                dead_corrections.append(fnp.zeros(width))

        half = fnp.array(self._sobol_points[: _MAIN_SAMPLES // 2, :width])
        x = fnp.concatenate([half, -half], axis=0)

        mc_rows = []
        mc_vars = []
        prev_idx = None
        x_before_fold = None

        for layer_idx, w in enumerate(mlp.weights):
            idx = active_indices[layer_idx]
            kink_idx = kink_indices[layer_idx]
            on_idx = on_indices[layer_idx]

            if len(idx) == 0:
                mc_rows.append(fnp.zeros(width))
                mc_vars.append(fnp.zeros(width))
                x = fnp.zeros((_MAIN_SAMPLES, 0))
                prev_idx = idx
                x_before_fold = None
                continue

            if layer_idx == 30 and len(on_idx) > 0 and prev_idx is not None:
                x_before_fold = x
                pilot_rows = max(2, min(_MAIN_SAMPLES, int(_MAIN_SAMPLES * _PILOT_FRACTION)))

                w_on_probe = w[prev_idx, :][:, on_idx]
                pre_on_pilot = x[:pilot_rows, :] @ w_on_probe
                pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                pilot_var = fnp.var(pre_on_pilot, axis=0)
                pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                on_np = on_idx.astype(int)
                kink_np = kink_idx.astype(int)
                keep_on = pilot_alpha > fnp.float32(_PILOT_ON_THRESH)
                demoted_np = on_np[~keep_on]
                on_idx = on_np[keep_on]
                kink_idx = fnp.sort(fnp.concatenate([kink_np, demoted_np]))
                on_indices[layer_idx] = on_idx
                kink_indices[layer_idx] = kink_idx

                dead_idx = dead_indices[layer_idx]
                if len(dead_idx) > 0:
                    w_dead_probe = w[prev_idx, :][:, dead_idx]
                    pre_dead_pilot = x[:pilot_rows, :] @ w_dead_probe
                    dead_mean = fnp.mean(pre_dead_pilot, axis=0)
                    dead_var = fnp.var(pre_dead_pilot, axis=0)
                    dead_alpha = dead_mean / fnp.sqrt(fnp.maximum(dead_var, 1e-12))

                    dead_np = dead_idx.astype(int)
                    promote_dead = dead_alpha > fnp.float32(_PILOT_DEAD_THRESH)
                    promoted_np = dead_np[promote_dead]
                    remaining_dead_np = dead_np[~promote_dead]
                    if len(promoted_np) > 0:
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                            analytical_rows[layer_idx][promoted_np], promoted_np, width
                        )
                    dead_indices[layer_idx] = remaining_dead_np
                    kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_np]))
                    idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = idx

                w_kink = w[prev_idx, :][:, kink_idx]
                x_kink = fnp.maximum(x @ w_kink, 0.0)
                kink_mean = fnp.mean(x_kink, axis=0)
                kink_var = fnp.var(x_kink, axis=0)

                mean_prev = fnp.mean(x, axis=0)
                var_prev = fnp.var(x, axis=0)
                w_on = w[prev_idx, :][:, on_idx]
                on_mean = mean_prev @ w_on
                on_var = fnp.sum(w_on * w_on * var_prev[:, None], axis=0)

                mc_rows.append(_scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width))
                mc_vars.append(_scatter(kink_var, kink_idx, width) + _scatter(on_var, on_idx, width))

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
                kink_var = fnp.var(x_kink, axis=0)

                prev_layer_mean = mc_rows[30]
                fold_active_idx = active_indices[30]
                w_to_on = w[fold_active_idx, :][:, on_idx]
                on_mean = prev_layer_mean[fold_active_idx] @ w_to_on
                prev_layer_var = mc_vars[30]
                on_var = fnp.sum(w_to_on * w_to_on * prev_layer_var[fold_active_idx, None], axis=0)

                mc_rows.append(_scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width))
                mc_vars.append(_scatter(kink_var, kink_idx, width) + _scatter(on_var, on_idx, width))
                prev_idx = idx
                x_before_fold = None
                continue

            if prev_idx is None:
                w_active = w[:, idx]
            else:
                w_active = w[prev_idx, :][:, idx]

            x = fnp.maximum(x @ w_active, 0.0)
            mc_rows.append(_scatter(fnp.mean(x, axis=0), idx, width))
            mc_vars.append(_scatter(fnp.var(x, axis=0), idx, width))
            prev_idx = idx

        w0 = mlp.weights[0]
        sigma_0 = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-12))
        row0 = sigma_0 * fnp.float32(0.3989422804014327)
        rows = [row0 + dead_corrections[0]] + [mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))]
        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)
