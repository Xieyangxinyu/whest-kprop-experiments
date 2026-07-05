"""Active-set Sobol QMC estimator with analytical classification.

Uses pre-computed Sobol quasi-random points (shipped as .npz) for better
space-filling than pseudo-random MC. Points loaded in setup() at zero FLOP cost.
Antithetic pairing doubles effective sample count for free.
"""

from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_MAIN_SAMPLES = 40960  # 20480 Sobol half-samples × 2 (antithetic)


def _scatter(values, idx, width):
    """Functionally scatter values at idx positions into a width-length zero vector.

    Uses identity-column read (free) + matmul (costs width×k FLOPs — negligible).
    Avoids immutable-array write restriction.
    """
    scatter_mat = fnp.eye(width, dtype=fnp.float32)[:, idx]  # (width, k) — READ indexing
    return scatter_mat @ values  # (width,)


def _normal_samples(rng, n_samples: int, width: int) -> fnp.ndarray:
    """Standard antithetic random samples (for pilot)."""
    n_pairs = max(1, (n_samples + 1) // 2)
    half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
    return fnp.concatenate([half, -half], axis=0)


class Estimator(BaseEstimator):
    """Active-set Sobol QMC + analytical classification + fold."""

    def __init__(self) -> None:
        self._setup_rng = None
        self._sobol_points = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        # Load pre-computed Sobol N(0,1) points (generated offline with scipy)
        sobol_path = Path(ctx.submission_dir) / "sobol_points.npz"
        data = fnp.load(str(sobol_path))
        self._sobol_points = data["points"]  # (n_half_samples, 256) float32

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width

        # Lazy-load Sobol points if setup() wasn't called (local testing)
        if self._sobol_points is None:
            data = fnp.load(str(Path(__file__).parent / "sobol_points.npz"))
            self._sobol_points = data["points"]

        # =================================================================
        # Stage 1: Analytical classification (no pilot — saves ~2.7B FLOPs)
        # Propagate mean/variance through layers to compute α = μ/σ
        # α < -3.0 → dead, α > 3.0 → always-on, else → kink (needs MC)
        # =================================================================
        _DEAD_THRESH = -3.0
        _ON_THRESH = 3.0

        active_indices = []
        kink_indices = []
        on_indices = []
        dead_indices = []
        dead_corrections = []  # analytical E[ReLU] for dead neurons (instead of 0)
        analytical_rows = []
        alpha_rows = []
        anal_mu_post = fnp.zeros(width)   # post-ReLU mean (analytical)
        anal_var_post = fnp.zeros(width)  # post-ReLU variance (analytical)

        for layer_idx, w in enumerate(mlp.weights):
            if layer_idx == 0:
                # Layer 0: input ~ N(0, I), pre = x @ W → N(0, ||w_j||²)
                var_pre = fnp.sum(w * w, axis=0)
                var_pre = fnp.maximum(var_pre, 1e-12)
                sigma_pre = fnp.sqrt(var_pre)
                mu_pre = fnp.zeros(width)
                alpha = mu_pre / sigma_pre  # = 0 for all at layer 0
            else:
                # Propagate: μ_pre = W^T @ μ_post_prev
                mu_pre = w.T @ anal_mu_post
                var_pre = fnp.sum(w * w * anal_var_post[:, None], axis=0)
                var_pre = fnp.maximum(var_pre, 1e-12)
                sigma_pre = fnp.sqrt(var_pre)
                alpha = mu_pre / sigma_pre

            # Classify from α
            dead_mask = alpha < _DEAD_THRESH
            on_mask = alpha > _ON_THRESH
            kink_mask = (~dead_mask) & (~on_mask)

            dead_idx = fnp.nonzero(dead_mask)[0]
            alpha_rows.append(alpha)
            dead_indices.append(dead_idx)
            active_indices.append(fnp.nonzero(~dead_mask)[0])
            kink_indices.append(fnp.nonzero(kink_mask)[0])
            on_indices.append(fnp.nonzero(on_mask)[0])

            # Analytical post-ReLU stats for next layer's propagation
            phi = flops.stats.norm.pdf(alpha)
            Phi = flops.stats.norm.cdf(alpha)
            anal_mu_post = mu_pre * Phi + sigma_pre * phi
            # Var[ReLU(z)] = (σ² + μ²)Φ(α) + μσφ(α) - [E[ReLU(z)]]²
            anal_var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - anal_mu_post * anal_mu_post
            anal_var_post = fnp.maximum(anal_var_post, 1e-12)
            analytical_rows.append(anal_mu_post)

            # Dead-neuron correction: use analytical E[ReLU] instead of 0
            if len(dead_idx) > 0:
                dead_corrections.append(_scatter(anal_mu_post[dead_idx], dead_idx, width))
            else:
                dead_corrections.append(fnp.zeros(width))

        # =================================================================
        # Stage 2: Main MC with pre-computed Sobol points + fold at layer 30
        # =================================================================
        # Use shipped Sobol half-samples + antithetic pairing.
        half = fnp.array(self._sobol_points[: _MAIN_SAMPLES // 2, :width])
        x_main = fnp.concatenate([half, -half], axis=0)
        mc_rows = []
        mc_vars = []
        mc_var_of_mean = []

        prev_idx = None
        x_before_fold = None

        for layer_idx, w in enumerate(mlp.weights):
            idx = active_indices[layer_idx]
            kink_idx = kink_indices[layer_idx]
            on_idx = on_indices[layer_idx]
            k_active = len(idx)
            k_on = len(on_idx)

            if k_active == 0:
                mc_rows.append(fnp.zeros(width))
                mc_vars.append(fnp.zeros(width))
                mc_var_of_mean.append(fnp.ones(width) * 1e-12)
                x_main = fnp.zeros((_MAIN_SAMPLES, 0))
                prev_idx = idx
                x_before_fold = None
                continue

            # --- Fold at layer 30: skip on columns ---
            if layer_idx == 30 and k_on > 0 and prev_idx is not None:
                x_before_fold = x_main

                pilot_rows = max(2, min(_MAIN_SAMPLES, int(_MAIN_SAMPLES * 0.05)))

                on_probe_mask = alpha_rows[layer_idx][on_idx] <= fnp.float32(4.0)
                trusted_on_idx = on_idx[~on_probe_mask]
                probe_on_idx = on_idx[on_probe_mask]
                if len(probe_on_idx) > 0:
                    w_on_probe = w[prev_idx, :][:, probe_on_idx]
                    pre_on_pilot = x_main[:pilot_rows, :] @ w_on_probe
                    pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                    pilot_var = fnp.var(pre_on_pilot, axis=0)
                    pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                    keep_on = pilot_alpha > fnp.float32(3.0)
                    demoted_idx = probe_on_idx[~keep_on]
                    kept_probe_on_idx = probe_on_idx[keep_on]
                else:
                    demoted_idx = on_idx[:0]
                    kept_probe_on_idx = on_idx[:0]

                on_idx = fnp.sort(fnp.concatenate([trusted_on_idx, kept_probe_on_idx]))
                kink_idx = fnp.sort(fnp.concatenate([kink_idx, demoted_idx]))
                on_indices[layer_idx] = on_idx
                kink_indices[layer_idx] = kink_idx

                dead_idx = dead_indices[layer_idx]
                if len(dead_idx) > 0:
                    dead_probe_mask = alpha_rows[layer_idx][dead_idx] >= fnp.float32(-4.0)
                    trusted_dead_idx = dead_idx[~dead_probe_mask]
                    probe_dead_idx = dead_idx[dead_probe_mask]
                    if len(probe_dead_idx) > 0:
                        w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                        pre_dead_pilot = x_main[:pilot_rows, :] @ w_dead_probe
                        dead_pilot_mean = fnp.mean(pre_dead_pilot, axis=0)
                        dead_pilot_var = fnp.var(pre_dead_pilot, axis=0)
                        dead_pilot_alpha = dead_pilot_mean / fnp.sqrt(fnp.maximum(dead_pilot_var, 1e-12))

                        promote_dead = dead_pilot_alpha > fnp.float32(-2.5)
                        promoted_idx = probe_dead_idx[promote_dead]
                        remaining_probe_dead_idx = probe_dead_idx[~promote_dead]
                    else:
                        promoted_idx = dead_idx[:0]
                        remaining_probe_dead_idx = dead_idx[:0]

                    remaining_dead_idx = fnp.sort(fnp.concatenate([trusted_dead_idx, remaining_probe_dead_idx]))
                    if len(promoted_idx) > 0:
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                            analytical_rows[layer_idx][promoted_idx], promoted_idx, width
                        )
                    dead_indices[layer_idx] = remaining_dead_idx
                    kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_idx]))
                    idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = idx

                w_kink = w[prev_idx, :][:, kink_idx]
                x_kink = fnp.maximum(x_main @ w_kink, 0.0)

                kink_mean = fnp.mean(x_kink, axis=0)
                kink_var = fnp.var(x_kink, axis=0)
                mean_prev = fnp.mean(x_main, axis=0)
                var_prev_mc = fnp.var(x_main, axis=0)
                w_on = w[prev_idx, :][:, on_idx]
                on_mean = mean_prev @ w_on
                on_var = fnp.sum(w_on * w_on * var_prev_mc[:, None], axis=0)

                row = _scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width)
                mc_rows.append(row)

                full_var = _scatter(kink_var, kink_idx, width) + _scatter(on_var, on_idx, width)
                mc_vars.append(full_var)

                full_vom = _scatter(kink_var / _MAIN_SAMPLES, kink_idx, width) + _scatter(on_var / _MAIN_SAMPLES, on_idx, width)
                mc_var_of_mean.append(full_vom)

                x_main = x_kink
                prev_idx = kink_idx
                continue

            # --- Layer 31: fold from layer 30 ---
            if layer_idx == 31 and x_before_fold is not None and len(on_indices[30]) > 0:
                fold_on_idx = on_indices[30]
                fold_prev_idx = active_indices[29]

                this_kink_idx = kink_idx
                this_on_idx = on_idx

                w_from_kink = w[prev_idx, :][:, this_kink_idx]
                pre_from_kink = x_main @ w_from_kink

                w_fold_layer = mlp.weights[30]
                w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]
                w_this_from_on = w[fold_on_idx, :][:, this_kink_idx]
                W_folded = w_fold_on @ w_this_from_on
                pre_from_on = x_before_fold @ W_folded

                x_kink_this = fnp.maximum(pre_from_kink + pre_from_on, 0.0)
                kink_mean = fnp.mean(x_kink_this, axis=0)
                kink_var = fnp.var(x_kink_this, axis=0)

                prev_layer_mean = mc_rows[30]
                fold_active_idx = active_indices[30]
                w_to_on = w[fold_active_idx, :][:, this_on_idx]
                on_mean = prev_layer_mean[fold_active_idx] @ w_to_on
                prev_layer_var = mc_vars[30]
                on_var = fnp.sum(
                    w_to_on * w_to_on * prev_layer_var[fold_active_idx, None], axis=0
                )

                row = _scatter(kink_mean, this_kink_idx, width) + _scatter(on_mean, this_on_idx, width)
                mc_rows.append(row)

                full_var = _scatter(kink_var, this_kink_idx, width) + _scatter(on_var, this_on_idx, width)
                mc_vars.append(full_var)

                full_vom = _scatter(kink_var / _MAIN_SAMPLES, this_kink_idx, width) + _scatter(on_var / _MAIN_SAMPLES, this_on_idx, width)
                mc_var_of_mean.append(full_vom)

                prev_idx = idx
                x_before_fold = None
                continue

            # --- Layer 29: promote borderline dead neurons before the layer-30 fold ---
            if layer_idx == 29 and prev_idx is not None:
                dead_idx = dead_indices[layer_idx]
                if len(dead_idx) > 0:
                    pilot_rows = max(2, min(_MAIN_SAMPLES, int(_MAIN_SAMPLES * 0.05)))
                    dead_probe_mask = alpha_rows[layer_idx][dead_idx] >= fnp.float32(-4.0)
                    trusted_dead_idx = dead_idx[~dead_probe_mask]
                    probe_dead_idx = dead_idx[dead_probe_mask]

                    if len(probe_dead_idx) > 0:
                        w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                        pre_dead_pilot = x_main[:pilot_rows, :] @ w_dead_probe
                        dead_pilot_mean = fnp.mean(pre_dead_pilot, axis=0)
                        dead_pilot_var = fnp.var(pre_dead_pilot, axis=0)
                        dead_pilot_alpha = dead_pilot_mean / fnp.sqrt(fnp.maximum(dead_pilot_var, 1e-12))

                        promote_dead = dead_pilot_alpha > fnp.float32(-2.5)
                        promoted_idx = probe_dead_idx[promote_dead]
                        remaining_probe_dead_idx = probe_dead_idx[~promote_dead]
                    else:
                        promoted_idx = dead_idx[:0]
                        remaining_probe_dead_idx = dead_idx[:0]

                    remaining_dead_idx = fnp.sort(fnp.concatenate([trusted_dead_idx, remaining_probe_dead_idx]))
                    if len(promoted_idx) > 0:
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                            analytical_rows[layer_idx][promoted_idx], promoted_idx, width
                        )
                    dead_indices[layer_idx] = remaining_dead_idx
                    kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_idx]))
                    idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = idx

            # --- Standard active-set forward ---
            if prev_idx is None:
                w_active = w[:, idx]
            else:
                w_active = w[prev_idx, :][:, idx]

            x_main = fnp.maximum(x_main @ w_active, 0.0)
            active_mean = fnp.mean(x_main, axis=0)
            active_var = fnp.var(x_main, axis=0)

            mc_rows.append(_scatter(active_mean, idx, width))

            mc_vars.append(_scatter(active_var, idx, width))

            mc_var_of_mean.append(_scatter(active_var / _MAIN_SAMPLES, idx, width))

            prev_idx = idx

        # =================================================================
        # Stage 3: Output MC means directly (pure MC, no α-blend)
        # At N=12288 Sobol, MC is as accurate as the blend and avoids
        # diagonal-covariance bias that hurts on some MLPs.
        # =================================================================
        # Layer 0: analytical (exact for N(0,I) input)
        w0 = mlp.weights[0]
        var_pre_0 = fnp.sum(w0 * w0, axis=0)
        sigma_pre_0 = fnp.sqrt(fnp.maximum(var_pre_0, 1e-12))
        row0 = sigma_pre_0 * fnp.float32(0.3989422804014327)  # σ × φ(0)

        # Assemble output: MC rows + dead-neuron analytical corrections
        rows = [row0 + dead_corrections[0]] + [mc_rows[l] + dead_corrections[l] for l in range(1, len(mc_rows))]

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)
