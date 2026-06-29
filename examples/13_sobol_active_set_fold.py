"""Sobol QMC Active-Set Estimator with Layer Fold (Submission #313687).

This is the best-scoring estimator (~3.3e-07 adjusted_final_layer_score).
It combines four key ideas:

1. **Analytical classification** — propagate diagonal mean/variance through
   ReLU layers to compute α = μ/σ per neuron. Classify as dead (α < -2.5),
   always-on (α > 2.5), or kink. No pilot pass needed (saves ~2.7B FLOPs).

2. **Sobol QMC inputs** — pre-generate 8192 Sobol N(0,1) points offline
   (scrambled, ndtri-transformed), ship as `sobol_points.npz`. Better space-
   filling than pseudo-random gives lower MC variance at same N.

3. **Antithetic pairing** — concatenate half and -half to get N=16384 effective
   samples. Guarantees the sample mean of inputs is exactly 0.

4. **Layer fold at layers 30–31** — for always-on neurons, compute
   E[x] = W @ mean_prev (exact, zero variance) instead of per-sample ReLU.
   At layer 31, fold the on-neuron contribution from layer 30 back through
   the saved pre-fold samples to get the correct kink pre-activations.

5. **Exact layer-0 formula** — E[ReLU(x^T w)] = ||w|| / sqrt(2π) when x ~ N(0,I).

Score: adjusted_final_layer_score ≈ 3.31e-07 on the grader.
"""

from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_MAIN_SAMPLES = 16384  # 8192 Sobol half-samples × 2 (antithetic)
_DEAD_THRESH = -2.5
_ON_THRESH = 2.5


def _scatter(values, idx, width):
    """Functionally place values at idx into a zero vector of length width.

    fnp arrays are immutable — no indexed assignment. This uses:
      eye(width)[:, idx]  →  (width, k) selector matrix (READ = free)
      selector @ values   →  (width,) result
    """
    return fnp.eye(width, dtype=fnp.float32)[:, idx] @ values


class Estimator(BaseEstimator):
    """Best submission: Sobol QMC + analytical classification + fold."""

    def __init__(self) -> None:
        self._sobol_points = None

    def setup(self, ctx: SetupContext) -> None:
        fnp.random.default_rng(ctx.seed)
        sobol_path = Path(ctx.submission_dir) / "sobol_points.npz"
        data = fnp.load(str(sobol_path))
        self._sobol_points = data["points"]  # (8192, 256) float32

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width

        # Lazy-load for local testing (setup() not called)
        if self._sobol_points is None:
            data = fnp.load(str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
            self._sobol_points = data["points"]

        # =================================================================
        # Stage 1: Analytical classification via diagonal propagation
        # =================================================================
        active_indices = []
        kink_indices = []
        on_indices = []
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

            # Classify
            dead_mask = alpha < _DEAD_THRESH
            on_mask = alpha > _ON_THRESH
            active_indices.append(fnp.nonzero(~dead_mask)[0])
            kink_indices.append(fnp.nonzero((~dead_mask) & (~on_mask))[0])
            on_indices.append(fnp.nonzero(on_mask)[0])

            # Post-ReLU moments for next layer
            phi = flops.stats.norm.pdf(alpha)
            Phi = flops.stats.norm.cdf(alpha)
            mu_post = mu_pre * Phi + sigma_pre * phi
            var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - mu_post * mu_post
            var_post = fnp.maximum(var_post, 1e-12)

        # =================================================================
        # Stage 2: Sobol MC through active subnetwork + fold at layers 30–31
        # =================================================================
        half = fnp.array(self._sobol_points[:, :width])
        x = fnp.concatenate([half, -half], axis=0)  # (16384, 256)

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

            # --- Fold at layer 30: on-neurons computed analytically ---
            if layer_idx == 30 and len(on_idx) > 0 and prev_idx is not None:
                x_before_fold = x  # save for layer 31's folded contribution

                # Kink neurons: standard per-sample ReLU
                w_kink = w[prev_idx, :][:, kink_idx]
                x_kink = fnp.maximum(x @ w_kink, 0.0)
                kink_mean = fnp.mean(x_kink, axis=0)
                kink_var = fnp.var(x_kink, axis=0)

                # On-neurons: E[ReLU(z)] = E[z] = W^T @ mean_prev (exact)
                mean_prev = fnp.mean(x, axis=0)
                var_prev = fnp.var(x, axis=0)
                w_on = w[prev_idx, :][:, on_idx]
                on_mean = mean_prev @ w_on
                on_var = fnp.sum(w_on * w_on * var_prev[:, None], axis=0)

                mc_rows.append(_scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width))
                mc_vars.append(_scatter(kink_var, kink_idx, width) + _scatter(on_var, on_idx, width))

                x = x_kink  # continue with kink samples only
                prev_idx = kink_idx
                continue

            # --- Layer 31: fold on-neurons from layer 30 back in ---
            if layer_idx == 31 and x_before_fold is not None and len(on_indices[30]) > 0:
                fold_on_idx = on_indices[30]
                fold_prev_idx = active_indices[29]

                # Kink contribution from layer-30 kink samples
                w_from_kink = w[prev_idx, :][:, kink_idx]
                pre_from_kink = x @ w_from_kink

                # On contribution: fold W_30[prev,on] @ W_31[on,kink] into one matmul
                w_fold_layer = mlp.weights[30]
                w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]
                w_this_from_on = w[fold_on_idx, :][:, kink_idx]
                W_folded = w_fold_on @ w_this_from_on
                pre_from_on = x_before_fold @ W_folded

                # Combined pre-activation for kink neurons at layer 31
                x_kink = fnp.maximum(pre_from_kink + pre_from_on, 0.0)
                kink_mean = fnp.mean(x_kink, axis=0)
                kink_var = fnp.var(x_kink, axis=0)

                # On-neurons at layer 31: use layer-30 mean directly
                prev_layer_mean = mc_rows[30]
                fold_active_idx = active_indices[30]
                w_to_on = w[fold_active_idx, :][:, on_idx]
                on_mean = prev_layer_mean[fold_active_idx] @ w_to_on

                mc_rows.append(_scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width))
                mc_vars.append(_scatter(kink_var, kink_idx, width))
                prev_idx = idx
                x_before_fold = None
                continue

            # --- Standard active-set forward ---
            if prev_idx is None:
                w_active = w[:, idx]
            else:
                w_active = w[prev_idx, :][:, idx]

            x = fnp.maximum(x @ w_active, 0.0)
            mc_rows.append(_scatter(fnp.mean(x, axis=0), idx, width))
            mc_vars.append(_scatter(fnp.var(x, axis=0), idx, width))
            prev_idx = idx

        # =================================================================
        # Stage 3: Assemble output — exact layer 0, MC for layers 1–31
        # =================================================================
        w0 = mlp.weights[0]
        sigma_0 = fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0), 1e-12))
        row0 = sigma_0 * fnp.float32(0.3989422804014327)  # ||w_j|| × φ(0)

        rows = [row0] + mc_rows[1:]
        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)
