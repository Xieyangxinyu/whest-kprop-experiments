"""Conservative local ablations at 16384 effective Sobol samples.

This keeps the estimator's sampled propagation path intact:
- non-dead neurons are sampled normally
- the existing layer-30 fold is preserved
- optional variants allocate extra final-layer samples to high-variance kink neurons
- optional variants use a late-layer pilot to refine always-on decisions
- optional variants use a late-layer pilot to refine dead and always-on decisions
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp, monte_carlo_layer_means


WIDTH = 256
DEPTH = 32
BUDGET = 272_000_000_000
N_SAMPLES = int(os.environ.get("N_SAMPLES", "16384"))
GT_SAMPLES = int(os.environ.get("GT_SAMPLES", "300000"))
SEEDS = tuple(int(seed) for seed in os.environ.get("SEEDS", "0,1,2,3,4").split(",") if seed)
# Comma-separated substrings; when set, only variants whose name contains one of them run.
VARIANT_FILTER = tuple(s for s in os.environ.get("VARIANT_FILTER", "").split(",") if s)


@dataclass(frozen=True)
class Variant:
    name: str
    dead_thresh: float = -3.0
    on_thresh: float = 2.5
    sample_mode: str = "normal"
    allocation_top_k: int = 0
    allocation_base_fraction: float = 1.0
    refine_layer29_dead: bool = False
    refine_dead_layers: tuple[int, ...] = ()
    refine_layer30_on: bool = False
    refine_layer30_dead: bool = False
    refine_layer31_on: bool = False
    refine_pilot_fraction: float = 0.25
    refine_on_thresh: float = 3.0
    refine_dead_thresh: float = -2.5
    refine_borderline_only: bool = False
    refine_on_probe_max: float = 4.0
    refine_dead_probe_min: float = -4.0
    dead_scale: float = 1.0
    final_analytical_blend: float = 0.0
    rotate_w0: bool = False


VARIANTS = (
    Variant("normal baseline"),
    Variant("normal fold-on 3.00", on_thresh=3.00),
    Variant("pilot refine l30 5%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00),
    Variant("pilot refine l30 10%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.10, refine_on_thresh=3.00),
    Variant("pilot refine l30 both 5%/-2.5", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50),
    Variant("pilot refine l30 both 5%/-2.0", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.00),
    Variant("pilot on borderline 4", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_borderline_only=True, refine_on_probe_max=4.00),
    Variant("pilot dead borderline -4", on_thresh=3.00, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot both borderline 4/-4", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("pilot both borderline 3.75/-3.75", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=3.75, refine_dead_probe_min=-3.75),
    Variant("pilot both borderline 3.5/-3.5", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=3.50, refine_dead_probe_min=-3.50),
    Variant("pilot l29 dead borderline -4", on_thresh=3.00, refine_layer29_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot l29+30 borderline 4/-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("pilot dead l28-30 border -4", on_thresh=3.00, refine_dead_layers=(28, 29), refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot dead l27-30 border -4", on_thresh=3.00, refine_dead_layers=(27, 28, 29), refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot dead l24-30 border -4", on_thresh=3.00, refine_dead_layers=(24, 25, 26, 27, 28, 29), refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot l28-30 + on30", on_thresh=3.00, refine_dead_layers=(28, 29), refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("pilot l27-30 + on30", on_thresh=3.00, refine_dead_layers=(27, 28, 29), refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 border 8% pilot", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.08, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 border 10% pilot", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.10, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 border + blend 0.05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_analytical_blend=0.05),
    Variant("l29+30 border + blend 0.10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_analytical_blend=0.10),
    Variant("l29+30 border + blend 0.20", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_analytical_blend=0.20),
    Variant("rot-w0 fold-on 3.00", on_thresh=3.00, rotate_w0=True),
    Variant("rot-w0 l29+30 borderline 4/-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotate_w0=True),
    Variant("pilot refine l31 5%/3.0", on_thresh=3.00, refine_layer31_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00),
    Variant("pilot refine l30+31 5%", on_thresh=3.00, refine_layer30_on=True, refine_layer31_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00),
    Variant("pilot refine l30 25%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.25, refine_on_thresh=3.00),
    Variant("pilot refine l30 25%/3.5", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.25, refine_on_thresh=3.50),
    Variant("pilot refine l30 50%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.50, refine_on_thresh=3.00),
)


def _chi_radius_mean(dim: int) -> float:
    return math.exp(0.5 * math.log(2.0) + math.lgamma((dim + 1) / 2.0) - math.lgamma(dim / 2.0))


def _scatter(values, idx, width):
    scatter_mat = fnp.eye(width, dtype=fnp.float32)[:, idx]
    return scatter_mat @ values


def _selected_final_update(
    mlp,
    sobol_points,
    start_pair: int,
    extra_pairs: int,
    active_indices,
    kink_indices,
    base_final_row,
    base_final_var,
    base_samples: int,
    top_k: int,
    width: int,
):
    if extra_pairs <= 0 or top_k <= 0:
        return base_final_row

    final_kink_idx = np.asarray(kink_indices[-1], dtype=np.int64)
    if final_kink_idx.size == 0:
        return base_final_row

    final_var = np.asarray(base_final_var)
    selected = final_kink_idx[np.argsort(final_var[final_kink_idx])[-top_k:]]
    selected.sort()
    selected_idx = fnp.array(selected.astype(np.int64))

    half = fnp.array(sobol_points[start_pair : start_pair + extra_pairs, :width])
    x = fnp.concatenate([half, -half], axis=0)

    prev_idx = None
    for layer_idx, w in enumerate(mlp.weights[:-1]):
        idx = active_indices[layer_idx]
        if len(idx) == 0:
            return base_final_row
        if prev_idx is None:
            w_active = w[:, idx]
        else:
            w_active = w[prev_idx, :][:, idx]
        x = fnp.maximum(x @ w_active, 0.0)
        prev_idx = idx

    w_selected = mlp.weights[-1][prev_idx, :][:, selected_idx]
    extra_final = fnp.maximum(x @ w_selected, 0.0)
    extra_mean = fnp.mean(extra_final, axis=0)
    extra_samples = extra_pairs * 2

    base_selected = base_final_row[selected_idx]
    combined = (base_selected * base_samples + extra_mean * extra_samples) / (base_samples + extra_samples)
    return base_final_row + _scatter(combined - base_selected, selected_idx, width)


def predict_variant(mlp, sobol_points, variant: Variant):
    width = mlp.width
    if variant.allocation_top_k > 0:
        n_pairs = max(1, int((N_SAMPLES // 2) * variant.allocation_base_fraction))
    else:
        n_pairs = N_SAMPLES // 2
    n_samples = n_pairs * 2
    extra_pairs = N_SAMPLES // 2 - n_pairs

    active_indices = []
    kink_indices = []
    on_indices = []
    dead_indices = []
    dead_corrections = []
    analytical_rows = []
    alpha_rows = []
    stats = {
        "pre30_dead_probed": 0,
        "pre30_dead_promoted": 0,
        "l30_on_probed": 0,
        "l30_on_demoted": 0,
        "l30_dead_probed": 0,
        "l30_dead_promoted": 0,
    }
    anal_mu_post = fnp.zeros(width)
    anal_var_post = fnp.zeros(width)

    for layer_idx, w in enumerate(mlp.weights):
        if layer_idx == 0:
            var_pre = fnp.sum(w * w, axis=0)
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            mu_pre = fnp.zeros(width)
            alpha = mu_pre / sigma_pre
        else:
            mu_pre = w.T @ anal_mu_post
            var_pre = fnp.sum(w * w * anal_var_post[:, None], axis=0)
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre

        dead_mask = alpha < variant.dead_thresh
        on_mask = alpha > variant.on_thresh
        kink_mask = (~dead_mask) & (~on_mask)

        dead_idx = fnp.nonzero(dead_mask)[0]
        alpha_rows.append(alpha)
        dead_indices.append(dead_idx)
        active_indices.append(fnp.nonzero(~dead_mask)[0])
        kink_indices.append(fnp.nonzero(kink_mask)[0])
        on_indices.append(fnp.nonzero(on_mask)[0])

        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        anal_mu_post = mu_pre * Phi + sigma_pre * phi
        anal_var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - anal_mu_post * anal_mu_post
        anal_var_post = fnp.maximum(anal_var_post, 1e-12)
        analytical_rows.append(anal_mu_post)

        if len(dead_idx) > 0:
            dead_corrections.append(_scatter(anal_mu_post[dead_idx], dead_idx, width) * variant.dead_scale)
        else:
            dead_corrections.append(fnp.zeros(width))

    half = fnp.array(sobol_points[:n_pairs, :width])
    if variant.rotate_w0:
        # Gaussian rotation invariance: align the first (highest-quality) Sobol
        # coordinates with W0's top left-singular directions. SVD runs outside
        # flopscope on raw NumPy; in a submission this cost would need tracking.
        u_rot, _, _ = np.linalg.svd(np.asarray(mlp.weights[0], dtype=np.float64))
        half = half @ fnp.array(u_rot.T.astype(np.float32))
    sample_scale = fnp.float32(1.0)
    if variant.sample_mode == "sphere":
        norm = fnp.sqrt(fnp.maximum(fnp.sum(half * half, axis=1, keepdims=True), 1e-12))
        half = half / norm
        sample_scale = fnp.float32(_chi_radius_mean(width))
    elif variant.sample_mode != "normal":
        raise ValueError(f"unknown sample_mode={variant.sample_mode!r}")

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
            x_main = fnp.zeros((n_samples, 0))
            prev_idx = idx
            x_before_fold = None
            continue

        if layer_idx == 30 and k_on > 0 and prev_idx is not None:
            x_before_fold = x_main
            if variant.refine_layer30_on or variant.refine_layer30_dead:
                pilot_rows = max(2, min(n_samples, int(n_samples * variant.refine_pilot_fraction)))

            if variant.refine_layer30_on:
                on_np = np.asarray(on_idx, dtype=np.int64)
                kink_np = np.asarray(kink_idx, dtype=np.int64)
                if variant.refine_borderline_only:
                    alpha_np = np.asarray(alpha_rows[layer_idx])
                    probe_mask = alpha_np[on_np] <= variant.refine_on_probe_max
                else:
                    probe_mask = np.ones(on_np.shape, dtype=bool)

                probe_on_np = on_np[probe_mask]
                trusted_on_np = on_np[~probe_mask]
                stats["l30_on_probed"] = int(probe_on_np.size)

                if probe_on_np.size > 0:
                    probe_on_idx = fnp.array(probe_on_np.astype(np.int64))
                    w_on_probe = w[prev_idx, :][:, probe_on_idx]
                    pre_on_pilot = x_main[:pilot_rows, :] @ w_on_probe
                    pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                    pilot_var = fnp.var(pre_on_pilot, axis=0)
                    pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                    keep_probe_on = np.asarray(pilot_alpha) > variant.refine_on_thresh
                    demoted_np = probe_on_np[~keep_probe_on]
                    kept_probe_on_np = probe_on_np[keep_probe_on]
                else:
                    demoted_np = np.zeros(0, dtype=np.int64)
                    kept_probe_on_np = np.zeros(0, dtype=np.int64)

                stats["l30_on_demoted"] = int(demoted_np.size)
                on_idx = fnp.array(np.sort(np.concatenate([trusted_on_np, kept_probe_on_np])).astype(np.int64))
                kink_idx = fnp.array(np.sort(np.concatenate([kink_np, demoted_np])).astype(np.int64))
                on_indices[layer_idx] = on_idx
                kink_indices[layer_idx] = kink_idx
                k_on = len(on_idx)

            if variant.refine_layer30_dead:
                dead_idx = dead_indices[layer_idx]
                if len(dead_idx) > 0:
                    dead_np = np.asarray(dead_idx, dtype=np.int64)
                    kink_np = np.asarray(kink_idx, dtype=np.int64)
                    on_np = np.asarray(on_idx, dtype=np.int64)
                    if variant.refine_borderline_only:
                        alpha_np = np.asarray(alpha_rows[layer_idx])
                        probe_mask = alpha_np[dead_np] >= variant.refine_dead_probe_min
                    else:
                        probe_mask = np.ones(dead_np.shape, dtype=bool)

                    probe_dead_np = dead_np[probe_mask]
                    trusted_dead_np = dead_np[~probe_mask]
                    stats["l30_dead_probed"] = int(probe_dead_np.size)

                    if probe_dead_np.size > 0:
                        probe_dead_idx = fnp.array(probe_dead_np.astype(np.int64))
                        w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                        pre_dead_pilot = x_main[:pilot_rows, :] @ w_dead_probe
                        pilot_mean = fnp.mean(pre_dead_pilot, axis=0)
                        pilot_var = fnp.var(pre_dead_pilot, axis=0)
                        pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                        promote_dead = np.asarray(pilot_alpha) > variant.refine_dead_thresh
                        promoted_np = probe_dead_np[promote_dead]
                        remaining_probe_dead_np = probe_dead_np[~promote_dead]
                    else:
                        promoted_np = np.zeros(0, dtype=np.int64)
                        remaining_probe_dead_np = np.zeros(0, dtype=np.int64)

                    remaining_dead_np = np.sort(np.concatenate([trusted_dead_np, remaining_probe_dead_np]))
                    stats["l30_dead_promoted"] = int(promoted_np.size)
                    if promoted_np.size > 0:
                        promoted_idx = fnp.array(promoted_np.astype(np.int64))
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                            analytical_rows[layer_idx][promoted_idx] * variant.dead_scale,
                            promoted_idx,
                            width,
                        )
                    dead_idx = fnp.array(remaining_dead_np.astype(np.int64))
                    kink_idx = fnp.array(np.sort(np.concatenate([kink_np, promoted_np])).astype(np.int64))
                    idx = fnp.array(np.sort(np.concatenate([np.asarray(kink_idx, dtype=np.int64), on_np])).astype(np.int64))
                    dead_indices[layer_idx] = dead_idx
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = idx
                    k_active = len(idx)

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

            full_vom = _scatter(kink_var / n_samples, kink_idx, width) + _scatter(on_var / n_samples, on_idx, width)
            mc_var_of_mean.append(full_vom)

            x_main = x_kink
            prev_idx = kink_idx
            continue

        if layer_idx == 31 and x_before_fold is not None and len(on_indices[30]) > 0:
            fold_on_idx = on_indices[30]
            fold_prev_idx = active_indices[29]

            this_kink_idx = kink_idx
            this_on_idx = on_idx

            if variant.refine_layer31_on and len(this_on_idx) > 0:
                pilot_rows = max(2, min(n_samples, int(n_samples * variant.refine_pilot_fraction)))
                w_fold_layer = mlp.weights[30]
                w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]

                w_from_kink_on = w[prev_idx, :][:, this_on_idx]
                pre_from_kink_on = x_main[:pilot_rows, :] @ w_from_kink_on
                w_this_from_on_on = w[fold_on_idx, :][:, this_on_idx]
                w_folded_on = w_fold_on @ w_this_from_on_on
                pre_from_on_on = x_before_fold[:pilot_rows, :] @ w_folded_on
                pre_on_pilot = pre_from_kink_on + pre_from_on_on

                pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                pilot_var = fnp.var(pre_on_pilot, axis=0)
                pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                on_np = np.asarray(this_on_idx, dtype=np.int64)
                kink_np = np.asarray(this_kink_idx, dtype=np.int64)
                keep_on = np.asarray(pilot_alpha) > variant.refine_on_thresh
                demoted_np = on_np[~keep_on]
                this_on_idx = fnp.array(on_np[keep_on].astype(np.int64))
                this_kink_idx = fnp.array(np.sort(np.concatenate([kink_np, demoted_np])).astype(np.int64))
                on_indices[layer_idx] = this_on_idx
                kink_indices[layer_idx] = this_kink_idx

            w_from_kink = w[prev_idx, :][:, this_kink_idx]
            pre_from_kink = x_main @ w_from_kink

            w_fold_layer = mlp.weights[30]
            w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]
            w_this_from_on = w[fold_on_idx, :][:, this_kink_idx]
            w_folded = w_fold_on @ w_this_from_on
            pre_from_on = x_before_fold @ w_folded

            x_kink_this = fnp.maximum(pre_from_kink + pre_from_on, 0.0)
            kink_mean = fnp.mean(x_kink_this, axis=0)
            kink_var = fnp.var(x_kink_this, axis=0)

            prev_layer_mean = mc_rows[30]
            fold_active_idx = active_indices[30]
            w_to_on = w[fold_active_idx, :][:, this_on_idx]
            on_mean = prev_layer_mean[fold_active_idx] @ w_to_on
            prev_layer_var = mc_vars[30]
            on_var = fnp.sum(w_to_on * w_to_on * prev_layer_var[fold_active_idx, None], axis=0)

            row = _scatter(kink_mean, this_kink_idx, width) + _scatter(on_mean, this_on_idx, width)
            mc_rows.append(row)

            full_var = _scatter(kink_var, this_kink_idx, width) + _scatter(on_var, this_on_idx, width)
            mc_vars.append(full_var)

            full_vom = _scatter(kink_var / n_samples, this_kink_idx, width) + _scatter(on_var / n_samples, this_on_idx, width)
            mc_var_of_mean.append(full_vom)

            prev_idx = idx
            x_before_fold = None
            continue

        if layer_idx < 30 and ((layer_idx == 29 and variant.refine_layer29_dead) or layer_idx in variant.refine_dead_layers) and prev_idx is not None:
            dead_idx = dead_indices[layer_idx]
            if len(dead_idx) > 0:
                pilot_rows = max(2, min(n_samples, int(n_samples * variant.refine_pilot_fraction)))
                dead_np = np.asarray(dead_idx, dtype=np.int64)
                kink_np = np.asarray(kink_idx, dtype=np.int64)
                on_np = np.asarray(on_idx, dtype=np.int64)
                if variant.refine_borderline_only:
                    alpha_np = np.asarray(alpha_rows[layer_idx])
                    probe_mask = alpha_np[dead_np] >= variant.refine_dead_probe_min
                else:
                    probe_mask = np.ones(dead_np.shape, dtype=bool)

                probe_dead_np = dead_np[probe_mask]
                trusted_dead_np = dead_np[~probe_mask]
                stats["pre30_dead_probed"] += int(probe_dead_np.size)

                if probe_dead_np.size > 0:
                    probe_dead_idx = fnp.array(probe_dead_np.astype(np.int64))
                    w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                    pre_dead_pilot = x_main[:pilot_rows, :] @ w_dead_probe
                    pilot_mean = fnp.mean(pre_dead_pilot, axis=0)
                    pilot_var = fnp.var(pre_dead_pilot, axis=0)
                    pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                    promote_dead = np.asarray(pilot_alpha) > variant.refine_dead_thresh
                    promoted_np = probe_dead_np[promote_dead]
                    remaining_probe_dead_np = probe_dead_np[~promote_dead]
                else:
                    promoted_np = np.zeros(0, dtype=np.int64)
                    remaining_probe_dead_np = np.zeros(0, dtype=np.int64)

                remaining_dead_np = np.sort(np.concatenate([trusted_dead_np, remaining_probe_dead_np]))
                stats["pre30_dead_promoted"] += int(promoted_np.size)
                if promoted_np.size > 0:
                    promoted_idx = fnp.array(promoted_np.astype(np.int64))
                    dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                        analytical_rows[layer_idx][promoted_idx] * variant.dead_scale,
                        promoted_idx,
                        width,
                    )
                dead_idx = fnp.array(remaining_dead_np.astype(np.int64))
                kink_idx = fnp.array(np.sort(np.concatenate([kink_np, promoted_np])).astype(np.int64))
                idx = fnp.array(np.sort(np.concatenate([np.asarray(kink_idx, dtype=np.int64), on_np])).astype(np.int64))
                dead_indices[layer_idx] = dead_idx
                kink_indices[layer_idx] = kink_idx
                active_indices[layer_idx] = idx
                k_active = len(idx)

        if prev_idx is None:
            w_active = w[:, idx]
        else:
            w_active = w[prev_idx, :][:, idx]

        x_main = fnp.maximum(x_main @ w_active, 0.0)
        active_mean = fnp.mean(x_main, axis=0)
        active_var = fnp.var(x_main, axis=0)

        mc_rows.append(_scatter(active_mean, idx, width))
        mc_vars.append(_scatter(active_var, idx, width))
        mc_var_of_mean.append(_scatter(active_var / n_samples, idx, width))

        prev_idx = idx

    w0 = mlp.weights[0]
    var_pre_0 = fnp.sum(w0 * w0, axis=0)
    sigma_pre_0 = fnp.sqrt(fnp.maximum(var_pre_0, 1e-12))
    row0 = sigma_pre_0 * fnp.float32(0.3989422804014327)

    if variant.sample_mode == "sphere":
        rows = [row0 + dead_corrections[0]] + [sample_scale * mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))]
    else:
        rows = [row0 + dead_corrections[0]] + [mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))]

    if variant.allocation_top_k > 0:
        rows[-1] = _selected_final_update(
            mlp,
            sobol_points,
            n_pairs,
            extra_pairs,
            active_indices,
            kink_indices,
            rows[-1],
            mc_vars[-1],
            n_samples,
            variant.allocation_top_k,
            width,
        )

    if variant.final_analytical_blend > 0.0:
        blend = fnp.float32(variant.final_analytical_blend)
        rows[-1] = (1.0 - blend) * rows[-1] + blend * analytical_rows[-1]

    predict_variant.last_stats = stats
    return fnp.stack(rows, axis=0)


predict_variant.last_stats = {}


def evaluate():
    sobol_points = np.load(Path(__file__).parent / "sobol_points.npz")["points"]
    if sobol_points.shape[0] < N_SAMPLES // 2:
        raise SystemExit(f"sobol_points.npz only has {sobol_points.shape[0]} half-samples")

    print(f"Conservative {N_SAMPLES}-sample ablation on seeds={SEEDS}, GT={GT_SAMPLES:,}")
    print(f"{'variant':<32} {'final_mse':>12} {'score':>12} {'flops':>12} {'util%':>7} {'all_mse':>12} {'pre30d':>7} {'on->k':>7} {'l30d':>6} {'probe':>9}")
    print("-" * 132)

    cases = []
    for seed in SEEDS:
        mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=seed)
        gt = np.asarray(monte_carlo_layer_means(mlp, GT_SAMPLES, seed=seed + 10_000))
        cases.append((seed, mlp, gt))

    variants = VARIANTS
    if VARIANT_FILTER:
        variants = tuple(v for v in VARIANTS if any(s in v.name for s in VARIANT_FILTER))

    results = []
    for variant in variants:
        final_mses = []
        all_mses = []
        flops_used = []
        pre30_dead_promoted = []
        on_demoted = []
        dead_promoted = []
        probed_total = []

        for _seed, mlp, gt in cases:
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
                pred = np.asarray(predict_variant(mlp, sobol_points, variant))
            stats = predict_variant.last_stats

            err = pred - gt
            final_mses.append(float(np.mean(err[-1] * err[-1])))
            all_mses.append(float(np.mean(err * err)))
            flops_used.append(ctx.flops_used)
            pre30_dead_promoted.append(stats.get("pre30_dead_promoted", 0))
            on_demoted.append(stats.get("l30_on_demoted", 0))
            dead_promoted.append(stats.get("l30_dead_promoted", 0))
            probed_total.append(stats.get("pre30_dead_probed", 0) + stats.get("l30_on_probed", 0) + stats.get("l30_dead_probed", 0))

        final_mse = float(np.mean(final_mses))
        all_mse = float(np.mean(all_mses))
        flops_mean = float(np.mean(flops_used))
        util = flops_mean / BUDGET
        score = final_mse * max(0.1, util)
        avg_pre30_dead_promoted = float(np.mean(pre30_dead_promoted))
        avg_on_demoted = float(np.mean(on_demoted))
        avg_dead_promoted = float(np.mean(dead_promoted))
        avg_probed_total = float(np.mean(probed_total))
        results.append((score, variant.name, final_mse, flops_mean, util, all_mse, avg_pre30_dead_promoted, avg_on_demoted, avg_dead_promoted, avg_probed_total))
        print(f"{variant.name:<32} {final_mse:12.3e} {score:12.3e} {flops_mean:12.2e} {util * 100:6.1f}% {all_mse:12.3e} {avg_pre30_dead_promoted:7.1f} {avg_on_demoted:7.1f} {avg_dead_promoted:6.1f} {avg_probed_total:9.1f}")

    print("\nRanked by adjusted score:")
    for rank, (score, name, final_mse, flops_mean, util, all_mse, avg_pre30_dead_promoted, avg_on_demoted, avg_dead_promoted, avg_probed_total) in enumerate(sorted(results), start=1):
        print(f"{rank:2d}. {name:<32} score={score:.3e} final_mse={final_mse:.3e} flops={flops_mean:.2e} util={util * 100:.1f}% all_mse={all_mse:.3e} pre30d={avg_pre30_dead_promoted:.1f} on->k={avg_on_demoted:.1f} l30d={avg_dead_promoted:.1f} probed={avg_probed_total:.1f}")


if __name__ == "__main__":
    evaluate()
