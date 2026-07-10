"""Test empirical/analytical tail rotations after the 30,720-sample prefix.

This is an offline experiment for the current sqrt-allocation estimator. The
prefix is run normally. If the sqrt rule asks for extra samples, the extra Sobol
block is optionally rotated using an input-space Gram matrix built from:

- an analytical diagonal-gated input-to-layer-30 sensitivity proxy
- an empirical cross-covariance between prefix inputs and sampled layer-30 kink
  activations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import special

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from whestbench import MLP

import estimator as estmod


BUDGET = 272_000_000_000


def _normalized_gram(matrix: np.ndarray) -> np.ndarray:
    gram_matrix = matrix @ matrix.T
    trace_value = float(np.trace(gram_matrix))
    if trace_value <= 1e-30:
        return np.eye(matrix.shape[0], dtype=np.float64) / float(matrix.shape[0])
    return gram_matrix / trace_value


def _analytical_layer30_matrix(mlp: MLP, structure: dict, target_indices: np.ndarray) -> np.ndarray:
    alpha_rows = structure["alpha_rows"]
    effective_matrix = np.asarray(mlp.weights[0], dtype=np.float64).copy()
    gate_prob = special.ndtr(np.asarray(alpha_rows[0], dtype=np.float64))
    effective_matrix *= gate_prob[None, :]
    for layer_idx in range(1, 31):
        effective_matrix = effective_matrix @ np.asarray(mlp.weights[layer_idx], dtype=np.float64)
        gate_prob = special.ndtr(np.asarray(alpha_rows[layer_idx], dtype=np.float64))
        effective_matrix *= gate_prob[None, :]
    if target_indices.size == 0:
        return effective_matrix
    return effective_matrix[:, target_indices]


def _rotation_from_grams(analytical_matrix: np.ndarray, empirical_matrix: np.ndarray, blend: float) -> np.ndarray:
    analytical_gram = _normalized_gram(analytical_matrix)
    empirical_gram = _normalized_gram(empirical_matrix)
    blended_gram = (1.0 - blend) * analytical_gram + blend * empirical_gram
    eigenvalues, eigenvectors = np.linalg.eigh(blended_gram)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order].astype(np.float32)


def _run_block_with_layer30_cross(
    estimator: estmod.Estimator,
    mlp: MLP,
    structure: dict,
    sample_values,
    n_samples: int,
    refine: bool,
) -> tuple[list, float, dict, np.ndarray | None, np.ndarray]:
    width = mlp.width
    active_indices = list(structure["active_indices"])
    kink_indices = list(structure["kink_indices"])
    on_indices = list(structure["on_indices"])
    dead_indices = list(structure["dead_indices"])
    dead_corrections = list(structure["dead_corrections"])
    analytical_rows = structure["analytical_rows"]
    alpha_rows = structure["alpha_rows"]

    original_samples_np = np.asarray(sample_values, dtype=np.float32)
    layer30_cross = None
    layer30_kink_np = np.zeros(0, dtype=np.int64)

    mc_rows = []
    mc_vars = []
    prev_idx = None
    x_before_fold = None
    current_values = sample_values

    for layer_idx, weight_matrix in enumerate(mlp.weights):
        active_idx = active_indices[layer_idx]
        kink_idx = kink_indices[layer_idx]
        on_idx = on_indices[layer_idx]

        if len(active_idx) == 0:
            mc_rows.append(fnp.zeros(width))
            mc_vars.append(fnp.zeros(width))
            current_values = fnp.zeros((n_samples, 0))
            prev_idx = active_idx
            x_before_fold = None
            continue

        if layer_idx == 30 and len(on_idx) > 0 and prev_idx is not None:
            x_before_fold = current_values

            if refine:
                pilot_rows = max(2, min(n_samples, int(n_samples * estmod._PILOT_FRACTION)))
                on_probe_mask = alpha_rows[layer_idx][on_idx] <= fnp.float32(estmod._ON_PROBE_MAX)
                trusted_on_idx = on_idx[~on_probe_mask]
                probe_on_idx = on_idx[on_probe_mask]
                if len(probe_on_idx) > 0:
                    weight_on_probe = weight_matrix[prev_idx, :][:, probe_on_idx]
                    pre_on_pilot = current_values[:pilot_rows, :] @ weight_on_probe
                    pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                    pilot_var = fnp.var(pre_on_pilot, axis=0)
                    pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                    keep_on = pilot_alpha > fnp.float32(estmod._PILOT_ON_THRESH)
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
                    dead_probe_mask = alpha_rows[layer_idx][dead_idx] >= fnp.float32(estmod._DEAD_PROBE_MIN)
                    trusted_dead_idx = dead_idx[~dead_probe_mask]
                    probe_dead_idx = dead_idx[dead_probe_mask]
                    if len(probe_dead_idx) > 0:
                        weight_dead_probe = weight_matrix[prev_idx, :][:, probe_dead_idx]
                        pre_dead_pilot = current_values[:pilot_rows, :] @ weight_dead_probe
                        dead_mean = fnp.mean(pre_dead_pilot, axis=0)
                        dead_var = fnp.var(pre_dead_pilot, axis=0)
                        dead_alpha = dead_mean / fnp.sqrt(fnp.maximum(dead_var, 1e-12))

                        promote_dead = dead_alpha > fnp.float32(estmod._PILOT_DEAD_THRESH)
                        promoted_idx = probe_dead_idx[promote_dead]
                        remaining_probe_dead_idx = probe_dead_idx[~promote_dead]
                    else:
                        promoted_idx = dead_idx[:0]
                        remaining_probe_dead_idx = dead_idx[:0]

                    remaining_dead_idx = fnp.sort(fnp.concatenate([trusted_dead_idx, remaining_probe_dead_idx]))
                    if len(promoted_idx) > 0:
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - estmod._scatter(
                            analytical_rows[layer_idx][promoted_idx], promoted_idx, width
                        )
                    dead_indices[layer_idx] = remaining_dead_idx
                    kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_idx]))
                    active_idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = active_idx

            weight_kink = weight_matrix[prev_idx, :][:, kink_idx]
            kink_values = fnp.maximum(current_values @ weight_kink, 0.0)
            kink_mean = fnp.mean(kink_values, axis=0)
            kink_var = fnp.var(kink_values, axis=0)

            layer30_kink_np = np.asarray(kink_idx, dtype=np.int64)
            kink_values_np = np.asarray(kink_values, dtype=np.float32)
            layer30_cross = (original_samples_np.T @ kink_values_np) / float(n_samples)

            prev_mean = fnp.mean(current_values, axis=0)
            prev_var = fnp.var(current_values, axis=0)
            weight_on = weight_matrix[prev_idx, :][:, on_idx]
            on_mean = prev_mean @ weight_on
            on_var = fnp.sum(weight_on * weight_on * prev_var[:, None], axis=0)

            mc_rows.append(estmod._scatter(kink_mean, kink_idx, width) + estmod._scatter(on_mean, on_idx, width))
            mc_vars.append(estmod._scatter(kink_var, kink_idx, width) + estmod._scatter(on_var, on_idx, width))

            current_values = kink_values
            prev_idx = kink_idx
            continue

        if layer_idx == 31 and x_before_fold is not None and len(on_indices[30]) > 0:
            fold_on_idx = on_indices[30]
            fold_prev_idx = active_indices[29]

            weight_from_kink = weight_matrix[prev_idx, :][:, kink_idx]
            pre_from_kink = current_values @ weight_from_kink

            fold_layer = mlp.weights[30]
            weight_fold_on = fold_layer[fold_prev_idx, :][:, fold_on_idx]
            weight_this_from_on = weight_matrix[fold_on_idx, :][:, kink_idx]
            weight_folded = weight_fold_on @ weight_this_from_on
            pre_from_on = x_before_fold @ weight_folded

            kink_values = fnp.maximum(pre_from_kink + pre_from_on, 0.0)
            kink_mean = fnp.mean(kink_values, axis=0)
            kink_var = fnp.var(kink_values, axis=0)

            prev_layer_mean = mc_rows[30]
            fold_active_idx = active_indices[30]
            weight_to_on = weight_matrix[fold_active_idx, :][:, on_idx]
            on_mean = prev_layer_mean[fold_active_idx] @ weight_to_on
            prev_layer_var = mc_vars[30]
            on_var = fnp.sum(weight_to_on * weight_to_on * prev_layer_var[fold_active_idx, None], axis=0)

            mc_rows.append(estmod._scatter(kink_mean, kink_idx, width) + estmod._scatter(on_mean, on_idx, width))
            mc_vars.append(estmod._scatter(kink_var, kink_idx, width) + estmod._scatter(on_var, on_idx, width))
            prev_idx = active_idx
            x_before_fold = None
            continue

        if refine and layer_idx == 29 and prev_idx is not None:
            dead_idx = dead_indices[layer_idx]
            if len(dead_idx) > 0:
                pilot_rows = max(2, min(n_samples, int(n_samples * estmod._PILOT_FRACTION)))
                dead_probe_mask = alpha_rows[layer_idx][dead_idx] >= fnp.float32(estmod._DEAD_PROBE_MIN)
                trusted_dead_idx = dead_idx[~dead_probe_mask]
                probe_dead_idx = dead_idx[dead_probe_mask]

                if len(probe_dead_idx) > 0:
                    weight_dead_probe = weight_matrix[prev_idx, :][:, probe_dead_idx]
                    pre_dead_pilot = current_values[:pilot_rows, :] @ weight_dead_probe
                    dead_mean = fnp.mean(pre_dead_pilot, axis=0)
                    dead_var = fnp.var(pre_dead_pilot, axis=0)
                    dead_alpha = dead_mean / fnp.sqrt(fnp.maximum(dead_var, 1e-12))

                    promote_dead = dead_alpha > fnp.float32(estmod._PILOT_DEAD_THRESH)
                    promoted_idx = probe_dead_idx[promote_dead]
                    remaining_probe_dead_idx = probe_dead_idx[~promote_dead]
                else:
                    promoted_idx = dead_idx[:0]
                    remaining_probe_dead_idx = dead_idx[:0]

                remaining_dead_idx = fnp.sort(fnp.concatenate([trusted_dead_idx, remaining_probe_dead_idx]))
                if len(promoted_idx) > 0:
                    dead_corrections[layer_idx] = dead_corrections[layer_idx] - estmod._scatter(
                        analytical_rows[layer_idx][promoted_idx], promoted_idx, width
                    )
                dead_indices[layer_idx] = remaining_dead_idx
                kink_idx = fnp.sort(fnp.concatenate([kink_idx, promoted_idx]))
                active_idx = fnp.sort(fnp.concatenate([kink_idx, on_idx]))
                kink_indices[layer_idx] = kink_idx
                active_indices[layer_idx] = active_idx

        if prev_idx is None:
            active_weight = weight_matrix[:, active_idx]
        else:
            active_weight = weight_matrix[prev_idx, :][:, active_idx]

        current_values = fnp.maximum(current_values @ active_weight, 0.0)
        mc_rows.append(estmod._scatter(fnp.mean(current_values, axis=0), active_idx, width))
        mc_vars.append(estmod._scatter(fnp.var(current_values, axis=0), active_idx, width))
        prev_idx = active_idx

    first_weight = mlp.weights[0]
    sigma_0 = fnp.sqrt(fnp.maximum(fnp.sum(first_weight * first_weight, axis=0), 1e-12))
    row0 = sigma_0 * fnp.float32(0.3989422804014327)
    rows = [row0 + dead_corrections[0]] + [mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))]
    final_diff = fnp.abs(rows[-1] - analytical_rows[-1])
    final_anal_diff_mean = float(fnp.mean(final_diff))

    refined_structure = {
        "active_indices": active_indices,
        "kink_indices": kink_indices,
        "on_indices": on_indices,
        "dead_indices": dead_indices,
        "dead_corrections": dead_corrections,
        "analytical_rows": analytical_rows,
        "alpha_rows": alpha_rows,
    }
    return rows, final_anal_diff_mean, refined_structure, layer30_cross, layer30_kink_np


def _predict_baseline(estimator: estmod.Estimator, mlp: MLP):
    structure = estimator._initial_structure(mlp, mlp.width)
    target_samples = estmod._choose_samples(structure["final_var_mean"])
    base_samples = estmod._BASE_SAMPLES
    base_values = estimator._sample_block(0, base_samples // 2, mlp.width)
    base_rows, _, refined_structure = estimator._run_block(mlp, structure, base_values, base_samples, refine=True)
    if target_samples <= base_samples:
        return fnp.stack(base_rows, axis=0), target_samples
    extra_samples = target_samples - base_samples
    extra_values = estimator._sample_block(base_samples // 2, extra_samples // 2, mlp.width)
    extra_rows, _, _ = estimator._run_block(mlp, refined_structure, extra_values, extra_samples, refine=False)
    rows = [(base_row * base_samples + extra_row * extra_samples) / target_samples for base_row, extra_row in zip(base_rows, extra_rows)]
    return fnp.stack(rows, axis=0), target_samples


def _predict_hybrid(estimator: estmod.Estimator, mlp: MLP, blend: float):
    structure = estimator._initial_structure(mlp, mlp.width)
    target_samples = estmod._choose_samples(structure["final_var_mean"])
    base_samples = estmod._BASE_SAMPLES
    base_values = estimator._sample_block(0, base_samples // 2, mlp.width)
    base_rows, _, refined_structure, empirical_matrix, target_indices = _run_block_with_layer30_cross(
        estimator, mlp, structure, base_values, base_samples, refine=True
    )
    if target_samples <= base_samples or empirical_matrix is None or empirical_matrix.shape[1] == 0:
        return fnp.stack(base_rows, axis=0), target_samples

    analytical_matrix = _analytical_layer30_matrix(mlp, refined_structure, target_indices)
    rotation_basis = _rotation_from_grams(analytical_matrix, empirical_matrix, blend)
    extra_samples = target_samples - base_samples
    extra_values = estimator._sample_block(base_samples // 2, extra_samples // 2, mlp.width)
    extra_values = extra_values @ fnp.array(rotation_basis.T.astype(np.float32))
    extra_rows, _, _ = estimator._run_block(mlp, refined_structure, extra_values, extra_samples, refine=False)
    rows = [(base_row * base_samples + extra_row * extra_samples) / target_samples for base_row, extra_row in zip(base_rows, extra_rows)]
    return fnp.stack(rows, axis=0), target_samples


def _score_prediction(prediction: np.ndarray, gt_all: np.ndarray, flops_used: float) -> tuple[float, float, float]:
    err_all = prediction - gt_all
    final_mse = float(np.mean(err_all[-1] * err_all[-1]))
    all_mse = float(np.mean(err_all * err_all))
    adjusted = final_mse * max(0.1, flops_used / BUDGET)
    return adjusted, final_mse, all_mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mini")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    labels = ["baseline"] + [f"blend{blend:g}" for blend in args.blends]
    metrics = {label: {"adjusted": [], "final": [], "all": [], "flops": [], "samples": []} for label in labels}

    for row_index, row in enumerate(dataset):
        weights_np = np.asarray(row["weights"], dtype=np.float32)
        weights = [fnp.array(weights_np[layer]) for layer in range(weights_np.shape[0])]
        mlp = MLP(width=weights_np.shape[1], depth=weights_np.shape[0], weights=weights)
        gt_all = np.asarray(row["all_layer_means"], dtype=np.float64)

        estimator = estmod.Estimator()
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            prediction, target_samples = _predict_baseline(estimator, mlp)
        adjusted, final_mse, all_mse = _score_prediction(np.asarray(prediction, dtype=np.float64), gt_all, float(ctx.flops_used))
        metrics["baseline"]["adjusted"].append(adjusted)
        metrics["baseline"]["final"].append(final_mse)
        metrics["baseline"]["all"].append(all_mse)
        metrics["baseline"]["flops"].append(float(ctx.flops_used))
        metrics["baseline"]["samples"].append(target_samples)

        for blend in args.blends:
            estimator = estmod.Estimator()
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
                prediction, target_samples = _predict_hybrid(estimator, mlp, blend)
            adjusted, final_mse, all_mse = _score_prediction(np.asarray(prediction, dtype=np.float64), gt_all, float(ctx.flops_used))
            label = f"blend{blend:g}"
            metrics[label]["adjusted"].append(adjusted)
            metrics[label]["final"].append(final_mse)
            metrics[label]["all"].append(all_mse)
            metrics[label]["flops"].append(float(ctx.flops_used))
            metrics[label]["samples"].append(target_samples)

        if (row_index + 1) % 5 == 0:
            print(f"evaluated {row_index + 1}/{len(dataset)}", flush=True)

    print(f"\nEmpirical tail rotation on split={args.split} n={len(dataset)}")
    for label in labels:
        print(
            f"{label:<10} score={np.mean(metrics[label]['adjusted']):.9e} "
            f"final={np.mean(metrics[label]['final']):.9e} "
            f"all={np.mean(metrics[label]['all']):.9e} "
            f"flops={np.mean(metrics[label]['flops']):.6e} "
            f"samples={np.mean(metrics[label]['samples']):.1f}"
        )


if __name__ == "__main__":
    main()