"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Stage 1 of the WhestBench ladder: just `flopscope` and the local engine. No CLI
knowledge required. Once `predict()` returns something interesting, climb to
Stage 2: `whest validate --estimator estimator.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator, SetupContext

_PILOT_SAMPLES = int(os.environ.get("WHEST_ACTIVE_PILOT_SAMPLES", "512"))
_MAIN_SAMPLES = int(os.environ.get("WHEST_ACTIVE_MAIN_SAMPLES", "12288"))
_FIRE_THRESHOLD = float(os.environ.get("WHEST_ACTIVE_FIRE_THRESHOLD", "0.02"))
_FIRE_THRESHOLD_GROWTH = float(os.environ.get("WHEST_FIRE_THRESHOLD_GROWTH", "0.0"))
_USE_ANTITHETIC = os.environ.get("WHEST_ACTIVE_USE_ANTITHETIC", "1") != "0"
_NLN_FALLBACK_BLEND = float(os.environ.get("WHEST_NLN_FALLBACK_BLEND", "0.15"))
_NLN_KEEP_FRACTION = float(os.environ.get("WHEST_NLN_KEEP_FRACTION", "0.15"))
_NLN_MIN_POSITIVES = float(os.environ.get("WHEST_NLN_MIN_POSITIVES", "8"))
_NLN_LOGVAR_CAP = float(os.environ.get("WHEST_NLN_LOGVAR_CAP", "4.0"))
_COV_CORR_THRESHOLD = float(os.environ.get("WHEST_SPARSE_COV_CORR_THRESHOLD", "0.06"))
_COV_MIN_DEGREE = float(os.environ.get("WHEST_SPARSE_COV_MIN_DEGREE", "8"))
_COV_CORE_DEGREE = float(os.environ.get("WHEST_SPARSE_COV_CORE_DEGREE", "16"))
_USE_SPARSE_COV = os.environ.get("WHEST_USE_SPARSE_COV", "1") != "0"
_MOMENT_MATCH_INPUTS = os.environ.get("WHEST_MOMENT_MATCH_INPUTS", "0") != "0"
_REUSE_PILOT_SAMPLES = os.environ.get("WHEST_REUSE_PILOT_SAMPLES", "1") != "0"
_PERIPHERAL_SHRINK = float(os.environ.get("WHEST_PERIPHERAL_SHRINK", "0.0"))
_BLOCK_RESTART_LAYER = int(os.environ.get("WHEST_BLOCK_RESTART_LAYER", "0"))
_BLOCK_RESTART_MODE = os.environ.get("WHEST_BLOCK_RESTART_MODE", "lognormal")
_SIMPLIFY_IDENTITY = os.environ.get("WHEST_SIMPLIFY_IDENTITY", "0") != "0"
_IDENTITY_MAX_DEGREE = float(os.environ.get("WHEST_IDENTITY_MAX_DEGREE", "2"))
_IDENTITY_MEAN_FRACTION = float(os.environ.get("WHEST_IDENTITY_MEAN_FRACTION", "0.5"))
_BLOCK_CLASS_START_LAYER = int(os.environ.get("WHEST_BLOCK_CLASS_START_LAYER", "9"))
_ADAPTIVE_EXTRA_SAMPLES = int(os.environ.get("WHEST_ADAPTIVE_EXTRA_SAMPLES", "0"))
_IMPORTANCE_SAMPLES = int(os.environ.get("WHEST_IMPORTANCE_SAMPLES", "0"))
_IMPORTANCE_ALPHA = float(os.environ.get("WHEST_IMPORTANCE_ALPHA", "0.75"))
_IMPORTANCE_WEIGHT_CLIP = float(os.environ.get("WHEST_IMPORTANCE_WEIGHT_CLIP", "8.0"))
_IMPORTANCE_DIRECTION_MODE = os.environ.get("WHEST_IMPORTANCE_DIRECTION_MODE", "gradient")
_USE_WHITEBOX_LOOKAHEAD = os.environ.get("WHEST_WHITEBOX_LOOKAHEAD", "0") != "0"
_LOOKAHEAD_FALLBACK_BLEND = float(os.environ.get("WHEST_LOOKAHEAD_FALLBACK_BLEND", "0.25"))
_USE_MAGNITUDE_CORE = os.environ.get("WHEST_USE_MAGNITUDE_CORE", "0") != "0"
_MAGNITUDE_CORE_FRACTION = float(os.environ.get("WHEST_MAGNITUDE_CORE_FRACTION", "0.75"))
_INV_SQRT_TWO_PI = 0.3989422804014327
_RELU_STANDARD_NORMAL_VAR = 0.5 - _INV_SQRT_TWO_PI * _INV_SQRT_TWO_PI


def _normal_samples(rng, n_samples: int, width: int) -> fnp.ndarray:
    if _USE_ANTITHETIC:
        n_pairs = max(1, (n_samples + 1) // 2)
        half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
        half = _maybe_moment_match_antithetic_half(half)
        samples = fnp.concatenate([half, -half], axis=0)
    else:
        samples = fnp.array(rng.standard_normal((n_samples, width)).astype(fnp.float32))

    if _MOMENT_MATCH_INPUTS and not _USE_ANTITHETIC:
        samples = samples - fnp.mean(samples, axis=0)
        second_moment = fnp.maximum(fnp.mean(samples * samples, axis=0), 1e-12)
        samples = samples / fnp.sqrt(second_moment)
    return samples


def _maybe_moment_match_antithetic_half(half: fnp.ndarray) -> fnp.ndarray:
    if not _MOMENT_MATCH_INPUTS:
        return half
    second_moment = fnp.maximum(fnp.mean(half * half, axis=0), 1e-12)
    return half / fnp.sqrt(second_moment)


def _first_layer_activations(rng, n_samples: int, w: fnp.ndarray, width: int) -> fnp.ndarray:
    activations, _ = _first_layer_activations_and_inputs(rng, n_samples, w, width)
    return activations


def _first_layer_activations_and_inputs(rng, n_samples: int, w: fnp.ndarray, width: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    if _USE_ANTITHETIC:
        n_pairs = max(1, (n_samples + 1) // 2)
        half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
        half = _maybe_moment_match_antithetic_half(half)
        z = half @ w
        samples = fnp.concatenate([half, -half], axis=0)
        activations = fnp.concatenate([fnp.maximum(z, 0.0), fnp.maximum(-z, 0.0)], axis=0)
        return activations, samples

    samples = _normal_samples(rng, n_samples, width)
    return fnp.maximum(samples @ w, 0.0), samples


def _actual_sample_count(n_samples: int) -> int:
    if _USE_ANTITHETIC:
        return 2 * max(1, (n_samples + 1) // 2)
    return n_samples


def _regular_main_samples() -> int:
    if _IMPORTANCE_SAMPLES <= 0:
        return _MAIN_SAMPLES
    return max(1, _MAIN_SAMPLES - _IMPORTANCE_SAMPLES)


def _layer_fire_threshold(layer_idx: int, depth: int) -> float:
    if depth <= 1:
        return _FIRE_THRESHOLD
    depth_fraction = layer_idx / float(depth - 1)
    return _FIRE_THRESHOLD * (1.0 + _FIRE_THRESHOLD_GROWTH * depth_fraction * depth_fraction)


def _use_block_classification(layer_idx: int) -> bool:
    return layer_idx + 1 >= _BLOCK_CLASS_START_LAYER


def _positive_slab_lognormal_mean(x: fnp.ndarray, pilot_mean: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
    fallback_mean, nln_keep, _, _, _ = _positive_slab_lognormal_stats(x, pilot_mean)
    return fallback_mean, nln_keep


def _positive_slab_lognormal_stats(
    x: fnp.ndarray, pilot_mean: fnp.ndarray
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    positive = x > 0.0
    positive_count = fnp.sum(positive, axis=0)
    sample_count = float(x.shape[0])

    log_x = fnp.where(positive, fnp.log(fnp.maximum(x, 1e-30)), 0.0)
    count_for_logs = fnp.maximum(positive_count, 1.0)
    log_mean = fnp.sum(log_x, axis=0) / count_for_logs
    log_centered = fnp.where(positive, log_x - log_mean, 0.0)
    log_var = fnp.sum(log_centered * log_centered, axis=0) / fnp.maximum(positive_count - 1.0, 1.0)
    log_var = fnp.minimum(log_var, _NLN_LOGVAR_CAP)

    firing_rate = (positive_count + 0.5) / (sample_count + 1.0)
    nln_mean = firing_rate * fnp.exp(log_mean + 0.5 * log_var)
    reliability = fnp.minimum(1.0, positive_count / _NLN_MIN_POSITIVES)
    fallback_mean = pilot_mean + _NLN_FALLBACK_BLEND * reliability * (nln_mean - pilot_mean)

    layer_scale = fnp.maximum(fnp.mean(pilot_mean), 1e-12)
    nln_keep = (positive_count >= _NLN_MIN_POSITIVES) & (nln_mean >= _NLN_KEEP_FRACTION * layer_scale)
    return fallback_mean, nln_keep, firing_rate, log_mean, log_var


def _sample_zero_inflated_lognormal(
    rng, n_samples: int, firing_rate: fnp.ndarray, log_mean: fnp.ndarray, log_var: fnp.ndarray
) -> fnp.ndarray:
    z = fnp.array(rng.standard_normal((n_samples, firing_rate.shape[0])).astype(fnp.float32))
    u = fnp.array(rng.random((n_samples, firing_rate.shape[0])).astype(fnp.float32))
    positive = u < firing_rate
    slab = fnp.exp(log_mean + fnp.sqrt(log_var) * z)
    return fnp.where(positive, slab, 0.0)


def _combine_means(old_mean: fnp.ndarray, old_count: float, new_mean: fnp.ndarray, new_count: float) -> fnp.ndarray:
    return (old_mean * old_count + new_mean * new_count) / (old_count + new_count)


def _weighted_mean(x: fnp.ndarray, weights: fnp.ndarray) -> fnp.ndarray:
    denom = fnp.maximum(fnp.sum(weights), 1e-12)
    return fnp.sum(x * weights[:, None], axis=0) / denom


def _importance_direction(mlp: MLP, active_indices, gate_rows, fallback_rows, cov_degree_rows) -> fnp.ndarray:
    final_idx = active_indices[-1]
    target = fallback_rows[-1][final_idx] * (1.0 + cov_degree_rows[-1][final_idx] / _COV_CORE_DEGREE)
    target = target / fnp.maximum(fnp.sqrt(fnp.sum(target * target)), 1e-12)

    grad = target
    for layer_idx in range(mlp.depth - 1, -1, -1):
        idx = active_indices[layer_idx]
        gate = gate_rows[layer_idx][idx]
        grad = grad * gate

        if layer_idx == 0:
            direction = mlp.weights[layer_idx][:, idx] @ grad
        else:
            prev_idx = active_indices[layer_idx - 1]
            direction = mlp.weights[layer_idx][prev_idx, :][:, idx] @ grad
            grad = direction

    return direction / fnp.maximum(fnp.sqrt(fnp.sum(direction * direction)), 1e-12)


def _elite_importance_direction(
    pilot_inputs: fnp.ndarray, final_pilot_x: fnp.ndarray, fallback_rows, cov_degree_rows
) -> fnp.ndarray:
    target = fallback_rows[-1] * (1.0 + cov_degree_rows[-1] / _COV_CORE_DEGREE)
    target = target / fnp.maximum(fnp.sqrt(fnp.sum(target * target)), 1e-12)
    rewards = final_pilot_x @ target
    elite_weight = fnp.maximum(rewards - fnp.mean(rewards), 0.0)
    direction = elite_weight @ pilot_inputs
    direction = direction / fnp.maximum(fnp.sum(elite_weight), 1e-12)
    return direction / fnp.maximum(fnp.sqrt(fnp.sum(direction * direction)), 1e-12)


def _tilted_first_layer_activations(
    rng, n_samples: int, w: fnp.ndarray, width: int, direction: fnp.ndarray
) -> tuple[fnp.ndarray, fnp.ndarray]:
    alpha = _IMPORTANCE_ALPHA
    shift_pre = (alpha * direction) @ w

    if _USE_ANTITHETIC:
        n_pairs = max(1, (n_samples + 1) // 2)
        eps_half = fnp.array(rng.standard_normal((n_pairs, width)).astype(fnp.float32))
        eps_pre = eps_half @ w
        eps_dot = eps_half @ direction
        z = fnp.concatenate([shift_pre + eps_pre, shift_pre - eps_pre], axis=0)
        log_weights = fnp.concatenate(
            [-alpha * eps_dot - 0.5 * alpha * alpha, alpha * eps_dot - 0.5 * alpha * alpha], axis=0
        )
    else:
        eps = fnp.array(rng.standard_normal((n_samples, width)).astype(fnp.float32))
        z = eps @ w + shift_pre
        log_weights = -alpha * (eps @ direction) - 0.5 * alpha * alpha

    log_weights = fnp.maximum(fnp.minimum(log_weights, _IMPORTANCE_WEIGHT_CLIP), -_IMPORTANCE_WEIGHT_CLIP)
    return fnp.maximum(z, 0.0), fnp.exp(log_weights)


def _positive_covariance_degree(x: fnp.ndarray, pilot_mean: fnp.ndarray) -> fnp.ndarray:
    centered = x - pilot_mean
    sample_count = max(1, x.shape[0] - 1)
    cov_raw = (centered.T @ centered) / float(sample_count)
    cov = flops.as_symmetric(0.5 * (cov_raw + cov_raw.T), symmetry=(0, 1))
    var = fnp.maximum(fnp.diag(cov), 1e-12)
    inv_std = 1.0 / fnp.sqrt(var)
    corr_raw = cov * fnp.outer(inv_std, inv_std)
    corr = flops.as_symmetric(0.5 * (corr_raw + corr_raw.T), symmetry=(0, 1))
    sparse_support = fnp.asarray(corr >= _COV_CORR_THRESHOLD)
    return fnp.sum(sparse_support, axis=0) - 1.0


def _core_sample_weight(cov_degree: fnp.ndarray) -> fnp.ndarray:
    if _PERIPHERAL_SHRINK <= 0.0:
        return fnp.ones(cov_degree.shape)
    core_weight = fnp.minimum(1.0, cov_degree / _COV_CORE_DEGREE)
    return 1.0 - _PERIPHERAL_SHRINK * (1.0 - core_weight)


def _first_layer_relu_mean(w: fnp.ndarray) -> fnp.ndarray:
    sigma = fnp.sqrt(fnp.maximum(fnp.sum(w * w, axis=0), 1e-30))
    return sigma * _INV_SQRT_TWO_PI


def _first_layer_relu_var(w: fnp.ndarray) -> fnp.ndarray:
    sigma2 = fnp.maximum(fnp.sum(w * w, axis=0), 1e-30)
    return sigma2 * _RELU_STANDARD_NORMAL_VAR


def _gaussian_relu_lookahead(
    prev_mean: fnp.ndarray, prev_var: fnp.ndarray, w: fnp.ndarray
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    pre_mean = prev_mean @ w
    pre_var = fnp.maximum(prev_var @ (w * w), 1e-12)
    pre_sigma = fnp.sqrt(pre_var)
    alpha = pre_mean / pre_sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)
    mean = pre_mean * Phi + pre_sigma * phi
    second = (pre_mean * pre_mean + pre_var) * Phi + pre_mean * pre_sigma * phi
    var = fnp.maximum(second - mean * mean, 0.0)
    return mean, var, Phi


class Estimator(BaseEstimator):
    """Active-set antithetic sampler for WHest activation means.

    A small full-network pilot pass discovers reliably active units in each
    layer. The pilot also fits a zero-inflated lognormal positive slab and a
    sparse covariance support matrix. The main pass then spends a larger
    antithetic Monte Carlo budget on only the active/covariance-coupled
    subnetwork, using pilot/lognormal means for dropped near-dead units.
    """

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        rng = fnp.random.default_rng(mlp.seed)
        width = mlp.width

        # Pilot pass: full network, small sample count, used to identify the
        # sparse live subnetwork and provide fallback means for dropped units.
        x = None
        pilot_inputs = None
        pilot_sample_count = float(_actual_sample_count(_PILOT_SAMPLES))
        fallback_rows = []
        pilot_rows = []
        cov_degree_rows = []
        gate_rows = []
        block_stats = None
        active_indices = []
        active_counts = []
        for layer_idx, w in enumerate(mlp.weights):
            if layer_idx == 0:
                x, pilot_inputs = _first_layer_activations_and_inputs(rng, _PILOT_SAMPLES, w, width)
                exact_mean = _first_layer_relu_mean(w)
                idx = fnp.arange(width)
                active_indices.append(idx)
                active_counts.append(width)
                fallback_rows.append(exact_mean)
                pilot_rows.append(exact_mean)
                cov_degree_rows.append(fnp.ones(width) * _COV_CORE_DEGREE)
                gate_rows.append(fnp.ones(width) * 0.5)
                continue

            x = fnp.maximum(x @ w, 0.0)
            pilot_mean = fnp.mean(x, axis=0)
            firing_rate = fnp.mean(x > 0.0, axis=0)
            if not _use_block_classification(layer_idx):
                idx = fnp.arange(width)
                active_indices.append(idx)
                active_counts.append(width)
                fallback_rows.append(pilot_mean)
                pilot_rows.append(pilot_mean)
                cov_degree_rows.append(fnp.ones(width) * _COV_CORE_DEGREE)
                gate_rows.append(firing_rate)
                continue

            fallback_mean, nln_keep, nln_rate, log_mean, log_var = _positive_slab_lognormal_stats(x, pilot_mean)
            keep = (firing_rate >= _layer_fire_threshold(layer_idx, mlp.depth)) | nln_keep
            layer_scale = fnp.maximum(fnp.mean(pilot_mean), 1e-12)
            if _USE_MAGNITUDE_CORE:
                keep = keep | (fallback_mean >= _MAGNITUDE_CORE_FRACTION * layer_scale)
            cov_degree = fnp.zeros(width)
            if _USE_SPARSE_COV:
                cov_degree = _positive_covariance_degree(x, pilot_mean)
                keep = keep | (cov_degree >= _COV_MIN_DEGREE)
            if _SIMPLIFY_IDENTITY:
                identity_drop = (cov_degree <= _IDENTITY_MAX_DEGREE) & (
                    fallback_mean <= _IDENTITY_MEAN_FRACTION * layer_scale
                )
                keep = keep & ~identity_drop
            idx = fnp.nonzero(keep)[0]
            active_indices.append(idx)
            active_counts.append(int(fnp.sum(keep)))
            fallback_rows.append(fallback_mean)
            pilot_rows.append(pilot_mean)
            cov_degree_rows.append(cov_degree)
            gate_rows.append(firing_rate)
            if _BLOCK_RESTART_LAYER == layer_idx + 1:
                block_stats = (layer_idx, nln_rate, log_mean, log_var, x)
        final_pilot_x = x

        # Main pass: spend most samples only on the active subnetwork.
        regular_samples = _regular_main_samples()
        restart_layer_idx = -1
        x_main = None
        if block_stats is not None:
            restart_layer_idx, nln_rate, log_mean, log_var, boundary_x = block_stats
            restart_idx = active_indices[restart_layer_idx]
            if _BLOCK_RESTART_MODE == "bootstrap":
                draw_idx = fnp.array(rng.integers(0, boundary_x.shape[0], size=_actual_sample_count(regular_samples)))
                x_main = boundary_x[draw_idx, :][:, restart_idx]
            else:
                x_main = _sample_zero_inflated_lognormal(
                    rng,
                    _actual_sample_count(regular_samples),
                    nln_rate[restart_idx],
                    log_mean[restart_idx],
                    log_var[restart_idx],
                )
        main_sample_count = float(_actual_sample_count(regular_samples))
        rows = []
        main_weight_slices = []
        prev_idx = None
        for layer_idx, w in enumerate(mlp.weights):
            idx = active_indices[layer_idx]
            active_count = active_counts[layer_idx]

            if layer_idx <= restart_layer_idx:
                rows.append(fallback_rows[layer_idx])
                prev_idx = idx
                continue

            if layer_idx == 0:
                x_main = _first_layer_activations(rng, regular_samples, w, width)
                rows.append(fallback_rows[layer_idx])
                prev_idx = idx
                continue

            if active_count == 0:
                rows.append(fallback_rows[layer_idx])
                x_main = fnp.zeros((x_main.shape[0], 0))
                prev_idx = idx
                continue

            w_active = w[prev_idx, :][:, idx]
            main_weight_slices.append((layer_idx, w_active))

            x_main = fnp.maximum(x_main @ w_active, 0.0)
            active_mean = fnp.mean(x_main, axis=0)
            if _REUSE_PILOT_SAMPLES:
                active_mean = (
                    active_mean * main_sample_count + pilot_rows[layer_idx][idx] * pilot_sample_count
                ) / (main_sample_count + pilot_sample_count)
            if _PERIPHERAL_SHRINK > 0.0:
                sample_weight = _core_sample_weight(cov_degree_rows[layer_idx][idx])
                fallback_active = fallback_rows[layer_idx][idx]
                active_mean = fallback_active + sample_weight * (active_mean - fallback_active)

            row = fallback_rows[layer_idx]
            row = row.copy()
            row[idx] = active_mean
            rows.append(row)
            prev_idx = idx

        if _IMPORTANCE_SAMPLES > 0 and restart_layer_idx < 0:
            if _IMPORTANCE_DIRECTION_MODE == "elite":
                direction = _elite_importance_direction(pilot_inputs, final_pilot_x, fallback_rows, cov_degree_rows)
            else:
                direction = _importance_direction(mlp, active_indices, gate_rows, fallback_rows, cov_degree_rows)
            x_importance, importance_weights = _tilted_first_layer_activations(
                rng, _IMPORTANCE_SAMPLES, mlp.weights[0], width, direction
            )
            importance_count = float(_actual_sample_count(_IMPORTANCE_SAMPLES))

            for layer_idx, w in enumerate(mlp.weights):
                if layer_idx == 0:
                    continue

                idx = active_indices[layer_idx]
                active_count = active_counts[layer_idx]
                if active_count == 0:
                    x_importance = fnp.zeros((x_importance.shape[0], 0))
                    continue

                prev_idx = active_indices[layer_idx - 1]
                w_active = w[prev_idx, :][:, idx]
                x_importance = fnp.maximum(x_importance @ w_active, 0.0)

                if layer_idx + 1 < _BLOCK_CLASS_START_LAYER:
                    continue

                importance_mean = _weighted_mean(x_importance, importance_weights)
                row = rows[layer_idx]
                row = row.copy()
                row[idx] = _combine_means(row[idx], main_sample_count + pilot_sample_count, importance_mean, importance_count)
                rows[layer_idx] = row

        if _ADAPTIVE_EXTRA_SAMPLES > 0:
            x_extra = None
            extra_count = float(_actual_sample_count(_ADAPTIVE_EXTRA_SAMPLES))
            for layer_idx, w in enumerate(mlp.weights):
                idx = active_indices[layer_idx]
                active_count = active_counts[layer_idx]

                if layer_idx == 0:
                    x_extra = _first_layer_activations(rng, _ADAPTIVE_EXTRA_SAMPLES, w, width)
                    continue

                if active_count == 0:
                    x_extra = fnp.zeros((x_extra.shape[0], 0))
                    continue

                prev_idx = active_indices[layer_idx - 1]
                w_active = w[prev_idx, :][:, idx]
                x_extra = fnp.maximum(x_extra @ w_active, 0.0)

                if layer_idx + 1 < _BLOCK_CLASS_START_LAYER:
                    continue

                extra_mean = fnp.mean(x_extra, axis=0)
                row = rows[layer_idx]
                row = row.copy()
                row[idx] = _combine_means(row[idx], main_sample_count + pilot_sample_count, extra_mean, extra_count)
                rows[layer_idx] = row

        return fnp.stack(rows, axis=0)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)  # phase-1 competition shape (warmup was 8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
