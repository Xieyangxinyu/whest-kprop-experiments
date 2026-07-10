"""Probe shifted rank-1 lattice Gaussian samples against shipped Sobol samples.

This is an offline experiment. It compares the current active-set estimator path
using antithetic Sobol half-points against rank-1 lattice points with a
Cranley-Patterson shift and inverse-normal transform.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy import special

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_variants as ev
import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from whestbench.domain import MLP


WIDTH = 256
BUDGET = ev.BUDGET
TAIL_CLIP = 1.0e-11


def first_primes(n: int) -> np.ndarray:
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return np.asarray(primes, dtype=np.float64)


_GENERATOR = np.mod(np.sqrt(first_primes(WIDTH)), 1.0).astype(np.float64)


def make_mlp(row) -> MLP:
    weights = [fnp.array(np.asarray(weight, dtype=np.float32)) for weight in row["weights"]]
    return MLP(width=weights[0].shape[0], depth=len(weights), weights=weights, seed=int(row["mlp_seed"]))


def lattice_gaussian_points(count: int, seed: int, generator: np.ndarray = _GENERATOR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shift = rng.random(WIDTH, dtype=np.float64)
    k = np.arange(count, dtype=np.float64)[:, None]
    u = np.mod(k * generator[None, :] + shift[None, :], 1.0)
    u = np.clip(u, TAIL_CLIP, 1.0 - TAIL_CLIP)
    return special.ndtri(u).astype(np.float32)


def evaluate_points(row, points: np.ndarray, sample_count: int, variant) -> tuple[float, float, float, int]:
    mlp = make_mlp(row)
    gt = np.asarray(row["all_layer_means"], dtype=np.float32)
    old_n_samples = ev.N_SAMPLES
    try:
        ev.N_SAMPLES = sample_count
        with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
            pred = np.asarray(ev.predict_variant(mlp, points, variant), dtype=np.float32)
    finally:
        ev.N_SAMPLES = old_n_samples
    err = pred - gt
    final_mse = float(np.mean(err[-1] * err[-1]))
    all_mse = float(np.mean(err * err))
    score = final_mse * max(0.1, ctx.flops_used / BUDGET)
    return final_mse, all_mse, score, int(ctx.flops_used)


def summarize(mode: str, metrics: list[tuple[float, float, float, int]], baseline_scores: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(metrics, dtype=np.float64)
    wins = len(metrics) if baseline_scores is None else int(np.sum(arr[:, 2] < baseline_scores))
    print(
        f"{mode:<18} {arr[:,0].mean():12.3e} {arr[:,2].mean():12.3e} "
        f"{arr[:,1].mean():12.3e} {arr[:,3].mean():12.2e} {wins:6d}"
    )
    return arr[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mini")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30720)
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    parser.add_argument("--seed-offset", type=int, default=12345)
    args = parser.parse_args()

    sobol_points = np.load(args.sobol)["points"].astype(np.float32)
    half_count = args.samples // 2
    if sobol_points.shape[0] < max(half_count, args.samples):
        print(
            f"warning: {args.sobol} has {sobol_points.shape[0]} points; "
            f"plain Sobol mode needs {args.samples}, antithetic modes need {half_count}."
        )

    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[i] for i in range(min(args.limit, len(dataset)))]

    antithetic_variant = replace(
        next(v for v in ev.VARIANTS if v.name == "pilot l29+30 borderline 4/-4"),
        dynamic_sample_mode="",
        sample_mode="normal",
    )
    plain_variant = replace(antithetic_variant, sample_mode="plain_sobol")

    print(f"shifted lattice RQMC probe split={args.split} limit={len(rows)} samples={args.samples}")
    print(f"{'mode':<18} {'final_mse':>12} {'score':>12} {'all_mse':>12} {'flops':>12} {'wins':>6}")

    modes: list[tuple[str, object, callable]] = [
        ("sobol_antithetic", antithetic_variant, lambda row: sobol_points[:half_count, :WIDTH]),
        ("lattice_antithetic", antithetic_variant, lambda row: lattice_gaussian_points(half_count, int(row["mlp_seed"]) + args.seed_offset)),
        ("lattice_plain", plain_variant, lambda row: lattice_gaussian_points(args.samples, int(row["mlp_seed"]) + args.seed_offset)),
    ]
    if sobol_points.shape[0] >= args.samples:
        modes.insert(1, ("sobol_plain", plain_variant, lambda row: sobol_points[: args.samples, :WIDTH]))

    baseline_scores = None
    for mode, variant, point_fn in modes:
        metrics = [evaluate_points(row, point_fn(row), args.samples, variant) for row in rows]
        scores = summarize(mode, metrics, baseline_scores)
        if mode == "sobol_antithetic":
            baseline_scores = scores


if __name__ == "__main__":
    main()
