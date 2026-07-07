"""Probe block sign-flip sample orbits against plain Sobol active-set variants.

This is an offline experiment. It tests whether fewer Sobol base points expanded
by deterministic coordinate-block sign flips can match or beat plain antithetic
Sobol at the same effective sample count.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

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


def make_mlp(row) -> MLP:
    weights = [fnp.array(np.asarray(weight, dtype=np.float32)) for weight in row["weights"]]
    return MLP(width=weights[0].shape[0], depth=len(weights), weights=weights, seed=int(row["mlp_seed"]))


def block_sign_patterns(blocks: int) -> np.ndarray:
    if blocks <= 0 or WIDTH % blocks != 0:
        raise ValueError(f"blocks must divide {WIDTH}, got {blocks}")
    block_width = WIDTH // blocks
    patterns = [np.ones(WIDTH, dtype=np.float32)]
    for block_idx in range(blocks):
        signs = np.ones(WIDTH, dtype=np.float32)
        start = block_idx * block_width
        signs[start : start + block_width] = -1.0
        patterns.append(signs)
    return np.stack(patterns, axis=0)


def build_orbit_points(sobol_points: np.ndarray, half_count: int, blocks: int) -> np.ndarray:
    base_count = max(1, half_count // (blocks + 1))
    base = sobol_points[:base_count, :WIDTH].astype(np.float32)
    patterns = block_sign_patterns(blocks)
    orbit = (base[:, None, :] * patterns[None, :, :]).reshape(base_count * (blocks + 1), WIDTH)
    return orbit[:half_count]


def evaluate_points(row, points: np.ndarray, sample_count: int) -> tuple[float, float, float, int]:
    mlp = make_mlp(row)
    gt = np.asarray(row["all_layer_means"], dtype=np.float32)
    variant = replace(
        next(v for v in ev.VARIANTS if v.name == "pilot l29+30 borderline 4/-4"),
        dynamic_sample_mode="",
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mini")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sobol", default=str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
    args = parser.parse_args()

    sobol_points = np.load(args.sobol)["points"]
    half_count = args.samples // 2
    if sobol_points.shape[0] < half_count:
        raise SystemExit(f"{args.sobol} has {sobol_points.shape[0]} half-points; need {half_count}")

    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[i] for i in range(min(args.limit, len(dataset)))]
    point_sets = {"plain": sobol_points[:half_count, :WIDTH].astype(np.float32)}
    for blocks in args.blocks:
        point_sets[f"block{blocks}"] = build_orbit_points(sobol_points, half_count, blocks)

    print(f"sign-flip orbit probe split={args.split} limit={len(rows)} samples={args.samples}")
    print(f"{'mode':<10} {'final_mse':>12} {'score':>12} {'all_mse':>12} {'flops':>12} {'wins':>6}")
    baseline_scores = None
    for mode, points in point_sets.items():
        metrics = [evaluate_points(row, points, args.samples) for row in rows]
        arr = np.asarray(metrics, dtype=np.float64)
        if mode == "plain":
            baseline_scores = arr[:, 2]
            wins = len(rows)
        else:
            wins = int(np.sum(arr[:, 2] < baseline_scores))
        print(
            f"{mode:<10} {arr[:,0].mean():12.3e} {arr[:,2].mean():12.3e} "
            f"{arr[:,1].mean():12.3e} {arr[:,3].mean():12.2e} {wins:6d}"
        )


if __name__ == "__main__":
    main()
