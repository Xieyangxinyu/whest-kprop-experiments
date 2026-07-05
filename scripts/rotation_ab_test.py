"""A/B test: baseline Sobol vs W0-singular-direction-rotated Sobol inputs.

Gaussian rotation invariance: if x = z @ U.T with U orthogonal and z the
Sobol-Gaussian points, x is still N(0, I)-distributed but the first Sobol
coordinates (highest equidistribution quality) now drive the top singular
directions of W0, since x @ W0 = z @ (S V.T) when W0 = U S V.T.

Offline diagnostic, pure NumPy. Metric: final-layer MSE vs 1M-sample MC
reference, full forward through all 32 layers (no active-set machinery, so
this isolates the sampling-geometry effect).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WIDTH, DEPTH = 256, 32
N_HALF = 20480


def build_weights(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = (2.0 / WIDTH) ** 0.5
    return [
        (rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
        for _ in range(DEPTH)
    ]


def final_mean(weights, x):
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0, dtype=np.float64)


def main():
    pts = np.load(REPO / "sobol_points.npz")["points"].astype(np.float32)
    half = pts[:N_HALF, :WIDTH]

    print(f"{'seed':>4} | {'baseline_MSE':>12} | {'rotated_MSE':>12} | {'ratio':>6}")
    ratios = []
    for seed in range(5):
        weights = build_weights(seed)

        rng = np.random.default_rng(10_000 + seed)
        acc = np.zeros(WIDTH, dtype=np.float64)
        n_ref = 0
        for _ in range(16):
            xb = rng.standard_normal((65536, WIDTH)).astype(np.float32)
            acc += final_mean(weights, xb) * 65536
            n_ref += 65536
        ref = acc / n_ref

        x_base = np.concatenate([half, -half], axis=0)
        mse_base = float(np.mean((final_mean(weights, x_base) - ref) ** 2))

        u, _, _ = np.linalg.svd(weights[0].astype(np.float64))
        half_rot = (half @ u.T.astype(np.float32))
        x_rot = np.concatenate([half_rot, -half_rot], axis=0)
        mse_rot = float(np.mean((final_mean(weights, x_rot) - ref) ** 2))

        ratio = mse_rot / mse_base
        ratios.append(ratio)
        print(f"{seed:>4} | {mse_base:.4e}   | {mse_rot:.4e}   | {ratio:.3f}",
              flush=True)

    g = float(np.exp(np.mean(np.log(ratios))))
    print(f"\ngeomean rotated/baseline MSE ratio over 5 seeds: {g:.3f} "
          f"(<1 means rotation helps)")


def control():
    """Random-rotation control: does ANY rotation help, or only W0-aligned?"""
    pts = np.load(REPO / "sobol_points.npz")["points"].astype(np.float32)
    half = pts[:N_HALF, :WIDTH]
    print(f"{'seed':>4} | {'svd_ratio':>9} | random-rotation ratios (3 draws)")
    for seed in range(5):
        weights = build_weights(seed)
        rng = np.random.default_rng(10_000 + seed)
        acc = np.zeros(WIDTH, dtype=np.float64)
        for _ in range(16):
            xb = rng.standard_normal((65536, WIDTH)).astype(np.float32)
            acc += final_mean(weights, xb) * 65536
        ref = acc / (16 * 65536)

        x_base = np.concatenate([half, -half], axis=0)
        mse_base = float(np.mean((final_mean(weights, x_base) - ref) ** 2))

        u, _, _ = np.linalg.svd(weights[0].astype(np.float64))
        hr = half @ u.T.astype(np.float32)
        mse_svd = float(np.mean((final_mean(weights, np.concatenate([hr, -hr])) - ref) ** 2))

        rand_ratios = []
        for k in range(3):
            g = np.random.default_rng(555 + k).standard_normal((WIDTH, WIDTH))
            q, _ = np.linalg.qr(g)
            hq = half @ q.astype(np.float32)
            m = float(np.mean((final_mean(weights, np.concatenate([hq, -hq])) - ref) ** 2))
            rand_ratios.append(m / mse_base)
        print(f"{seed:>4} | {mse_svd/mse_base:9.3f} | "
              + ", ".join(f"{r:.3f}" for r in rand_ratios), flush=True)

if __name__ == "__main__":
    main()
    control()
