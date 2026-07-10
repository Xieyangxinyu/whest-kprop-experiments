"""Diagnostic: is the lognormal tail binding for Sobol at width=256, depth=32?

Per the external review's gating check:
1. Convergence slope of final-layer estimates vs n (log-log). Slope ~ -1
   => QMC fine, tail not binding. Slope ~ -0.5 => tail owns the error.
2. Max-sample contribution fraction per final-layer neuron mean. > a few
   percent => unresolved spike.
3. sigma_h = std of log ||a^(31)|| (lognormal scale of the latent gain).

Pure NumPy, offline diagnostic — not scored code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def build_weights(width: int, depth: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    return [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]


def forward_means(weights, x):
    """Return (per-layer means, final-layer activations, layer-31 pre-final norms)."""
    rows = []
    a_prev = None
    for w in weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(x.mean(axis=0))
        a_prev = x if a_prev is None else a_prev  # placeholder
    return rows, x


def final_layer_mean(weights, x):
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x  # final activations per sample


def main():
    width, depth = 256, 32
    data = np.load(REPO / "sobol_points.npz")
    pts = data["points"].astype(np.float32)
    print(f"sobol points shape: {pts.shape}", flush=True)
    max_half = min(pts.shape[0], 20480)

    for mlp_seed in (0, 1):
        weights = build_weights(width, depth, mlp_seed)
        print(f"\n=== MLP seed {mlp_seed} ===", flush=True)

        # --- Reference: 1M MC samples, batched ---
        rng = np.random.default_rng(10_000 + mlp_seed)
        acc = np.zeros(width, dtype=np.float64)
        n_ref = 0
        for _ in range(16):
            xb = rng.standard_normal((65536, width)).astype(np.float32)
            acc += final_layer_mean(weights, xb).sum(axis=0, dtype=np.float64)
            n_ref += 65536
        ref = acc / n_ref
        print(f"reference: {n_ref:,} MC samples, mean of final-layer means "
              f"= {ref.mean():.6f}", flush=True)

        # --- Sobol convergence slope (antithetic, like the estimator) ---
        print("\n  n_eff   | sobol_final_MSE | mc_final_MSE(avg3) | drift(sobol)")
        ns = [512, 2048, 8192, max_half]
        sob_mses, mc_mses = [], []
        for n_half in ns:
            half = pts[:n_half, :width]
            x = np.concatenate([half, -half], axis=0)
            fin = final_layer_mean(weights, x)
            est = fin.mean(axis=0)
            sob_mse = float(np.mean((est - ref) ** 2))
            drift = float(np.mean(est - ref))
            # MC comparison at same n_eff, 3 seeds
            mc_ms = []
            for s in range(3):
                r2 = np.random.default_rng(777 + s)
                xm = r2.standard_normal((2 * n_half, width)).astype(np.float32)
                em = final_layer_mean(weights, xm).mean(axis=0)
                mc_ms.append(float(np.mean((em - ref) ** 2)))
            mc_mse = float(np.mean(mc_ms))
            sob_mses.append(sob_mse)
            mc_mses.append(mc_mse)
            print(f"  {2*n_half:7,} | {sob_mse:.3e}       | {mc_mse:.3e}          "
                  f"| {drift:+.2e}", flush=True)

        def slope(ns_, es_):
            ln = np.log([2 * n for n in ns_])
            le = np.log(es_)
            return float(np.polyfit(ln, le, 1)[0]) / 2  # MSE slope /2 = error slope

        print(f"  error-vs-n slope: sobol {slope(ns, sob_mses):+.2f}, "
              f"mc {slope(ns, mc_mses):+.2f}   (-1 good QMC, -0.5 plain MC)")

        # --- Tail diagnostics at full n ---
        half = pts[:max_half, :width]
        x = np.concatenate([half, -half], axis=0)
        a = x
        norms_by_layer = []
        for w in weights:
            a = np.maximum(a @ w, 0.0)
            norms_by_layer.append(np.linalg.norm(a, axis=1))
        fin = a  # (n, width) final activations
        h31 = 2.0 * np.log(np.maximum(norms_by_layer[-2], 1e-30))
        sigma_h = float(h31.std())
        print(f"  sigma_h (std of 2*log||a^(31)||): {sigma_h:.3f} "
              f"(ESS penalty exp(sigma_h^2) = {np.exp(sigma_h**2):.2f})")

        sums = fin.sum(axis=0)
        maxs = fin.max(axis=0)
        frac = np.where(sums > 0, maxs / np.maximum(sums, 1e-30), 0.0)
        print(f"  max-sample share of neuron mean: median {np.median(frac)*100:.3f}%, "
              f"p95 {np.percentile(frac, 95)*100:.3f}%, max {frac.max()*100:.3f}%")


if __name__ == "__main__":
    main()
