"""Low-rank conditional final-on estimator probe.

For selected final-on neurons, sample only a low-dimensional input subspace and
integrate the orthogonal residual approximately with diagonal Gaussian moment
propagation. This tests whether the final-on error can be attacked without a full
extra network sample pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import special

from sample_k3_projection_probe import _load_samples
from two_stage_final_on_rotation import BUDGET, DEPTH, WIDTH, _base_case, _mse, _selected


def _basis(case, selected: np.ndarray) -> np.ndarray:
    effective = case.weights[0].astype(np.float64).copy()
    effective *= (case.alpha_rows[0] > -3.0).astype(np.float64)[None, :]
    for layer_idx in range(1, 30):
        effective = effective @ case.weights[layer_idx].astype(np.float64)
        effective *= (case.alpha_rows[layer_idx] > -3.0).astype(np.float64)[None, :]
    layer30_on = case.on_indices[30]
    if layer30_on.size == 0 or selected.size == 0:
        return np.eye(WIDTH, dtype=np.float64)
    sensitivity = effective @ case.weights[30][:, layer30_on].astype(np.float64)
    sensitivity = sensitivity @ case.weights[31][layer30_on, :][:, selected].astype(np.float64)
    u, _, _ = np.linalg.svd(sensitivity, full_matrices=False)
    return u


def _relu_moments(pre_mean: np.ndarray, pre_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pre_var = np.maximum(pre_var, 1e-12)
    sigma = np.sqrt(pre_var)
    alpha = pre_mean / sigma
    phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2.0 * np.pi)
    Phi = special.ndtr(alpha)
    mean = pre_mean * Phi + sigma * phi
    ez2 = (pre_mean * pre_mean + pre_var) * Phi + pre_mean * sigma * phi
    var = np.maximum(ez2 - mean * mean, 1e-12)
    return mean, var


def _conditional_pre_mean(case, coords: np.ndarray, selected: np.ndarray, rank: int) -> np.ndarray:
    if selected.size == 0:
        return np.zeros(0, dtype=np.float64)
    u = _basis(case, selected)
    rank = min(rank, u.shape[1], coords.shape[1])
    if rank <= 0:
        mean = np.zeros((coords.shape[0], WIDTH), dtype=np.float64)
        var = np.ones((coords.shape[0], WIDTH), dtype=np.float64)
    else:
        ur = u[:, :rank]
        mean = coords[:, :rank].astype(np.float64) @ ur.T
        residual_diag = np.maximum(1.0 - np.sum(ur * ur, axis=1), 1e-8)
        var = np.repeat(residual_diag[None, :], coords.shape[0], axis=0)

    for layer_idx, w in enumerate(case.weights):
        w64 = w.astype(np.float64)
        pre_mean = mean @ w64
        pre_var = var @ (w64 * w64)
        if layer_idx == DEPTH - 1:
            return pre_mean[:, selected].mean(axis=0)
        mean, var = _relu_moments(pre_mean, pre_var)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="8,9,10,11")
    parser.add_argument("--base-samples", default="8192,16384")
    parser.add_argument("--cond-samples", default="256,512,1024")
    parser.add_argument("--rank", default="4,8,16")
    parser.add_argument("--select", default="alpha,col_norm,oracle")
    parser.add_argument("--k", default="8,16,32")
    parser.add_argument("--blend", default="0.25,0.5,1.0")
    parser.add_argument("--sobol", default=str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    base_samples = [int(item) for item in args.base_samples.split(",") if item]
    cond_samples = [int(item) for item in args.cond_samples.split(",") if item]
    ranks = [int(item) for item in args.rank.split(",") if item]
    select_modes = [item for item in args.select.split(",") if item]
    ks = [int(item) for item in args.k.split(",") if item]
    blends = [float(item) for item in args.blend.split(",") if item]

    for n_base in base_samples:
        cases = [_base_case(index, n_base) for index in indices]
        base_mse = float(np.mean([_mse(case.base_final, case.true_final) for case in cases]))
        base_flops = float(np.mean([case.base_flops for case in cases]))
        print(f"\nN_base={n_base} base_mse={base_mse:.3e} base_util={base_flops / BUDGET:.3f}")
        for n_cond in cond_samples:
            coords = _load_samples(args.sobol, n_cond).astype(np.float64)
            print(f"  N_cond={n_cond}")
            best = None
            for mode in select_modes:
                for k in ks:
                    selecteds = [_selected(case, mode, k) for case in cases]
                    for rank in ranks:
                        replacements = [
                            _conditional_pre_mean(case, coords, selected, rank)
                            for case, selected in zip(cases, selecteds)
                        ]
                        for blend in blends:
                            mses = []
                            for case, selected, replacement in zip(cases, selecteds, replacements):
                                pred = case.base_final.copy()
                                if selected.size > 0:
                                    pred[selected] = (1.0 - blend) * pred[selected] + blend * replacement
                                mses.append(_mse(pred, case.true_final))
                            row = (float(np.mean(mses)), mode, k, rank, blend)
                            if best is None or row[0] < best[0]:
                                best = row
            print(
                f"    best mse={best[0]:.3e} select={best[1]} k={best[2]} "
                f"rank={best[3]} blend={best[4]:.2f}"
            )


if __name__ == "__main__":
    main()