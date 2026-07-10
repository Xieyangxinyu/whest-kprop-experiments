"""Offline two-stage final-on rotation experiment.

Stage A: run a cheap broad active-set estimate with N_base samples.
Stage B: select top-k risky final-on neurons, estimate only those final-on
pre-activation means using N_rot Sobol samples, optionally rotated into a
final-on sensitivity basis, then blend into the Stage A final row.

This script uses higher-moment/public-full files for ground truth and is not a
submission estimator. It answers whether the broad-cheap + targeted-expensive
shape is promising before porting anything into estimator.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP

from estimator import Estimator
from learn_cumulant_closure import _moment_path, _weights
from sample_k3_projection_probe import _final_pre_samples, _load_samples


BUDGET = 272_000_000_000
WIDTH = 256
DEPTH = 32


@dataclass
class BaseCase:
    index: int
    weights: np.ndarray
    mlp: MLP
    true_final: np.ndarray
    base_final: np.ndarray
    analytical_final: np.ndarray
    alpha_rows: list[np.ndarray]
    on_indices: list[np.ndarray]
    final_on: np.ndarray
    base_flops: float


def _mlp(weights: np.ndarray) -> MLP:
    return MLP(width=WIDTH, depth=DEPTH, weights=[fnp.array(weights[layer].astype(np.float32)) for layer in range(DEPTH)])


def _base_case(index: int, n_base: int) -> BaseCase:
    weights = _weights(index).astype(np.float32)
    mlp = _mlp(weights)
    data = np.load(_moment_path(index))
    true_final = np.asarray(data["mean"][-1], dtype=np.float64)
    estimator = Estimator()
    structure = estimator._initial_structure(mlp, WIDTH)
    x = estimator._sample_block(0, n_base // 2, WIDTH)
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
        rows, _, refined = estimator._run_block(mlp, structure, x, n_base, refine=True)
        base_final = np.asarray(rows[-1], dtype=np.float64)
    alpha_rows = [np.asarray(alpha, dtype=np.float64) for alpha in refined["alpha_rows"]]
    on_indices = [np.asarray(indices, dtype=np.int64) for indices in refined["on_indices"]]
    final_on = on_indices[-1]
    return BaseCase(
        index=index,
        weights=weights,
        mlp=mlp,
        true_final=true_final,
        base_final=base_final,
        analytical_final=np.asarray(refined["analytical_rows"][-1], dtype=np.float64),
        alpha_rows=alpha_rows,
        on_indices=on_indices,
        final_on=final_on,
        base_flops=float(ctx.flops_used),
    )


def _basis(case: BaseCase, selected: np.ndarray) -> np.ndarray:
    effective = case.weights[0].astype(np.float64).copy()
    effective *= (case.alpha_rows[0] > -3.0).astype(np.float64)[None, :]
    for layer_idx in range(1, 30):
        effective = effective @ case.weights[layer_idx].astype(np.float64)
        effective *= (case.alpha_rows[layer_idx] > -3.0).astype(np.float64)[None, :]
    layer30_on = case.on_indices[30]
    if layer30_on.size == 0 or selected.size == 0:
        return np.eye(WIDTH, dtype=np.float32)
    sensitivity = effective @ case.weights[30][:, layer30_on].astype(np.float64)
    sensitivity = sensitivity @ case.weights[31][layer30_on, :][:, selected].astype(np.float64)
    values, vectors = np.linalg.eigh(sensitivity @ sensitivity.T)
    return vectors[:, np.argsort(values)[::-1]].astype(np.float32)


def _selected(case: BaseCase, mode: str, k: int) -> np.ndarray:
    final_on = case.final_on
    if final_on.size == 0:
        return final_on
    if mode == "oracle":
        score = (case.base_final - case.true_final) ** 2
        order = final_on[np.argsort(score[final_on])[::-1]]
    elif mode == "anal_diff":
        score = np.abs(case.base_final[final_on] - case.analytical_final[final_on])
        order = final_on[np.argsort(score)[::-1]]
    elif mode == "col_norm":
        norms = np.sqrt(np.sum(case.weights[31][:, final_on].astype(np.float64) ** 2, axis=0))
        order = final_on[np.argsort(norms)[::-1]]
    elif mode == "alpha":
        score = case.alpha_rows[-1][final_on]
        order = final_on[np.argsort(score)[::-1]]
    elif mode == "alpha_low":
        score = case.alpha_rows[-1][final_on]
        order = final_on[np.argsort(score)]
    elif mode == "alpha_col_norm":
        norms = np.sqrt(np.sum(case.weights[31][:, final_on].astype(np.float64) ** 2, axis=0))
        score = case.alpha_rows[-1][final_on] * norms
        order = final_on[np.argsort(score)[::-1]]
    elif mode == "anal_diff_col_norm":
        norms = np.sqrt(np.sum(case.weights[31][:, final_on].astype(np.float64) ** 2, axis=0))
        score = np.abs(case.base_final[final_on] - case.analytical_final[final_on]) * norms
        order = final_on[np.argsort(score)[::-1]]
    else:
        raise ValueError(mode)
    return order[: min(k, order.size)]


def _replacement(case: BaseCase, samples: np.ndarray, selected: np.ndarray, rotate: bool) -> np.ndarray:
    if selected.size == 0:
        return np.zeros(0, dtype=np.float64)
    if rotate:
        rotation = _basis(case, selected)
        samples = samples @ rotation
    pre = _final_pre_samples(case.weights, samples.astype(np.float32))
    return pre[:, selected].mean(axis=0)


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="8,9,10,11")
    parser.add_argument("--base-samples", default="8192,16384")
    parser.add_argument("--rot-samples", default="8192,16384,32768")
    parser.add_argument("--select", default="anal_diff,col_norm,oracle")
    parser.add_argument("--k", default="8,16")
    parser.add_argument("--blend", default="0.25,0.5,1.0")
    parser.add_argument("--sobol", default=str(Path(__file__).resolve().parent.parent / "sobol_points.npz"))
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    base_samples = [int(item) for item in args.base_samples.split(",") if item]
    rot_samples = [int(item) for item in args.rot_samples.split(",") if item]
    select_modes = [item for item in args.select.split(",") if item]
    ks = [int(item) for item in args.k.split(",") if item]
    blends = [float(item) for item in args.blend.split(",") if item]

    for n_base in base_samples:
        cases = [_base_case(index, n_base) for index in indices]
        base_mse = float(np.mean([_mse(case.base_final, case.true_final) for case in cases]))
        base_flops = float(np.mean([case.base_flops for case in cases]))
        print(f"\nN_base={n_base} base_mse={base_mse:.3e} base_flops={base_flops:.2e} util={base_flops / BUDGET:.3f}")
        for n_rot in rot_samples:
            samples = _load_samples(args.sobol, n_rot).astype(np.float32)
            print(f"  N_rot={n_rot}")
            best = None
            for mode in select_modes:
                for k in ks:
                    for rotate in (False, True):
                        replacements = []
                        selecteds = []
                        for case in cases:
                            selected = _selected(case, mode, k)
                            selecteds.append(selected)
                            replacements.append(_replacement(case, samples, selected, rotate))
                        for blend in blends:
                            mses = []
                            for case, selected, replacement in zip(cases, selecteds, replacements):
                                pred = case.base_final.copy()
                                if selected.size > 0:
                                    pred[selected] = (1.0 - blend) * pred[selected] + blend * replacement
                                mses.append(_mse(pred, case.true_final))
                            mse = float(np.mean(mses))
                            row = (mse, mode, k, "rot" if rotate else "none", blend)
                            if best is None or row[0] < best[0]:
                                best = row
            print(f"    best mse={best[0]:.3e} select={best[1]} k={best[2]} {best[3]} blend={best[4]:.2f}")


if __name__ == "__main__":
    main()