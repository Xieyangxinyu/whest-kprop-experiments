"""Probe adaptive signed corrections for final-on/high-alpha neurons.

Offline diagnostic on baked public data. Tests small corrections on risky final-on
neurons using deployable sign proxies, then compares to oracle sign choices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flopscope as flops
import flopscope.numpy as fnp
import whestbench
from estimator import Estimator
from whestbench.domain import MLP

BUDGET = 272_000_000_000
WIDTH = 256
DEPTH = 32


def make_mlp(row) -> MLP:
    weights = [fnp.array(np.asarray(weight, dtype=np.float32)) for weight in row["weights"]]
    return MLP(width=WIDTH, depth=DEPTH, weights=weights, seed=int(row["mlp_seed"]))


def predict_current(row, sobol_points: np.ndarray):
    estimator = Estimator()
    estimator._sobol_points = fnp.array(sobol_points)
    mlp = make_mlp(row)
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
        pred = np.asarray(estimator.predict(mlp, BUDGET), dtype=np.float64)
    structure = estimator._initial_structure(mlp, WIDTH)
    return pred, structure, int(ctx.flops_used)


def top_mask(score: np.ndarray, keep: float) -> np.ndarray:
    count = max(1, int(round(keep * score.size)))
    selected = np.argpartition(score, -count)[-count:]
    mask = np.zeros(score.size, dtype=bool)
    mask[selected] = True
    return mask


def risky_mask(mode: str, pred: np.ndarray, structure: dict, keep: float) -> np.ndarray:
    final_on = np.asarray(structure["on_indices"][-1], dtype=np.int64)
    alpha = np.asarray(structure["alpha_rows"][-1], dtype=np.float64)
    analytical = np.asarray(structure["analytical_rows"][-1], dtype=np.float64)
    mask = np.zeros(WIDTH, dtype=bool)
    if final_on.size == 0:
        return mask
    if mode == "all_on":
        mask[final_on] = True
    elif mode == "alpha_ge6":
        mask[final_on[alpha[final_on] >= 6.0]] = True
    elif mode == "top_alpha":
        local = final_on[top_mask(alpha[final_on], keep)]
        mask[local] = True
    elif mode == "top_pred":
        local = final_on[top_mask(pred[-1, final_on], keep)]
        mask[local] = True
    elif mode == "top_adiff":
        local = final_on[top_mask(np.abs(pred[-1, final_on] - analytical[final_on]), keep)]
        mask[local] = True
    else:
        raise ValueError(mode)
    return mask


def correction_direction(kind: str, pred: np.ndarray, structure: dict, mask: np.ndarray) -> np.ndarray:
    analytical = np.asarray(structure["analytical_rows"][-1], dtype=np.float64)
    direction = np.zeros(WIDTH, dtype=np.float64)
    if not np.any(mask):
        return direction
    if kind == "constant":
        scale = np.mean(np.abs(pred[-1, mask]))
        direction[mask] = max(scale, 1e-12)
    elif kind == "pred":
        direction[mask] = pred[-1, mask]
    elif kind == "anal_gap_up":
        direction[mask] = np.maximum(analytical[mask] - pred[-1, mask], 0.0)
    elif kind == "adiff_mag":
        direction[mask] = np.abs(pred[-1, mask] - analytical[mask])
    else:
        raise ValueError(kind)
    return direction


def sign_for_proxy(proxy: str, pred: np.ndarray, structure: dict, mask: np.ndarray) -> float:
    analytical = np.asarray(structure["analytical_rows"][-1], dtype=np.float64)
    if proxy == "up":
        return 1.0
    if proxy == "down":
        return -1.0
    if proxy == "mean_pred_minus_anal":
        value = float(np.mean(pred[-1, mask] - analytical[mask])) if np.any(mask) else 0.0
        # If sampled pred is above analytical, nudge down; otherwise nudge up.
        return -1.0 if value > 0 else 1.0
    if proxy == "mean_anal_minus_pred":
        value = float(np.mean(analytical[mask] - pred[-1, mask])) if np.any(mask) else 0.0
        return 1.0 if value > 0 else -1.0
    raise ValueError(proxy)


def fit_lambda(base: list[np.ndarray], directions: list[np.ndarray], truth: list[np.ndarray]) -> float:
    residual = np.concatenate([target - pred for pred, target in zip(base, truth)])
    direction = np.concatenate(directions)
    denom = float(direction @ direction)
    if denom <= 1e-30:
        return 0.0
    return float((direction @ residual) / denom)


def mean_mse(preds: list[np.ndarray], truths: list[np.ndarray]) -> float:
    return float(np.mean([np.mean((pred - truth) ** 2) for pred, truth in zip(preds, truths)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=70)
    parser.add_argument("--keep", default="0.10,0.20,0.40")
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    args = parser.parse_args()

    keeps = [float(item) for item in args.keep.split(",") if item]
    sobol_points = np.load(args.sobol)["points"]
    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[i] for i in range(min(args.limit, len(dataset)))]

    preds = []
    truths = []
    structures = []
    for index, row in enumerate(rows):
        pred, structure, _ = predict_current(row, sobol_points)
        preds.append(pred[-1])
        truths.append(np.asarray(row["final_means"], dtype=np.float64))
        structures.append(structure)
        print(f"loaded {index+1}/{len(rows)} {row['mlp_name']}", flush=True)

    train_n = min(args.train_count, len(rows) - 1)
    train_slice = slice(0, train_n)
    test_slice = slice(train_n, None)
    base_train = mean_mse(preds[train_slice], truths[train_slice])
    base_test = mean_mse(preds[test_slice], truths[test_slice])
    print(f"\nbase train={base_train:.3e} test={base_test:.3e}")
    print(f"{'mask':<12} {'keep':>5} {'kind':<12} {'proxy':<22} {'lambda':>9} {'train':>12} {'test':>12} {'oracle':>12} {'frac':>7}")

    configs = []
    for mask_mode in ("all_on", "alpha_ge6", "top_alpha", "top_pred", "top_adiff"):
        mode_keeps = [1.0] if mask_mode in {"all_on", "alpha_ge6"} else keeps
        for keep in mode_keeps:
            for kind in ("constant", "pred", "anal_gap_up", "adiff_mag"):
                for proxy in ("up", "down", "mean_pred_minus_anal", "mean_anal_minus_pred"):
                    configs.append((mask_mode, keep, kind, proxy))

    for mask_mode, keep, kind, proxy in configs:
        directions = []
        fractions = []
        for pred, structure in zip(preds, structures):
            mask = risky_mask(mask_mode, np.asarray([pred]), structure, keep)
            # risky_mask expects pred with final row at [-1]
            mask = risky_mask(mask_mode, pred[None, :], structure, keep)
            base_direction = correction_direction(kind, pred[None, :], structure, mask)
            directions.append(sign_for_proxy(proxy, pred[None, :], structure, mask) * base_direction)
            fractions.append(float(np.mean(mask)))
        lam = fit_lambda(preds[train_slice], directions[train_slice], truths[train_slice])
        lam = float(np.clip(lam, -2.0, 2.0))
        oracle_lam = fit_lambda(preds[test_slice], directions[test_slice], truths[test_slice])
        oracle_lam = float(np.clip(oracle_lam, -2.0, 2.0))
        train_pred = [pred + lam * direction for pred, direction in zip(preds[train_slice], directions[train_slice])]
        test_pred = [pred + lam * direction for pred, direction in zip(preds[test_slice], directions[test_slice])]
        oracle_pred = [pred + oracle_lam * direction for pred, direction in zip(preds[test_slice], directions[test_slice])]
        test_mse = mean_mse(test_pred, truths[test_slice])
        # Print only if it is not totally awful or beats base train/test enough to inspect.
        if test_mse < base_test * 1.02:
            print(
                f"{mask_mode:<12} {keep:5.2f} {kind:<12} {proxy:<22} {lam:+9.4f} "
                f"{mean_mse(train_pred, truths[train_slice]):12.3e} {test_mse:12.3e} "
                f"{mean_mse(oracle_pred, truths[test_slice]):12.3e} {np.mean(fractions):7.3f}"
            )


if __name__ == "__main__":
    main()
