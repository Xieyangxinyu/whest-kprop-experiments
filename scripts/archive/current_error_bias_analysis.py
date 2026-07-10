"""Analyze current estimator final-layer error bias on a public dataset split.

This script uses baked ground truth from the public dataset and reports error
patterns by final neuron class, alpha bins, prediction magnitude, analytical
agreement, and simple feature correlations.
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


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def summarize_group(label: str, err: np.ndarray) -> None:
    if err.size == 0:
        return
    print(
        f"{label:<26} n={err.size:6d} mean={np.mean(err):+.3e} "
        f"mse={np.mean(err*err):.3e} mae={np.mean(np.abs(err)):.3e} "
        f"under={np.mean(err < 0):.3f}"
    )


def bin_report(name: str, values: np.ndarray, err: np.ndarray, quantiles=(0, .1, .25, .5, .75, .9, 1.0)) -> None:
    edges = np.quantile(values, quantiles)
    print(f"\n== bins by {name} ==")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        summarize_group(f"{name} [{lo:.3g},{hi:.3g}]", err[mask])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sobol", default=str(REPO_ROOT / "sobol_points.npz"))
    args = parser.parse_args()

    sobol_points = np.load(args.sobol)["points"]
    dataset = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split=args.split)
    rows = [dataset[i] for i in range(min(args.limit, len(dataset)))]

    all_err = []
    all_pred = []
    all_truth = []
    all_alpha = []
    all_anal = []
    all_final_class = []
    all_l30_class = []
    all_adiff = []
    per_mlp = []

    for idx, row in enumerate(rows):
        pred, structure, flops_used = predict_current(row, sobol_points)
        truth = np.asarray(row["final_means"], dtype=np.float64)
        final_pred = pred[-1]
        err = final_pred - truth
        alpha = np.asarray(structure["alpha_rows"][-1], dtype=np.float64)
        analytical = np.asarray(structure["analytical_rows"][-1], dtype=np.float64)
        final_class = np.full(WIDTH, 1, dtype=np.int8)  # kink
        final_class[np.asarray(structure["dead_indices"][-1], dtype=np.int64)] = 0
        final_class[np.asarray(structure["on_indices"][-1], dtype=np.int64)] = 2
        l30_class = np.full(WIDTH, 1, dtype=np.int8)
        l30_class[np.asarray(structure["dead_indices"][-2], dtype=np.int64)] = 0
        l30_class[np.asarray(structure["on_indices"][-2], dtype=np.int64)] = 2
        adiff = final_pred - analytical

        all_err.append(err)
        all_pred.append(final_pred)
        all_truth.append(truth)
        all_alpha.append(alpha)
        all_anal.append(analytical)
        all_final_class.append(final_class)
        all_l30_class.append(l30_class)
        all_adiff.append(adiff)
        per_mlp.append(
            {
                "name": row["mlp_name"],
                "mse": float(np.mean(err * err)),
                "mean_err": float(np.mean(err)),
                "under": float(np.mean(err < 0)),
                "final_on": int(np.sum(final_class == 2)),
                "final_kink": int(np.sum(final_class == 1)),
                "l30_on": int(np.sum(l30_class == 2)),
                "flops": flops_used,
            }
        )
        print(f"loaded {idx+1}/{len(rows)} {row['mlp_name']}", flush=True)

    err = np.concatenate(all_err)
    pred = np.concatenate(all_pred)
    truth = np.concatenate(all_truth)
    alpha = np.concatenate(all_alpha)
    analytical = np.concatenate(all_anal)
    final_class = np.concatenate(all_final_class)
    l30_class = np.concatenate(all_l30_class)
    adiff = np.concatenate(all_adiff)

    print("\n== overall final error ==")
    summarize_group("all", err)
    for label, cls in (("dead", 0), ("kink", 1), ("on", 2)):
        summarize_group(f"final {label}", err[final_class == cls])
    for label, cls in (("l30 dead", 0), ("l30 kink", 1), ("l30 on", 2)):
        summarize_group(label, err[l30_class == cls])

    print("\n== correlations with error and squared error ==")
    features = {
        "pred": pred,
        "truth": truth,
        "alpha": alpha,
        "analytical": analytical,
        "pred_minus_anal": adiff,
        "abs_pred_minus_anal": np.abs(adiff),
        "abs_pred": np.abs(pred),
        "final_on": (final_class == 2).astype(float),
        "final_kink": (final_class == 1).astype(float),
    }
    for name, values in features.items():
        print(f"{name:<22} corr_err={corr(values, err):+.3f} corr_sqerr={corr(values, err*err):+.3f}")

    bin_report("alpha", alpha, err)
    bin_report("prediction", pred, err)
    bin_report("pred_minus_anal", adiff, err)
    bin_report("abs_pred_minus_anal", np.abs(adiff), err)

    print("\n== worst MLPs by final MSE ==")
    for item in sorted(per_mlp, key=lambda x: x["mse"], reverse=True)[:15]:
        print(
            f"{item['name']:<24} mse={item['mse']:.3e} mean_err={item['mean_err']:+.3e} "
            f"under={item['under']:.3f} final_on={item['final_on']:3d} final_kink={item['final_kink']:3d} "
            f"l30_on={item['l30_on']:3d} flops={item['flops']:.2e}"
        )


if __name__ == "__main__":
    main()
