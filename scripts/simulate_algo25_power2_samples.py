from __future__ import annotations

import argparse
import importlib.util
import os
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import whestbench
from scipy import special
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"
BUDGET = 272_000_000_000
DEFAULT_START_SAMPLES = 32768
DEFAULT_FINAL_SAMPLES = 65536


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _parse_variants(raw: str) -> list[tuple[str, bool, int | None]]:
    variants = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        lowered = token.lower()
        if lowered in {"joekuo", "joe-kuo", "deterministic"}:
            variants.append(("joekuo", False, None))
        elif lowered.startswith("owen:"):
            seed = int(lowered.split(":", 1)[1])
            variants.append((f"owen{seed}", True, seed))
        elif lowered.startswith("owen") and lowered[4:].isdigit():
            seed = int(lowered[4:])
            variants.append((f"owen{seed}", True, seed))
        else:
            raise ValueError(f"Unknown variant {token!r}; use joekuo or owen:<seed>.")
    return variants


def generate_normal_sobol(width: int, half_rows: int, *, scramble: bool, seed: int | None) -> np.ndarray:
    if not _is_power_of_two(half_rows):
        raise ValueError(f"half_rows must be a power of two, got {half_rows}")
    engine = qmc.Sobol(d=width, scramble=scramble, seed=seed)
    uniform = engine.random_base2(m=int(np.log2(half_rows)))
    clipped = np.clip(uniform, 1e-7, 1.0 - 1e-7)
    return special.ndtri(clipped).astype(np.float32, copy=False)


def load_algo25(points: np.ndarray):
    os.environ["WHEST_DIAG_SAMPLE_DIVISOR"] = "1"
    os.environ["WHEST_DIAG_ANTITHETIC"] = "1"
    estimator_path = ROOT / "examples" / "25_best_of_both_worlds.py"
    spec = importlib.util.spec_from_file_location(
        f"algo25_power2_{time.time_ns()}", estimator_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    estimator._sobol_points = points
    return module, estimator


def predict_power2(module, estimator, source_mlp, start_samples: int, final_samples: int):
    if start_samples >= final_samples:
        raise ValueError("start_samples must be smaller than final_samples")
    if not _is_power_of_two(start_samples) or not _is_power_of_two(final_samples):
        raise ValueError("start_samples and final_samples must both be powers of two")
    if start_samples % 2 or final_samples % 2:
        raise ValueError("effective sample counts must be even for antithetic sampling")

    mlp = module._fast_mlp(source_mlp)
    width = mlp.width
    structure = estimator._initial_structure(mlp, width)

    start_x = estimator._sample_block(0, start_samples // 2, width)
    start_rows, refined = estimator._run_block(mlp, structure, start_x, start_samples, refine=True)

    extra_samples = final_samples - start_samples
    extra_x = estimator._sample_block(start_samples // 2, extra_samples // 2, width)
    extra_rows = estimator._run_block(mlp, refined, extra_x, extra_samples, refine=False)[0]

    final_rows = [
        (start_row * start_samples + extra_row * extra_samples) / final_samples
        for start_row, extra_row in zip(start_rows, extra_rows)
    ]
    return (
        module.fnp.stack(start_rows, axis=0),
        module.fnp.stack(final_rows, axis=0),
        module.fnp.stack(extra_rows, axis=0),
    )


def summarize(detail: pd.DataFrame, start_samples: int, final_samples: int) -> pd.DataFrame:
    rows = []
    for variant, group in detail.groupby("variant"):
        delta_col = f"delta_{final_samples}_vs_{start_samples}"
        rows.append(
            {
                "variant": variant,
                "schedule": f"pow2_{start_samples}_{final_samples}",
                "mean_mse_start": group[f"mse_{start_samples}"].mean(),
                "mean_mse_final": group[f"mse_{final_samples}"].mean(),
                "median_mse_final": group[f"mse_{final_samples}"].median(),
                "mean_delta_final_start": group[delta_col].mean(),
                "median_delta_final_start": group[delta_col].median(),
                "worsened_final_vs_start": int((group[delta_col] > 0.0).sum()),
                "mean_extra_block_mse": group[f"extra_{final_samples - start_samples}_mse"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mse_final")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate Algorithm 25 with power-of-two Sobol antithetic sample cadences."
    )
    parser.add_argument("--split", default="mini", choices=("mini", "full"))
    parser.add_argument("--n-mlps", type=int, default=100)
    parser.add_argument("--start-samples", type=int, default=DEFAULT_START_SAMPLES)
    parser.add_argument("--final-samples", type=int, default=DEFAULT_FINAL_SAMPLES)
    parser.add_argument("--variants", default="joekuo,owen:404")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "bench_logs" / "sample_diagnostics")
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    if args.start_samples % 2 or args.final_samples % 2:
        raise ValueError("Power-of-two antithetic schedules require even effective sample counts")
    if not _is_power_of_two(args.start_samples) or not _is_power_of_two(args.final_samples):
        raise ValueError("Use power-of-two effective sample counts, such as 32768 and 65536")

    variants = _parse_variants(args.variants)
    dataset = whestbench.load_dataset(DATASET, revision=REVISION, split=args.split)
    n_mlps = min(args.n_mlps, len(dataset))
    rows = [dataset[index] for index in range(n_mlps)]
    mlps = list(islice(whestbench.iter_mlps(dataset), n_mlps))

    detail_rows = []
    start = time.perf_counter()
    for name, scramble, seed in variants:
        points = generate_normal_sobol(
            mlps[0].width,
            args.final_samples // 2,
            scramble=scramble,
            seed=seed,
        )
        module, estimator = load_algo25(points)
        variant_name = f"{name}_pow2_antithetic"
        for index, (source_mlp, row) in enumerate(zip(mlps, rows)):
            pred_start, pred_final, pred_extra = predict_power2(
                module,
                estimator,
                source_mlp,
                args.start_samples,
                args.final_samples,
            )
            truth = np.asarray(row["final_means"], dtype=np.float64)
            start_final = np.asarray(pred_start[-1], dtype=np.float64)
            final = np.asarray(pred_final[-1], dtype=np.float64)
            extra = np.asarray(pred_extra[-1], dtype=np.float64)
            mse_start = float(np.mean((start_final - truth) ** 2))
            mse_final = float(np.mean((final - truth) ** 2))
            mse_extra = float(np.mean((extra - truth) ** 2))
            detail_rows.append(
                {
                    "variant": variant_name,
                    "mlp_index": index,
                    "mlp_name": row["mlp_name"],
                    f"mse_{args.start_samples}": mse_start,
                    f"mse_{args.final_samples}": mse_final,
                    f"delta_{args.final_samples}_vs_{args.start_samples}": mse_final - mse_start,
                    f"extra_{args.final_samples - args.start_samples}_mse": mse_extra,
                }
            )
            print(
                f"{variant_name:<24} {index + 1:04d}/{n_mlps:04d} {row['mlp_name']:<24} "
                f"mse{args.start_samples}={mse_start:.3e} "
                f"mse{args.final_samples}={mse_final:.3e}",
                flush=True,
            )

    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail, args.start_samples, args.final_samples)
    elapsed = time.perf_counter() - start

    print("\n== summary ==")
    print(summary.to_string(index=False))
    print(f"elapsed_s={elapsed:.3f}")

    if args.write_csv:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"algo25_power2_{args.split}_n{n_mlps}_{args.start_samples}_{args.final_samples}"
        detail_path = args.output_dir / f"{stem}.csv"
        summary_path = args.output_dir / f"{stem}_summary.csv"
        detail.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"detail_csv={detail_path}")
        print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()