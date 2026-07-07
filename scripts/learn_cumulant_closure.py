"""Learn sparse closures for omitted higher-cumulant contractions.

This script uses Keenan Pepper's public higher-moment dataset as supervision.
For each layer transition and output neuron it decomposes the true next
pre-activation cumulants into stored O(W^2) blocks plus omitted blocks:

    kappa3(pre_next) = b3 + b21 + b111
    kappa4(pre_next) = b4 + b31 + b22 + b_omitted

The goal is to test whether the omitted pieces are predictable from block-level
features that a deployable moment estimator could plausibly compute, and whether
small gates can decide when to keep/zero a learned correction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


MOMENT_REPO = "keenanpepper/arc-whestbench-higher-moments-2026"
OFFICIAL_REPO = "aicrowd/arc-whestbench-public-2026"
OFFICIAL_REVISION = "v1-phase1"
WIDTH = 256
DEPTH = 32
N_FULL_SHARDS = 28


@dataclass
class Blocks:
    features: np.ndarray
    full3: np.ndarray
    stored3: np.ndarray
    omitted3: np.ndarray
    full4: np.ndarray
    stored4: np.ndarray
    omitted4: np.ndarray


def _moment_path(index: int) -> str:
    return hf_hub_download(MOMENT_REPO, f"full/mlp_{index:05d}.npz", repo_type="dataset")


@lru_cache(maxsize=1)
def _first_full_shard_weights() -> np.ndarray:
    path = hf_hub_download(
        OFFICIAL_REPO,
        f"data/full-00000-of-{N_FULL_SHARDS:05d}.parquet",
        revision=OFFICIAL_REVISION,
        repo_type="dataset",
    )
    table = pq.read_table(path, columns=["weights"])
    return np.asarray([table.column("weights")[i].as_py() for i in range(table.num_rows)], dtype=np.float64)


def _weights(index: int) -> np.ndarray:
    weights = _first_full_shard_weights()
    if index >= weights.shape[0]:
        raise ValueError("this quick probe currently expects indices from full shard 0")
    return weights[index]


def _connected(npz, prefix: str = "") -> dict[str, np.ndarray]:
    g = lambda key: np.asarray(npz[prefix + key], dtype=np.float64)
    mu = g("mean")
    m2 = g("m2")
    m3 = g("m3")
    m4 = g("m4")
    m11 = g("M11")
    m21 = g("M21")
    m22 = g("M22")
    m31 = g("M31")
    m12 = m21.transpose(0, 2, 1)

    mi = mu[:, :, None]
    mj = mu[:, None, :]
    m2i = m2[:, :, None]
    m2j = m2[:, None, :]
    m3i = m3[:, :, None]
    cov = m11 - mi * mj
    var = np.diagonal(cov, axis1=1, axis2=2)
    vi = var[:, :, None]
    vj = var[:, None, :]

    c21 = m21 - 2.0 * mi * m11 - m2i * mj + 2.0 * mi * mi * mj
    cen22 = (
        m22
        - 2.0 * mj * m21
        - 2.0 * mi * m12
        + mj * mj * m2i
        + mi * mi * m2j
        + 4.0 * mi * mj * m11
        - 3.0 * mi * mi * mj * mj
    )
    c22 = cen22 - vi * vj - 2.0 * cov * cov
    cen31 = m31 - mj * m3i - 3.0 * mi * m21 + 3.0 * mi * mj * m2i + 3.0 * mi * mi * m11 - 3.0 * mi**3 * mj
    c31 = cen31 - 3.0 * vi * cov
    mu3 = m3 - 3.0 * mu * m2 + 2.0 * mu**3
    central4 = m4 - 4.0 * mu * m3 + 6.0 * mu * mu * m2 - 3.0 * mu**4
    k4 = central4 - 3.0 * var * var
    return {"cov": cov, "c21": c21, "c22": c22, "c31": c31, "mu3": mu3, "k4": k4, "var": var, "mean": mu}


def _pre_marginal_cumulants(npz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(npz["pre_mean"], dtype=np.float64)
    m2 = np.asarray(npz["pre_m2"], dtype=np.float64)
    m3 = np.asarray(npz["pre_m3"], dtype=np.float64)
    m4 = np.asarray(npz["pre_m4"], dtype=np.float64)
    var = np.maximum(m2 - mean * mean, 1e-12)
    k3 = m3 - 3.0 * mean * m2 + 2.0 * mean**3
    central4 = m4 - 4.0 * mean * m3 + 6.0 * mean * mean * m2 - 3.0 * mean**4
    k4 = central4 - 3.0 * var * var
    return k3, k4, var


def _extract_blocks(index: int) -> Blocks:
    npz = np.load(_moment_path(index))
    weights = _weights(index)
    post = _connected(npz, prefix="")
    pre_k3, pre_k4, pre_var = _pre_marginal_cumulants(npz)

    features = []
    full3 = []
    stored3 = []
    omitted3 = []
    full4 = []
    stored4 = []
    omitted4 = []

    for layer_idx in range(DEPTH - 1):
        w = weights[layer_idx + 1]
        w2 = w * w
        w3 = w2 * w
        w4 = w2 * w2
        diag_c21 = np.diagonal(post["c21"][layer_idx])
        diag_c22 = np.diagonal(post["c22"][layer_idx])
        diag_c31 = np.diagonal(post["c31"][layer_idx])

        b3 = (w3 * post["mu3"][layer_idx, :, None]).sum(axis=0)
        c21w = post["c21"][layer_idx] @ w
        b21_unrestricted = (w2 * c21w).sum(axis=0)
        b21_diag = (w3 * diag_c21[:, None]).sum(axis=0)
        b21 = 3.0 * (b21_unrestricted - b21_diag)
        s3 = b3 + b21
        f3 = pre_k3[layer_idx + 1]
        o3 = f3 - s3

        b4 = (w4 * post["k4"][layer_idx, :, None]).sum(axis=0)
        c31w = post["c31"][layer_idx] @ w
        b31 = 4.0 * ((w3 * c31w).sum(axis=0) - (w4 * diag_c31[:, None]).sum(axis=0))
        c22w2 = post["c22"][layer_idx] @ w2
        b22 = 3.0 * ((w2 * c22w2).sum(axis=0) - (w4 * diag_c22[:, None]).sum(axis=0))
        s4 = b4 + b31 + b22
        f4 = pre_k4[layer_idx + 1]
        o4 = f4 - s4

        weight_l2 = np.sqrt(np.maximum(w2.sum(axis=0), 1e-12))
        weight_l3 = np.cbrt(np.maximum(np.abs(w) ** 3, 0.0).sum(axis=0))
        weight_l4 = np.maximum(w4.sum(axis=0), 1e-30) ** 0.25
        prev_mean_abs = np.full(WIDTH, np.mean(np.abs(post["mean"][layer_idx])))
        prev_var_mean = np.full(WIDTH, np.mean(post["var"][layer_idx]))
        layer_frac = np.full(WIDTH, layer_idx / (DEPTH - 2))
        feat = np.stack(
            [
                np.ones(WIDTH),
                layer_frac,
                b3,
                b21,
                s3,
                np.abs(b3),
                np.abs(b21),
                b4,
                b31,
                b22,
                s4,
                np.abs(s4),
                weight_l2,
                weight_l3,
                weight_l4,
                pre_var[layer_idx + 1],
                prev_mean_abs,
                prev_var_mean,
            ],
            axis=1,
        )
        features.append(feat)
        full3.append(f3)
        stored3.append(s3)
        omitted3.append(o3)
        full4.append(f4)
        stored4.append(s4)
        omitted4.append(o4)

    return Blocks(
        np.concatenate(features, axis=0),
        np.concatenate(full3),
        np.concatenate(stored3),
        np.concatenate(omitted3),
        np.concatenate(full4),
        np.concatenate(stored4),
        np.concatenate(omitted4),
    )


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train_x - center) / scale, (test_x - center) / scale


def _expand_features(features: np.ndarray) -> np.ndarray:
    eps = 1e-9
    block_values = features[:, 2:12]
    ratios = np.stack(
        [
            features[:, 4] / (np.abs(features[:, 2]) + np.abs(features[:, 3]) + eps),
            features[:, 10] / (np.abs(features[:, 7]) + np.abs(features[:, 8]) + np.abs(features[:, 9]) + eps),
            features[:, 3] / (np.abs(features[:, 2]) + eps),
            features[:, 8] / (np.abs(features[:, 7]) + eps),
            features[:, 9] / (np.abs(features[:, 7]) + eps),
        ],
        axis=1,
    )
    products = np.stack(
        [
            features[:, 2] * features[:, 3],
            features[:, 7] * features[:, 8],
            features[:, 7] * features[:, 9],
            features[:, 8] * features[:, 9],
            features[:, 15] * features[:, 1],
        ],
        axis=1,
    )
    return np.concatenate(
        [
            features,
            np.abs(block_values),
            np.sign(block_values) * np.sqrt(np.abs(block_values) + eps),
            np.log1p(np.abs(block_values)),
            ratios,
            products,
        ],
        axis=1,
    )


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    rhs = x.T @ y
    return np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), rhs)


def _relative_mse(pred: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> tuple[float, float, float]:
    base = float(np.mean((baseline - target) ** 2))
    model = float(np.mean((pred - target) ** 2))
    zero = float(np.mean(target * target))
    return base, model, zero


def _report(name: str, full: np.ndarray, stored: np.ndarray, omitted: np.ndarray, pred_omitted: np.ndarray) -> None:
    base, model, zero = _relative_mse(stored + pred_omitted, full, stored)
    oracle_scale = float(np.mean(omitted * omitted))
    corr = float(np.corrcoef(pred_omitted, omitted)[0, 1]) if np.std(pred_omitted) > 0 else 0.0
    print(
        f"{name:<8} full_mse stored={base:.3e} learned={model:.3e} zero_full={zero:.3e} "
        f"omitted_mse={oracle_scale:.3e} corr={corr:+.3f}"
    )
    for keep in (0.05, 0.10, 0.20, 0.40):
        threshold = np.quantile(np.abs(pred_omitted), 1.0 - keep)
        mask = np.abs(pred_omitted) >= threshold
        pred_gated = np.where(mask, pred_omitted, 0.0)
        oracle_gated = np.where(mask, omitted, 0.0)
        true_threshold = np.quantile(np.abs(omitted), 1.0 - keep)
        true_mask = np.abs(omitted) >= true_threshold
        recall = float(np.mean(mask[true_mask])) if np.any(true_mask) else 0.0
        pred_mse = float(np.mean((stored + pred_gated - full) ** 2))
        oracle_mask_mse = float(np.mean((stored + oracle_gated - full) ** 2))
        print(
            f"  keep_top={keep:0.2f} pred_mse={pred_mse:.3e} "
            f"oracle_mask_mse={oracle_mask_mse:.3e} recall_top={recall:.3f} active={np.mean(mask):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--feature-mode", choices=["simple", "expanded"], default="expanded")
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    blocks = [_extract_blocks(index) for index in indices]
    train_blocks = blocks[: args.train_count]
    test_blocks = blocks[args.train_count :]
    if not train_blocks or not test_blocks:
        raise SystemExit("need non-empty train and test splits")

    train_x = np.concatenate([block.features for block in train_blocks])
    test_x = np.concatenate([block.features for block in test_blocks])
    if args.feature_mode == "expanded":
        train_x = _expand_features(train_x)
        test_x = _expand_features(test_x)
    train_x, test_x = _standardize(train_x, test_x)

    train_o3 = np.concatenate([block.omitted3 for block in train_blocks])
    test_o3 = np.concatenate([block.omitted3 for block in test_blocks])
    train_full3 = np.concatenate([block.full3 for block in train_blocks])
    test_full3 = np.concatenate([block.full3 for block in test_blocks])
    train_stored3 = np.concatenate([block.stored3 for block in train_blocks])
    test_stored3 = np.concatenate([block.stored3 for block in test_blocks])

    train_o4 = np.concatenate([block.omitted4 for block in train_blocks])
    test_o4 = np.concatenate([block.omitted4 for block in test_blocks])
    train_full4 = np.concatenate([block.full4 for block in train_blocks])
    test_full4 = np.concatenate([block.full4 for block in test_blocks])
    train_stored4 = np.concatenate([block.stored4 for block in train_blocks])
    test_stored4 = np.concatenate([block.stored4 for block in test_blocks])

    coef3 = _fit_ridge(train_x, train_o3, args.ridge)
    coef4 = _fit_ridge(train_x, train_o4, args.ridge)
    print(f"loaded train={indices[:args.train_count]} test={indices[args.train_count:]} features={args.feature_mode}")
    _report("kappa3", test_full3, test_stored3, test_o3, test_x @ coef3)
    _report("kappa4", test_full4, test_stored4, test_o4, test_x @ coef4)
    print("coef3", np.array2string(coef3, precision=3, suppress_small=False))
    print("coef4", np.array2string(coef4, precision=3, suppress_small=False))


if __name__ == "__main__":
    main()