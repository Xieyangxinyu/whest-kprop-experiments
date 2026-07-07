"""Attribute useful K3 correction to dead/kink/on block patterns.

For each transition and output neuron, the true K3 contraction is known from
pre_m3. The stored O(W^2) blocks provide the diagonal (3) and pair (2,1)
contributions. This script decomposes those stored pieces by source neuron class
patterns and evaluates which pattern groups help downstream ReLU mean accuracy.

It cannot directly materialize the fully off-diagonal KKK/OOK/OKK tensors, but it
can test the natural block hypotheses:
  - zero dead-involving stored pieces
  - on-only identity/pair pieces
  - kink-involving dense stored pieces
  - oracle omitted K3 as a residual upper bound
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from learn_cumulant_closure import _connected, _extract_blocks, _moment_path, _weights
from learn_k4_downstream_gate import DEPTH, WIDTH, _edgeworth_mean


CLASS_DEAD = 0
CLASS_KINK = 1
CLASS_ON = 2
CLASS_NAMES = {CLASS_DEAD: "D", CLASS_KINK: "K", CLASS_ON: "O"}


def _classes(npz, layer_idx: int, threshold: float) -> np.ndarray:
    mean = np.asarray(npz["pre_mean"][layer_idx], dtype=np.float64)
    m2 = np.asarray(npz["pre_m2"][layer_idx], dtype=np.float64)
    variance = np.maximum(m2 - mean * mean, 1e-12)
    alpha = mean / np.sqrt(variance)
    out = np.full(WIDTH, CLASS_KINK, dtype=np.int8)
    out[alpha < -threshold] = CLASS_DEAD
    out[alpha > threshold] = CLASS_ON
    return out


def _stored_k3_parts(npz, weights: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    post = _connected(npz, prefix="")
    parts = defaultdict(list)
    for layer_idx in range(DEPTH - 1):
        source_class = _classes(npz, layer_idx, threshold)
        w = weights[layer_idx + 1]
        c21 = post["c21"][layer_idx]
        mu3 = post["mu3"][layer_idx]
        diag_c21 = np.diagonal(c21)

        # Diagonal third cumulants: pattern DDD/KKK/OOO.
        for cls, label in CLASS_NAMES.items():
            mask = source_class == cls
            value = ((w[mask, :] ** 3) * mu3[mask, None]).sum(axis=0)
            parts[f"{label}{label}{label}_diag"].append(value)

        # Pair blocks: i,i,j. Pattern is class(i), class(i), class(j).
        for cls_i, label_i in CLASS_NAMES.items():
            mask_i = source_class == cls_i
            if not np.any(mask_i):
                value = np.zeros(WIDTH)
                for cls_j, label_j in CLASS_NAMES.items():
                    parts[f"{label_i}{label_i}{label_j}_pair"].append(value)
                continue
            w_i = w[mask_i, :]
            c21_i = c21[mask_i, :]
            diag_i = diag_c21[mask_i]
            for cls_j, label_j in CLASS_NAMES.items():
                mask_j = source_class == cls_j
                if not np.any(mask_j):
                    parts[f"{label_i}{label_i}{label_j}_pair"].append(np.zeros(WIDTH))
                    continue
                c21_ij = c21_i[:, mask_j]
                w_j = w[mask_j, :]
                unrestricted = ((w_i * w_i) * (c21_ij @ w_j)).sum(axis=0)
                diag_overlap = np.zeros(WIDTH)
                if cls_i == cls_j:
                    diag_overlap = ((w_i ** 3) * diag_i[:, None]).sum(axis=0)
                parts[f"{label_i}{label_i}{label_j}_pair"].append(3.0 * (unrestricted - diag_overlap))

    return {name: np.asarray(values) for name, values in parts.items()}


def _pre_arrays(npz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_mean = np.asarray(npz["pre_mean"], dtype=np.float64)[1:]
    pre_m2 = np.asarray(npz["pre_m2"], dtype=np.float64)[1:]
    true_post = np.asarray(npz["mean"], dtype=np.float64)[1:]
    pre_var = np.maximum(pre_m2 - pre_mean * pre_mean, 1e-12)
    return pre_mean, pre_var, true_post


def _score(name: str, pred_k3: np.ndarray, pre_mean: np.ndarray, pre_var: np.ndarray, true_post: np.ndarray, region: np.ndarray) -> float:
    pred = _edgeworth_mean(pre_mean[region], pre_var[region], pred_k3[region], np.zeros(np.count_nonzero(region)))
    return float(np.mean((pred - true_post[region]) ** 2))


def _region(mode: str) -> np.ndarray:
    layer_ids = np.arange(DEPTH - 1)[:, None].repeat(WIDTH, axis=1)
    if mode == "all":
        return np.ones((DEPTH - 1, WIDTH), dtype=bool)
    if mode == "late":
        return layer_ids >= 24
    if mode == "final":
        return layer_ids == DEPTH - 2
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    indices = [int(item) for item in args.indices.split(",") if item]
    pattern_scores = defaultdict(list)
    group_scores = defaultdict(list)

    for index in indices:
        npz = np.load(_moment_path(index))
        weights = _weights(index)
        block = _extract_blocks(index)
        pre_mean, pre_var, true_post = _pre_arrays(npz)
        parts = _stored_k3_parts(npz, weights, args.threshold)
        full3 = block.full3.reshape(DEPTH - 1, WIDTH)
        stored3 = block.stored3.reshape(DEPTH - 1, WIDTH)
        zero = np.zeros_like(full3)
        groups = {
            "zero": zero,
            "stored_all": stored3,
            "full_oracle": full3,
            "diag_only": sum(value for name, value in parts.items() if name.endswith("_diag")),
            "pair_only": sum(value for name, value in parts.items() if name.endswith("_pair")),
            "on_only": sum(value for name, value in parts.items() if set(name[:3]) <= {"O"}),
            "kink_involving": sum(value for name, value in parts.items() if "K" in name[:3]),
            "dead_involving": sum(value for name, value in parts.items() if "D" in name[:3]),
            "stored_no_dead": sum(value for name, value in parts.items() if "D" not in name[:3]),
            "stored_no_on_only": sum(value for name, value in parts.items() if not set(name[:3]) <= {"O"}),
        }
        for region_name in ("all", "late", "final"):
            mask = _region(region_name)
            for group_name, pred_k3 in groups.items():
                group_scores[(region_name, group_name)].append(_score(group_name, pred_k3, pre_mean, pre_var, true_post, mask))
            base = _score("zero", zero, pre_mean, pre_var, true_post, mask)
            for part_name, pred_k3 in parts.items():
                mse = _score(part_name, pred_k3, pre_mean, pre_var, true_post, mask)
                pattern_scores[(region_name, part_name)].append(mse - base)

    for region_name in ("all", "late", "final"):
        print(f"\n== {region_name} ==")
        for group_name in ("zero", "stored_all", "full_oracle", "diag_only", "pair_only", "on_only", "kink_involving", "dead_involving", "stored_no_dead", "stored_no_on_only"):
            values = group_scores[(region_name, group_name)]
            print(f"{group_name:<18} mse={np.mean(values):.3e}")
        ranked = sorted(
            ((np.mean(values), name) for (region, name), values in pattern_scores.items() if region == region_name),
            key=lambda item: item[0],
        )
        print("best individual stored patterns by delta vs zero:")
        for delta, name in ranked[:8]:
            print(f"  {name:<12} delta={delta:+.3e}")
        print("worst individual stored patterns by delta vs zero:")
        for delta, name in ranked[-8:]:
            print(f"  {name:<12} delta={delta:+.3e}")


if __name__ == "__main__":
    main()