"""Probe: treat on/high-alpha neurons as identity (skip ReLU) in layers 1..29.

Questions:
  Q1  How many neurons per layer exceed alpha thresholds {3, 3.5, 4, 4.5, 5}
      (analytic diagonal alpha, same as the estimator's classifier)?
  Q2  Paired final-mean bias of identity treatment (no maximum on those
      columns) at each threshold, vs the exact-ReLU baseline on the SAME
      samples. Deterministic treatment effect; bar = delta2 must stay well
      under ~1% of production raw MSE scale (~2e-9 on 2e-7).
  Q3  Variant D (hot mean-substitution: replace on-column samples with the
      column mean) at t=4 — expected fatal via variance deletion; quantify.
  Q4  Economics: billed saving = N * |O_l(t)| * 1/elem per layer (maximum
      skipped on the hot slab; slab split via existing free slicing), vs
      142G billed at N=61,440.  Composition crossover check: |O_l| vs
      k_active^2/(2*k_active) ~ k/2 per hop.

Offline numpy probe (no flopscope billing here; dtype irrelevant).
"""
import math

import numpy as np
import whestbench as wb

_erf = np.frompyfunc(math.erf, 1, 1)

THRESHOLDS = [3.0, 3.5, 4.0, 4.5, 5.0]
N_NETS = 10
N_SAMPLES = 8192
SEEDS = [0, 1]

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")


def analytic_alpha(weights):
    """Diagonal Gaussian moment propagation, mirroring estimator.py lines 336-363."""
    alphas = []
    mu_post = None
    var_post = None
    for li, w in enumerate(weights):
        if li == 0:
            mu_pre = np.zeros(w.shape[1])
            var_pre = np.sum(w * w, axis=0)
        else:
            mu_pre = w.T @ mu_post
            var_pre = np.sum(w * w * var_post[:, None], axis=0)
        var_pre = np.maximum(var_pre, 1e-12)
        sigma = np.sqrt(var_pre)
        a = mu_pre / sigma
        alphas.append(a)
        phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1.0 + _erf(a / np.sqrt(2)).astype(np.float64))
        mu_post = mu_pre * Phi + sigma * phi
        var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma * phi - mu_post * mu_post
        var_post = np.maximum(var_post, 1e-12)
    return alphas


def forward(weights, x0, id_masks=None, meansub_masks=None):
    """Forward pass. id_masks[l]: bool per neuron -> skip relu (identity).
    meansub_masks[l]: bool -> replace column with its empirical mean."""
    x = x0
    for li, w in enumerate(weights):
        z = x @ w
        if li == len(weights) - 1:
            return np.maximum(z, 0.0).mean(axis=0)
        if id_masks is not None and id_masks[li] is not None:
            m = id_masks[li]
            x = np.where(m[None, :], z, np.maximum(z, 0.0))
        else:
            x = np.maximum(z, 0.0)
        if meansub_masks is not None and meansub_masks[li] is not None:
            mm = meansub_masks[li]
            col_mean = x.mean(axis=0)
            x = np.where(mm[None, :], col_mean[None, :], x)
    return None


results = {t: [] for t in THRESHOLDS}
results_d = []
counts = {t: np.zeros(32) for t in THRESHOLDS}
k_active_all = np.zeros(32)
base_mse_gt = []

for i in range(N_NETS):
    mlp = wb.mlp_at(ds, i)
    weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    gt = np.asarray(ds[i]["final_means"], dtype=np.float64)
    alphas = analytic_alpha(weights)
    for t in THRESHOLDS:
        for li, a in enumerate(alphas):
            counts[t][li] += (a > t).sum()
    for li, a in enumerate(alphas):
        k_active_all[li] += (a > -3.0).sum()  # ~active set size proxy

    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        half = rng.standard_normal((N_SAMPLES // 2, weights[0].shape[0]))
        x0 = np.concatenate([half, -half], axis=0)  # antithetic, like production

        base = forward(weights, x0)
        base_mse_gt.append(np.mean((base - gt) ** 2))

        for t in THRESHOLDS:
            masks = [alphas[li] > t if 1 <= li <= 29 else None for li in range(32)]
            treated = forward(weights, x0, id_masks=masks)
            results[t].append(np.mean((treated - base) ** 2))

        # Variant D at t=4 only
        mmasks = [alphas[li] > 4.0 if 1 <= li <= 29 else None for li in range(32)]
        treated_d = forward(weights, x0, meansub_masks=mmasks)
        results_d.append(np.mean((treated_d - base) ** 2))

print("=== Q1: neurons with alpha > t (mean per layer, 10 nets) ===")
print(f"{'layer':>5}", *[f"t>{t}" for t in THRESHOLDS], "k_act", sep="\t")
for li in range(32):
    print(f"{li:>5}", *[f"{counts[t][li]/N_NETS:.1f}" for t in THRESHOLDS],
          f"{k_active_all[li]/N_NETS:.0f}", sep="\t")
tot = {t: counts[t][1:30].sum() / N_NETS for t in THRESHOLDS}
print("total |O| layers 1-29 per net:", {t: round(v, 1) for t, v in tot.items()})

print("\n=== Q2: identity-skip paired delta^2 (mean over nets x seeds) ===")
for t in THRESHOLDS:
    arr = np.array(results[t])
    print(f"t={t}: mean {arr.mean():.3e}  max {arr.max():.3e}  (n={len(arr)})")

print("\n=== Q3: hot mean-substitution t=4 ===")
arr = np.array(results_d)
print(f"mean {arr.mean():.3e}  max {arr.max():.3e}")

print(f"\nbaseline-vs-GT final MSE at N={N_SAMPLES}: {np.mean(base_mse_gt):.3e}")
print("production raw scale reference: ~2e-7; 1% bar: ~2e-9")

print("\n=== Q4: economics at N=61,440, billed total ~142G ===")
for t in THRESHOLDS:
    saving = 61440 * tot[t] / 1e9
    print(f"t={t}: skip-maximum saving ~{saving:.3f}G = {100*saving/142:.4f}% of billed")
kbar = k_active_all[1:30].mean() / N_NETS
print(f"mean k_active layers1-29: {kbar:.0f}; composition crossover needs |O_l| > ~{kbar/2:.0f}/layer")
