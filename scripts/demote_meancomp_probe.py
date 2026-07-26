"""Probe: mean-compensated demotion (bench_logs/submission_learnings_2026-07-25.md).

The demotion knee (analytic-alpha probe band <= -1.5, sampled-alpha demote
<= -2.33) cuts sample-propagation FLOPs but costs raw MSE (+1.3%, fragile to
+3.3% at the -2.45 neighbor). Open variant: add the demoted columns' analytic
mean contribution to the consuming layer's pre-activation as a constant row
(mu_post[D] @ w[D, :]) to cancel the first-order bias while keeping the cut.

Questions:
  Q1  Does compensation make the knee raw-neutral vs the production baseline
      (initial dead dropped, no demotion)?  Metric: final-layer mean MSE vs
      the exact full forward on the SAME samples (paired).
  Q2  Does compensation kill the threshold fragility (-2.33 vs -2.45)?
  Q3  Free upside: compensating the INITIAL dead set too (with or without
      the knee) — does it reduce baseline bias?
  Q4  FLOP proxy: mean active-column reduction per layer -> % of the
      sample-propagation matmul cost removed.

Offline numpy probe (no flopscope billing; mirrors scripts/identity_hot_probe.py).
"""
import math

import numpy as np
import whestbench as wb

_erf = np.frompyfunc(math.erf, 1, 1)

N_NETS = 10
N_SAMPLES = 8192
SEEDS = [0, 1]
PROBE_ROWS = 1024          # per antithetic half (2048 total), ~20% like the recheck stage
DEAD_THRESH = -3.0         # production analytic dead cut
PROBE_MAX = -1.5           # knee: analytic band eligible for demotion probing
DEMOTE_THRESHES = [-2.33, -2.45]
DEMOTE_LAYERS = range(1, 30)  # layers 1..29, matching _DEMOTE_ACTIVE_DEAD_*

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")


def analytic_moments(weights):
    """Diagonal Gaussian moment propagation, mirroring estimator.py."""
    alphas, mus = [], []
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
        mus.append(mu_post.copy())
    return alphas, mus


def forward_exact(weights, x0):
    x = x0
    for li, w in enumerate(weights):
        z = x @ w
        if li == len(weights) - 1:
            return np.maximum(z, 0.0).mean(axis=0)
        x = np.maximum(z, 0.0)
    return None


def sample_alpha(z, rows):
    """Antithetic-balanced sampled alpha from the first `rows` of each half."""
    half = z.shape[0] // 2
    zp = np.concatenate([z[:rows], z[half:half + rows]], axis=0)
    m = zp.mean(axis=0)
    s = np.sqrt(np.maximum(zp.var(axis=0), 1e-12))
    return m / s


def forward_variant(weights, x0, alphas, mus, demote_thresh=None,
                    comp_demoted=False, comp_dead=False, comp_source="analytic"):
    """Production-style forward: initial dead (alpha < -3) dropped everywhere;
    optional knee demotion in layers 1..29; optional constant-row mean
    compensation for demoted and/or dead columns. Returns (final_mean,
    active_counts). Demoted/dead columns are zeroed rather than removed —
    identical algebra to removal + compensation row.

    comp_source: 'analytic' (diagonal mu_post), 'probe' (post-ReLU mean over
    the demote probe's antithetic rows — what the estimator can compute for
    free at demote time), 'oracle' (all-row post-ReLU mean — noise floor of
    the mechanism). Dead-column comp is always analytic (never materialized)."""
    x = x0
    comp = 0.0  # constant row added to this layer's pre-activation
    counts = []
    half = x0.shape[0] // 2
    for li, w in enumerate(weights):
        z = x @ w + comp
        if li == len(weights) - 1:
            pred = np.maximum(z, 0.0).mean(axis=0)
            # dead_corrections analog: analytic mean for final-layer dead
            dead_f = alphas[li] < DEAD_THRESH
            pred[dead_f] = mus[li][dead_f]
            return pred, counts
        dead = alphas[li] < DEAD_THRESH
        dropped = dead.copy()
        demoted = np.zeros_like(dead)
        if demote_thresh is not None and li in DEMOTE_LAYERS:
            band = (~dead) & (alphas[li] <= PROBE_MAX)
            if band.any():
                sa = sample_alpha(z[:, band], PROBE_ROWS)
                demoted[np.where(band)[0][sa <= demote_thresh]] = True
                dropped |= demoted
        relu_z = np.maximum(z, 0.0)
        x = relu_z.copy()
        x[:, dropped] = 0.0
        counts.append(int((~dropped).sum()))
        comp_vec = np.zeros(w.shape[1])
        use = np.zeros_like(dead)
        if comp_demoted and demoted.any():
            if comp_source == "analytic":
                comp_vec[demoted] = mus[li][demoted]
            elif comp_source == "probe":
                zp = np.concatenate(
                    [relu_z[:PROBE_ROWS, demoted], relu_z[half:half + PROBE_ROWS, demoted]], axis=0
                )
                comp_vec[demoted] = zp.mean(axis=0)
            else:  # oracle
                comp_vec[demoted] = relu_z[:, demoted].mean(axis=0)
            use |= demoted
        if comp_dead:
            comp_vec[dead] = mus[li][dead]
            use |= dead
        if use.any():
            comp = comp_vec[use] @ weights[li + 1][use, :]
        else:
            comp = 0.0
    return None, counts


VARIANTS = [
    ("base_prod (dead dropped)", dict()),
    ("knee -2.33 bare", dict(demote_thresh=-2.33)),
    ("knee -2.33 +analytic comp", dict(demote_thresh=-2.33, comp_demoted=True)),
    ("knee -2.33 +probe comp", dict(demote_thresh=-2.33, comp_demoted=True, comp_source="probe")),
    ("knee -2.33 +oracle comp", dict(demote_thresh=-2.33, comp_demoted=True, comp_source="oracle")),
    ("knee -2.45 bare", dict(demote_thresh=-2.45)),
    ("knee -2.45 +probe comp", dict(demote_thresh=-2.45, comp_demoted=True, comp_source="probe")),
    ("comp dead only (no knee)", dict(comp_dead=True)),
]

mse = {name: [] for name, _ in VARIANTS}
act_counts = {name: [] for name, _ in VARIANTS}

for i in range(N_NETS):
    mlp = wb.mlp_at(ds, i)
    weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    alphas, mus = analytic_moments(weights)
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        half = rng.standard_normal((N_SAMPLES // 2, weights[0].shape[0]))
        x0 = np.concatenate([half, -half], axis=0)
        ref = forward_exact(weights, x0)
        for name, kw in VARIANTS:
            pred, counts = forward_variant(weights, x0, alphas, mus, **kw)
            mse[name].append(np.mean((pred - ref) ** 2))
            act_counts[name].append(counts)

base = np.array(mse["base_prod (dead dropped)"])
base_counts = np.array(act_counts["base_prod (dead dropped)"], dtype=float)
print(f"=== paired final-mean MSE vs exact forward (n={len(base)} net-seeds) ===")
print(f"{'variant':<30} {'mean MSE':>12} {'max':>12} {'vs base':>9} {'matmul dF':>10}")
for name, _ in VARIANTS:
    arr = np.array(mse[name])
    cnts = np.array(act_counts[name], dtype=float)
    # consumer matmul cost ~ sum over layers of k_in * n_out; input width fixed
    rel = (cnts.mean(axis=0).sum()) / (base_counts.mean(axis=0).sum()) - 1.0
    print(f"{name:<30} {arr.mean():>12.3e} {arr.max():>12.3e} "
          f"{arr.mean() / base.mean() - 1.0:>+8.1%} {rel:>+9.2%}")
print("\nproduction raw scale ~2e-7; a variant is 'raw-neutral' if its MSE-vs-exact")
print("stays within ~1% of base_prod's (the shared MC error dominates both).")
