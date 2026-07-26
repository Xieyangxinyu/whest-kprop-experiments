"""Truncation-bias curve for hot-block low-rank matmul (coldslice_blocks.ipynb
loose thread).

The continuation block's dense hot activation block is low-rank in value
(99.9% energy at r=48 vs matmul break-even r* ~ 82-93). Replacing X_hot with
its rank-r projection (basis fit on PILOT rows — the deployable analogue)
would cut the hot matmul from 2Nkn to 2Nr(k+n). The notebook's own caveat:
truncation is lossy and ReLU converts the residual into bias on the scored
mean. This probe measures that bias.

Protocol mirrors scripts/demote_meancomp_probe.py: 10 mini nets x 2 seeds,
N=8192 antithetic, pilot = first 1024 rows of each half. At layers 1..28
(inputs to consumers 2..29), post-ReLU activations of hot columns (pilot
fire rate >= 0.03) are projected onto the top-r right-singular subspace fit
on the pilot rows; cold columns stay exact. Layers 29-31 untouched. Metric:
paired final-mean MSE vs the exact forward on the same samples (reference
scale: demote probe base = 3.45e-08; ~1% production bar = 2e-09).
"""
import numpy as np
import whestbench as wb

N_NETS = 10
N_SAMPLES = 8192
SEEDS = [0, 1]
PILOT_ROWS = 1024          # per antithetic half
FIRE_THRESH = 0.03
RANKS = [8, 16, 32, 48, 64, 96]
TRUNC_LAYERS = range(1, 29)   # producers whose consumers are layers 2..29

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")


def forward(weights, x0, rank=None, energy_log=None):
    x = x0
    half = x0.shape[0] // 2
    for li, w in enumerate(weights):
        z = x @ w
        if li == len(weights) - 1:
            return np.maximum(z, 0.0).mean(axis=0)
        x = np.maximum(z, 0.0)
        if rank is not None and li in TRUNC_LAYERS:
            pilot = np.concatenate([x[:PILOT_ROWS], x[half:half + PILOT_ROWS]], axis=0)
            fire = (pilot > 0).mean(axis=0)
            hot = fire >= FIRE_THRESH
            k = int(hot.sum())
            if k <= rank:
                continue
            ph = pilot[:, hot]
            # top-r right-singular basis of the pilot hot block
            _, _, vt = np.linalg.svd(ph, full_matrices=False)
            B = vt[:rank]                      # (r, k)
            xh = x[:, hot]
            xh_t = (xh @ B.T) @ B
            if energy_log is not None:
                num = np.linalg.norm(xh - xh_t) ** 2
                den = np.linalg.norm(xh) ** 2
                energy_log.append(1.0 - num / max(den, 1e-30))
            x = x.copy()
            x[:, hot] = xh_t
    return None


results = {r: [] for r in RANKS}
energy = {r: [] for r in RANKS}
for i in range(N_NETS):
    mlp = wb.mlp_at(ds, i)
    weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        haf = rng.standard_normal((N_SAMPLES // 2, weights[0].shape[0]))
        x0 = np.concatenate([haf, -haf], axis=0)
        ref = forward(weights, x0)
        for r in RANKS:
            elog = []
            pred = forward(weights, x0, rank=r, energy_log=elog)
            results[r].append(np.mean((pred - ref) ** 2))
            energy[r].append(np.mean(elog))

print(f"=== hot-block truncation: paired final-mean MSE vs exact (n={N_NETS * len(SEEDS)}) ===")
print("reference: demote-probe base delta 3.45e-08; ~1% production bar 2e-09; break-even r* ~82-93")
print(f"{'rank':>5} {'mean delta2':>13} {'max':>11} {'mean energy kept':>17}")
for r in RANKS:
    arr = np.array(results[r])
    print(f"{r:>5} {arr.mean():>13.3e} {arr.max():>11.3e} {np.mean(energy[r]):>16.5%}")
