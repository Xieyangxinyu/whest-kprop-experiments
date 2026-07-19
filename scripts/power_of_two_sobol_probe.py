"""Probe: does a power-of-2 Sobol prefix (2^14 / 2^15 half-samples) beat the
shipped 30,720-half prefix on RAW final-layer MSE?

Hypothesis (2026-07-19): scrambled-Sobol prefixes are balanced (t,m,d)-nets
only at power-of-2 lengths; the shipped 30,720 = 2^11*15 prefix may leave
accuracy on the table vs n_half = 16,384 (2^14) or 32,768 (2^15).

Protocol (per repo memory):
- 12 He-init 256x32 nets (fleet aggregate; single-net deltas are noise).
- 8 independent scrambled-Sobol realizations, SAME 8 seeds across nets
  (mirrors the fleet sharing one artifact); balance effect is judged on the
  scramble-mean with SEM over scrambles; scramble spread = realization-luck
  band (the shipping risk for a fresh artifact).
- Antithetic pairing exactly like the estimator: x and -x halves.
- Cumulative-prefix evaluation: one forward pass per (net, scramble) gives
  MSE(n) at every checkpoint.
- GT: 2^21 MC samples/net (float64 accumulation), GT noise floor estimated
  as mean(per-neuron var)/N_gt and reported (additive constant across n).
- Checkpoints bracket 2^14 and 2^15 at +/-256 to resolve a local dip vs a
  smooth b/n fit through non-power-of-2 points.

Pure NumPy+scipy offline diagnostic - not scored code.
Run: uv run --with scipy python scripts/power_of_two_sobol_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

WIDTH = 256
DEPTH = 32
N_NETS = int(os.environ.get("N_NETS", "12"))
N_SCRAMBLES = int(os.environ.get("N_SCRAMBLES", "8"))
MAX_HALF = 32768  # 2^15
GT_SAMPLES = int(os.environ.get("GT_SAMPLES", str(2**21)))
GT_BATCH = 131072
OUT = Path(__file__).resolve().parent.parent / "bench_logs" / "power_of_two_sobol_probe_2026-07-19.npz"

CHECKPOINTS = [
    8192, 10240, 12288, 14336, 15360, 16128, 16384, 16640, 18432,
    20480, 24576, 28672, 30464, 30720, 30976, 32512, 32768,
]


def build_weights(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = (2.0 / WIDTH) ** 0.5
    return [
        (rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
        for _ in range(DEPTH)
    ]


def final_acts(weights, x):
    for w in weights:
        x = np.maximum(x @ w, 0.0)
    return x


def gt_mean_and_floor(weights, seed: int):
    """High-sample MC ground truth + estimated GT noise floor."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(WIDTH, dtype=np.float64)
    sq = np.zeros(WIDTH, dtype=np.float64)
    n = 0
    while n < GT_SAMPLES:
        xb = rng.standard_normal((GT_BATCH, WIDTH)).astype(np.float32)
        f = final_acts(weights, xb).astype(np.float64)
        acc += f.sum(axis=0)
        sq += (f * f).sum(axis=0)
        n += GT_BATCH
    mean = acc / n
    var = sq / n - mean * mean
    floor = float(np.mean(var) / n)  # per-neuron GT sampling variance, averaged
    return mean, floor


def main():
    t0 = time.time()
    ckpt = np.array(CHECKPOINTS)
    print(f"nets={N_NETS} scrambles={N_SCRAMBLES} max_half={MAX_HALF} gt={GT_SAMPLES:,}",
          flush=True)

    # Sobol point sets (shared across nets, like the fleet artifact).
    sobol_z = []
    for s in range(N_SCRAMBLES):
        eng = qmc.Sobol(d=WIDTH, scramble=True, seed=1000 + s)
        u = eng.random(MAX_HALF)
        z = ndtri(u).astype(np.float32)
        sobol_z.append(z)
        print(f"  sobol scramble {s}: |z|max={np.abs(z).max():.2f}", flush=True)

    # mse[scramble, net, checkpoint]
    mse = np.zeros((N_SCRAMBLES, N_NETS, len(ckpt)))
    gt_floors = np.zeros(N_NETS)

    for i in range(N_NETS):
        weights = build_weights(i)
        gt, floor = gt_mean_and_floor(weights, 900_000 + i)
        gt_floors[i] = floor
        for s in range(N_SCRAMBLES):
            z = sobol_z[s]
            fp = final_acts(weights, z).astype(np.float64)
            fm = final_acts(weights, -z).astype(np.float64)
            csum = np.cumsum(fp + fm, axis=0)  # (MAX_HALF, WIDTH)
            for j, n_half in enumerate(ckpt):
                est = csum[n_half - 1] / (2.0 * n_half)
                mse[s, i, j] = float(np.mean((est - gt) ** 2))
        print(f"net {i}: gt_floor={floor:.3e} "
              f"mse@30720={mse[:, i, CHECKPOINTS.index(30720)].mean():.3e} "
              f"mse@32768={mse[:, i, CHECKPOINTS.index(32768)].mean():.3e} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    np.savez(OUT, mse=mse, checkpoints=ckpt, gt_floors=gt_floors)
    print(f"saved {OUT}", flush=True)

    # ---- Analysis ----
    fleet = mse.mean(axis=1)                     # (scramble, ckpt)
    m = fleet.mean(axis=0)                       # scramble-mean
    sem = fleet.std(axis=0, ddof=1) / np.sqrt(N_SCRAMBLES)
    floor = gt_floors.mean()

    print("\nn_half   total    MSE(fleet)   SEM       n*(MSE-floor)")
    for j, n_half in enumerate(ckpt):
        tag = " <-- 2^14" if n_half == 16384 else (" <-- 2^15" if n_half == 32768 else
              (" <-- SHIP" if n_half == 30720 else ""))
        print(f"{n_half:6d} {2*n_half:7d}  {m[j]:.4e}  {sem[j]:.1e}  "
              f"{n_half*(m[j]-floor):.4e}{tag}")

    # b/n fit through non-power-of-2, non-adjacent points
    fit_idx = [j for j, n in enumerate(ckpt)
               if n not in (16128, 16384, 16640, 32512, 32768)]
    b = np.mean([(m[j] - floor) * ckpt[j] for j in fit_idx])
    print(f"\nb/n fit (non-2^m points): b = {b:.4e}")
    for n_half in (16384, 30720, 32768):
        j = CHECKPOINTS.index(n_half)
        pred = b / n_half + floor
        print(f"  n={n_half}: measured {m[j]:.4e} vs b/n-pred {pred:.4e} "
              f"-> ratio {m[j]/pred:.4f} (SEM ratio {sem[j]/pred:.4f})")

    # Paired dip test: 2^m vs +/-256 neighbors, per scramble
    for center, lo, hi in ((16384, 16128, 16640), (32768, 32512, None)):
        jc = CHECKPOINTS.index(center)
        jl = CHECKPOINTS.index(lo)
        neigh = fleet[:, jl] if hi is None else 0.5 * (fleet[:, jl] + fleet[:, CHECKPOINTS.index(hi)])
        # scale neighbor to center n via b/n on the reducible part
        n_neigh = lo if hi is None else center  # symmetric avg ~ center
        neigh_scaled = (neigh - floor) * (n_neigh / center) + floor
        d = fleet[:, jc] - neigh_scaled
        print(f"paired dip @ {center}: mean delta {d.mean():+.3e} "
              f"({d.mean()/m[jc]*100:+.2f}%), t = {d.mean()/ (d.std(ddof=1)/np.sqrt(len(d))):+.2f}")

    # Realization-luck band at ship-relevant n
    for n_half in (30720, 32768):
        j = CHECKPOINTS.index(n_half)
        f = fleet[:, j]
        print(f"realization band @ {n_half}: fleet MSE min {f.min():.3e} max {f.max():.3e} "
              f"spread {(f.max()-f.min())/f.mean()*100:.1f}% of mean")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
