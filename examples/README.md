# Examples — A Curriculum

Read in order. Each file is a complete, runnable Stage 1 estimator.

| File | Difficulty | What it teaches |
|---|---|---|
| [01_random.py](01_random.py) | introductory | The `BaseEstimator` interface and the contract: `predict(mlp, budget) -> fnp.ndarray of shape (depth, width)` |
| [02_mean_propagation.py](02_mean_propagation.py) | easy | First-order analytical: propagate per-neuron mean and diagonal variance through ReLU layers |
| [03_covariance_propagation.py](03_covariance_propagation.py) | medium | Track full covariance, not just diagonal variance — costlier but more accurate |
| [04_shipped_weights.py](04_shipped_weights.py) | easy | Ship a precomputed `weights.npz` next to your estimator and load it via `submission_dir` in `setup()` |
| [09_monte_carlo_sampling.py](09_monte_carlo_sampling.py) | intermediate | Direct Gaussian-input MC sampling — spends FLOPs to estimate the target directly |
| [11_antithetic_monte_carlo_sampling.py](11_antithetic_monte_carlo_sampling.py) | intermediate | x/−x Gaussian input pairs for variance reduction at no extra FLOP cost |
| [12_active_set_sampling.py](12_active_set_sampling.py) | advanced | Pilot pass discovers sparse live layers, then main sampling runs on the active subnetwork |
| [13_sobol_active_set_fold.py](13_sobol_active_set_fold.py) | advanced | Sobol QMC + analytical classification + layer fold |
| [14_two_ended_layer30_refinement.py](14_two_ended_layer30_refinement.py) | advanced | Previous best family: layer-29 plus layer-30 borderline pilot refinement on top of Sobol active-set folding |
| [15_dynamic_allocation.py](15_dynamic_allocation.py) | **best** | Algorithm 15 / submission 315204: staged all-layer classification, smooth analytical-variance allocation (`30720..61440`, anchor `49152`), and final-scored-row bookkeeping |
| [16_argpartition_rowsparse.py](16_argpartition_rowsparse.py) | **best** | Algorithm 16 / submissions 315640 and 315718: block-split packed row-sparse propagation with guarded Strassen dense paths, row buckets through NNZ<=80, mask reuse, and `put_along_axis` row-order restore |

The main `estimator.py` may be ahead of this curriculum during active experiments. The
numbered examples preserve stable milestone surfaces: Algorithm 15 is the last pre-row-sparse
baseline, and Algorithm 16 is the row-sparse/Strassen surface that transferred cleanly.

## Experiment archive (2026-07-14 → 2026-07-19)

These preserve the exact bytes of later submitted surfaces and notable dead
variants (archival records — the submission copies load `sobol_points.npz`
from their own directory, so run them from a folder that has the artifact).

| File | Status | What it records |
|---|---|---|
| [29b_syncbatch.py](29b_syncbatch.py) | tried | Algorithm 29b sync-batch surface (submission 315-era wall-time work) |
| [29d_fulltrim.py](29d_fulltrim.py) | tried | Algorithm 29d full-trim surface |
| [archived_capins_insurance_cap.py](archived_capins_insurance_cap.py) | dead | Algorithm 21 surface with a 49,152 wave-pacing insurance sample cap (allocation-campaign era) |
| [30_floormode.py](30_floormode.py) | dead | Algorithm 30 floor-mode (submission 317172: flopscope counters read 0 remotely; ran fixed max-N and accidentally set the then-best) |
| [31_fixed61440.py](31_fixed61440.py) | milestone | Algorithm 31 fixed N=61,440 (submission 317197, adjusted 1.3086e-7 — previous best; rollback surface) |
| [32_pow2_65536_fresh_seed.py](32_pow2_65536_fresh_seed.py) | dead | Algorithm 32 power-of-2 N=65,536 with a fresh seed-1005 Sobol artifact (submission 317412: +20.2% — realization luck dominates; offline best-of-8 seed selection did not transfer) |
| [33_pow2_32768_prefix.py](33_pow2_32768_prefix.py) | dead | Algorithm 33 power-of-2 N=32,768 via same-scramble 2^14-half prefix (submission 317415: +24.4%, raw 18-20% above the 1/n line — no (t,m,d)-net balance dip) |
| [34_antithetic_pilot.py](34_antithetic_pilot.py) | **best** | Algorithm 34 antithetic pilot probes (submission 317421, adjusted 1.3028e-7 — current best; = the current `estimator.py`) |
| [34b_pilot_fullblock.py](34b_pilot_fullblock.py) | dead | Pilot probes over all 5,120 half-rows: raw +0.31%, flops +1.04% |
| [34c_pilot_strided.py](34c_pilot_strided.py) | dead | Even-spread (strided) pilot rows: raw +0.32% at equal flops — Sobol-prefix balance is why probes work |
| [34d_pilot_halved.py](34d_pilot_halved.py) | dead | Halved pilot (256/1,024 rows): raw +0.99% for flops -0.18% — current sizes are the minimum safe point |

Full context: `bench_logs/submission_learnings_2026-07-18.md` and `_2026-07-19.md`.

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

```bash
uv run python estimator.py --baseline mean_propagation
```
