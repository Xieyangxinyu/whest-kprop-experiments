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

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

```bash
uv run python estimator.py --baseline mean_propagation
```
