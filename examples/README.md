# Examples — A Curriculum

Read in order. Each file is a complete, runnable Stage 1 estimator.

| File | Difficulty | Expected MSE (default MLP, n=100k) | What it teaches |
|---|---|---|---|
| [01_random.py](01_random.py) | introductory | ~0.29 (random baseline) | The `BaseEstimator` interface and the contract: `predict(mlp, budget) -> fnp.ndarray of shape (depth, width)` |
| [02_mean_propagation.py](02_mean_propagation.py) | easy | ~0.003 | First-order analytical: propagate per-neuron mean and diagonal variance through ReLU layers |
| [03_covariance_propagation.py](03_covariance_propagation.py) | medium | ~0.0007 | Track full covariance, not just diagonal variance — costlier but more accurate |
| [04_shipped_weights.py](04_shipped_weights.py) | easy | n/a (zeros baseline) | Ship a precomputed `weights.npz` next to your estimator and load it via `submission_dir` in `setup()` |
| [05_paper_series_covariance.py](05_paper_series_covariance.py) | advanced | ~0.0006 | Track full covariance with the paper's K=4 ReLU covariance series instead of the K=1 gain approximation |
| [06_paper_series_covariance_pruned.py](06_paper_series_covariance_pruned.py) | advanced | experimental | K=4 covariance propagation plus alpha-based active-set pruning to exploit dead-neuron structure |
| [07_normal_lognormal_mean_propagation.py](07_normal_lognormal_mean_propagation.py) | experimental | ~0.0016 on local seed=0, n=100k | Diagonal mean propagation with a normal-lognormal prior-predictive ReLU moment correction; improves 02 locally but not covariance methods |
| [08_normal_lognormal_series_covariance.py](08_normal_lognormal_series_covariance.py) | experimental | ~0.00021 on local seed=0, n=100k at full strength | K=4 covariance propagation with normal-lognormal-corrected marginal moments; useful as a negative result/tuning scaffold, not better than 05 locally |
| [09_monte_carlo_sampling.py](09_monte_carlo_sampling.py) | experimental | ~5.7e-6 final-layer MSE on local seed=42, n-mlps=10, n=100k | Direct Gaussian-input sampling; spends FLOPs to estimate the target directly and beat the analytic examples in local scored comparisons |
| [10_hybrid_sampling_covariance.py](10_hybrid_sampling_covariance.py) | experimental | ~9.2e-6 final-layer MSE on local seed=42, n-mlps=10, n=100k | Antithetic sampling shrunk toward K=4 covariance propagation; improves 4096-sample MC but not active sampling |
| [11_antithetic_monte_carlo_sampling.py](11_antithetic_monte_carlo_sampling.py) | experimental | ~5.5e-6 final-layer MSE on local seed=42, n-mlps=10, n=100k | Direct sampling with x/-x Gaussian input pairs; modest variance reduction over plain MC at 8192 samples |
| [12_active_set_sampling.py](12_active_set_sampling.py) | experimental | ~3.7e-6 final-layer MSE on local seed=42, n-mlps=10, n=100k | Pilot pass discovers sparse live layers, then main sampling runs on the active subnetwork; best sampler in local comparisons |

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

```bash
uv run python estimator.py --baseline mean_propagation
```
