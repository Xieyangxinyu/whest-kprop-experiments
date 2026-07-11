# Algorithm 17 (Probe-Free, Dead -4.0) Submission Prep - 2026-07-11

## Candidate

`estimator.py` staged from `examples/17_probe_free_wideband.py`: the 315718
surface with the staged pilot-probe machinery removed entirely and the dead
threshold widened -3.0 -> -4.0 (bias protection by inclusion instead of probe
promotion; on-side stays +3.0 since the fold is linear-exact for on-neurons).
Matmul machinery (block-split packed row-sparse, guarded 1-level Strassen,
layer-30/31 fold, sqrt-hardness continuation) unchanged. Stack: whestbench
0.12.0rc5 / flopscope 0.8.0rc5 (matches current grader pins).

## Motivation (notebook algorithm16_deep_dive.ipynb section 7b/7c)

- Probes reclassify only ~44 of 8192 neuron-layers (6 promotions, 38 demotions
  on seed-0); cost ~0.32% of budget plus per-layer Python residual.
- Probe-only classification (no analytic pass) tested and REJECTED: seed-42
  adjusted 4.00e-7, raw 6.97e-7 (+63% vs baseline) - the analytic screen is
  load-bearing; probing all 256 cols with 5% rows is noisier and costlier.
- Widening the dead band replaces the probes' bias protection at ~+3-4% width.

## A/B evidence (n=3 subprocess, same machine, same stack)

| arm | seed-42 adj / raw | seed-0 adj / raw |
|---|---|---|
| baseline 315718 surface | 3.10e-7 / 4.27e-7 | 1.69e-7 / 2.46e-7 |
| probe-free, dead -3.5 | 3.00e-7 / 4.05e-7 | 1.68e-7 / 2.40e-7 |
| probe-free, dead -4.0 (THIS) | 2.92e-7 / 3.96e-7 | NOT RUN (stopped) |
| probe-only (no analytic) | 4.00e-7 / 6.97e-7 | NOT RUN (stopped) |

Raw MSE improves as well as adjusted on every probe-free point measured - the
probes were mis-demoting on these seeds, not just costing compute.

## Validation gates (all pass)

- `whest validate`: OK.
- Parity seed-42 n=3: local adjusted 2.83e-7 / subprocess 3.01e-7, RAW MSE
  IDENTICAL 3.96e-7, 0/3 failed both. Deterministic, subprocess-safe.
- Budget headroom: ~110G analytic vs 272G budget; no budget/time exhaustion.
- Package: `whest package --estimator .` -> submission-algo17-probefree-m40.tar.gz,
  5 files (estimator.py, sobol_points.npz, requirements.txt, LICENSE,
  manifest.json), 28 MB < 52 MB cap. `whest validate-package`: valid.
- .whestignore extended (again) to exclude .est*.npz / *_cache.npz /
  relu_moment_propagation.py / prokpt.txt - the 07-09 fix had not persisted.

## Known gaps (accepted by user direction)

- Seed-0 group NOT run for the -4.0 point (stopped mid-run; the -3.5 point beat
  baseline there, direction consistent).
- No public-mini A/B for this surface.
- Docstring in estimator.py records the measured numbers.

## Status

SUPERSEDED, NEVER SUBMITTED AS-IS. This surface was merged with the remote
finer-row-buckets line (`origin/algorithm-17-finer-row-buckets-2026-07-11`,
submission 315824) into Algorithm 18 and submitted as 315844 on 2026-07-11.
See `bench_logs/submission_learnings_2026-07-11.md`. The stale
`submission-algo17-probefree-m40.tar.gz` artifact should not be submitted.
