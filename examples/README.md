# Examples — A Curriculum

Read in order. Each file is a complete, runnable Stage 1 estimator.

| File | Difficulty | What it teaches |
|---|---|---|
| [01_random.py](01_random.py) | introductory | The `BaseEstimator` interface and the contract: `predict(mlp, budget) -> fnp.ndarray of shape (depth, width)` |
| [02_mean_propagation.py](02_mean_propagation.py) | easy | First-order analytical: propagate per-neuron mean and diagonal variance through ReLU layers |
| [03_covariance_propagation.py](03_covariance_propagation.py) | medium | Track full covariance, not just diagonal variance — costlier but more accurate |
| [04_shipped_weights.py](04_shipped_weights.py) | easy | Ship a precomputed `weights.npz` next to your estimator and load it via `submission_dir` in `setup()` |
| [05_cumulant_propagation.py](05_cumulant_propagation.py) | advanced | Cumulant-propagation-inspired ReLU moments with full covariance plus marginal third/fourth cumulants |
| [06_full_k3_pruned.py](06_full_k3_pruned.py) | advanced | Factorized SIMPLE K3 propagation with repeated slices, fourth-order radial state, pK-to-K conversion, and active-set pruning |
| [07_low_rank_covariance.py](07_low_rank_covariance.py) | advanced | Low-rank covariance factors with active-set pruning and diagonal variance repair |
| [08_submission_317459.py](08_submission_317459.py) | advanced | Algorithm 37 / submission 317459: mask-producing where-threading on top of Algorithm 34 |

Numbers 09–16 were removed in commit `01123e7` ("cleanup") and are recoverable from git
history if needed.

The main `estimator.py` is well ahead of this curriculum — see the submission archive
below for its lineage.

## Experiment archive (2026-07-14 → 2026-07-19)

These preserve the exact bytes of later submitted surfaces and notable dead
variants (archival records — the submission copies load `sobol_points.npz`
from their own directory, so run them from a folder that has the artifact).

| File | Status | What it records |
|---|---|---|
| [34_antithetic_pilot.py](34_antithetic_pilot.py) | milestone | Algorithm 34 antithetic pilot probes (submission 317421, adjusted 1.3028e-7 — best of that window). Also the last surface carrying the gather-packed `_packed_matmul` row-sparse path |
| [40_strassen2_and_threshold.py](40_strassen2_and_threshold.py) | tried | Two-level Strassen plus threshold work, built on Algorithm 34 at N=16,384 |

The rest of this archive (29b, 29d, 30–33, 34b–34d, 35, 36, 37, `archived_capins_…`) was
removed in commit `01123e7` ("cleanup"); the findings survive in the bench logs and the
bytes are recoverable from git history. Full context:
`bench_logs/submission_learnings_2026-07-18.md` and `_2026-07-19.md`.

Scores in the table above are **not comparable** to the ones below it: flopscope 0.9.x
repriced the cost model (data movement is metered, `take` bills 4/element, float64 bills
2x), so every pre-0.9 measurement was invalidated and the whole surface was re-derived.

## Submission archive (2026-07-24 → 2026-07-26)

Files named by AIcrowd submission id, each a **verbatim copy of the graded bytes**
(no added headers). All are float32-billed, fixed N = 61,440, on flopscope 0.9.1.

| File | Score | What it records |
|---|---|---|
| [318752.py](318752.py) | 1.7679e-07 | Two-level Strassen, packed row-sparse path deleted; first surface of the 0.9.1 era |
| [318803.py](318803.py) | improved | Algorithm 43: cold-column slicing — sort columns coldest-first, group rows by cold support, skip the all-zero rectangle. Resubmit of 318793, which was **rejected `IMPORT_FAILED`** for importing raw numpy |
| [318873.py](318873.py) | 1.662e-07 | Algorithm 44: two-level carve — cold set split into ultra-cold/warm-cold slabs, rows grouped by support *pattern*, one rectangle becomes three |
| [318937.py](318937.py) | 1.6785e-07 | Algorithm 46 dtype hygiene — regressed |
| [318964.py](318964.py) | 1.6736e-07 | Algorithm 51 tuned base |
| [318978.py](318978.py) | 1.6689e-07 | Algorithm 54 fold-Strassen |
| [319031.py](319031.py) | 1.6605e-07 | Verbatim re-grade of 318978 — a grading-variance probe. Identical bytes scored 0.5% apart, which is the noise floor every A/B below is judged against |
| [319266.py](319266.py) | **1.5755e-07** | **Current best; = the current `estimator.py`.** Packed prefix replaces the carve in both blocks |

The 2026-07-26 step (319031 → 319266, −5.1%) came in two graded submissions:

1. **Packed prefix replaces the pattern carve** (−3.94%). The carve's own cold corrections
   were 5–11% dense yet multiplied in full, and the low-fire band past the cold set rode
   the dense block at full width. Merged, that prefix is ~4.5% dense — far below the ~w/3
   gather crossover — so per-row gathering pays after all. The carve survives as the k=0
   bucket. Prefix width is sized by *expected NNZ*, not a fire-rate cut; a fire-rate cut
   regressed +1.49% FLOPs.
2. **Same packing in the pilot block, plus a wider NNZ ladder** (−1.23%). The pilot pass
   must restore row order (Sobol probe rows, antithetic pairing), and sizes its prefix from
   `Phi(alpha)` because no fire census exists yet in that pass.

Full context: `bench_logs/submission_learnings_2026-07-25.md` and `_2026-07-26.md`.

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

The `--baseline` flag is part of the starter *template*'s `__main__` block. The repo-root
`estimator.py` is the competition entry and no longer carries one, so
`uv run python estimator.py --baseline mean_propagation` exits 0 printing nothing. Score it
through the harness instead:

```bash
uv run whest run --estimator estimator.py --runner local \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini \
  --seed 42 --n-mlps 10
```

The curriculum examples above still have their `__main__` blocks and remain directly
runnable.
