# Submission Learnings - 2026-07-05

## Submission Learning Loop

- After each `whest submit`, ask whether the leaderboard result improved, regressed, tied, or is still grading.
- Record the submission id, variant, local expectation, leaderboard/public evidence, decision, and lesson before starting the next variant.
- Treat leaderboard regressions as stronger evidence than synthetic-seed sweeps, and explicitly mark local-only overfits.
- Keep appending to this dated file for today's work instead of scattering notes.

## Current Best Artifact

- Keep `40960` effective Sobol samples: `20480` half-points plus antithetic pairs.
- Keep budget use near the proven `45%` envelope; `49152` samples was too large on leaderboard despite local raw-MSE gains.
- Current submission file should slice `sobol_points.npz` by `_MAIN_SAMPLES // 2` so a larger shipped point file cannot silently increase compute.

## Negative Results

- Sobol scramble seed selection overfit local validation. Seed `314159` looked robust locally but was worse on leaderboard; reverted to seed `12345`.
- Sphere/radial-collapsed sampling helped slightly at `16384` samples but lost at `40960`, so do not use it for the current compute envelope.
- Kink-focused top-k sample allocation was worse locally because reducing the base pass hurt all final neurons more than extra samples helped selected neurons.
- More samples alone are not enough: the champion uses less compute and much lower raw final-layer MSE, so the remaining gap is algorithmic.

## Promising Threads

- Late-layer fold behavior matters. Stricter always-on thresholds around `3.0` reduced local fold bias.
- Next serious direction should be late-layer bias diagnostics and a targeted residual/control-variate correction, validated on real public challenge MLPs when possible.
- Treat synthetic-seed sweeps as weak evidence; use leaderboard/public-dataset checks for changes that can overfit sample geometry.

## Submission Log

### Submission 314954 - Two-ended layer-30 refinement

- Result: still grading.
- Change: added a 5% layer-30 pilot to demote weak always-on neurons and promote weak dead neurons to kink.
- Local expectation: small but consistent synthetic-seed improvement over `normal fold-on 3.00` at `40960` samples.
- Leaderboard/public evidence: pending.
- Decision: investigate after result is available.
- Lesson: record outcome before starting another submission variant, because the expected gain is small enough that leaderboard evidence should decide.
