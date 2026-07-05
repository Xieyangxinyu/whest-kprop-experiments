# Submission Learnings - 2026-07-05

## Submission Learning Loop

- After each `whest submit`, ask whether the leaderboard result improved, regressed, tied, or is still grading.
- Record the submission id, variant, local expectation, leaderboard/public evidence, decision, and lesson before starting the next variant.
- Treat leaderboard regressions as stronger evidence than synthetic-seed sweeps, and explicitly mark local-only overfits.
- Keep appending to this dated file for today's work instead of scattering notes.
- Multistage refinement plan: `bench_logs/multistage_refinement_plan_2026-07-05.md`.

## Current Best Artifact

- Keep `40960` effective Sobol samples: `20480` half-points plus antithetic pairs.
- Keep budget use near the proven `45%` envelope; `49152` samples was too large on leaderboard despite local raw-MSE gains.
- Current submission file should slice `sobol_points.npz` by `_MAIN_SAMPLES // 2` so a larger shipped point file cannot silently increase compute.
- Submission `314954` with two-ended layer-30 refinement improved to public adjusted score `2.27e-7`, slightly better than the prior best.
- Submission `314957` with layer-29 plus layer-30 borderline refinement improved again to public adjusted score `2.25e-7`.

## Negative Results

- Sobol scramble seed selection overfit local validation. Seed `314159` looked robust locally but was worse on leaderboard; reverted to seed `12345`.
- Sphere/radial-collapsed sampling helped slightly at `16384` samples but lost at `40960`, so do not use it for the current compute envelope.
- Kink-focused top-k sample allocation was worse locally because reducing the base pass hurt all final neurons more than extra samples helped selected neurons.
- More samples alone are not enough: the champion uses less compute and much lower raw final-layer MSE, so the remaining gap is algorithmic.

## Promising Threads

- Late-layer fold behavior matters. Stricter always-on thresholds around `3.0` reduced local fold bias.
- Two-ended layer-30 pilot refinement is leaderboard-confirmed as a small improvement.
- Layer-29 plus layer-30 borderline refinement is also leaderboard-confirmed; future work should explore narrow earlier dead-neuron refinements, but avoid broad early windows without stronger validation.
- Next serious direction should be late-layer bias diagnostics and a targeted residual/control-variate correction, validated on real public challenge MLPs when possible.
- Treat synthetic-seed sweeps as weak evidence; use leaderboard/public-dataset checks for changes that can overfit sample geometry.

## Submission Log

### Submission 314954 - Two-ended layer-30 refinement

- Result: improved slightly.
- Change: added a 5% layer-30 pilot to demote weak always-on neurons and promote weak dead neurons to kink.
- Local expectation: small but consistent synthetic-seed improvement over `normal fold-on 3.00` at `40960` samples.
- Leaderboard/public evidence: public adjusted score `2.27e-7`, slightly better than before.
- Decision: keep and investigate sharper/cheaper Stage 1 refinement variants.
- Lesson: the two-ended Stage 1 refinement signal was small locally but did transfer to the public leaderboard; prefer conservative classification refinements over sample-geometry seed tuning.

### Submission 314957 - Layer-29 plus layer-30 borderline refinement

- Result: improved slightly.
- Change: added layer-29 borderline dead promotion to the existing layer-30 borderline dead/on refinement.
- Local expectation: best local candidate on both seed groups; `5.530e-7` on seeds `0..4` and `2.756e-7` on seeds `5..9` in synthetic validation.
- Leaderboard/public evidence: public adjusted score `2.25e-7`; final-layer MSE `4.94e-7`; all-layers MSE `1.09e-6`; budget used `45.82%`; mean effective compute `1.25e11`.
- Decision: keep as current best and investigate narrower earlier dead-neuron refinements.
- Lesson: layer-29 dead refinement transferred to the public leaderboard; broader early-layer windows were unstable locally, so only test earlier layers with narrow gates and public confirmation.
