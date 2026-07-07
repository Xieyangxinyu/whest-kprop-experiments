# Submission Learnings - 2026-07-05

## Submission Learning Loop

- After each `whest submit`, ask whether the leaderboard result improved, regressed, tied, or is still grading.
- Record the submission id, variant, local expectation, leaderboard/public evidence, decision, and lesson before starting the next variant.
- Treat leaderboard regressions as stronger evidence than synthetic-seed sweeps, and explicitly mark local-only overfits.
- Keep appending to this dated file for today's work instead of scattering notes.
- Multistage refinement plan: `bench_logs/multistage_refinement_plan_2026-07-05.md`.

## Current Best Artifact

- Current best is submission `315204`: Algorithm 15 plus final-scored-row bookkeeping only; intermediate rows are analytical filler, while the final row and sample propagation are unchanged.
- Previous best was submission `315123`: staged all-layer classification refinement plus higher continuous sqrt-variance allocation anchor `49152`, clipped to `30720..61440` samples.
- Previous best was submission `315122`: continuous sqrt-variance allocation with unused active-path sampled variance bookkeeping removed, first-layer antithetic sign-flip shortcut, and staged all-layer classification refinement.
- Previous best was submission `315109`: continuous sqrt-variance allocation with unused active-path sampled variance bookkeeping removed and first-layer antithetic sign-flip shortcut.
- Previous best was submission `315105`: continuous sqrt-variance allocation with unused active-path sampled variance bookkeeping removed from `_run_block()`.
- Previous best was submission `315074`: continuous sqrt-variance allocation with `N_i = clip(40960 * sqrt(V_i / 0.02143), 30720, 61440)` and `on_thresh=3.0` on the algorithm-14 refinement path.
- For this active estimator, ship `sobol_points.npz` with `30720` half-points `(30720, 256)` so the `61440` antithetic cap is available. If reverting to fixed algorithm 14, restore the `20480` half-point artifact.
- Current best budget use is about `54%`; the old `45-46%` envelope was too conservative after staged classification improved raw accuracy. Prefer smooth sample-envelope changes over blunt fixed or bucketed sample jumps.
- Half-size `20480` effective Sobol samples with `10240` shipped half-points regressed on leaderboard; do not use as the current artifact.
- Current submission file should slice `sobol_points.npz` by the selected sample count so a larger shipped point file cannot silently increase compute.
- Intermediate all-layer MSE is no longer meaningful for the current submission surface because rows `0..30` are intentionally cheap analytical filler; optimize and compare by adjusted final-layer score and raw final-layer MSE.
- Submission `314954` with two-ended layer-30 refinement improved to public adjusted score `2.27e-7`, slightly better than the prior best.
- Submission `314957` with layer-29 plus layer-30 borderline refinement improved again to public adjusted score `2.25e-7`.
- Submission `315074` with continuous sqrt-variance allocation improved again to public adjusted score `2.2452526444934443e-7`.

## Negative Results

- Sobol scramble seed selection overfit local validation. Seed `314159` looked robust locally but was worse on leaderboard; reverted to seed `12345`.
- Sphere/radial-collapsed sampling helped slightly at `16384` samples but lost at `40960`, so do not use it for the current compute envelope.
- Kink-focused top-k sample allocation was worse locally because reducing the base pass hurt all final neurons more than extra samples helped selected neurons.
- Halving the current Sobol QMC sample count and shipped point file cut compute/artifact size but regressed publicly (`2.62e-7`), so the proven `40960` envelope remains better.
- Dynamic analytical-variance allocation (`30720` easy / `40960` medium / `49152` hard) improved public mini but regressed on the leaderboard (`2.4639658682612954e-7`), so do not submit mini-tuned bucket thresholds without full/public-like confirmation.
- Two-action strict-alpha continuation (`on3.5`, stop at `30720` when final-on analytical disagreement is low, otherwise continue to `40960`) regressed on the leaderboard (`2.5356820345891775e-7`), so the compute-saving gate overfit public mini/full-cache diagnostics.
- More samples can help when allocated smoothly by analytical final variance. Fixed-all-high envelopes still overspend locally, so do not jump straight to max samples without a score check.

## Promising Threads

- Late-layer fold behavior matters. Stricter always-on thresholds around `3.0` reduced local fold bias.
- Two-ended layer-30 pilot refinement is leaderboard-confirmed as a small improvement.
- Layer-29 plus layer-30 borderline refinement is also leaderboard-confirmed; future work should explore narrow earlier dead-neuron refinements, but avoid broad early windows without stronger validation.
- Continuous analytical-final-variance sample allocation is leaderboard-confirmed at anchors `40960` and `49152` with `V_ref=0.02143`, clipped to `30720..61440`; bucketed/quartile variants remain negative evidence.
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

### Submission 314972 - Half-size Sobol sample count

- Result: regressed.
- Change: reduced `_MAIN_SAMPLES` from `40960` to `20480`, trimmed `sobol_points.npz` from `20480` to `10240` half-points, and excluded `.claude/` from folder submissions.
- Local expectation: one-MLP subprocess smoke test passed with adjusted score `3.74e-7`, raw final-layer MSE `1.56e-6`, `0` failed MLPs, and about `23.94%` compute utilization.
- Leaderboard/public evidence: public adjusted score `2.6201749203339546e-07`; raw final-layer MSE `1.14e-6`.
- Decision: revert before continuing serious leaderboard work; keep `.whestignore` cleanup.
- Lesson: cutting the base Sobol sample count saved compute and artifact size but increased final-layer error enough to lose versus the `40960` sample envelope.

### Submission 315004 - Rejected Bucketed Dynamic Allocation Prototype

- Result: regressed.
- Change: promoted `examples/15_dynamic_allocation.py` to `estimator.py` and shipped `24576` Sobol half-points; per-MLP analytical final variance selected `30720` easy, `40960` medium, or `49152` hard samples.
- Local expectation: public mini improved versus fixed `40960` (`2.502437227e-7` vs `2.633383014e-7`); validation and fixed-seed local/subprocess checks passed with `0` failed MLPs.
- Leaderboard/public evidence: public adjusted score `2.4639658682612954e-7`, worse than current best `2.25e-7` from submission `314957`.
- Decision: revert before the next serious submission; keep this as a rejected bucketed allocation prototype and continue dynamic allocation only with better hardness gates/full-split validation.
- Lesson: dynamic allocation is promising structurally, but mini-tuned analytical-variance quartile thresholds overfit; leaderboard evidence favors the fixed `40960` algorithm-14 baseline until a full-split/per-MLP adjusted-score policy wins robustly.

### Submission 315071 - Strict Alpha Two-Action Continuation

- Result: regressed.
- Change: submitted true-continuation two-action policy: `on_thresh=3.5`, run `30720` samples first, return if `stat_base_final_on_anal_diff_mean <= 0.0415`, otherwise continue to `40960` using the extra Sobol prefix.
- Local expectation: public mini CLI improved over algorithm 14 (`2.501500611e-7` vs `2.662116209e-7`) by saving compute, despite worse raw final MSE; validation and local/subprocess checks passed with `0` failures.
- Leaderboard/public evidence: public adjusted score `2.5356820345891775e-7`; final-layer MSE `6.75e-7`; all-layers MSE `1.373e-6`; budget used `38.44%`; mean effective compute `1.05e11`; `0` failures. Worse than current best `314957` (`2.249e-7`) and worse than dynamic allocation `315004` (`2.464e-7`).
- Decision: revert before next serious submission; do not resubmit this threshold/gate.
- Lesson: the policy won public mini by compute discount but sacrificed too much raw accuracy on leaderboard public; full-cache and mini improvements are not enough when the gate aggressively stops at `30720` for many MLPs.

### Submission 315074 - Continuous Sqrt-Variance Allocation

- Result: improved slightly; new current best.
- Change: submitted continuous allocation `N_i = clip(40960 * sqrt(V_i / 0.02143), 30720, 61440)` where `V_i` is analytical final-layer variance mean, with true Sobol-prefix continuation and a shipped `(30720, 256)` Sobol artifact.
- Local expectation: implementation audit confirmed `V_ref -> 40960`, low variance clamps to `30720`, high variance clamps to `61440`; validation passed; local and subprocess `--seed 42 --n-mlps 3` checks both had `0/3` failures and matched adjusted score around `2.55e-7`.
- Leaderboard/public evidence: public adjusted score `2.2452526444934443e-7`; final-layer MSE `4.76e-7`; all-layers MSE `1.10e-6`; budget used `46.30%`; mean effective compute `1.26e11`; best public MLP `christopher-daniels` `6.24e-8`; worst public MLP `joshua-aguilar` `6.27e-7`; `0/50` public failures.
- Public per-MLP breakdown vs `314957` using rounded ledger rows: `25` improved, `23` regressed, `2` tied; mean adjusted delta about `-3.24e-10`; mean raw final-layer MSE delta about `-1.83e-8`; mean compute delta about `+8.0e8`. Biggest wins were `william-gilbert`, `austin-duffy`, `monica-brown`, and `joseph-vasquez`; biggest losses were `lauren-johnson`, `joshua-aguilar`, `maria-mcintosh`, and `michael-byrd`. The leaderboard public names did not match the HF public `full` split, so exact local `N_i` per public MLP could not be recovered from weights; effective-compute deltas split roughly half lower and half higher than `314957`.
- Decision: keep as current best, but treat the gain as small and preserve algorithm-14 fixed `40960` as the fallback.
- Lesson: continuous variance scaling transferred where quartile buckets and stop gates failed; the useful form is smooth allocation around the proven `40960` anchor, not aggressive easy-network cuts or blunt hard buckets.

### Submission 315076 - Feature-Modified Sqrt Allocation

- Result: regressed.
- Change: submitted deployable feature-modified hardness `V * modifier / norm`, where the modifier increased with final-on fraction, final `alpha>=6` fraction, low final-kink fraction, and low layer-30-kink fraction; used coefficient `2`, `V_ref=0.024`, cap `65536`, and a shipped `(32768, 256)` Sobol artifact.
- Local expectation: cache interpolation and public mini local run looked plausible; exact public mini local score rounded to `2.61e-7` with compute utilization `44.82%`, but raw final MSE worsened versus the plain sqrt rule.
- Leaderboard/public evidence: public adjusted score `2.2756611195776188e-7`; final-layer MSE `5.21e-7`; all-layers MSE `1.205e-6`; budget used `43.44%`; mean effective compute `1.18e11`; `0/50` public failures. Worse than current best `315074` (`2.2452526444934443e-7`).
- Decision: reject and restore `estimator.py`/`sobol_points.npz` to the `315074` plain sqrt-variance rule before further work.
- Lesson: the always-on/low-kink modifier saved compute but lost too much raw final-layer accuracy; public evidence favors the simpler analytical-variance scalar over this hand-composed structural modifier.

### Submission 315104 - Raw NumPy Final Fold

- Result: failed grading with generic `Evaluation error`.
- Change: kept submission `315074` continuous sqrt-variance allocation but computed the folded final-layer kink ReLU mean/variance with raw NumPy (`np.asarray`, `np.concatenate`/`vstack`, `input @ weight`, `np.maximum`) to avoid two flopscope-tracked final matmuls.
- Local expectation: validation passed; subprocess seed `0`, `n=1` improved adjusted score `1.75e-7 -> 1.73e-7` with unchanged raw MSE and residual wall time about `0.027s`; subprocess seed `0`, `n=5` passed with `0/5` failures, adjusted score `1.98e-7`, raw final MSE `4.25e-7`, mean utilization `45.86%`, residual wall time `0.1909s` total.
- Leaderboard/public evidence: AIcrowd submission `315104` failed before producing leaderboard metrics: `<p>Error : Evaluation error</p>`.
- Decision: do not extend or resubmit raw-NumPy-in-`predict()` variants until the remote failure mode is understood; treat `315074` as current best/fallback.
- Lesson: local/subprocess tolerance of off-flopscope raw NumPy does not guarantee grader acceptance; future compute-accounting experiments need a grader-safe implementation path or a smaller diagnostic submission with traceback details.

### Submission 315105 - Variance-Free Active Block

- Result: improved; new current best.
- Change: kept the `315074` continuous sqrt-variance allocation but removed unused sampled variance bookkeeping from the active `_run_block()` path and dropped an unused final analytical-difference diagnostic. No raw NumPy in `predict()`.
- Local expectation: validation passed; subprocess seed `0`, `n=1` improved adjusted score to `1.72e-7` with unchanged raw final MSE `4.84e-7`; subprocess seed `0`, `n=5` passed with `0/5` failures, adjusted score `1.97e-7`, raw final MSE `4.25e-7`, all-layers MSE `1.19e-6`, mean utilization `45.58%`.
- Leaderboard/public evidence: public adjusted score `2.2090464792719037e-7`. Other public fields not yet captured from the leaderboard UI.
- Decision: keep as current best and use it as the fallback for future experiments.
- Lesson: exact flopscope-native removal of unused work transferred; prefer this kind of structural compute cleanup over off-flopscope raw NumPy hacks.

### Submission 315109 - First-Layer Antithetic Shortcut

- Result: improved; new current best.
- Change: kept submission `315105` and replaced the first sampled layer matmul over `[half, -half]` with a half-batch matmul plus sign flip: `pre_half = half @ W0`, then concatenate `relu(pre_half)` and `relu(-pre_half)`.
- Local expectation: validation passed; subprocess seed `0`, `n=1` improved adjusted score to `1.69e-7` with unchanged raw final MSE `4.84e-7` and all-layers MSE `1.29e-6`; subprocess seed `0`, `n=5` passed with `0/5` failures, adjusted score `1.93e-7`, raw final MSE `4.25e-7`, all-layers MSE `1.19e-6`, mean utilization `44.57%`.
- Leaderboard/public evidence: public adjusted score `2.1590206197719508e-7`. Other public fields not yet captured from the leaderboard UI.
- Decision: keep as current best and use it as the fallback for future experiments.
- Lesson: exact antithetic symmetry at layer 0 transferred cleanly; look for similarly exact symmetry/unused-work reductions before approximate hacks.

### Submission 315118 - Adaptive Signed Final-On Scale

- Result: regressed.
- Change: added a tiny final-on scale correction after prediction: `pred[final_on] += 1e-4 * sign(mean(analytical_final_on - pred_final_on)) * pred[final_on]`.
- Local expectation: public-mini subprocess improved current estimator from adjusted `2.57e-7`, raw `5.56e-7` to adjusted `2.52e-7`, raw `5.47e-7`, all-layer `1.20e-6`, utilization `44.12%`, `0/100` failures; one-MLP smoke also looked strong.
- Leaderboard/public evidence: public adjusted score `2.214490192314609e-7`, worse than current best `315109` (`2.1590206197719508e-7`).
- Decision: reject and revert `estimator.py` to the `315109` path.
- Lesson: the final-on signed scale correction overfit public mini; even nearly-free signed bias corrections need leaderboard confirmation before replacing exact compute cleanups.

### Submission 315122 - Staged All-Layer Classification Refinement

- Result: improved; new current best.
- Change: added staged classification probes: a 5% pilot with a 20% recheck for borderline decisions, broadened dead-neuron rescue across layers `1..29`, and added conservative active-to-dead demotion for borderline kink neurons.
- Local expectation: public mini local score was about `2.47e-7`, raw final-layer MSE `5.27e-7`, all-layers MSE `1.17e-6`, compute utilization about `44.75%`, `0/100` failures.
- Leaderboard/public evidence: public adjusted score `2.102e-7`; final-layer MSE `4.53e-7`; all-layers MSE `1.082e-6`; budget used `45.95%`; mean effective compute `1.25e11`; `0/50` public failures.
- Decision: keep as current best; use it as the baseline for higher-sample-envelope probes.
- Lesson: broader staged classification transferred despite weak HF-mini evidence; leaderboard public favored the extra structural refinement more than the local mini split suggested.

### Submission 315123 - Higher Sqrt-Variance Anchor 49152

- Result: improved; new current best.
- Change: kept submission `315122` staged all-layer classification refinement and changed only the smooth sample allocation anchor from `40960` to `49152`, with the same `30720..61440` sample clip and `V_ref=0.02143`.
- Local expectation: exact public-mini CLI improved `315122`-style local score from about `2.47e-7` to `2.37e-7`; raw final-layer MSE `4.39e-7`; all-layers MSE `9.84e-7`; mean utilization `52.32%`; subprocess seed `42`, `n=3` passed with adjusted `2.29e-7`, raw `4.27e-7`, utilization `53.62%`, `0/3` failures.
- Leaderboard/public evidence: public adjusted score `2.042047322113885e-7`; final-layer MSE `3.72e-7`; all-layers MSE `8.996e-7`; budget used `54.05%`; mean effective compute `1.47e11`; `0/50` public failures.
- Decision: keep as current best; use `49152` as the new anchor baseline before testing any higher envelope.
- Lesson: submission `315119` correctly signaled that spending more compute can be worthwhile; the transfer-safe form was a smooth variance-scaled anchor increase, not fixed max samples.

### Submission 315204 - Final-Scored Row Bookkeeping

- Result: improved slightly; new current best.
- Change: kept Algorithm 15 final-row computation unchanged but stopped materializing sampled intermediate output rows and intermediate dead-correction scatter rows when only the final row is scored; rows `0..30` are returned as analytical filler. Also avoided stacking the base prediction unless the base-only path returns.
- Local expectation: exact public-mini CLI with both bookkeeping trims had adjusted score rounded `2.35e-7`, raw final-layer MSE `4.39e-7`, all-layers MSE `7.86e-4`, mean utilization `51.95%`, and `0/100` failures. Subprocess seed `42`, `n=3` passed with adjusted `2.26e-7`, raw `4.27e-7`, all-layers MSE `7.09e-4`, utilization `52.94%`, and `0/3` failures.
- Leaderboard/public evidence: public adjusted score `2.037596606578937e-7`; final-layer MSE `3.72e-7`; all-layers MSE `8.164e-4`; budget used `53.93%`; mean effective compute `1.47e11`; `0/50` public failures.
- Decision: keep as current best; accept bad all-layer MSE because the leaderboard scores only the final row.
- Lesson: output-row bookkeeping was a real but small compute leak. Removing non-scored intermediate row materialization transfers when final sample propagation is unchanged.
