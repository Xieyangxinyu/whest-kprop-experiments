# Submission Learnings - 2026-07-09

## Current Decision

- Keep submission `315416` as Algorithm 16: argpartition packed row-sparse propagation on the base/refinement block only, with extra continuation blocks left dense.
- Reason: `315416` graded successfully with public score `1.8089072203069811e-7`; more aggressive continuation/split variants crossed remote runtime/scoring boundaries or graded worse.
- Next iteration should branch from Algorithm 16, not from the split always-on dense experiment.

## Submission Log

### Submission 315415 - Full Packed Row-Sparse Active Propagation

- Result: failed partially.
- Change: exact packed row-sparse active propagation across ordinary active layers `1..29`, including continuation blocks; used flopscope-native packing and reduced-width contraction.
- Local expectation: public-mini first 3 improved adjusted score from `2.97e-7` to `2.37e-7` with unchanged raw final MSE, but backend wall time was high.
- Leaderboard/public evidence: submission `315415` reported `5` failed MLPs.
- Decision: do not resubmit the full-range/full-block variant unchanged; investigate wall-time-safe gates.
- Lesson: effective-compute savings can be real while total wall time still causes remote failures.

### Submission 315416 - Argpartition Base-Block Packed Row-Sparse

- Result: improved; new best among recent row-sparse submissions.
- Change: replaced row packing `argsort` with `argpartition`, kept fused `fnp.einsum`, and disabled packing for extra continuation blocks (`_PACKED_ROWSPARSE_EXTRA_BLOCKS = False`). Base/refinement block still packs ordinary active layers `1..29` when `k <= 0.75 * prev_width`.
- Local expectation: public-mini first 10 subprocess check had `0/10` failures, adjusted score `1.91e-7`, raw final MSE `4.56e-7`, mean multiplier `0.40068691`, total estimator backend time `70.705s` (`~7.1s/MLP`). Public-mini first 3 adjusted score `2.58e-7` vs disabled packed baseline `2.97e-7`.
- Leaderboard/public evidence: public score `1.8089072203069811e-7`, score secondary/raw final MSE `3.724917161207486e-7`, graded successfully.
- Decision: keep as safe row-sparse baseline; test whether argpartition continuation-block packing can recover more of the failed full-packing win without wall-time failures.
- Lesson: `argpartition` and skipping continuation-block packing preserve most of the compute win while reducing remote failure risk.

### Candidate - Argpartition Full-Block Packed Row-Sparse

- Result: failed grading.
- Change: same as `315416`, but re-enables packed row-sparse propagation on extra continuation blocks (`_PACKED_ROWSPARSE_EXTRA_BLOCKS = True`). This differs from failed `315415` by using `argpartition` and fused `fnp.einsum` instead of full sorting/heavier packing.
- Local expectation: first-10 public-mini runtime toggle had `0/10` failures, adjusted score `1.798201268612198e-7`, raw final MSE `4.5558933265965605e-7`, mean multiplier `0.37905223702902596`; max local wall time `22.77s`. Stress set of max/near-max target-sample public-mini MLPs had `0/8` failures and max wall time `23.00s`.
- Leaderboard/public evidence: submitted as `315417`; failed with `Evaluation could not complete; please retry`.
- Decision: reject full-block argpartition without the always-on split; total grader runtime is still too high.
- Lesson: the apparent boundary is sorting/packing wall time, not the sparse arithmetic idea itself; `argpartition` likely moves the full-block variant back under wall-time limits.

### Submission 315420 - Split Always-On Dense + Kink Packed Row-Sparse

- Result: failed grading.
- Change: previous-layer always-on columns are multiplied as a dense block, while previous-layer kink columns use the row-packed sparse kernel. Keeps `argpartition`, fused `fnp.einsum`, and continuation-block packing enabled.
- Local expectation: first-10 public-mini check had `0/10` failures, adjusted score `1.694214355049e-7`, raw final MSE `4.556272699574e-7`, mean multiplier `0.358928`, max wall time `13.26s`, backend sum `100.16s`. This is better than `315416` local (`1.930949257536e-7`, backend sum `72.35s`) and much faster than failed `315417` local (`1.798201268612e-7`, backend sum `176.62s`). High-sample stress set had `0/8` failures with max wall time `26.32s`.
- Leaderboard/public evidence: submitted as `315420`; failed with `Error while scoring your submission`.
- Decision: one retry requested, then if it fails again retreat toward `315416` or cap target samples for high-variance networks.
- Lesson: separating dense always-on signal from sparse kink signal may be the right runtime/compute tradeoff for this idea.