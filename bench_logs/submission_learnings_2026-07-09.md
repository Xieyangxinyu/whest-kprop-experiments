# Submission Learnings - 2026-07-09

## Current Decision

- Current best is submission `315521`: Algorithm 17 block-split packed row-sparse with guarded Strassen dense sampled matmuls.
- Public score: `1.4895563350388822e-7`; raw final MSE: `3.724955968209542e-7`.
- Safe fallback is submission `315416`: Algorithm 16 argpartition base-block packed row-sparse (`1.8089072203069811e-7`).
- Submission `315512` hand-built early-layer schedule and submission `315515` fixed `0.85` threshold both regressed and should not be retried.

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

### Submission 315507 - Full Sampled-Stage Block Split

- Result: improved; new current best.
- Change: block-split row-sparse active propagation runs for both sampled propagation stages: base/refinement block and extra continuation block. High-fire columns use dense matmul; low-fire columns use packed sparse matmul. Analytical structure, first-layer antithetic shortcut, layer-30/31 special paths, and final averaging remain unchanged.
- Local expectation: first-10 public-mini check had `0/10` failures, adjusted score `1.736087396920e-7`, raw final MSE `4.556189452387e-7`, mean multiplier `0.367739`, mean effective compute `100.03G`, backend sum `62.30s`, max wall `8.66s`. High-continuation stress set had `0/8` failures and max wall `13.92s`.
- Leaderboard/public evidence: public score `1.6418987912821085e-7`, score secondary/raw final MSE `3.724918838088342e-7`, graded successfully. Better than confirmed safe baseline `315416` (`1.8089072203069811e-7`).
- Decision: keep as current best; next boundary probe is applying block split to the sampled kink matmuls inside the existing layer-30/31 fold path.
- Lesson: block splitting made full continuation sparse propagation locally viable; this tests whether it transfers remotely.

### Submission 315509 - Full Block Split plus Fold-Kink Block Split

- Result: improved; new current best.
- Change: keeps submission `315507` full sampled-stage block split, and also applies block-split matmul to the sampled kink paths inside the existing layer-30/31 fold. The layer-30 on-neuron mean/fold contribution remains unchanged.
- Local expectation: first-10 public-mini check had `0/10` failures, adjusted score `1.729532355024e-7`, raw final MSE `4.556216993024e-7`, mean multiplier `0.366037`, mean effective compute `99.56G`, backend sum `62.76s`, max wall `8.70s`. This is a small local improvement over `315507` local (`1.736087396920e-7`, mean effective `100.03G`).
- Leaderboard/public evidence: public score `1.590833103387916e-7`, score secondary/raw final MSE `3.7249296212849005e-7`, graded successfully. Better than `315507` (`1.6418987912821085e-7`) and `315416` (`1.8089072203069811e-7`).
- Decision: keep as current best and use this fold+kink block-split surface as the baseline for the next iteration.
- Lesson: fold+kink block splitting is an exact micro-optimization with small effective-compute upside and modest extra backend time.

### Submission 315512 - Layer-Scheduled Block Split

- Result: regressed; local-only overfit.
- Change: keeps `315509` full sampled-stage block split plus fold-kink block split, and adds a conservative early-layer schedule from a 20-MLP public-mini sweep: pack-all on layers `1,2,4,5,9,10`; higher split thresholds on layers `3,6,7,8`; default `0.75` split elsewhere.
- Local expectation: first-10 public-mini check had `0/10` failures, adjusted score `1.702862926985e-7`, raw final MSE `4.556266418376e-7`, mean multiplier `0.360587`, mean effective compute `98.08G`, backend sum `75.00s`, max wall `10.12s`. High-continuation stress set had `0/8` failures, adjusted `3.041652453535e-7`, mean effective `132.81G`, max wall `18.65s`.
- Leaderboard/public evidence: public score `1.615389109901443e-7`, score secondary/raw final MSE `3.7249477742307134e-7`, graded successfully but worse than current best `315509` (`1.590833103387916e-7`).
- Decision: reject and revert to the uniform `0.75` block-split threshold from `315509`; do not retry this hand-built early-layer schedule without broader evidence.
- Lesson: early-layer schedule tuning overfit the local public-mini slice even though it reduced local effective compute; leaderboard favored the simpler uniform threshold.

### Offline Full-Split Block-Split Policy Probe

- Result: proxy promising, but the direct estimator implementation regressed locally.
- Change: added `scripts/learn_block_split_policy.py`, which runs one sampled forward pass per MLP, caches per-layer design-matrix support statistics, and evaluates dense / pack-all / split-threshold action costs without rerunning candidate matmuls. The experiment uses the `full` public split with a 100-MLP holdout.
- Local evidence: full split (`1000` MLPs), `1024` samples, `900/100` train/holdout by MLP. On holdout contexts: fixed `split0.75` width `154.259`; fixed `packall` width `149.854`; oracle width `145.680`; learned depth-3 tree width `148.308` (`0.9614x` of fixed `split0.75`). The learned tree selected mostly `packall`, `split0.95`, `split0.90`, and a few `split0.85` actions.
- Estimator implementation check: hardcoded the learned tree in `estimator.py` and ran first-10 public-mini subprocess profile. It regressed versus `315509` local: adjusted `1.779972058635e-7` vs `1.729532355024e-7`, mean effective `102.28G` vs `99.56G`, backend sum `123.13s` vs `62.76s`. Reverted to the `315509` fixed `0.75` surface.
- Decision: do not submit the learned tree as implemented. Any future contextual policy must avoid recomputing several candidate split costs inside `predict()` or must learn a rule from features already computed for the chosen action.
- Lesson: more data confirms the early-layer proxy issue, but candidate-evaluation overhead can erase the proxy win in the real estimator.

### Submission 315515 - Fixed 0.85 Block Split

- Result: regressed; local-only/proxy overfit.
- Change: keeps `315509` full block split plus fold-kink block split, but changes the fixed dense/high-fire threshold from `0.75` to `0.85`. This tests a simple fixed threshold suggested by the full-split proxy without learned-policy overhead.
- Local expectation: first-10 public-mini check had `0/10` failures, adjusted score `1.707989953489e-7`, raw final MSE `4.556196913086e-7`, mean multiplier `0.361994`, mean effective compute `98.46G`, backend sum `73.20s`, max wall `10.69s`. High-continuation stress set had `0/8` failures, adjusted `3.026585424401e-7`, mean effective `132.35G`, max wall `17.00s`.
- Leaderboard/public evidence: public score `1.6300780646592332e-7`, score secondary/raw final MSE `3.7249472711664567e-7`, graded successfully but worse than current best `315509` (`1.590833103387916e-7`).
- Decision: reject and revert to the uniform `0.75` block-split threshold from `315509`.
- Lesson: fixed `0.85` improved local/proxy metrics but did not transfer; simple threshold tuning is also overfit-prone.

### Submission 315517 - Chunk 8192 Block Split

- Result: failed setup; packaging mistake.
- Change: keeps submission `315509` / Algorithm 17 uniform `0.75` block split and increases `_PACKED_ROWSPARSE_CHUNK_ROWS` from `2048` to `8192` to reduce packed-kernel call count and charged residual overhead.
- Local expectation: paired first-10 public-mini subprocess check improved adjusted score from `1.739684539162e-7` (`2048`) to `1.708857025381e-7` (`8192`) with essentially unchanged raw final MSE (`4.55622e-7`). Mean effective compute improved from `100.15G` to `98.33G`; mean FLOPs rose slightly from `93.02G` to `93.88G`; residual sum fell from `0.713s` to `0.445s`. First-36 stress check had `0/36` failures, no budget/time/residual exhaustion, max wall `14.46s`, max residual `0.160s`.
- Leaderboard/public evidence: failed in setup with `FileNotFoundError` for `/tmp/submission/.../sobol_points.npz`; no score, no quota counted. Root cause: submitted a single-file artifact, but the estimator loads shipped `sobol_points.npz` from `ctx.submission_dir`.
- Decision: reject only the artifact; resubmit the same estimator as a folder package including `sobol_points.npz`.
- Lesson: any submission using this estimator must package the folder or otherwise include `sobol_points.npz`; single-file `estimator.py` artifacts are invalid.

### Submission 315518 - Chunk 8192 Block Split with Sobol Artifact

- Result: improved; new current best.
- Change: same estimator as `315517`, but packaged as a folder artifact containing `estimator.py`, `requirements.txt`, and `sobol_points.npz`.
- Local expectation: same as `315517`: paired first-10 public-mini adjusted `1.708857025381e-7` for `8192` vs `1.739684539162e-7` for `2048`; first-36 stress had `0/36` failures, max wall `14.46s`, max residual `0.160s`.
- Leaderboard/public evidence: public score `1.5676438766098022e-7`, score secondary/raw final MSE `3.7249364169156253e-7`, inferred multiplier `0.42085117734918676`; graded successfully and improved over `315509` (`1.590833103387916e-7`).
- Decision: keep chunk `8192` as the confirmed baseline for Strassen experiments.
- Lesson: larger packed chunks transferred; residual/call-overhead savings beat the slight FLOP increase remotely.

### Submission 315521 - Strassen Sampled-Dense Matmuls

- Result: improved; new current best.
- Change: starts from `315518` and adds guarded one-level Strassen for large sampled dense matmuls: high-fire block-split dense branch, packed-kernel dense fallback, first-layer antithetic dense half, layer-31 folded-on sampled contribution, and generic dense sampled fallback. `_sample_alpha` classification probes remain plain matmul after the block-split probe regressed locally.
- Local expectation: first-10 public-mini improved adjusted score from `1.708857025381e-7` (`315518` local chunk `8192`) to `1.607992476973e-7`; mean effective compute from `98.33G` to `92.56G`; raw final MSE essentially unchanged. First-36 stress improved from `1.850651848793e-7` to `1.794391039294e-7`, with `0/36` failures, max wall `17.39s`, max residual `0.393s`, and no budget/time/combined exhaustion. The `_sample_alpha` block-split variant was worse on first-10 (`1.643482704723e-7`, mean effective `95.43G`) due to extra take/einsum overhead, so it was not submitted.
- Leaderboard/public evidence: public score `1.4895563350388822e-7`, score secondary/raw final MSE `3.724955968209542e-7`, inferred multiplier `0.39988562220639096`; graded successfully and improved over `315518` (`1.5676438766098022e-7`).
- Decision: keep as current best and freeze this as the Algorithm 17 estimator surface before further experiments.
- Lesson: one-level Strassen transferred strongly for large dense sampled matmuls, while probe-path block splitting and naive density fallback were local regressions.

### Local Probe - Strassen Density Fallback 0.25

- Result: regressed locally; reverted.
- Change: added an average-density guard inside `_packed_matmul` to skip row packing and use dense-Strassen when a chunk's average nonzero density exceeded `0.25`.
- Local evidence: first-5 public-mini with Strassen threshold `0.75` worsened from adjusted `1.575469150647e-7`, mean effective `97.84G` to adjusted `1.785511969058e-7`, mean effective `110.14G`. The guard reduced `argpartition` calls (`953` to `533`) and `einsum` FLOPs (`171.23G` to `63.07G`) but increased dense `matmul` FLOPs (`262.49G` to `446.25G`).
- Decision: do not submit; revert the guard.
- Lesson: the generic 10-15% sparse-density rule does not transfer directly to this packed activation kernel; even moderately sparse packed chunks can beat dense-Strassen in charged FLOPs.