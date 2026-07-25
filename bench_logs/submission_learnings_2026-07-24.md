# Submission Learnings - 2026-07-24

## Context

- The repriced flopscope (PR #150) went live on the grader today. Billed gather
  FLOPs roughly double; `take`-heavy paths are hit hardest. Newer 61,440-sample
  refinements are going over budget / timing out, and the leaderboard best
  reverted to submission 314957 (July 5, layer-29+30 borderline refinement,
  40,960 samples, 45.8% budget under old pricing).
- Repo `estimator.py` at HEAD is NOT the 317421 ship bytes: commit 7ccf108
  ("clean up", Jul 21) dropped `_TOTAL_SAMPLES` 61,440 -> 16,384, and the
  Algorithm 40 commit changed packing thresholds (3/4 -> 1/5) and added
  2-level Strassen recursion. The verified 317421 bytes live in
  `submissions/algo34-antithetic-pilot/` (md5-checked against `7648ff2`).
- Algorithm 40 leans harder on packing/gathers — exactly what PR #150
  repriced. Re-bench before submitting anything from HEAD.

## Submission Log

### Submission 318620 - algo34-n40960 (Algorithm 36)

- Result: still grading
- Change: byte-identical to Algorithm 34 / 317421 except
  `_TOTAL_SAMPLES` 61,440 -> 40,960 (20,480 half-samples x 2 antithetic,
  prefix of the same shipped 30,720-half Sobol artifact; no rebuild).
  Package `submission-20260725-022122.tar.gz`, staged in
  `submissions/algo34-n40960/`.
- Local expectation: subprocess runner, 3 MLPs, 0 failures; raw final-layer
  MSE 7.22e-7, multiplier 0.678 — but local flopscope is still 0.8.0rc5
  (old pricing), so the multiplier does NOT reflect PR #150. Scaling
  317172's grader utilization (0.441 at 61,440) predicts ~0.29 old-pricing
  / ~0.6 if gather billing doubles — under budget either way.
- Hypothesis: cutting N restores budget headroom under the repriced
  flopscope and beats the zeroed/over-budget 61,440 surfaces; target is to
  retake the lead from 314957.
- Decision: awaiting leaderboard outcome.
- Lesson (pending): fixed-N surfaces must be re-sized whenever flopscope
  pricing changes; local validation cannot detect grader-side repricing.
