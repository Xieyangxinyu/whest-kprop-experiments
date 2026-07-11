# Submission Learnings - 2026-07-11

## Submission Log

### Submission 315844 - Algorithm 18 (probe-free + finer row buckets)

- Result: still grading
- Change: merged the two independent Algorithm 17 lines into one surface:
  (a) local probe-free wide dead band (pilot probes deleted, dead threshold
  -3.0 -> -4.0, on-side kept +3.0); (b) remote finer antithetic row buckets
  from submission 315824 (`origin/algorithm-17-finer-row-buckets-2026-07-11`):
  buckets 112/144/160/176, `searchsorted` bucket splits, `empty_like`
  row-order restore, layer 1 routed directly to `_packed_matmul`.
  Merge commit `1481c3d`; artifact `submission-algo18-probefree-finerbuckets.tar.gz`
  (5 files, validate-package OK).
- Local expectation: seed-42 n=3 subprocess adjusted 2.76e-7 (raw 3.9620e-7,
  multiplier 0.699, eff compute 1.90e11 / 2.72e11). Raw MSE is IDENTICAL to
  probe-free alone (bucket changes are exact rewrites of the same matmul), so
  the entire gain vs probe-free's 2.92e-7 is compute-multiplier savings from
  the finer buckets. Local/subprocess parity exact on raw MSE; 0/3 failed.
  Reference points, same seed/protocol: baseline 315718 3.10e-7 (n=3),
  2.946e-7 (n=10); probe-free alone 2.92e-7 (n=3).
- Leaderboard/public evidence: graded, public adjusted `1.3864410008848122e-7`,
  raw final MSE (score_secondary) `3.7222e-7`. REGRESSED vs both parents
  (API-fetched 2026-07-11): 315824 (probes + finer buckets) `1.3489e-7` raw
  `3.72491e-7`; 315718 (probes, coarse buckets) `1.3830e-7` raw `3.72497e-7`.
  Decomposition: raw MSE slightly IMPROVED (-0.07%), so the entire +2.8%
  regression vs 315824 is compute multiplier (implied 0.3725 vs 0.3621) —
  probe-free spends more effective compute on the grader than the probes did.
- Decision: investigate (3-arm public-mini per-MLP ablation: 315824 vs algo18
  vs algo18 with dead=-3.0); 315824 remains the leaderboard frontier.
- Lesson: probe deletion's win was a slow-dev-box artifact — locally the
  probes' Python residual dominates, on the fast grader their FLOP savings
  (≈38 demotions shrink matmuls) dominate; residual-vs-FLOP trades flip sign
  with machine speed, same caveat as 315416. NEVER declare a leaderboard
  comparison from bench-log memory alone: 315718/315824 scores were never
  recorded, and the initial "new best" call here was wrong —
  `AIcrowdClient.get_submission_status` gives exact prior scores in seconds.
  Also: grading queue sat >10 min in `submitted`, so `whest submit --watch`'s
  default 600s timeout detaches — poll get_submission_status instead of
  re-running submit.

## Process notes

- The planned 10-MLP multi-seed A/B of probe-free vs baseline was stopped
  early by user direction (too slow; ~11.5 min per 10-MLP arm). Only the
  baseline seed-42 n=10 point completed: adjusted 2.946e-7, raw 4.018e-7,
  multiplier 0.726, 0/10 failed. Kept here as a reference for future sweeps.
- "Test the submission" fallback protocol used instead: whest validate +
  seed-42 n=3 local/subprocess parity + validate-package. Raw-MSE parity and
  zero failure flags were the load-bearing gates.
