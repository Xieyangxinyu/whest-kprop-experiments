# Submission Learnings - 2026-07-11

## Submission Log

### Submission 315819 - Antithetic Pair-Complement Layer-1

- Result: failed
- Change: packaged folder submission with `estimator.py`, `requirements.txt`, and `sobol_points.npz`; keeps `_PAIR_COMPLEMENT_LAYER1 = True` and removes the rejected structured 2:4 residual experiment from the submission surface.
- Local expectation: validation passed; local seed42 n3 profile passed with `0/3` failures, adjusted score `1.51e-7`, raw final MSE `4.27e-7`, mean multiplier `0.35127893`; subprocess seed42 n3 profile passed with `0/3` failures, adjusted score `1.53e-7`, raw final MSE `4.27e-7`, mean multiplier `0.35576073`.
- Leaderboard/public evidence: AIcrowd rejected import with `ModuleNotFoundError: No module named 'numpy'` at `import numpy as np`.
- Decision: do not resubmit this artifact. Either declare `numpy` explicitly and test packaging, or remove the raw-NumPy/private-op path for a safe fallback submission.
- Lesson: local/subprocess validation used the local environment and did not catch missing remote dependency; private custom-op implementations that import raw NumPy need explicit dependency handling or should stay out of submissions.

### Safe Fallback Check - Remove Private Raw-NumPy Shortcut

- Result: local/subprocess validation passed, not submitted yet.
- Change: removed `import numpy`, private `flopscope._validation.require_budget`, and `_pair_complement_matmul`; `layer_idx=1` now uses the supported packed path again.
- Local expectation: `whest validate` passed; local seed42 n3 profile passed with `0/3` failures, adjusted score `1.58e-7`, raw final MSE `4.27e-7`, mean multiplier `0.36615559`.
- Subprocess evidence: seed42 n3 profile passed with `0/3` failures, adjusted score `1.52e-7`, raw final MSE `4.27e-7`, mean multiplier `0.35504670`.
- Decision: safe fallback is submission-ready but gives up the private antithetic compute saving.
- Lesson: supported-only estimator still has the Algorithm 16/17 block-split/packed/Strassen edge, but not the extra antithetic custom-op edge.

### Submission 315824 - Finer Antithetic Row Buckets

- Result: improved
- Change: kept the private raw-NumPy shortcut removed; added packed row buckets `112, 144, 160, 176` to reduce padding around half-density antithetic layer-1 rows.
- Local expectation: focused MLP 14 `layer_idx=1` check saved `117,400,514` FLOPs with max diff `4.53e-06`; `whest validate` passed; local seed42 n3 profile passed with `0/3` failures, adjusted score `1.55e-7`, raw final MSE `4.27e-7`, mean multiplier `0.35948175`.
- Subprocess evidence: seed42 n3 profile passed with `0/3` failures, adjusted score `1.47e-7`, raw final MSE `4.27e-7`, mean multiplier `0.34484810`.
- Leaderboard/public evidence: graded successfully with public score `1.3489169563638498e-07`.
- Decision: keep as current best safe/native row-sparse surface; private raw-NumPy pair-complement remains rejected for submission.
- Lesson: the submission-safe way to exploit antithetic half-density is reducing row-bucket padding, not private custom op accounting.

### Submission 315843 - Packed Bucket Boundary Cleanup

- Result: improved
- Change: kept 315824 surface; removed dead `_ =`/unused `_run_block()` return plumbing, removed redundant appended `prev_width` bucket, used `fnp.searchsorted(sorted_nnz, limit, side="right")`, and changed fully-overwritten reorder buffer from `zeros_like` to `empty_like`.
- Local expectation: exact-output vs 315824 artifact on fixed synthetic MLPs; one-MLP attribution on `kathleen-munoz` showed `-14.6M` deterministic FLOPs and repeated subprocess A/B mean adjusted delta `-2.64e-09` with residual noise.
- Leaderboard/public evidence: submitted as `submission-cleanup-2026-07-11.tar.gz`; AIcrowd submission id `315843`; graded successfully with public adjusted score `1.3487136997788326e-07`, public raw final MSE / secondary score `3.724907188029647e-07`.
- Decision: keep as current best safe/native row-sparse surface; the cleanup transferred but only by a tiny margin over 315824 (`1.3489169563638498e-07`).
- Lesson: exact-output cleanup and `searchsorted` bucket boundaries can transfer, but public movement is very small; keep future cleanup variants similarly exact and deterministic on FLOPs.

### Submission 315849 - No-Bucket8 Strassen/Half-Sample Cleanup

- Result: regressed versus current best, but beat 315824 slightly.
- Change: packaged bucket-16 row-bucket surface with SC16-style immediate Strassen accumulation, half-sample-only Sobol loading, and `_DENSE_STRASSEN_MIN_ROWS = 1024`; did not include targeted bucket-8 row buckets.
- Local expectation: `whest validate` passed; subprocess seed42 n3 passed with `0/3` failures, adjusted `1.4935626492445296e-7`, raw final MSE `4.2683230579617276e-7`, mean effective compute `95.009G`.
- Leaderboard/public evidence: AIcrowd submission id `315849`; graded successfully with public adjusted score `1.34888792540272e-07`, public raw final MSE / secondary score `3.724907685409562e-07`.
- Decision: do not replace current best `315843` (`1.3487136997788326e-07`); the non-bucket cleanup transferred versus `315824` (`1.3489169563638498e-07`), but not enough to beat packed bucket boundary cleanup.
- Lesson: SC16 accumulation/half-sample/1024-row threshold are submission-safe small positives, but the no-bucket8 surface is not the current best; continue from 315843 or a properly validated targeted bucket-8 artifact.

### Submission 315851 - No 8-Limit Strassen/Half-Sample Isolate

- Result: improved; new current best.
- Change: packaged bucket-16 surface with row buckets `(0,16,32,48,64,80,96,112,128,144,160,176,192)`, SC16-style immediate Strassen accumulation, half-sample-only Sobol loading, and `_DENSE_STRASSEN_MIN_ROWS = 4096`.
- Local expectation: `whest validate` passed; subprocess seed42 n3 passed with `0/3` failures, adjusted `1.4537996599469996e-7`, raw final MSE `4.2683230579617276e-7`, mean effective compute `92.592G`; local profile showed removing the `8` row limit preserved final MSE and reduced argpartition/einsum calls (`1113 -> 971`).
- Leaderboard/public evidence: AIcrowd submission id `315851`; graded successfully with public adjusted score `1.348653018476005e-07`, public raw final MSE / secondary score `3.724907429614177e-07`.
- Decision: keep as current best over `315843` (`1.3487136997788326e-07`), despite slightly worse raw final MSE than `315843` (`3.724907188029647e-07`).
- Lesson: removing the special `8` row-bucket limit transferred; the adjusted win is likely from lower effective compute/backend overhead, not raw-MSE improvement.