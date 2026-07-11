# Submission Learnings - 2026-07-10

## Submission Log

### Submission 315718 - Put-Along-Axis Row Unsort

- Result: improved/good; exact leaderboard metrics not captured locally.
- Change: submitted the current Algorithm 17 surface with mask reuse plus `zeros_like`/`put_along_axis` row-order restore instead of inverse `argsort(row_order)`. Artifact was built in folder mode as `submission-put-unsort-2026-07-10.tar.gz` and contains `estimator.py`, `requirements.txt`, `sobol_points.npz`, and `manifest.json`.
- Local expectation: public-mini n36 subprocess adjusted `1.594630365522e-7`, raw final MSE `4.556868091754e-7`, mean multiplier `0.336219`, mean effective compute `91.45G`, `0/36` failures. Pre-submit fixed seed 42 n3 had local adjusted `1.486462911327e-7` and subprocess adjusted `1.534403795503e-7`, both with raw final MSE `4.268462608555e-7` and `0/3` failures.
- Leaderboard/public evidence: AIcrowd `315718` graded successfully and the result was reported good; fill exact public adjusted score/raw MSE/budget when available.
- Decision: keep and fold into the stable Algorithm 16 example surface.
- Lesson: exact unsort replacement is locally positive enough to test publicly.

## Local Probes

### Put-Along-Axis Row Unsort

- Change: replaced row-bucket inverse `argsort(row_order)` restore with `zeros_like(pre_sorted)` plus `put_along_axis(pre_chunk, row_order[:, None], pre_sorted, axis=0)` in `_packed_matmul`, on top of the existing mask-reuse path.
- Local evidence: public-mini n36 subprocess scored adjusted `1.594630365522e-7`, raw final MSE `4.556868091754e-7`, mean multiplier `0.336219`, mean effective compute `91.45G`, `0/36` failures. This improved over `clean075_plus080_n36` adjusted `1.616650205472e-7`, raw final MSE `4.556847111006e-7`, multiplier `0.340122`, effective `92.51G`.
- Decision: keep; it is an exact row-order restore and looks like a small compute win locally.

### Max-Squared Hardness Allocation Proxy

- Change: tested `N = clip(49152 * max(V / V_ref, 1)^2, 30720, 61440)` as a current-artifact proxy for an 80K hard-MLP cap idea. The current `sobol_points.npz` has only `30720` half-points, so exact 80K antithetic samples would require a larger artifact.
- Local evidence: public-mini n10 subprocess scored adjusted `1.545107562002e-7`, raw final MSE `4.162548933095e-7`, multiplier `0.377395`, effective `102.65G`, `0/10` failures. Baseline put-unsort n10 scored adjusted `1.523889052262e-7`, raw final MSE `4.556151935731e-7`, multiplier `0.328515`, effective `89.36G`.
- Decision: reject this allocation shape for now; it buys raw MSE but overspends enough to regress adjusted score. Exact 80K max-squared is likely riskier unless paired with a more selective hard gate.
