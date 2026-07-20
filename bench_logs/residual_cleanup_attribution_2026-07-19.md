# Residual Cleanup Attribution - 2026-07-19

Purpose: keep a compact record of exact-output / residual-wall / FLOP-cleanup ideas tested around Algorithm 34/36 so future work does not re-bundle rejected pieces or over-read noisy local timing.

## Baseline

Public-confirmed baseline is submission 317459:

- Surface: Algorithm 34 plus ReLU mask threading, original eye-matmul `_scatter`.
- Public: adjusted `1.285e-7`, raw final MSE `2.98e-7`, budget `43.28%`, effective compute `1.18e11`.
- Decision: keep as public-confirmed baseline unless a later public submission improves.

## Public Results

| Submission | Change | Public result | Decision |
|---|---|---|---|
| 317455 | `fnp.put` scatter only | Adjusted `1.319e-7`, raw `2.98e-7`, budget `44.43%` | Reject; zero tracked FLOPs did not transfer because backend/residual overhead dominated. |
| 317456 | `fnp.put` scatter + where threading | Adjusted `1.310e-7`, raw `2.98e-7`, budget `44.07%` | Reject bundle; `fnp.put` remains suspect. |
| 317459 | where threading only | Adjusted `1.285e-7`, raw `2.98e-7`, budget `43.28%` | Keep; only public-confirmed cleanup win. |
| 317460 | broad cleanup stack | Smoke-test `ESTIMATOR_EXCEPTION` | Reject; broke shallow-depth smoke invariants. |
| 317462 | smoke-fixed broad cleanup stack | Adjusted `1.309e-7`, raw `2.98e-7`, budget `44.11%` | Reject; broad cleanup did not transfer. |
| 317468 | conservative salvage bundle: array-only sample block + exact `sorted_nnz[-1]` max + remove `final_var_mean` | Adjusted `1.311e-7`, raw `2.98e-7`, budget `44.17%` | Reject bundle; component attribution shows not all pieces help. |
| 317472 | isolated array-only sample block | Adjusted `1.313e-7`, raw `2.98e-7`, budget `44.21%` | Reject for now; local residual win did not transfer. |

## Component Attribution

| Component | Local evidence | Public evidence | Current decision |
|---|---|---|---|
| ReLU mask threading | `where(mask, pre, 0)` plus mask reuse removes redundant `x > 0` comparisons; local mixed but exact-output. | 317459 improved vs 317421 with same raw MSE. | Keep. |
| `fnp.put` scatter | 0 tracked FLOPs vs eye-matmul, but adds backend/data-mutation path. | 317455 and 317456 regressed. | Reject; keep original eye-matmul scatter. |
| Array-only `_sample_block` | Isolated n=3 vs 317459 locally: FLOPs `-7.86M/MLP`, wall `-1.82s/MLP`, residual `-0.141s/MLP`, effective `-14.08B/MLP`, same raw MSE. | 317472 regressed to `1.313e-7`; budget `44.21%`. | Reject as public-scoring variant despite local win. |
| `sorted_nnz[-1]` replacing `fnp.max(nnz_per_row)` | Isolated n=1 lowered tracked FLOPs by ~`1.9M` but worsened wall/residual/effective locally. | Bundled in 317468, which regressed. | Reject for now; do not bundle. |
| Remove unused `final_var_mean` | Isolated n=1 saved only ~`256` FLOPs and worsened residual/effective locally. | Bundled in 317468, which regressed. | Reject; savings too small. |
| `fnp.var` -> mean-of-square | n=1 looked good, but n=3 reversed: FLOPs `-3.02M/MLP`, residual `+0.0066s/MLP`, effective `+654M/MLP`, same raw MSE. | Not submitted. | Reject until larger repeated local evidence says otherwise. |
| Layer-0 closed form | n=1 preserved final MSE but worsened residual/effective; tiny all-layer arithmetic drift. | Not submitted. | Reject. |
| Rows-only extra block | n=1 preserved final MSE but worsened residual/effective significantly. | Not submitted. | Reject. |
| Exact bucket-k (`k = min(limit, prev_width)`) | n=3 vs array-only: `einsum` FLOPs `-166M` total, no new calls, final MSE drift ~`4.7e-12`; residual worsened enough that effective compute rose ~`474M/MLP`. | Not submitted. | Hold; analytically clean but local effective compute did not improve on this machine. |

## Interpretation

The public grader repeatedly rejected micro-cleanups that only shave tiny tracked FLOPs or rely on residual-wall attribution. The only cleanup that transferred publicly was ReLU mask threading. Local residual wins are not reliable enough for submission unless the change is both isolated and large enough to dominate timing-window noise.

Bundling small residual cleanups has not produced a threshold effect so far:

- Broad bundle 317462 regressed.
- Conservative bundle 317468 regressed.
- Isolated array-only 317472 regressed.

Future bundling should require a different kind of evidence: either a large hot-path contraction reduction, a clear raw final-MSE gain, or repeated fixed-seed/multi-machine evidence that effective compute improves. Do not bundle rejected micro-cleanups just to make the deterministic FLOP delta larger.

## Workflow Rules Going Forward

1. Start from public-confirmed 317459 unless a newer public result improves.
2. Test one scientific change per temp folder with `estimator.py` and `sobol_points.npz` together.
3. Smoke-test shallow depths plus depth 32 before packaging.
4. Use fixed-seed subprocess JSON and compare saved files, not terminal tails.
5. Require unchanged raw final MSE for exact-output ideas and lower effective compute on n=3/n=5 before considering a public probe.
6. Treat public results as final for residual-sensitive micro-cleanups; local residual attribution is too noisy to override repeated public regressions.