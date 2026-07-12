# Strassen (Algorithm 17) Submission Prep - 2026-07-09

## Candidate

estimator.py with one-level flopscope-native Strassen on the dominant
sample-propagation matmuls (`_USE_STRASSEN=True`, `_PACKED_ROWSPARSE=False`,
`_STRASSEN_MIN_DIM=96`). Legitimacy cleared by user + team (Strassen = genuine
fewer scalar multiplies, unmodified flopscope counts them; no prohibited behavior).

## Validation gates (all pass)

- `whest validate`: OK (shape, finite, setup). Tiny width-4 MLP correctly falls
  back to plain matmul below MIN_DIM.
- Parity seed 42 n=3: local adjusted 2.29e-7 / subprocess 2.33e-7, RAW MSE
  IDENTICAL 4.27e-7, mult ~0.54, 0/3 failed both. Subprocess-safe, no hidden state.
- Public-mini first-10 SAME-MACHINE A/B:
  - Strassen: adjusted 2.33e-7, mult 0.492, residual 2.0s, 0/10 failed
  - 315416 (packing): adjusted 3.60e-7, mult 0.779, residual 10.77s, 0/10 failed
  - RAW MSE identical 4.56e-7 (exact). Strassen -35% here.
- Seed-0 n=3 subprocess (earlier): Strassen 1.29e-7 vs packing 1.98e-7.

## CRITICAL caveat (leaderboard uncertainty)

The win magnitude depends on the GRADER machine's residual cost for packing's
`fnp.take` gather (flopscope charges gather as free indexing -> its wall-time is
pure residual). This dev box is SLOW (315416 mult 0.779 here vs ~0.486 recorded
leaderboard) so packing's gather residual is inflated -> big Strassen win locally.
On the faster grader, packing's residual is smaller (its leaderboard mult 0.486 vs
~0.38 analytical -> ~28% residual tax), and Strassen has ~10% HIGHER analytical
flops. Net: Strassen LIKELY still wins on the grader (~5-10%) but the margin is
uncertain and could be marginal. Strassen is EXACT + 0-failed, so worst case is a
small score change, never a failure. NOT yet validated on both seed groups
(0-4/5-9); public-mini is the representative check and it passed.

## Package

`whest package --estimator . --output submission-strassen-algo17.tar.gz`:
4 files (estimator.py, sobol_points.npz, requirements.txt, manifest.json), 28MB,
under 52MB cap. Fixed .whestignore to exclude local *_cache.npz / .est*.npz (were
pushing it to 80MB) and relu_moment_propagation.py. requirements.txt lists scipy
(unused - estimator imports only math/pathlib/flopscope/whestbench; harmless).

## Status

PREPARED, NOT SUBMITTED. Awaiting user go/no-go (outward-facing action is theirs).
estimator.py currently staged as Algorithm 17 (Strassen). To revert to 315416:
`git checkout estimator.py` (staged candidate also at
scratchpad/estimator_strassen_candidate.py).
