---
name: whest-submission-learning-loop
description: "Use when: a WHest AIcrowd submission has just been submitted, a submission id is available, leaderboard results are out, public score changed, a variant improved or regressed, or we need to record submission learnings and avoid repeating failed ideas."
argument-hint: "submission id, leaderboard score, public result, or variant summary"
---

# WHest Submission Learning Loop

## When to Use

Use this skill immediately after `whest submit`, when a leaderboard result arrives, or before starting a new WHest variant after a submission.

This is a post-submission workflow. It complements `whest-submission-review`, which is for pre-submit validation.

## Required Loop

1. Capture the submission id, variant name, estimator changes, sample count, and expected compute envelope.
2. Ask the user for the leaderboard outcome if it is not already known:
   - improved
   - regressed
   - tied/no meaningful movement
   - still grading
3. If available, record the public adjusted score, raw final-layer MSE, budget used, and any notable per-MLP or failure details.
4. Update the dated learning log at `bench_logs/submission_learnings_YYYY-MM-DD.md`.
5. Mark the hypothesis as one of:
   - confirmed
   - falsified
   - inconclusive
   - overfit/local-only
6. Before proposing or implementing the next variant, summarize what the result says not to try again.

## Learning Log Rules

- Reuse the current dated file for the day instead of scattering notes across new files.
- Append concise entries under a `Submission Log` section.
- Keep entries short enough to scan during future work.
- Prefer leaderboard/public-dataset evidence over synthetic-seed sweeps.
- Explicitly record negative results, especially ideas that looked good locally but regressed on the leaderboard.

## Entry Template

```md
### Submission <id> - <short variant name>

- Result: improved | regressed | tied | still grading
- Change: <one-line estimator or artifact change>
- Local expectation: <brief local evidence>
- Leaderboard/public evidence: <score, budget used, raw MSE if known>
- Decision: keep | revert | investigate | do not retry
- Lesson: <one sentence>
```

## Retrieving Scores

**Never use `webfetch` or browser scraping on AIcrowd submission pages.** The submission page is a React SPA — the score data is embedded in obfuscated JavaScript bundles and won't render without a headful browser. `webfetch` returns only skeleton HTML with no score values.

Instead, retrieve scores using one of these methods:

1. **`whest submit --watch` (preferred):** The CLI streams the grading status and prints the final score directly:
   ```
   ✓ Graded — score 1.7678734432755242e-07
   ```

2. **`whest submit --watch` on a previously-submitted id:** If you already have a submission id and want to re-watch:
   ```bash
   uv run whest submit --watch --submission-id <id>
   ```

3. **Leaderboard page:** The public leaderboard at `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboard` is also a React SPA and cannot be scraped with `webfetch`. Scores must be read manually by the user in a real browser.

4. **Post-submission CLI output:** The immediate output of `whest submit --watch` is the canonical automated source. If the user closed the terminal before grading finished, re-watch with `--submission-id`.

## Guardrails

- Do not continue a sequence of submissions without updating the learning log.
- Do not rely on a synthetic validation win if the leaderboard contradicted it.
- If a change is reverted, write down both the tempting local evidence and the reason it was rejected.
- Keep local scratch files out of submission packages with `.whestignore`.
