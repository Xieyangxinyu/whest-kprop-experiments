# Multistage Refinement Plan - 2026-07-05

## Goal

Improve the current `40960`-sample active-set Sobol estimator by using early samples to refine Stage 1 decisions and guide later estimation, without repeating local-overfit traps like Sobol seed selection.

Primary metric: public `adjusted_final_layer_score`.

Secondary local metrics:
- raw final-layer MSE
- budget utilization / score multiplier
- all-layers MSE
- per-class final-layer error by dead / kink / on

## Current Baseline

Current best public direction:
- `40960` effective samples: `20480` Sobol half-points plus antithetic pairs.
- Stage 1 diagonal analytical propagation classifies neurons as dead / kink / on.
- Stage 2 active-set Sobol MC samples non-dead neurons.
- Layer-30 fold skips confident on neurons.
- Dead-neuron analytical correction is added back.
- Submission `314954` added two-ended layer-30 refinement and improved public score slightly to `2.27e-7`.

Important negative results:
- `49152` samples used too much compute and was worse on leaderboard.
- Sobol scramble seed `314159` overfit local validation and regressed on leaderboard.
- Sphere/radial collapsed sampling helped at `16384` but lost at `40960`.
- Final-neuron top-k sample allocation was worse because it starved the base estimate.

## Working Hypothesis

Stage 1 is a strong cheap prior, not something to replace. Its main weakness is late-layer model bias from diagonal propagation, especially around classification thresholds.

Multistage sampling should therefore do two things:

1. Refine structural decisions:
   - demote weak `on -> kink`
   - promote weak `dead -> kink`
   - keep very confident dead/on decisions analytical

2. Use pilot information to improve later estimates without starving the full final-layer average:
   - avoid reducing the base sample count for all neurons
   - reuse pilot samples in the final estimate
   - prefer conservative corrections and stratification over arbitrary tilting

## Design Constraints

- Keep effective sample count near `40960` unless leaderboard data says otherwise.
- Keep budget use near the proven `45%` envelope.
- Do not choose QMC seeds by synthetic validation alone.
- Do not submit a change based only on one synthetic seed group.
- Prefer changes that preserve normal Sobol sampling unless proper weights or stratification are available.
- Record every submission result in `bench_logs/submission_learnings_YYYY-MM-DD.md` before starting the next variant.

## Experiment Track A - Sharpen Stage 1 Classification

### A1. Instrument two-ended layer-30 refinement

Question: how many neurons are actually changed by the 5% pilot, and which side contributes the gain?

Collect per MLP:
- number of `on -> kink` demotions at layer 30
- number of `dead -> kink` promotions at layer 30
- final-layer MSE delta vs `normal fold-on 3.00`
- FLOP delta

Decision rule:
- If most gains come from `dead -> kink`, focus next on cheaper dead probes.
- If most gains come from `on -> kink`, focus next on fold threshold and on-probe cost.
- If changes are rare but high impact, make the pilot targeted to borderline neurons only.

### A2. Target only borderline layer-30 neurons

Instead of probing all layer-30 dead/on neurons:
- probe on-neurons with analytical alpha in `[3.0, 4.0]`
- probe dead-neurons with analytical alpha in `[-4.0, -2.5]`

Expected benefit:
- lower pilot FLOPs
- similar correction quality

Validation:
- local seeds `0..4` and `5..9`
- compare adjusted score and raw final MSE against current two-ended all-probe variant

### A3. Try layer-window refinement

Extend conservative two-ended refinement to late layers only:
- layer 29
- layer 30
- optionally layer 31 only if no regression

Risk:
- more pilot FLOPs
- dynamic mask changes can alter downstream active sets and increase compute

Decision rule:
- Only keep if public-like validation shows score improvement after multiplier.

## Experiment Track B - Pilot-Guided Corrections

### B1. Late-layer residual diagnostics

Use pilot/full local runs to bin final-layer error by:
- final neuron class: dead / kink / on
- analytical alpha bin
- pilot alpha bin
- analytical variance
- layer-30 promoted/demoted status

Goal:
find systematic signed residuals that can be corrected cheaply.

Do not submit any correction until the bias is consistent across fresh seed groups.

### B2. Conservative residual correction

If diagnostics show stable bias, add a small correction only for the affected bin.

Candidate form:
- correction = `lambda * analytical_mu` for a specific class/bin
- or correction = `lambda * (pilot_mean - analytical_mean)` for promoted/demoted neurons

Guardrail:
- tune `lambda` on one seed group, validate on a disjoint group.
- treat leaderboard contradiction as decisive.

## Experiment Track C - Guided Next Batch Sampling

### C1. Stratified activation-pattern sampling

Use pilot samples to identify late-layer activation strata, then ensure the next batch covers underrepresented strata.

Safer than arbitrary tilting because strata can have explicit weights.

Open design questions:
- which low-dimensional stratum label to use?
- layer-30 active count bucket?
- number of near-boundary neurons?
- sign pattern of a small influential neuron subset?

### C2. Boundary-focused sampling with correction

Oversample near uncertain late ReLU boundaries only if we can attach weights or a controlled correction.

Risk:
- unweighted tilting can look good locally and fail on leaderboard.

Initial stance:
- do not implement until Track B diagnostics identify specific boundary-driven error.

## Immediate Next Step

Implement instrumentation in `eval_variants.py` for the two-ended layer-30 pilot:
- print average demotions/promotions per variant
- optionally print per-seed counts
- compare current all-probe variant against borderline-only probing

Then update this plan with the observed counts and the next decision.

## Iteration Log

### 2026-07-05

- Created plan after submission `314954` improved to public score `2.27e-7`.
- Current best direction is two-ended Stage 1 refinement, not sample-count scaling or QMC seed tuning.
- Next experiment should make the layer-30 pilot cheaper or more targeted before another submission.
