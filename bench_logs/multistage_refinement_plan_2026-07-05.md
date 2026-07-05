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
- Submission `314957` added layer-29 borderline dead refinement on top of layer-30 borderline refinement and improved public score to `2.25e-7`.

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

Next candidate should refine earlier dead-neuron gates conservatively, starting from the leaderboard-confirmed layer-29 + layer-30 probe:
- at layer 29, probe dead-neurons with analytical alpha in `[-4.0, -3.0]` and promote to kink if pilot alpha `> -2.5`
- at layer 30, probe on-neurons with analytical alpha in `[3.0, 4.0]` and demote to kink if pilot alpha `<= 3.0`
- at layer 30, probe dead-neurons with analytical alpha in `[-4.0, -3.0]` and promote to kink if pilot alpha `> -2.5`

Rationale: local validation improved over layer-30-only borderline refinement on both seed groups while probing only about `40-45` total columns, and submission `314957` confirmed a small public improvement.

## Iteration Log

### 2026-07-05

- Created plan after submission `314954` improved to public score `2.27e-7`.
- Current best direction is two-ended Stage 1 refinement, not sample-count scaling or QMC seed tuning.
- Instrumented `eval_variants.py` with layer-30 demotion/promotion counts.
- All-probe two-ended refinement changed about `5` on-neurons and `0.8-1.0` dead-neurons on average, but probed about `158-164` columns.
- Borderline-only `4/-4` refinement changed nearly the same useful neurons while probing only about `26-29` columns, and was best or tied-best on both local seed groups.
- Isolated probes suggest the dead-side correction carries most of the raw-MSE gain; on-side still adds a small extra improvement when combined.
- Tested dead refinement further back than layer 30. Layer-29 + layer-30 borderline refinement was consistently better than layer-30-only on both seed groups (`5.530e-7` vs `5.550e-7`; `2.756e-7` vs `2.784e-7`).
- Wider dead-only windows were unstable: `l24-30` was best on seeds `5..9` but much worse on seeds `0..4`, so do not submit that without stronger evidence.
- Current next candidate is `l29+30 borderline 4/-4`; avoid layer-31 refinement for now because it repeatedly regressed.
- Submission `314957` confirmed `l29+30 borderline 4/-4` on the public leaderboard: adjusted score `2.25e-7`, final-layer MSE `4.94e-7`, budget used `45.82%`.
