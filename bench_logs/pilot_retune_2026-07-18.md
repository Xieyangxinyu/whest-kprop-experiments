# Pilot-fraction retune at floor-mode N=15360: KILLED + attribution retraction - 2026-07-18

## Hypothesis (pre-registered)

The low-N classification-bias doubling (6.6e-8 -> 1.49e-7, nets 1-8) is
pilot-probe starvation (768/3072 rows at N=15360 vs shipped 1536/6144);
restoring absolute probe rows restores fleet bias^2 to <= 8e-8 at <= 2%
flop cost, making floor-mode project -10..-15% vs 316405.

Bars: ADOPT bias <= 8e-8 & var*N in [1.7e-2, 2.4e-2] & C <= 2.72e10;
MARGINAL (8e-8, 1.2e-7] -> recomputed projection must beat 1.25e-7;
KILL bias > 1.2e-7 at 4x rows.

## Protocol

Script: .tmp/router_opt/pilot_retune_probe.{py,pkl}. Nets 1-8, fixed
N=15360 (all stage constants), R=5 scrambled-Sobol replicates (seeds
1000-1004, direct .random(7680)), paired across configs; decomposition
per error_attribution.py (bias_part = mean_j max((E_j-gt_j)^2 - V_j/R, 0)).
Configs (_PILOT_FRACTION, _PILOT_RECHECK_FRACTION): stock (0.05, 0.20)
= 768/3072 rows; x2 (0.10, 0.40) = 1536/6144; x4 (0.20, 0.80) =
3072/12288. Runner: in-process predicts, `uv run --with scipy` (scipy no
longer in the synced venv).

## Result: KILL, flat to 4 decimal places

| config | rows | var_part | bias_part | var*N | flops | dC |
|---|---|---|---|---|---|---|
| stock | 768/3072 | 1.259e-6 | 1.493e-7 | 1.934e-2 | 2.768e10 | +0.00% |
| x2 | 1536/6144 | 1.256e-6 | 1.493e-7 | 1.929e-2 | 2.820e10 | +1.88% |
| x4 | 3072/12288 | 1.256e-6 | 1.492e-7 | 1.929e-2 | 2.879e10 | +4.01% |

Baseline replicates the 07-17 discriminator exactly (1.4935e-7 / 1.93e-2).
Bias is INVARIANT to pilot rows: the starvation hypothesis is falsified.
Extra pilot rows are pure flop cost (+1.9%/+4.0%).

## Attribution retraction: the "bias doubling" was a measurement artifact

The ratio measured_bias / (V/R) is 0.60 at high N and 0.59 at low N —
constant. A zero-true-bias null simulation (draw R=5 replicates ~
N(gt, V_j) from each net's measured per-output V, recompute bias_part;
2000 sims/net) gives the estimator's clipped-noise floor:

- Low-N fleet: measured 1.493e-7, null floor 1.349e-7 -> TRUE excess ~1.4e-8.
- High-N fleet: measured 6.56e-8, null floor 5.29e-8 -> TRUE excess ~1.3e-8.

**True structural bias is ~1.3-1.4e-8 at BOTH N. The 6.6e-8 -> 1.49e-7
"doubling" was the decomposition noise floor scaling with V/R (V doubles
when N halves).** The 07-17 "classification-bias doubling / pilot
starvation" attribution is RETRACTED. Same retraction applies to the
high-N noclass comparison (6.6e-8 vs 1.36e-7 "classification helps" —
both are ~floor; classification's true bias effect at high N is ~nil).

Per-net exceptions with real bias (above p95 of the null):
- net 6 at low N: +2.8e-7 excess (3.46e-7 vs floor 6.2e-8), absent at
  high N (2.2e-8 ~ floor). Targeted noclass ablation @15360: bias
  3.40e-7 noclass vs 3.46e-7 stock -> NOT classification. Real,
  structural, N-dependent, in the pure sampling+fold path (candidate
  mechanism: sample-moment-dependent fold analytics, O(1/N)-in-empirical-
  measure). This is likely the mechanism class behind the 32-net A/B's
  cratering nets (19, 2, 14, 26).
- net 8 at high N: +1.5e-7 excess — the one real structural-bias net at
  shipped N.

## Consequences

1. **Pilot retune is DEAD twice over**: bias is flat in probe rows, and
   the deficit it was meant to fix mostly does not exist. Do not revisit
   pilot fractions/rows for score at any N.
2. **Floor-mode go/no-go gate: NO-GO on score.** The -15% upside was the
   phantom bias; the honest projection stays ~1.31e-7, FLAT vs 316405
   (1.3245e-7). Floor-mode's remaining case is entirely non-score:
   pricing-divergence immunity (0.1 clamp) + ~3-min jobs outside both
   death regimes. Whether that is worth a submission slot is a decision
   for after the 316405 re-grade discriminator.
3. **Protocol rule (extends the >=13-net-aggregates rule): any
   replicate-decomposition bias_part claim MUST be reported net of the
   simulated zero-bias null floor** (floor ~ 0.5-0.6 x V/R at R=5).
   Raising R shrinks the floor ~1/R. Past bias_part magnitudes quoted
   without this correction are upper bounds, not measurements.
4. Residual real low-N pathology (net 6 class) is structural and
   unfixable by classification knobs; if floor-mode is ever built for
   robustness, its per-net tail risk stands (~3.5e-8 fleet contribution
   on nets 1-8, worse on wider sets per the A/B craters).

## Next steps (unchanged order from 07-17, gate now resolved)

1. Resolve pricing divergence: verbatim 316405 re-grade (1 slot).
2. Floor-mode: build only if robustness case justifies a slot post-(1);
   no score gain expected.
3. The 1.7x product gap: Strassen two-level ruling remains the only
   unmined lever. Watch the board.
