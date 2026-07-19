# Piecewise-region / analytic-propagation family: SCOPING - 2026-07-17

## Motivation

After tonight's kills (RB/EB read-out smoothing; merge-and-multiply
1000x over bar) plus 07-14's (low-rank 1900x; learned corrections +1%),
the modification-of-algo28 menu is empty. The merge probe's structural
finding — deep activations collapse to a ~few-dim cone (~50 occupied LSH
cells at l>=24 / 16k samples) while kink boundaries stay locally dense —
plus the board's ~4x raw-per-compute frontier motivate scoping the one
unfalsified family: estimators that treat the net as EXACTLY piecewise
linear and move integration from samples to analytics.

Structural facts to build on:
- No biases anywhere: f(x) = A_P x on each activation-pattern region;
  regions are CONES (P(cx)=P(x), c>0) — patterns live on the sphere;
  radius separates (E[x 1_C] = E[r] E[u 1_C(u)], r~chi(256)).
- z_1 = W0^T x is EXACTLY Gaussian with known cov W0^T W0 (free layer).
- Deep geometry ~few-dim (merge probe); early layers high-entropy
  (flip rates 19-38% at l=0-4).

## Candidate inventory vs prior evidence

| candidate | mechanism | prior-kill exposure | decisive unknown |
|---|---|---|---|
| C1 region enumeration + exact/EP cone integration | E[f] = sum_P A_P E[x 1_CP], analytic orthant-class integrals over dominant cones | none directly; merge kill does NOT apply (no centroid approx) | does realized-pattern MASS concentrate? (early-layer entropy says likely no) |
| C2 composite-map control variate | per cheap layer-1 signature group g, PRECOMPUTED A_g = prod(diag(m_l) W_l); surrogate g(x)=A_g x costs ~2 matmuls vs 32; CV: E_ADF-free, mean(f - g) correction on full samples + E[g] exact per group? (E[g] over group = A_g E[x|group], group = z1 sign-cell of a KNOWN Gaussian -> analytic!) | rho2=0.06 kill was ONE global linear map; K pattern-indexed maps are a different animal; MLMC kill was cheap-NET coupling, not composite maps | rho2(f, g) at K groups; composite build cost |
| C3 mixture-of-Gaussians assumed-density filter | propagate K-component Gaussian mixture with exact per-component full-cov ReLU moment updates; zero sampling variance | single-Gaussian full-cov anchor abs error 6.8e-5 (needs ~200x from mixture); NOT killed as PRIMARY estimator at floor economics (cheap_forward log flagged the re-read) | bias(K) curve; priced cost of cov propagation |

Cost notes (honest, with the 316894 caveat that LOCAL PRICING IS
CURRENTLY UNTRUSTED for the multiplier — any go decision needs the
316405 re-grade discriminator first):
- C2 surrogate eval ~2x256^2-class/sample vs f's ~54.5k x 32; composite
  build K x 32 x 2 x 256^3 = K x 1.07e9 flops (K=16 -> 1.7e10 = 6% of B;
  active-subnetwork composition ~150-wide roughly halves this).
- C3: full-cov layer update ~2x256^3 + orthant-corrected cov ~O(256^2)
  per component-layer; K=100 -> ~1e11 = 37% of B, zero variance.
- CV variance math for C2: with cost ratio ~16x and correction on N_f
  full samples + surrogate on N_g >> N_f, achievable variance factor
  ~ (1-rho2) + rho2 * (N_f/N_g-ish); leaders' 4x needs rho2 >~ 0.8;
  rho2 = 0.5 -> ~2x.

## Pre-registered probes and bars (in execution order)

P1 (pattern-mass census, decides C1, sizes C2/C3): on 8 mini nets,
N=8192: distinct realized full-depth patterns, per-layer distinct-mask
counts, mass share of top-k patterns.
  - C1 CONTINUE iff top-4096 patterns carry >= 80% of sample mass.
  - Expected: KILL (early-layer entropy).

P2 (composite-map CV, decides C2): group samples by s sign bits of
top-eigendirections of z1 (analytic Gaussian!), K=2^s in {8,32,128};
A_g from the group CENTROID's realized pattern (K cheap propagations);
rho2_eff = 1 - Var(f-g)/Var(f), variance-weighted over outputs, fleet.
  - CONTINUE to design iff fleet rho2_eff >= 0.5 at some K.
  - MARGINAL 0.3-0.5: only if rho2 rises steeply in K (then finer
    signatures/multi-probe assignment is the design problem).
  - KILL < 0.3 at K=128: pattern-indexed linear maps join the rho2=0.06
    surrogate graveyard and C2 is dead.

P3 (mixture-ADF bias curve, decides C3; run only if P2 < 0.8 since C3
is the zero-variance endgame): diagonal + full-cov mixture ADF offline,
K in {1,8,64}, split heuristic = largest-mass component along its
max-|alpha| kink coordinate; measure final-mean MSE vs mini GT.
  - CONTINUE iff MSE(K=64) <= 3e-7 (raw-competitive with zero variance
    => adjusted ~0.1-floor x 3e-7 = 3e-8 = beyond-frontier IF cost
    lands near-floor; even 1e-6 at floor = 1e-7 would beat 316405).
  - KILL if MSE(K=64) > 1e-6 or K-curve flattens (closure bias floor).

Fixed for all probes: nets 1-8 mini, antithetic Gaussian N=8192 (probe
class, offline numpy, no flopscope), GT = dataset final_means, seeds
fixed, paired contrasts where possible.

## Results

ALL THREE CANDIDATES KILLED at pre-registered bars (scripts/data:
.tmp/router_opt/region_probe.{py,pkl}, mixture_adf_probe.py):

- **P1/C1 KILLED**: every sample's full-depth pattern is UNIQUE
  (8192/8192 distinct, max multiplicity 1); even single-layer masks are
  ~all distinct at every depth (8191-8192/8192). The low-dim cone
  geometry does NOT quantize patterns; there is no mass concentration
  whatsoever. Region enumeration is unsalvageable.
- **P2/C2 KILLED (spectacularly)**: composite frozen-pattern linear maps
  are not merely uncorrelated with f, they are catastrophically
  mis-scaled: fleet rho2_eff ~ -50 at every K (surrogate variance ~50x
  f's). Mechanism: per-sample ReLU clipping is adaptive gain control on
  a near-critical system; freezing any pattern removes the attenuation
  and the map explodes. Same mechanism as the low-rank var_ratio 2-7.6x
  (07-14). Pattern-indexed linear surrogates join the rho2 graveyard for
  a deeper reason than correlation: the function's stability IS the
  nonlinearity.
- **P3/C3 KILLED**: mixture-ADF K-curve is FLAT: fleet MSE 7.00e-5 (K=1)
  -> 6.43e-5 (K=64). 64x components buy 8%; the 3e-7 bar needs ~200x.
  Closure bias regenerates at every layer; input-space splitting cannot
  touch it.

## Economics correction discovered during scoping (load-bearing)

Two fixed-N runs (30720/61440, net 2) give LOCAL F1 = 1.678e6/sample and
**F0 ~ 1.0e9 (1% of a net's compute), NOT the fleet-fitted 1.83e10**.
The fitted F0 was a model-unit artifact: flop-implied fleet mean N is
~57k actual samples (b_eff ~ 1.98e-2), not 44.9k model units. Flop mix
at N=61440: matmul 5.85e10 + einsum 4.25e10 = 97% of total; everything
else ~3%.

Consequences:
1. **Our adjusted score is intrinsically ~flat in N**: adjusted ~
   F1*b_eff/B + small terms = 1.22e-7 intrinsic product floor; we sit
   +8% above it (a-term + F0). There is no overhead to strip — the
   allocation-wash results were telling us this all along.
2. **The leaders' edge is a 1.7x F1*b PRODUCT gap, not 4x**: floor-rider
   adjusted 7.7e-8 implies F1*b ~ 1.95e4 vs ours 3.33e4. The "3.5-4x
   variance-per-compute" framing conflated operating points. 1.7x is
   the real, less-mystical target — but it must come from F1 or b,
   both heavily mined.
3. **Floor-mode variant of OUR estimator projects 1.296e-7** (-2% vs
   316405's 1.3245e-7; -4% vs the re-priced 316894 reality): cap N by
   the 0.1-floor budget (N ~ 15.6k), strip the adaptive rule (moot under
   a hard cap). Non-score properties are the real case: (a) the 0.1
   CLAMP makes the score IMMUNE to the grader pricing divergence that
   just cost 316894 +2%; (b) wall drops ~4x (~12s grader/net, ~2-3 min
   jobs) — fully outside the 15-min-reaper and 60s-cap death regimes.
   Caveat: b_eff extrapolation from 57k down to 15.6k samples needs a
   local A/B (32-net harness) before any build.

## Low-N b_eff A/B (32 nets, N=15360 vs 30720, shipped artifact)

Pre-registered bars: ratio raw(15360)/raw(30720) in [1.85, 2.15]
(pure-1/N ~1.97); projected floor adjusted must beat 1.3245e-7.

- **Headline FAILED the bar**: fleet raw 1.8447e-6 / 6.3085e-7 =
  ratio 2.924; naive floor projection 1.538e-7 (+16%). Several nets
  crater at low N (net 19: 22x; nets 2/14/26: 4-5.5x).
- **Mechanism discriminator (5 scrambled-Sobol replicates @15360, 8
  nets) REVERSES the interpretation**: variance 1.2592e-6 (89%) /
  bias^2 1.4935e-7 (11%); **var x N = 1.93e-2 = the high-N constant**
  (2.2e-2). Per-sample variance is INTACT at low N — b_eff holds. The
  A/B excess decomposes into:
  (a) shipped-artifact SHORT-PREFIX degradation: shipped raw@15360 is
      +31% over the scrambled-replicate class (it was -9% at 30720).
      The 7,680-half prefix of the full-length-validated artifact is a
      bad realization — fixable by re-baking an artifact at the floor
      length (scrambled-Sobol replicates already achieve the good
      constant).
  (b) classification-bias doubling (6.6e-8 -> 1.49e-7): pilot probes
      starve at 768/3072 rows — partially fixable by retuning pilot
      fractions for low N (costs flops, small).
- Revised floor-mode projection with a re-baked artifact:
  raw_mini ~ 1.41e-6 -> leaderboard ~1.19e-6 -> adjusted ~1.19e-7
  (**-10% vs 316405**); with bias retuned toward 6.6e-8: ~1.12e-7
  (**-15%**). Floor clamp immunity to the pricing divergence and ~3-min
  job walls (outside both death regimes) stand.
- Build prerequisites: (1) re-bake sobol artifact at floor length
  (power-of-2 half-count preferred; scrambled fresh block, prefix
  validation); (2) pilot-fraction retune at N~15k; (3) per-net N rule
  fitting C_i <= 0.1B (F1 spread 1.58-1.92e6 means fixed-N wastes ~10%
  on easy nets or busts the floor on hard ones — a deterministic
  flop-count-based rule, NOT wall-time); (4) 32-net gate, then
  user-gated submit. Data: .tmp/router_opt/lown_beff_ab.{py,pkl}.

## Artifact re-bake (10 candidates, screen 0-31 / holdout 32-63) — HYPOTHESIS WITHDRAWN

Two-stage anti-overfit protocol; the holdout INVERTED the screen ranking:
- Screen @15360 nets 0-31: candidates 1.156e-6 .. 1.872e-6, shipped
  1.845e-6 (2nd worst) — looked like the +31% prefix penalty.
- Holdout nets 32-63: screen-best direct-20260720 1.717e-6, runner-up
  direct-20260719 1.666e-6, **shipped 1.592e-6 = BEST**.
- 7168-prefix @14336 (nets 0-31): winner 1.443e-6 vs shipped 2.096e-6 —
  same screen-net-set direction, same suspicion of net-set luck.

Conclusions:
1. The A/B's "shipped short-prefix is a bad realization (+31%)" claim is
   WITHDRAWN — it was substantially net-set-specific single-realization
   luck. No transferable artifact-quality difference is measurable at
   32-net single-realization resolution: between-artifact deltas share
   ONE realization across all nets (effective df ~ 1), so fleet size
   does not average the noise away. Artifact comparisons need
   replicate-grade designs (multi-seed per candidate), like everything
   else (extends the ">=13-net aggregates" rule).
2. Artifact choice for floor-mode: RETAIN the shipped prefix (proven at
   full length across 21 graded submissions; no demonstrated
   replacement win). Candidate file sobol_points_floor7680_fp32.npz
   (direct-20260719) kept in .tmp/router_opt/ only as a spare.
3. Floor-mode projection REVISED (honest, artifact-neutral): fleet raw
   @15360 across artifacts/net-sets sits ~1.5-1.7e-6 mini; low-N bias
   on the wider net set is ~2-3e-7 (bigger than nets-1-8's 1.5e-7).
   Projection: ~1.31e-7 adjusted (FLAT vs 316405) before pilot retune;
   ~1.12e-7 (-15%) only if the retune restores bias to the high-N
   6.6e-8 level. **The pilot retune is now the load-bearing fix and the
   go/no-go gate for the floor-mode build.**

## Verdict and next steps

The exact/analytic piecewise family is closed with prejudice — the
near-critical adaptive-gain mechanism now unifies ALL failures across
both campaigns (surrogates, stratification, closure, merging, frozen
maps). Remaining rational moves, in order:
1. Resolve the pricing divergence (verbatim 316405 re-grade, 1 slot).
2. Local A/B of floor-mode (b_eff at low N); if b holds, build the
   floor-mode variant — modest score, large robustness/reliability win.
3. The 1.7x product gap: only unmined F1 lever = the blocked Strassen
   ruling (1.28x); only unmined b lever = none known. Watch the board.
