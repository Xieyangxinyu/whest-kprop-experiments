# Identity treatment of on/high-alpha neurons + on-neuron control variate — 2026-07-25

Question (user): can on/high-fire neurons be treated as identity (no ReLU
sampling) in layers earlier than the 30/31 fold, and is there an alpha
region where it pays? Follow-up: can the low-rank expected-gate structure
be exploited as a control variate for on-neurons specifically?

Scripts: `scripts/identity_hot_probe.py`, `scripts/on_neuron_cv_probe.py`,
`scripts/on_cv_sobol_transfer.py`. 10 mini nets (6 for the transfer probe),
offline numpy, antithetic pairs, paired same-sample deltas.

## Identity-skip (skip `maximum` on hot columns, layers 1–29)

- **Bias is threshold-controlled and a safe region exists**: paired final-mean
  delta² vs exact baseline — t=3.0: 5.1e-8 (FATAL, matches the 07-09 fold-bias
  probe); t=3.5: 3.5e-9 (marginal-fatal); t=4.0: 4.4e-10 mean / 1.7e-9 max
  (borderline vs the ~2e-9 1% bar); **t=4.5: 5.6e-11 / 2.6e-10 max (safe)**;
  t=5.0: 3.0e-12.
- **Economics kill it**: |O(t)| per net over layers 1–29 = 1108 (t=3), 718
  (t=4), 578 (t=4.5). `maximum` bills 1/elem vs the matmul's ~2k/elem, so the
  skip saves only **0.02–0.03% of billed FLOPs** (~0.03–0.04G of 142G at
  N=61,440) even with free slab ordering. ~1e-11-class adjusted move. HYGIENE
  at best — fold into a future change touching the same code, never a
  submission.
- **Weight-space composition (cross-depth fold) stays dead under 0.9.x**:
  max per-layer |O| at t=4 is ~57 (t=3: ~73), crossover needs > ~108
  (k_active/2 at mean k_active 217). Re-confirms the 07-09 structural
  falsification with current pricing and current alpha census.
- **Hot mean-substitution (demote-analog on the hot side) is catastrophic**:
  paired delta² 4.7e-3 — deleting the variance that on-columns transmit
  downstream is the anchor-family failure. Dead on mechanism, as predicted.
- Alpha census (mean count per layer above t): grows monotonically with depth,
  ~0 before layer 3, ~40–73 at layers 20–31 (t=3), ~22–59 (t=4). Identity
  structure is a deep-layer phenomenon.

## On-neuron control variate (the genuinely new question)

- **The prize was real**: final pair-variance share = **80.1% on-neurons /
  19.9% kink** (10 nets, variance of pair-averaged final estimates,
  _ON_THRESH=3 classification). The 07-08 cv_rho_gate kill (rho²=0.06)
  measured KINK targets only; the on-target (linear functional
  w_j·mean(x_pen)) was never CV-tested.
- Antithetic parity: pair-averaging relu(u·a) gives exactly |u·a|/2, so
  even-part surrogates with exact known means exist: |u·dir| (folded
  Gaussian, E=√(2/π)·‖dir‖) and (u·dir)² along expected-gate directions —
  this is where sample_data_viz §22's low-rank finding (top-2 dirs = 81% of
  MEAN energy) re-enters.
- **Gaussian upper bound: rho² ≈ 0.05** — per-neuron folded direction
  |u·(M w_j)| only 0.011; shared top-8 SVD features (|u·U_i|, (u·U_i)²)
  0.0495; combined 0.0505. Implied raw-variance cut **4.05% upper bound**.
  The on-target hits the SAME rock as the kink target: the even-part
  fluctuation of deep activations is high-dim/rough and barely loads on the
  low-rank mean subspace. Fourth independent confirmation of the
  variance-reduction-tapped mechanism (after linear-CV 0.06, MLMC r_V=1.0,
  stratification instability).
- **Sobol-transfer arm — FALSIFIED for deployment** (6 nets × 24 scrambled
  realizations, N=4,096 pairs, variance-weighted on-total, beta fit on pooled
  Gaussian pairs = deployable pilot-fit analogue, exact analytic E[F]):
  corrected/uncorrected variance ratio **Sobol 0.992 ± 0.029 (null)** vs
  **Gaussian 0.932** (positive control, ~1.7σ, matches the 4–5% bound) vs
  shuffled-feature null 1.078 (mismatched-beta inflation, calibrates the
  noise). Mechanism confirmed directly: Sobol integrates the feature means
  **5–7× better than MC** (per-net feature-mean variance shrink 0.14–0.21),
  so the fluctuation handle the CV corrects against is already integrated
  away by the block. Same rock as the stratified-Sobol kill of 2026-07-07.

## Implementation follow-up (same day): the 0.03% saving is UNREALIZABLE

Attempted to ship the identity-skip on the 318873 surface. Billing
micro-bench (`scripts/relu_skip_microbench.py`, flopscope 0.9.1 = grader pricing,
N=61,440, n=217, kh=20, per ReLU site):

| implementation                                | billed     | vs plain max |
|-----------------------------------------------|-----------:|-------------:|
| plain `maximum(pre)` (baseline)               | 13,332,480 |            — |
| slab-max + column `concatenate`               | 25,436,160 |         +91% |
| full-max + `put_along_axis` raw hot tail      | 14,561,360 |  +9% (+kh·N) |
| slab-max + scatter left back into `pre`       | 24,208,148 |         +82% |
| zeros + two column scatters                   | 25,437,028 |         +91% |
| slab-max only, output left in two pieces      | 12,103,680 |  −9% (−kh·N) |

Slices and `zeros` are free; `maximum`/`concatenate`/`put_along_axis` all
bill ≥1/elem of what they touch. Flopscope arrays are IMMUTABLE (no
`__setitem__`, no `out=`), so exempting a column slab from the ReLU forces a
reassembly that costs ≥ the kh·N it saves. The two-piece variant is the only
saver, but consuming split activations at the next layer needs a full
(N, n_next) partial-product add (~13.3M ≫ kh·N ≈ 1.2M) — the same add that
kills cross-depth folding. The Q4 economics above priced the `maximum` skip
as if the slab split were free; under 0.9.x immutability it never is.

## Verdicts

- Identity-skip in earlier layers: statistically SAFE at alpha>4.5 but
  **UNIMPLEMENTABLE at a billed saving under flopscope 0.9.x** — every
  faithful implementation is net-NEGATIVE (cheapest: +kh·N ≈ +0.02–0.03%
  billed, i.e. the hoped saving with the sign flipped, plus the 5.6e-11
  bias). Strictly dominated by plain `maximum`. Not hygiene — DEAD as a
  standalone change AND as a fold-in; only revivable if flopscope ever adds
  in-place/masked ops or stops billing concat.
- Composition/fold across depth: still structurally dead (|O| max ~57 vs
  crossover ~108).
- Hot mean-substitution: dead (4.7e-3 delta²).
- **On-neuron CV: DEAD for deployment.** Gaussian-only gain (~5–7%) does not
  survive the Sobol block (0.8% ± 2.9%, null). With the on-target now
  measured, the linear/expected-gate variance-reduction family is closed for
  BOTH final-layer variance components (kink 19.9% + on 80.1%).
