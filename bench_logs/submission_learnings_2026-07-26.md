# Submission Learnings - 2026-07-26

## Packed-prefix path replaces the pattern carve (`_PREFIX_PACK`)

Env: flopscope 0.9.1+np2.2.6, whestbench 0.13.0, budget 272G, mini split @v1-phase1.

### What changed

`estimator.py` gains `_plan_prefix_pack` + `_prefix_pack_relu_matmul` behind `_PREFIX_PACK`
(now **on**; `_cold_slice_relu_matmul2` / `_plan_cold_slices` stay intact as the fallback,
so reverting is one flag).

Instead of splitting the cold set into slabs and skipping the slabs a row misses, rows are
sorted by their **NNZ over one wide prefix** and each bucket gathers only its live columns.
The carve survives as the `k=0` bucket. Ladder `(0, 2, 6)`: bucket g keeps `ladder[g]`
columns, rows above 6 fall back to a dense prefix matmul.

Exactness: a bucket's rows have `nnz <= k`, so `argpartition` on the support mask returns a
superset of that row's nonzeros — the extra picks are zeros. Same trick as `_packed_matmul`
in `examples/34_antithetic_pilot.py:366-369`.

### Why the leftovers were worth attacking

Profiling the carve's own output found the cold **corrections** are 5–11% dense and
multiplied in full — at layer 15 the "warm" group is 18,300 rows x 27 columns carrying a
mean of 1.40 nonzeros. Merging that with the low-fire band past the cold set gives one block
that is 4.5% dense over ~50–70 columns, far below the ~w/3 gather crossover.

Measured carve value for reference: `_COLD_SLICE=False` vs `True` on net 0 is
151.47G -> 140.55G (**-10.92G**), predictions identical to 6.9e-6.

### Phase 0: ladder sweep (net 0, 27 sliced layers, modelled)

Every non-empty bucket costs one host sync; today's 4-group carve costs 3/layer = 81/net.

| ladder | saved | % billed | syncs/net |
|---|---|---|---|
| `(0,2,6)` | 6.04G | 4.30% | **81 — same as today** |
| `(0,2,4,8)` | 7.20G | 5.12% | 107 |
| `(0,1,2,4,8,16,32)` | 7.65G | 5.44% | 137 |

Picked `(0,2,6)`: 79% of the maximum saving at zero net sync cost. The +26 syncs for the
next 1.16G is a losing trade at 318873's measured rate (~28 syncs ~ 0.1s ~ 10G).

### LESSON: size the prefix by expected NNZ, not by a fire-rate cut

First implementation used `_PREFIX_FIRE_THRESH = 0.30` and **regressed +1.49% FLOPs**. A
fixed fire-rate cut ignores the quantity the ladder actually has to cover: at layer 4 it
picked s=98 where mean nnz is 13 and **97% of rows** landed above the ladder, every one
falling back to a dense prefix matmul.

Fix: walk the sorted fire rates, take the cumulative sum (= mean nonzeros a row carries over
that prefix), and stop at `_PREFIX_NNZ_TARGET = 2.5`. Same single sync, adaptive per layer.
Sweep of target x ladder at equal sync count put the optimum at target 2.5 / `(0,2,6)`
(4.78G, 3.40% of billed; mean s=52, 2.5% of rows in the dense tail).

Note the oracle-vs-realizable gap: choosing s per layer with hindsight gives 6.04G, the
pilot-driven rule gives 4.78G. Quote the realizable number.

### Phase 3: FLOP-only A/B, 10 mini nets (0,10,...,90), seed 0

- FLOPs   141.73G -> 135.51G (**-4.39%**)
- raw MSE 3.790327e-07 -> 3.788100e-07 (-0.059%, fp noise)
- adjusted 1.953974e-07 -> 1.868930e-07 (**-4.35%**)
- max |final-row delta| 2.19e-05, against an RMS error scale of 6.16e-04 (3.6% of it)
- Improved on all 10 nets individually (-3.43% to -4.98% FLOPs)

### Phase 4: `whest run --runner local --seed 42 --n-mlps 10` (lambda*R INCLUDED)

| field | carve | packed | delta |
|---|---|---|---|
| adjusted_final_layer_score | 2.47994e-07 | 2.38952e-07 | **-3.65%** |
| final_layer_mse | 4.54145e-07 | 4.54614e-07 | +0.10% |
| mean_score_multiplier | 0.556008 | 0.533913 | -3.97% |
| mean_effective_compute | 1.51234e+11 | 1.45224e+11 | -3.97% |
| worst_mlp_adjusted | 8.517e-07 | 8.24658e-07 | -3.18% |
| n_failed_mlps | 0 | 0 | — |

**Residual did not bite:** per-net residual 0.057s -> 0.059s (+2ms ~ +0.2G) against a 6.27G
FLOP saving. This is the whole reason the 3-bucket ladder was chosen — sync count held at
today's level. Per-net wall rose 3.78s -> 4.32s, but that is flopscope simulation overhead,
not the billed residual.

Subprocess parity (`--runner subprocess --seed 42 --n-mlps 3`): 0 failures, all-zero
`failure_breakdown`, no budget exhaustion.

### Decision

**Adopt** — `_PREFIX_PACK = True`. -3.65% adjusted clears the ~1% grader noise floor
(319031: repeat variance on identical bytes ~0.5%) by a wide margin, and the direction is
consistent on every net.

### GRADED — NEW BEST 1.5951e-07

Submitted as `submission-algo46-prefixpack.tar.gz` (staged in
`submissions/algo46-prefixpack/`), description "algo46 packed-prefix: NNZ-bucketed prefix
replaces pattern carve (ladder 0/2/6, nnz target 2.5)".

- **Graded 1.5951465648336642e-07**, previous best 1.6605e-07 (319031) -> **-3.94%**
- Local `whest run` predicted -3.65%; the grader delivered -3.94%. **The prediction
  transferred almost exactly** — first time a local multiplier-led estimate has done so in
  this campaign.
- Submission id not printed by `whest submit --watch` in whestbench 0.13.0; read it from
  the AIcrowd submissions page if it needs citing.

**LESSON — which local gains transfer.** 315844/315892 and 316260 taught that
locally-priced multiplier gains are fragile, and 318756 confirmed it when raw MSE was
traded for FLOPs. This one held because it is **exactness-preserving**: raw MSE moved
+0.10% (fp reassociation only), so the entire gain is a structural FLOP cut with nothing
bet on the hidden suite's error distribution. The distinction worth carrying forward is
not "local vs grader" but **"does the change trade raw MSE?"** — if it does not, local
FLOP measurements transfer.

### PACKAGING TRAP (cost nothing this time, would have burned the slot)

`whest package --estimator submissions/algo46-prefixpack/estimator.py` produced a **9.0 KB**
archive containing only `estimator.py` — it silently dropped `sobol_points.npz`, which
`estimator.py:389` loads from `submission_dir`. **`whest validate-package` passed it
anyway.** Point `--estimator` at the *folder*:

```bash
whest package --estimator submissions/algo46-prefixpack --output ... --yes   # 29.16 MB
tar tzf ...   # estimator.py, requirements.txt, sobol_points.npz, manifest.json
```

`validate-package` checks archive structure, not whether the estimator's data dependencies
are present. Always `tar tzf` and check the byte size against a known-good prior archive
(algo45-carve is 29.16 MB).

Open follow-ups, not pursued:
- 4-bucket ladder `(0,2,4,8)` is worth ~+0.8% more FLOPs for +26 syncs/net — retest only if
  a grader measurement shows syncs are cheaper than 318873 implied.
- The hot block is untouched and stays ~60% dense; the only remaining lever there is the
  rank-8 truncation idea, which is lossy and needs a truncation-bias curve first.

---

## algo47: base-block packing + wider ladder (bundle on top of algo46)

Two extensions to the graded 1.5951e-07 surface, measured separately and together.

### 1. Base/pilot block packing (`_PREFIX_BASE`)

The pilot pass (10,240 samples) was still on the old single-level carve
(`_cold_slice_relu_matmul_restore`). Same packed-prefix code now applies there, with
`restore=True` — the pilot pass MUST keep row order (Sobol-prefix probe rows and the
antithetic pairing), so it pays the inverse-permutation scatter the carve variant already
paid. Cost-neutral on that axis.

Prefix width can't use a fire census there (that pass is what builds one), so it comes from
the analytic alpha: `Phi(alpha)` is the per-column fire rate and its running sum is the mean
nonzeros over the prefix — same `_PREFIX_NNZ_TARGET` stop. Costs ~p ops/layer instead of the
2*n*p a direct measurement off `x` would need. **`flops.stats.norm.cdf(...).astype(float32)`
— the cast is load-bearing** (f64 leak promotes everything downstream to 2x billing).

### 2. Wider ladder, swept on 5 nets

Joint (target x ladder) sweep over nets 0/10/20/30/40, baselined on the shipped
target 2.5 / `(0,2,6)`:

| target | ladder | vs shipped | syncs/net |
|---|---|---|---|
| 3.0 | `(0,1,2,4,8,16)` | -1.66% | 140 |
| 3.0 | `(0,1,2,4,8)` | -1.61% | 140 |
| 3.0 | `(0,2,4,8,16)` | -1.32% | 112 |
| 2.5 | `(0,2,6)` (shipped) | — | 84 |

Consistent across nets (worst -1.29%, best -1.88%). The 6th bucket adds nothing over 5
(-1.66 vs -1.61), so `(0,1,2,4,8)` at target 3.0 — same saving, one fewer group.

### FLOP-only A/B, 10 mini nets seed 0

| arm | dFLOPs | d raw MSE | d adjusted | max abs delta |
|---|---|---|---|---|
| +base | -1.30% | +0.002% | -1.27% | 1.19e-06 |
| +ladder | -1.21% | +0.047% | -1.00% | 1.20e-05 |
| **+both** | **-2.77%** | +0.049% | **-2.53%** | 1.22e-05 |

Additive, slightly super-additive.

### `whest run` local, seed 42, n=10 (lambda*R included), vs the GRADED algo46

| arm | adjusted | vs algo46 | flops | residual |
|---|---|---|---|---|
| algo46 (graded 1.5951e-07) | 2.38952e-07 | — | 139.31G | 0.0591s |
| base only | 2.37089e-07 | -0.78% | 137.48G | 0.0656s |
| **bundle** | **2.34483e-07** | **-1.87%** | 135.31G | 0.0708s |

raw MSE flat (4.5461e-07 -> 4.5469e-07). 0 failures. Subprocess n=3: 2.8639e-07 vs algo46's
2.9321e-07 (-2.33%), 0 failures, all-zero breakdown.

### LESSON: syncs are cheap but not free — and now they are priced

algo46 held syncs at 81/net and residual moved +2ms, which said nothing about marginal sync
cost. This bundle takes syncs to ~140/net and residual went 0.0591s -> 0.0708s (+11.7ms
~ +1.17G). That ate **26% of the 4.0G FLOP saving** — the FLOP-only estimate of -2.53%
became -1.87% under lambda*R.

Working number: **~0.2ms per host sync**, i.e. ~20 MFLOP each. Far below the ~350 MFLOP/sync
that 318873's "28 syncs ~ 0.1s" implied — that figure bundled a scatter and overstated syncs
badly. Use 0.2ms/sync for future ladder/bucket decisions.

Base-only is the sync-cheap half (-0.78% for +6.5ms) but not worth shipping alone; the
bundle is 2.4x better and still clears the ~0.5% noise floor by ~3.7x.

### GRADED — NEW BEST 1.5755e-07 (submission 319266)

`submission-algo47-bundle.tar.gz` (staged in `submissions/algo47-bundle/`), shipped bytes
byte-identical to commit 3181727 and archived verbatim as `examples/319266.py`.

- **Graded 1.5755419799438493e-07**, previous best 1.5951e-07 (algo46) -> **-1.23%**
- Cumulative over the day: 1.6605e-07 (319031) -> 1.5755e-07 = **-5.12%**

### LESSON — transfer rate depends on whether the gain is FLOP-pure or residual-mediated

Two data points from today, same surface, same day, same grader:

| submission | local predicted | grader delivered | transfer |
|---|---|---|---|
| algo46 (syncs held flat, pure FLOP cut) | -3.65% | -3.94% | **108%** |
| algo47 (FLOP cut MINUS a residual cost) | -1.87% | -1.23% | **66%** |

algo46 changed FLOPs and left residual alone (+2ms). algo47's net local gain was
`4.0G FLOPs saved - 1.17G residual cost`, i.e. a quarter of it was a lambda*R bet on local
machine speed — and that quarter is the part that under-delivered. Consistent with the
standing 315844/315892 and 316260 lesson, now with a magnitude attached.

Refinement to the exactness rule: **raw-MSE-flat is necessary but not sufficient for full
transfer.** Also ask what fraction of the local gain is `lambda*R` rather than FLOPs, and
discount that fraction. Both submissions were raw-MSE-flat; only the FLOP-pure one landed
at its predicted size.

n=2 — do not over-fit this, but prefer sync-neutral designs when the FLOP saving is
comparable.

---

## Submission Log — N=81,920 artifact test (algo56/algo57, submitted late 07-26)

Context: user-directed test of a larger sampling budget. Two artifacts at 40,960 stored
half-rows (81,920 effective antithetic samples), both on the graded 319266 bytes with only
`_TOTAL_SAMPLES = 81920` changed. Validation-only gate by design (no local A/B): `whest
validate` pass, subprocess n=2 seed-42 clean, ~177G billed/net (65% of the 272G budget),
tarballs 38.87 MB with all four files. Submission ids not printed by whestbench 0.13.0 —
read from the AIcrowd page if they need citing.

### Seed-recovery attempt for the protected artifact: FALSIFIED (scipy 1.15.3)

Candidates 2, int.from_bytes(b"sobol") big/little (495766970220 / 465425703539), ASCII-concat
11511198111108, letter-position 191521512, and "2"+ASCII 211511198111108, each under both
`random(32768)` and `random_base2(15)`: none reproduce the shipped draw — values diverge from
row 0, so this is not a trim/shape artifact. The shipped file is `(30,720, 256)` (exactly the
consumed rows; the writeup's "(32,768, 256)" was wrong and is fixed in a6acc6d). The protected
draw still cannot be extended as a true continuation of its own sequence.

### Submission (id TBD) — algo56-n81920-concat

- Result: **tied** (nominal −0.61%, inside the ~0.5–1% repeat-noise floor from 319031)
- Change: N=81,920; artifact = shipped 30,720-row protected draw + 10,240-row extension
  (seed 495766970220, first 10,240 of a 2^14 draw); prefix byte-identical to shipped
- Local expectation: none claimed (validation-only); smoke on nets 0/10 improved on both
- Leaderboard: **1.5659658271033176e-07** vs 319266's 1.5755e-07, same-day pair → −0.61%
- Decision: do not promote on this evidence; a verbatim re-grade of either side would be
  needed to separate them, and the expected value doesn't justify the slot
- Lesson: at 0.9.x pricing the +33% FLOP cost of the N-raise almost exactly cancels the
  added-rows variance gain even with the protected prefix held fixed — fixed-61,440 remains
  the efficient point, now measured cleanly (prefix-controlled) rather than lottery-confounded.

### Submission (id TBD) — algo57-n81920-seed2

- Result: **regressed +21.8%**
- Change: N=81,920; artifact = fresh full seed-2 draw (first 40,960 of a 2^16 draw)
- Local expectation: net 0 looked spectacular (1.53e-7 vs baseline 9.20e-7) — pure lottery
- Leaderboard: **1.918874139765107e-07**
- Decision: do not retry fresh-draw artifacts; third confirmation of the 317412 rule
- Lesson: hidden-realization variance (~30%) dominates any local draw signal; a fresh
  scramble is a lottery ticket regardless of N.

### What not to try again

- Fresh Sobol scrambles as "maybe better" artifacts (three leaderboard confirmations now).
- N-raises financed by billed FLOPs on this cost model — the multiplier eats the variance
  gain; only a FLOP-neutral way to add samples would reopen this axis.
- Seed archaeology on the shipped artifact under scipy 1.15.3 — the candidate space of
  memorable seeds is now well covered; treat the artifact as version-locked data.

### Batch: three more independent 82k draws (algo58/59/60, seeds 3/5/7)

User-directed lottery sampling on the same N=81,920 estimator bytes; validation-only gate
(all passed: validate, subprocess n=1 clean, 38.87 MB tarballs, 4 files).

| artifact | graded | vs 319266 (1.5755e-07) |
|---|---|---|
| seed 2 (algo57, earlier) | 1.9189e-07 | +21.8% |
| seed 3 (algo58) | 2.0090e-07 | +27.5% |
| seed 5 (algo59) | 1.5724e-07 | −0.2% (tie) |
| seed 7 (algo60) | 1.9602e-07 | +24.4% |
| concat/protected prefix (algo56) | 1.5660e-07 | −0.61% (tie) |

- Fresh-draw spread at n=4: min 1.572, max 2.009 (max/min 1.28); median draw ~+23% worse
  than the incumbent. **Quantifies the artifact lottery at ~25-30% spread** — first clean
  multi-draw measurement at fixed bytes.
- The incumbent artifact is confirmed a strong draw: only 1 of 4 fresh scrambles ties it,
  none beat it beyond noise.
- The two ties (seed 5 fresh, concat) both sit within noise of 319266 despite +33% billed
  FLOPs — consistent with the N-raise economics being a wash: a good 82k draw ≈ the good
  61,440 incumbent.
- Observation, not a rule change: the 1-net local smoke happened to rank seed 5 best and
  it was best on hidden too (n=4, one local net) — insufficient against the standing
  317412 evidence; do not start trusting local draw screening on this.
- Axis disposition: five slots spent on 82k artifacts, zero net gain. CLOSED unless a
  FLOP-neutral sample-addition mechanism appears.

### Submission (id TBD) — algo62-n65536-seed5, and the seed-5 N-curve

- Result: **regressed +9.5%** vs standing best (graded **1.7259010523185348e-07**)
- Change: seed-5 draw, `_TOTAL_SAMPLES = 65536` (2^16 total; first 32,768 artifact rows)
- Seed-5 N-curve now: 65,536 -> 1.7259e-7; 81,920 -> 1.5724e-7 (−8.9% from more samples)
- Lesson: the seed-5 draw needs its full 82k budget to tie the incumbent's 61,440 — the
  protected artifact is strong per-sample, not just lucky in aggregate. N-cuts on a fixed
  draw lose more to variance than the multiplier returns (consistent with 318756).

### Local test — sobol_burley (hash-based Owen scrambling), no slot spent

Artifact from cessen/sobol_burley v0.5 (Rust, contained toolchain in scratchpad): seed 0,
40,960 x 256, u clipped to [2^-25, 1-2^-25] (crate emitted no exact 0/1), norm.ppf, f32 —
`sobol_points_81920_burley0.npz`, staged as algo61-n81920-burley0 (validates, 0 failures).

10-net local A/B at N=81,920, seed 42, identical nets:

| artifact | adjusted | raw MSE |
|---|---|---|
| scipy seed-5 | 2.31e-07 | 2.69e-07 |
| burley seed-0 | 3.02e-07 | 3.46e-07 (+29%) |

- +29% is INSIDE the ~28% single-realization spread measured on hidden today, and seed 5
  is a locally-lucky scipy draw — this does NOT establish the Burley construction is worse,
  only that this one Burley realization is mediocre on these 10 nets.
- A structural verdict needs a multi-seed local mean (e.g. 5 Burley seeds vs 5 scipy seeds,
  10 nets each) — all local, no slots. Not run yet.

### Submission (id TBD) — algo61-n81920-burley0

- Result: **regressed +28.5%** (graded **2.0250067813837198e-07**)
- Change: sobol_burley (hash-based Owen scrambling) seed-0 artifact at N=81,920
- Leaderboard evidence: lands at the bottom of the fresh-draw band (1.572–2.009 scipy);
  one realization — says nothing structural about the Burley construction
- Note: local 10-net gap vs seed-5 (+29%) matched the hidden gap (+29%) almost exactly;
  with seed-5 best-local/best-hidden, that is now 2-for-2 rank agreement today — still far
  from overturning the 317412 no-local-predictive-value rule, but worth tracking
- 82k ticket tally (6): concat 1.566 | seed5 1.572 | seed2 1.919 | seed7 1.960 |
  seed3 2.009 | burley0 2.025. Standing best remains 319266 @ 1.5755e-7 (61,440, protected
  artifact). Axis remains CLOSED for score; Burley structural question needs the multi-seed
  local mean, not slots.

### Submission (id TBD) — algo63-n84992-seed5: NOMINAL NEW BEST 1.5514e-07

- Result: **improved** — graded **1.5514080482970139e-07**, −1.53% vs 319266 (1.5755e-7),
  −1.34% vs its prefix-identical seed-5 parent at 81,920 (1.5724e-7)
- Change: seed-5 draw extended to 42,496 rows (verified true continuation; first 40,960
  rows byte-identical to the 82k artifact), `_TOTAL_SAMPLES = 84992`
- Caveats before promoting: the −1.34% prefix-controlled delta is ~2.7x the measured
  repeat-noise (0.5%) but CONTRADICTS the economics prediction (+3.75% samples should be
  a slight net loss at the measured FLOP/variance exchange rate) — consistent with the
  appended 1,536 halves being a lucky increment, i.e. the within-artifact increment
  lottery, not a reproducible N-curve effect
- Seed-5 N-curve: 65,536 -> 1.7259 | 81,920 -> 1.5724 | 84,992 -> 1.5514
- Fully reproducible construction: scipy 1.15.3, qmc.Sobol(d=256, scramble=True, seed=5),
  random_base2(16), first 42,496 rows, norm.ppf, f32
- Decision hook: if promoted, a verbatim re-grade would separate luck-of-window from a
  real edge (~0.5% repeat noise vs 1.5% margin)
