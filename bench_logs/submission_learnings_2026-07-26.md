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
