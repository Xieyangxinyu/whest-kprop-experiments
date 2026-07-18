# Submission Learnings — 2026-07-14

Overnight Algorithm 25 campaign (closed-form allocation from
`allocation_optimality.ipynb` §13, rebuilt on the algo21/315892 surface,
extended fp16 Sobol artifact). Four submissions between 00:45 and 02:01
local; results pulled via the AIcrowd API the following morning.

Baselines for comparison:
- 315892 (algo21, same base surface): adjusted 1.337748e-7, raw
  3.724968e-7, multiplier 0.35913.
- 315998 (algo24 c64 packing): adjusted 1.006111e-7 — graded frontier on
  paper, but retired 2026-07-14 (organizers correcting the flopscope
  complex-cost bug), so 315892 is the honest reference.

## Submission Log

### Submission 316255 - Algorithm 25 v1 (closed-form alloc, cap 122,880)

- Result: FAILED — "Evaluation error" (setup-stage), score null
- Change: algo21 surface + closed-form N_i = sqrt(b̂_i·F0/(a·F1)) from the
  base-block antithetic pair-mean variance, clip [30720, 122880], extended
  fp16 Sobol artifact (27.7 MB), budget guard 0.8×budget.
- Local expectation: §13 grader-priced −13.6%/−1.8σ vs production rule
  (n=10 full-split nets); whest validate PASS, local run clean.
- Leaderboard/public evidence: grader died in setup(). Root cause found in
  session: `points.astype("float32", copy=False)` — the evaluator's
  RemoteArray.astype REJECTS the `copy=` kwarg. Local validate/run never
  exercises RemoteArray, so this passed every local gate.
- Decision: fixed in v2 (bare `astype("float32")`).
- Lesson: the grader's RemoteArray is not a numpy array — keyword-argument
  surface differs. Any new-in-setup array op needs the most conservative
  call signature; local validation cannot catch RemoteArray API mismatches.

### Submission 316258 - Algorithm 25 v2 (astype fix + 42s wall clamp)

- Result: FAILED — "Evaluation could not complete; please retry", score null
- Change: v1 + bare astype, combined-compute guard (flops + residual ≤
  0.88×budget), wall clamp projecting continuation time from the measured
  base block with _WALL_TARGET=42s (vs the 60s per-predict limit).
- Local expectation: same §13 evidence; wall clamp believed sufficient
  headroom.
- Leaderboard/public evidence: failed AFTER setup (past 316255's death
  point), ~10+ min into grading. Message suggests timeout/infra, not a
  contract error.
- Decision: retried verbatim as 316259 to test transience.
- Lesson: see 316259.

### Submission 316259 - Algorithm 25 v2 retry (identical artifact)

- Result: FAILED — identical "Evaluation could not complete; please retry"
- Change: none (byte-identical retry of 316258).
- Local expectation: if 316258 was evaluator infrastructure flake, this
  passes.
- Leaderboard/public evidence: same failure ⇒ NOT transient. The
  cap-122,880 / 42s-wall config repeatably kills the evaluation. Most
  consistent story: grader hardware is slower than local, so 42s-target
  plans overrun the per-predict (or whole-run) wall limit.
- Decision: do not retry the aggressive config. Do NOT burn further
  submissions probing which limit it is.
- Lesson: "please retry" from the grader is not necessarily transient —
  one verbatim retry is the maximum spend on that hypothesis. Wall-time
  headroom calibrated locally does not transfer; the grader box is
  materially slower.

### Submission 316260 - Algorithm 25 v3 conservative (cap 92,160, 30s wall)

- Result: REGRESSED — graded, adjusted 1.58697e-7 (+18.6% vs 315892), raw
  6.02340e-7 (+61.7% vs 315892), multiplier 0.26347 (−26.6% compute)
- Change: v2 with _MAX_SAMPLES=92160 and _WALL_TARGET=30.0s.
- Local expectation: §13 said closed form + high cap wins grader-priced
  −13.6%/−1.8σ; conservative clamps were expected to keep most of that.
- Leaderboard/public evidence: the score decomposition tells the story:
  compute DROPPED 26.6% while raw MSE ROSE 61.7% — realized per-net N
  landed far BELOW the production rule's, not above it. On slower grader
  hardware the 30s wall clamp (projected from measured base-block elapsed)
  bound nearly everywhere and slashed continuation samples; the closed
  form never got to allocate its 62k–122.9k picks. Net adjusted −18.6%
  loss vs just shipping the rule.
- Decision: revert — Algorithm 25 is falsified as a shippable artifact.
  Ship surface stays algo21/315892-class. Do not retry closed-form
  allocation, cap raises, or any wall-time-projected N clamping.
- Refinement (post-hoc code+score analysis): the raw hit is BIGGER than a
  uniform N cut explains. Working the multiplier back through measured
  costs (f1≈1.745e6 FLOPs/sample, residual ~1.9e10) gives realized mean
  N ≈ 32–35k ≈ base-block-only; a uniform fleet cut to that level
  predicts raw ~4.5–4.9e-7, observed 6.02e-7. The excess is the clamp's
  SHAPE: `elapsed` covers _initial_structure + the refine=True base block
  (estimator.py:698-711), so big/slow nets — exactly the high-variance
  nets the rule gives 61k samples — take longest through the base block
  and get clamped hardest. Per-net N becomes ANTI-correlated with need
  (worst-possible allocation shape), further jittered by grader timing
  noise. The clamp is also doubly conservative by construction: it prices
  the continuation at the base block's per-sample rate, but the base
  block includes one-time refine/probe work the continuation never
  repeats.
- Lesson: two independent kills in one submission: (1) the §13 win hinged
  on fixed-residual pricing AND on the grader running our wall-time
  economics — neither held; (2) any N-selection that depends on measured
  grader wall-time inherits the grader/local speed gap as a first-order
  allocation error (60% raw regression), far worse than the ≤5.3% oracle
  gain allocation could ever buy — and because slow nets are the
  sample-hungry nets, a wall clamp doesn't just lower N, it INVERTS the
  allocation.

## What this closes

- **Allocation is now leaderboard-falsified, not just offline-tapped.**
  The §13 closed form was the ONE open suggestive lead
  (allocation-ceiling-result memory); it lost −18.6% on the public set.
  The 5.3% honest-oracle ceiling was never reachable at grader prices.
- **Wall-time-adaptive sample counts are a dead pattern.** The grader is
  slower than local; anything that converts measured elapsed into N
  transfers the speed gap into an accuracy loss.
- **Three failure modes cost 3 submissions before a graded result:**
  RemoteArray API mismatch (setup), repeatable evaluator timeout
  (aggressive config), then the regression. Pre-submit gates cover none of
  them — a NEW-mechanism artifact should ship its most conservative
  configuration FIRST, then escalate, not the reverse.
- Frontier status: 315998 (c64 packing) is still the number on the board
  but does not survive the announced flopscope correction; the honest
  working frontier is 315892 (algo21). Repo-root estimator.py still
  carries the _COMPLEX_PACK surface and should be reverted before the
  correction lands.

## Do-not-retry list additions

- Closed-form / pilot-b̂ / any per-net sample allocation (now grader-falsified).
- Cap > 61,440 in any form (122,880 config repeatably fails evaluation;
  92,160 graded but the samples never materialized under the wall clamp).
- Wall-time-projected N clamping (grader/local speed gap becomes raw MSE).
- `copy=` (or any non-essential kwarg) on array methods touched in setup().

## Afternoon campaign: Algorithm 26 (adaptive fire threshold)

### Submissions 316325 + 316326 - algo26 v1 (BROKEN PACKAGING, x2)

- Result: FAILED — "Setup error in your submission", score null (16:15Z,
  16:21Z; the 16:21 one was a re-package that repeated the same mistake)
- Change: algo21 surface + per-call adaptive fire threshold
  (bench_logs/algo26_adaptive_fire_thresh_2026-07-14.md).
- Root cause (found in-session, deterministic): the tarballs
  (`submission-algo26-adaptive-fire.tar.gz` 8.6 KB, 11:13 local;
  `submission-20260714-162134.tar.gz` 8.6 KB, 11:21 local) contain ONLY
  `estimator.py` + manifest — no `sobol_points.npz`. They were built with
  `whest package --estimator estimator.py` (a FILE path). Per
  `whestbench/packaging.py::resolve_submission`, a file argument means
  SINGLE-FILE MODE: only that file ships, siblings are ignored. Folder mode
  (`--estimator <dir>`) is what bundles the artifact. `setup()` then does
  `fnp.load(submission_dir / "sobol_points.npz")` → file not found →
  setup error. Local `whest package`/`validate` cannot catch it: the
  packager only imports the module; setup() with the packaged folder is
  never run.
- Decision: always package the submission FOLDER. Verify with
  `tar -tzf` (5 entries, ~29 MB) or `whest validate-package` before
  submitting; a 4-file estimator+npz package must never be ~8 KB.
- Lesson: `--estimator estimator.py` and `--estimator .` build DIFFERENT
  archives; the CLI's own "Next: whest submit --estimator estimator.py"
  hint is a footgun for artifact-bearing submissions.

### Submission 316327 - algo26 v1 (complete package)

- Result: FAILED — "Evaluation could not complete; please retry", score
  null (16:24:52Z; tarball `submission-20260714-162434.tar.gz`, complete:
  estimator + fp32 sobol npz sha c6b4a836 = same artifact as 315892).
- Change vs graded 315892: ONLY the adaptive-fire rule (same artifact,
  same caps 30720/61440, no allocation changes).
- Local evidence: validate PASS (including on the extracted tarball),
  seed-42 local/subprocess parity exact, flops −0.996%, local wall DOWN.
- Failure class: same message as 316258/316259, which was proven
  non-transient (repeatable evaluator death, likely run-level timeout).
  Prime suspect is the rule's REMOTE op traffic: ~45 extra small backend
  ops + one synchronous scalar pull (`int(fnp.argmin(total))`) per split
  call ≈ 60 sync round-trips/net × 100 nets on the grader's RemoteArray
  pipeline — locally free, remotely latency-bound. This was pre-flagged as
  the residual risk in the algo26 bench log (315898 lesson). Secondary
  suspect: a RemoteArray slow-path/unsupported op among the new calls
  (`searchsorted`, `take`, 2-D `norm.cdf`) — but an unsupported op should
  present as "Evaluation error", not "could not complete".
- User report: an MSE is visible on the submission page despite the error
  (API shows score null) — consistent with grading progressing partway
  then dying, i.e. mid-run timeout, NOT a contract/shape error.
- Decision: investigate before any resubmission. If algo26 is retried at
  all, the scalar pull per split call must go (batch the 60 argmin pulls,
  or precompute thresholds per net from the layer-1 fire census in one
  shot); better: test remote-op-count sensitivity with a cheap probe
  submission budgeted separately. One verbatim retry maximum (316258/9
  rule) — and only if we decide the op-traffic story is wrong.
- Lesson: "locally wall-negative" does not bound grader wall when the
  change multiplies BACKEND OP COUNT / sync pulls; the remote pipeline
  charges per round-trip, not per FLOP.
- Base-rate caveat (from the same-day API scan, 316200-316330): ~14 of
  ~90 non-ours submissions today failed with "could not complete" /
  "Error while scoring" — the evaluator is visibly flaky today, so ONE
  failure of a conservative-config package is weak evidence against the
  artifact. A single verbatim retry (the 316258/9 protocol cap) is
  justified before touching the estimator.

### Submission 316368 - Algorithm 27 (gather-restore, copy-traffic cut)

- Result: still grading (submitted 20:04Z)
- Change: 315892 surface + packed row-order restore via ONE global
  inverse-permutation gather (fnp.take + argsort) instead of per-chunk
  empty_like + put_along_axis scatter. Copy-class traffic 8.09 -> 5.42
  GB/net (residual_copy_census_2026-07-14.md). Zero-chunk branch hardened
  to reuse argsort (no fnp.arange - no RemoteArray surface untested
  locally). Package: folder mode, 5 entries, 29,154,627 B, artifact sha
  c6b4a836 (= 315892's), validate-package + extracted-validate PASS.
- Local expectation: raw BIT-IDENTICAL to 315892 (array_equal on mini
  nets; seed-42 raw 4.2685332838724815e-07 exact across local/subprocess);
  flops +0.028% (added global argsort).
- Pre-registered read of the result: multiplier vs 315892's 0.35913
  directly prices the scatter/alloc class - expect adjusted -4..-6.5% if
  scatters price like concats (~1.5-2.4 FLOP-eq/B), ~flat (+0.03%) if
  only concats are priced. Either outcome gates the Strassen row-blocking
  follow-up.
- Result (20:35Z): FAILED — "Evaluation could not complete; please retry",
  score null. Same infra-class message as 316327 and as ~10 other
  participants' failures today. Artifact evidence says this is unlikely to
  be the estimator: the change REMOVES ~230 backend ops/net, adds no sync
  pulls, uses only grader-proven op vocabulary, and the package mirrors
  graded 315892 byte-for-byte except estimator.py.
- INVESTIGATION (20:40-21:00Z): cause identified as POST-SCORING INFRA
  FAILURE, not the artifact. Evidence: (1) user reports the submission
  page shows an adjusted score — "a new honest best without packing" —
  i.e. the evaluation ran to completion and scored; the job died between
  scoring and result-commit (API record has score:null; the UI number is
  rendered from evaluation logs; no API endpoint exposes it — probed
  ?meta=true, api.aicrowd.com, /grades). (2) Window scan 316330-316400:
  the identical "could not complete" hit ANOTHER participant (316367) 5
  minutes before ours; ~10 failures across other participants through the
  evening; 316369 GRADED 10 seconds after our failure — per-job worker
  roulette on a degraded pool, ongoing since ~09:38Z. (3) The scatter-
  pricing hypothesis is therefore likely CONFIRMED by the UI-visible
  score (below 1.3377e-7, honest surface) — pending a committed grade.
- UI panel numbers (user-read, grade never committed): SCORED 50/50,
  PUBLIC MEAN 1.36e-7, IQR 8.87e-8 - 1.81e-7, best 4.95e-8
  (donald-henderson), worst 3.05e-7 (erin-maddox). vs 315892's 1.3377e-7:
  +1.7% ~ FLAT (identical-resubmit multiplier noise today is +/-1%).
  User confirms: competitive, not a best.
- VERDICT: the pre-registered FLAT outcome. put_along_axis/empty_like
  scatters are in the backend-FREE class - they were never costing us.
  Recalibrated residual map: concat-only census bytes (~5.4 GB/net) at
  ~2.2 FLOP-eq/B reproduce the implied 1.2e10 residual almost exactly.
  Scatter-removal is worthless; the CONCAT class is the whole residual.
- Decision: estimator.py REVERTED to pristine algo21 (the gather variant
  carries +0.028% guaranteed flops for zero grader benefit); variant
  preserved in submissions/algo27-gather-restore/ + tarball. Verbatim
  retry NOT spent - the UI panel already delivered the calibration the
  retry was for; committing a flat grade has no leaderboard value
  (315892 stands).
- Lessons: (1) "Evaluation could not complete" can occur AFTER a
  complete 50/50 scoring pass (commit-phase death) - the submission page
  panel carries the scores even when the API records null; read it
  before spending a retry. (2) The public set is 50 nets with 6x per-net
  adjusted spread. (3) Residual attack surface narrows to CONCATS ONLY:
  Strassen assembly 2.30 GB/net + packed group/chunk concats 2.58 GB/net
  + sample blocks 0.11; the row-blocked pipeline (kills the axis=0
  assembly class) is the remaining live lever, ~-2.5% adjusted if its
  concat bytes price at ~2.2/B.

### Submission 316405 - Algorithm 28 (row-blocked Strassen pipeline)

- Result: GRADED — IMPROVED. Adjusted 1.324541e-7 (315892: 1.337748e-7,
  -0.99%), raw 3.724609e-7 (315892: 3.725000e-7, -0.01% = the designed
  fp-neutrality), multiplier 0.35562 (315892: 0.35913, -0.98%).
  NEW HONEST FRONTIER (packing-free): 316405 > 315892.
- Concat-price recalibration: -2.46 GB/net bought only ~1.0e9 FLOP-eq =
  ~0.39 FLOP-eq/byte — 6x BELOW the 315898-derived 2.2-2.4/B. Reconciled
  reading: 315898's +1.72% was dominated by its many small per-layer
  concats (op-count/dispatch), not bytes; large contiguous copies are
  cheap. Consequences: (1) the remaining Strassen axis=1 concats
  (1.15 GB, 112 ops/net) are worth only ~-0.5% — marginal, column-
  blocking NOT justified; (2) the bulk of the ~1.2e10 residual is NOT
  copy bytes at all — it is per-op/dispatch overhead territory, which
  algo20 showed is hard to move by reorganization. The copy-traffic
  attack is now largely TAPPED after banking this ~1%.
- Change: 315892 surface + (a) row-blocked sample pipeline where the
  antithetic (+half,-half) block boundary IS the Strassen row split
  (_dense_matmul_2blk returns top/bottom unassembled - kills the axis=0
  assembly concat), (b) packed single-global-concat + inverse gather.
  Priced concat traffic 5.40 -> 2.94 GB/net (-46%). estimator_rowblock.py;
  package algo28, folder mode, 5 entries, artifact sha c6b4a836.
- Local gates: fleet raw -0.0063% (16 nets, fp-noise class, NOT
  bit-identical - fire-tie flips + chunk-boundary Strassen eligibility);
  flops +0.026%; validate + validate-package + extracted-validate PASS;
  subprocess seed-42 n=3 clean, raw 4.26914e-7 (+0.009% vs baseline).
- Pre-registered read: adjusted -5.5% (concats ~2.2 FLOP-eq/B) to -3%
  (1.5/B); ~flat only if concat pricing is ALSO weak, which would leave
  315898's +1.72% unexplained. Raw should land ~3.725e-7 unchanged.
- Process note: user instruction landed mid-submit ("after validating,
  ask me before submitting") - arrived after 316405 was accepted; rule
  saved to memory and applies to all future packages.

### Submission 316412 - Algorithm 29 (all-s router on the algo28 surface)

- Result: still grading (submitted 23:23:44Z; user-approved after gates;
  health check: 10/10 latest graded incl. 316405)
- Change: 316405 surface + per-call all-s split router (algo26 cost model
  vectorized over every split point; census capture 97.7% of oracle;
  no fnp.arange - s via cumsum(ones); one sync pull/call).
- Local gates: fleet flops -1.0321% vs algo28 (16 nets, ALL improved;
  census predicted -1.04%); fleet raw -0.0004%; validate +
  validate-package + extracted-validate PASS; subprocess clean 3/3.
- Pre-registered expectation: adjusted ~1.311e-7 (-1.0% vs 316405),
  raw ~3.7246e-7 unchanged. First grader test of router op-traffic
  (risk rated low post-316405).
- Result: FAILED — "Evaluation could not complete", dead at <=15m25s
  (created 23:23:44Z, detected 23:39:09Z) in a HEALTHY window (all 7
  neighbors graded). User: do not resubmit.
- VERDICT — the router wall question is ANSWERED, against the router:
  * ~15-MINUTE HARD JOB DEADLINE established: 316405 (no router)
    completed in <=10m51s; 316412 (+router) and 316368 (degraded-evening
    workers) both died at ~15m20-25s. This retroactively unifies EVERY
    "could not complete" we have hit (316258/9: 2x samples >> deadline;
    316327: router + slow afternoon; 316368: slow evening at commit;
    316412: router in a healthy window).
  * The router class (one sync pull + ~45 small ops PER SPLIT CALL,
    ~3,000 pulls + ~270k ops per 50-net job) adds >=+4.5 min job wall
    (>=~90ms per sync round-trip) — grader-fatal despite -1.03% flops.
    2/2 router artifacts died (316327, 316412); non-router twins passed.
- Decision: algo26/29 per-call routing is CLOSED in its current form.
  Reopening condition: pull-free or <=1-pull-per-net routing (e.g. route
  only the continuation block with ONE batched decision from base-block
  fire vectors — captures ~40% of the -1% win — or device-side selection
  without host shape knowledge, which flopscope's API may not permit).
  The -1.03% flop win is real but unshippable at 60 pulls/net.
- Lesson: local wall gates CANNOT see RemoteArray round-trip latency;
  the pre-submit signal for wall risk is HOST SYNC COUNT x job size vs
  the ~4-min headroom at ~90ms/pull (i.e. budget ~2,600 pulls/job MAX
  above the algo28 baseline).
- User page read: the panel score REGRESSED as well — the sync-pull
  latency is not only a deadline risk, it lands in residual_wall_time_s
  and is PRICED into the multiplier (~60 pulls x ~90ms = ~5s/net =
  ~3e10 FLOP-eq at lambda ~5.8e9/s = multiplier +~0.1). The router is
  dead THREE ways: deadline kill, priced latency swamping the -1.03%
  flop win (net REGRESSION even when scored), and a modest ceiling.
  This is the algo25 lesson recurring in a new form: in-predict elapsed
  time IS the grader's residual meter; anything that blocks Python per
  call - measurement OR routing - pays for itself at lambda.

### Submission 316416 - Algorithm 28 reproducibility check (byte-identical resubmit)

- Result: FAILED — "could not complete" (submitted 00:44:59Z Jul 15,
  healthy queue). Same artifact graded 10m51s earlier (316405) and now
  died — we sit at ~11 min vs the ~15-min deadline, inside the
  worker-speed roulette band. Score reproduction NOT yet confirmed;
  artifact-identity is proven by sha, not by regrade.
- The 15-min limit is NOT documented client-side (limits.py and docs
  have no total-job constant) — it is an empirical grader/platform
  property from 3 timing points. Worth asking organizers.

### Yangxinyu Algorithm-25 (row-dense fallback) re-evaluated on algo28

- Graded prior evidence: 316005 vs 315998 = -0.32% adjusted at identical
  raw on the c64 surface — REAL but 20x below its local -7.6% signal.
  Since its flops were ~neutral, that -0.33% multiplier prices op-count:
  ~470 fewer backend ops/net -> ~7e5 FLOP-eq PER OP. First graded
  dispatch-price point; implies our ~9k wrapped ops/net ~ half the
  1.2e10 residual, and implies the ~113 per-chunk int(fnp.max) host
  pulls/net (~5,700/job) may account for MOST of our ~11-min job wall.
- Port to algo28 (estimator_rowdense.py), 8-net A/B:
  - cleanups (drop bucket-8, Strassen immediate-accumulate, half-only
    sample load): BIT-IDENTICAL 8/8, flops -0.0072%, -160 ops/net.
    ADOPT-candidate (free, ~-0.1%-class via op pricing).
  - MAX_K 3/4 -> 1/2 row-dense fallback: flops +7.19% on the honest fp32
    surface (their surface's complex-cost pricing made dense fallback
    ~free; ours prices it fully). REJECT - net ~+6.8% adjusted.

### Submission 316417 - Algorithm 29b (sync-batch + exact cleanups)

- Result: still grading (submitted 01:32:30Z Jul 15; queue quiet, no
  adverse health signal)
- Change vs 316405: (1) removed the per-chunk int(fnp.max) host sync
  (dead all-zero short-circuit; the limit-0 group handles it exactly);
  (2) ONE vectorized searchsorted + one tolist() pull per chunk replaces
  ~13 int() syncs (conservative per-element fallback if RemoteArray
  lacks tolist); (3) Yangxinyu cleanups: bucket-8 drop, Strassen
  immediate-accumulate, half-only sample load. Their MAX_K 1/2 row-dense
  fallback EXCLUDED (+7.19% flops on honest pricing).
- Gates: BIT-IDENTICAL 8/8 nets vs algo28 (raw byte-equal through the
  packaged artifact: subprocess 4.269141508454292e-07 exact); flops
  -0.0089%; host syncs 950 -> 124/net (-826); validate, validate-package,
  extracted-validate, subprocess flags all clean; artifact sha c6b4a836.
- Pre-registered read: adjusted -0.5..-1% if the 316005 op-price
  (~7e5 FLOP-eq/op) holds for syncs; job runtime should DROP vs 316405's
  10m51s (sync latency savings, magnitude unknown) - the runtime is
  itself a measurement of grader sync latency: (10m51s - t_29b)/826
  syncs/net. Failure would be deadline roulette, not artifact (bit-
  identical math to a graded surface).
- Result: FAILED at ~15m12s — the deadline again (3rd consecutive:
  316416 15m34s, 316417 15m12s, vs 316405's 10m51s success this
  afternoon). Two inferences: (1) per-sync latency on the
  searchsorted/max class must be SMALL (~2-5ms, not 90ms — at 90ms the
  baseline's 47,500 syncs/job could never have finished in 10m51s;
  algo29's death was likely the argmin pipeline-FLUSH semantics, not
  scalar-pull latency), so the -826 syncs/net bought little TIME (its
  ~-0.5-1% PRICING value per 316005 remains plausible but ungraded);
  (2) tonight's workers are running ~40%+ slower than this afternoon's
  — time-of-day load. The deadline case for organizers is now
  overwhelming: 4 deadline deaths incl. a byte-identical pair
  (316405 graded / 316416 failed).
- Decision: STOP submitting tonight (3 consecutive deadline deaths =
  the current window is hostile to our ~11-min runtime class). Retry
  29b in a morning window (316260/315892-era submissions all graded in
  mornings). Score-free runtime cuts are now essentially exhausted:
  the remaining ~6 min/job is einsum gather + matmul backend compute,
  reducible only by paying flops. Escalate the 15-min limit to
  organizers with the identical-artifact pair as the exhibit.

## Runtime model rev 3 + insurance sizing (Jul 15 early, user panel data)

- User-read per-net panel (our submission): WALL 32.0-44.4s/net, mean
  ~37s, flops 6.6e10-1.04e11. Sum ~31 min over 50 nets vs 10m51s job =>
  grader runs BATCHES OF 5 (user-confirmed): job ~ 10 waves x max(wall
  of 5). ~31% of nets exceed rule-N 49152 => ~84% of waves are
  tail-paced. Grader per-WORKER speed ~2.3x slower than local laptop
  (corrects the "grader ~ local" note - that was parallelism masking).
- RULES QUOTE (user): "60-second hard wall-clock cap per MLP - zero-
  prediction fallback (S5.5)". Tail nets at 44.4s cross 60s on a 1.35x
  slow worker => silent per-net score craters are a SECOND slow-night
  risk besides the ~15-min batch limit (still undocumented).
- Insurance sweep (census-exact + fleet A/Bs, all on the 29b surface):
  - dense-gate 3/4->2/3: MIRAGE (-0.1% gather; occupancy sits at
    k<=0.63s). Gate knob dead at any useful size.
  - full trim (anchor 40960 + cap 49152): flops -14.7%, local wall
    -16.4%, fleet raw +22.2% (32 nets) => projected adjusted ~+4.2%.
    DOMINATED - drop.
  - cap-only 49152 (= 29c): affected nets 10/32; on them raw +17.4%,
    flops -16.3% => per-net adjusted product ~0.98 (score ~NEUTRAL,
    N-flatness measured); fleet ~ -0.6%..+0.5%. Tail wall 44->~36s
    (slow-worker ~50s < 60 cap), batch ~12.5-13 min on slow workers.
- DECISION QUEUE: 29c = the insurance submission (score-neutral,
  reliability-positive); 29b full-N = the frontier attempt for healthy
  windows. Both need user go per the submission gate rule.
- "Why are others faster": leaders spend 2.8x less effective compute
  AND run GEMM-speed dense paths; our packed gather/einsum machinery is
  90% of wall at ~10-20 GFLOP/s effective, buying the ~25% flop
  discount (~15-20% of score). Slowness is earned, not waste.

## Evaluator-failure forensics (20:30Z, "why do our submissions keep failing")

Timing facts:
- 316368: created 20:04:24Z, failure detected 20:19:44Z (30s-grain poll) —
  the ENTIRE lifecycle (queue + run + scoring 50/50 nets + death) fits in
  15m20s. The grader was NOT slow for this job (<=18s/net incl. overhead);
  it scored everything and died at the commit/finalize phase.
- 316327 (algo26): same signature — user saw MSE on its page despite the
  error => also scored, died post-scoring.
- 316258/316259 (algo25 v2, cap 122,880 ~ 2x samples ~ 2x runtime): died
  ~10+ min in, mid-run, repeatably. 316260 (sample-slashed config, much
  shorter run): graded. Runtime correlates with death across our history.
- Field state at 20:27-20:28Z: FOUR different submissions insta-failed
  within 48 seconds of creation ("Evaluation error") — the evaluator is
  hard-down right now, failing jobs on arrival. 316371 (another
  participant, 20:14Z) also died "could not complete" within 15 min.

Synthesis: today's failures are dominated by evaluator infrastructure
degradation (began ~09:38Z, worsened into a full outage by ~20:27Z). Our
disproportionate hit rate (1 committed grade in 6 full-pipeline runs
today) has a mechanistic component the user correctly suspected as
timeout-flavored: our jobs are among the LONGEST-RUNNING on the board
(heaviest backend wall — e.g. the packing gather moves 78 GB/net), so per
submission we expose a ~13-15 min window to a sick worker pool that
cheap ~2-minute jobs (the mult-0.13 leaders) mostly dodge. But it is NOT
a fixed our-job-too-slow limit: 316368 proved the run itself completes
comfortably; the kills land mid-run OR at commit depending on when the
worker dies.

Actions:
- HOLD all submissions until the evaluator stabilizes. Health check
  before any submit: scan the last ~10 submission IDs — healthy = mostly
  "graded", no insta-fail bursts, no "could not complete" cluster.
- The 316368 verbatim retry is UNNECESSARY (panel already delivered the
  flat-result calibration; committing a flat grade has no value).
- Reliability argument added to the wall-reduction ledger: cutting
  backend wall shrinks the per-submission exposure window ~5x, on top of
  any pricing effect — strengthens the row-blocked Strassen case.
- Worth reporting to organizers (forum) with IDs: 316327/316368 scored
  50/50 then failed at commit; 20:27-28Z insta-fail burst — their side
  has the tracebacks.

## Competitive landscape (public scores, API scan 316200-316330, 2026-07-14)

Decomposition: multiplier = adjusted/raw; scoring is
`max(0.1, C_m/B) * raw` (documented floor at 0.1).

- **Frontier cluster** (same raw fingerprint drifting 3.26e-7 → 2.87e-7
  across 6 submissions, mult steady 0.126-0.133): adjusted 3.81e-8.
  vs our 315892: adjusted 1.338e-7, raw 3.725e-7, mult 0.359. They get
  1.3x BETTER raw at 2.8x LESS effective compute — naively ~3.6x better
  variance-per-effective-FLOP if sampling-limited.
- **Raw-MSE existence proofs** (distinct raw fingerprints, others):
  1.162e-7 @ mult 0.653 (316314); 1.962e-7 with mult walked 0.977 →
  0.868 → 0.776 same day (316290/301/318, plus 2.336e-7 @ 0.716);
  2.127e-7 @ 0.509 (316221); 2.65-2.76e-7 @ ~0.27 (316322/328,
  adjusted ~7.2e-8). At least 4-5 participants have raw < 3e-7; we are
  at 3.72e-7.
- **Floor economics**: compute below 10% of budget is free; the leaders
  sit at 0.126-0.133 (near-floor) and iterate on RAW at fixed mult. Our
  -1%-FLOP class of wins (algo26) is noise against a 2.8x compute gap —
  and cutting our samples 2.8x would blow raw up ~proportionally,
  landing adjusted right back at ~1.3e-7. FLOP-shaving cannot reach the
  frontier from our surface.
- **Two visible routes on the board**: (a) high-compute variance/bias
  breakthrough (raw 1.16e-7 exists; at OUR mult 0.359 that raw would be
  adjusted 4.2e-8 ≈ frontier); (b) near-floor efficiency (frontier
  cluster). Both require per-sample value we don't have.
- **Scope correction to closed doors**: "variance-per-FLOP fully tapped"
  (2026-07-08/09) is TRUE ONLY within the MC-on-algo21 architecture
  (linear/expected-gate surrogates, rho^2=0.06). The leaderboard is an
  existence proof of ~3.5x better variance-per-compute by structurally
  different estimators. The next campaign should target per-sample value
  (e.g., analytical/anchor hybrids re-examined under floor economics —
  full-cov's +30% residual multiplier reads differently when the target
  operating point is the 0.1 floor), not FLOP price.
