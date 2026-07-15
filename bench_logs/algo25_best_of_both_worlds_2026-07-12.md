# Algorithm 25 Best-of-Both-Worlds Probe - 2026-07-12

## Change

Algorithm 25 keeps Algorithm 24 / submission 315998's complex64 row-axis sample
packing and Algorithm 21's per-layer block-split thresholds, then restores
wall-safe exact packed-path cleanups from submission 315856:

- Keep `_PACKED_ROWSPARSE_CHUNK_ROWS = 16384` for wall-time safety.
- Remove redundant row bucket `8`, keeping bucket-16 grouping.
- Use immediate Strassen accumulation instead of storing `prod1..prod7`.
- Return only stored Sobol half-samples from `_sample_block`; layer 0 already
  reconstructs the antithetic half exactly after the first matmul.

The first probe also tested restoring 315856's `24576` chunk size. That version
preserved MSE and improved local multiplier, but raised local backend wall time
too much on top of complex packing; it is kept here as a future candidate after
other packed-path wall-time reductions.

## Fixed Slice

Command shape:

```bash
uv run whest run --estimator <estimator> \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
  --split mini --runner subprocess --seed 42 --n-mlps 3 --profile
```

Initial rounded report check against pre-edit Algorithm 24:

| variant | adjusted | raw final MSE | all-layer MSE | mean multiplier | estimator residual |
|---|---:|---:|---:|---:|---:|
| Algorithm 24 current | `1.44e-7` | `5.45e-7` | `9.54e-4` | `0.27067970` | `0.201163s` |
| Algorithm 25 | `1.43e-7` | `5.45e-7` | `9.54e-4` | `0.26848098` | `0.183395s` |

Corrected JSON A/B used a temp copy of `HEAD:estimator.py` for Algorithm 24 and
the first Algorithm 25 `24576`-chunk probe. Aggregate public-mini n=3 results:

| metric | Algorithm 24 | Algorithm 25 | delta |
|---|---:|---:|---:|
| adjusted_final_layer_score | `1.6014165294504084e-7` | `1.4792142393483428e-7` | `-7.630887%` |
| final_layer_mse | `5.452320124277321e-7` | `5.452320124277321e-7` | `0.000000%` |
| all_layers_mse | `9.536515960159401e-4` | `9.536515960159401e-4` | `0.000000%` |
| mean_score_multiplier | `0.2942721162233982` | `0.27660996421632883` | `-6.001979%` |
| mean_effective_compute | `80.04201561276431G` | `75.23791026684145G` | `-6.001979%` |
| estimator flops_used | `200758339427` | `200740971136` | `-0.008651%` |
| estimator calls | `27637` | `26228` | `-5.098238%` |
| estimator residual_wall_time_s | `0.3936770741129294` | `0.24972759664524347` | `-36.565370%` |
| flopscope_backend_time_s | `42.48250955378171` | `50.51145330350846` | `+18.899410%` |

## Readout

Predictions are unchanged at report precision: final-layer MSE and all-layer MSE
match exactly in the JSON A/B. Deterministic tracked FLOPs improve only slightly
because Algorithm 24's complex packing already dominates the charge reduction.
The stronger local score gain comes from lower residual/call overhead, which is
noisy locally and should be treated as a submission-calibration question rather
than a guaranteed public multiplier delta. The call-count reduction and exact
raw parity make Algorithm 25 a low-risk follow-up candidate.

## Wall-Time Follow-Up

The `24576` chunk version was not wall-time safe locally. Operation-level profile
showed the regression was dominated by zero-FLOP memory traffic rather than
mathematical FLOPs:

| op | Algorithm 24 backend | Algorithm 25 `24576` backend | delta |
|---|---:|---:|---:|
| `fnp.take` | `19.1522s` | `25.2120s` | `+6.0599s` |
| `fnp.einsum` | `9.0913s` | `10.9106s` | `+1.8193s` |
| `put_along_axis` | `3.4974s` | `3.5727s` | `+0.0753s` |
| `concatenate` | `1.2460s` | `1.3010s` | `+0.0550s` |

Variant sweep on the same fixed slice:

| variant | adjusted | raw MSE | multiplier | wall | backend | residual | flops | calls | take backend | einsum backend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Algorithm 24 | `1.601e-7` | `5.452e-7` | `0.294272` | `44.483s` | `42.483s` | `0.394s` | `200.758G` | `27637` | `19.152s` | `9.091s` |
| Algorithm 25 `24576` | `1.479e-7` | `5.452e-7` | `0.276610` | `52.386s` | `50.511s` | `0.250s` | `200.741G` | `26228` | `25.212s` | `10.911s` |
| Algorithm 25 wall-safe (`16384`, no bucket 8) | `1.437e-7` | `5.452e-7` | `0.270845` | `44.130s` | `42.421s` | `0.203s` | `200.738G` | `27152` | `20.045s` | `8.105s` |
| bucket 8 with `24576` | `1.451e-7` | `5.452e-7` | `0.272833` | `55.430s` | `53.620s` | `0.219s` | `200.741G` | `26658` | `27.142s` | `11.937s` |
| stored Strassen with `24576` | `1.726e-7` | `5.452e-7` | `0.313092` | `49.870s` | `47.831s` | `0.547s` | `200.741G` | `26228` | `23.898s` | `9.572s` |

Decision for current `estimator.py`: keep the wall-safe `16384` chunk size while
retaining the exact-output no-bucket-8, immediate-Strassen, and half-only-Sobol
cleanups. This kept raw/all-layer MSE identical, improved the fixed-slice
adjusted score vs Algorithm 24, and brought total wall back to Algorithm 24
levels (`44.130s` vs `44.483s` for n=3).

### Row-Dense Fallback in the Sparse Column Block

Layer-by-layer diagnostics showed the high-`k` row-density problem is almost
entirely in the first few packed layers after column grouping:

| layer | mean k | rows >64 | rows >96 | rows >128 | sparse flops |
|---:|---:|---:|---:|---:|---:|
| `1` | `135.50` | `100%` | `~100%` | `47.6%` | `5.50G` |
| `2` | `135.92` | `100%` | `100%` | `49.5%` | `5.52G` |
| `3` | `137.38` | `100%` | `~100%` | `57.5%` | `5.53G` |
| `4` | `134.44` | `100%` | `~100%` | `41.5%` | `5.35G` |
| `5` | `64.27` | `18.8%` | `0%` | `0%` | `2.48G` |
| `6` | `63.28` | `9.9%` | `0%` | `0%` | `2.43G` |

Lowered the packed sparse dense-fallback threshold from `k > 3/4 width` to
`k > 1/2 width`. This happens after the column fire split: high-fire columns
already go dense, and low-fire columns only fall back to dense for rows that are
still more than half nonzero inside that sparse-column block.

Fixed public-mini subprocess checks:

| n | variant | adjusted | raw MSE | multiplier | wall | flops | matmul flops | einsum flops | failures |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `3` | Algorithm 24 | `1.601e-7` | `5.452e-7` | `0.294272` | `44.483s` | `200.758G` | `76.680G` | `115.978G` | `0/3` |
| `3` | Algorithm 25 `k>1/2` | `1.399e-7` | `5.452e-7` | `0.263777` | `34.048s` | `196.467G` | `105.004G` | `83.141G` | `0/3` |
| `5` | Algorithm 24 | `1.168e-7` | `4.195e-7` | `0.262022` | `63.035s` | `289.429G` | `109.203G` | `168.664G` | `0/5` |
| `5` | Algorithm 25 `k>1/2` | `1.005e-7` | `4.195e-7` | `0.229909` | `51.578s` | `283.906G` | `146.691G` | `125.355G` | `0/5` |

Decision: keep `k > width/2` fallback in current `estimator.py`. It preserves
raw/all-layer metrics at report precision, improves adjusted score by about
`12.6%` on n=3 and `13.9%` on n=5 versus Algorithm 24, and reduces wall time.

Tested an aggregate row-dense prototype that merges dense-column and sparse-
column row-dense work into one full-width dense matmul for row-dense rows. It is
faster than current, but regresses adjusted score versus current:

| n | current `k>1/2` | aggregate prototype | aggregate delta |
|---:|---:|---:|---:|
| `3` | `1.399e-7` | `1.413e-7` | `+0.98%` |
| `5` | `1.005e-7` | `1.012e-7` | `+0.64%` |

Do not port aggregate dense blocks yet; current `k > width/2` fallback is the
submission candidate.

### Cap-Sample Wall-Time Check

Current sample policy caps at `_MAX_SAMPLES = 61440`. On the public mini split,
13/100 MLPs hit the cap:

`11, 14, 19, 20, 28, 50, 53, 60, 63, 71, 84, 95, 96`.

Direct current-estimator timing on all 13 cap MLPs, with `sobol_points.npz`
loaded via a local setup context, stayed comfortably below the 60s wall limit:

| metric | value |
|---|---:|
| cap MLPs timed | `13` |
| max wall time | `25.026s` |
| mean wall time | `17.549s` |
| cap MLPs over 60s | `0` |

Per-cap-row wall times:

| idx | name | wall |
|---:|---|---:|
| `11` | `alan-campbell` | `25.026s` |
| `14` | `thomas-parsons` | `16.709s` |
| `19` | `jessica-benson` | `12.648s` |
| `20` | `alexandra-adkins` | `15.866s` |
| `28` | `todd-jenkins` | `19.409s` |
| `50` | `joshua-collins` | `22.792s` |
| `53` | `alexandra-reid` | `18.517s` |
| `60` | `david-tyler` | `18.341s` |
| `63` | `frederick-vargas` | `15.470s` |
| `71` | `savannah-hayes` | `16.151s` |
| `84` | `jessica-henderson` | `16.817s` |
| `95` | `cathy-butler` | `14.221s` |
| `96` | `lindsey-chase` | `16.174s` |

This is not a full official subprocess telemetry run on a targeted cap subset;
the CLI rejected an ad-hoc `Dataset.save_to_disk` subset without baked metadata,
and a long prefix run did not produce a usable JSON artifact. Still, the direct
cap timing plus subprocess n=3/n=5 checks with zero failures suggest cap sample
size is not currently the wall-time failure mode on this machine. Recheck with a
proper baked cap subset before a high-stakes final submission if time permits.

Important lesson: `fnp.take` itself is not optional in the current flopscope path.
A probe replacing `fnp.take` with direct advanced indexing eliminated `take`
backend time but failed all 3 MLPs with `combined_budget_exhausted` (`raw=1.121`,
multiplier `1.0`). The practical target is not removing all gathers, but making
the packed gather path move less memory.

Next hypothesis: larger chunks can probably still help after cutting packed-path
wall time elsewhere. Revisit `24576` only after reducing the main memory traffic
sources, especially `fnp.take(weights, order, axis=0)`, per-chunk sort/unsort,
and per-group `einsum` temporaries. Promising directions are adaptive group
splitting by `group_rows * k * out_width`, avoiding full-chunk sorted copies, or
slightly denser routing for high-`k` groups where dense matmul is wall-cheaper
even if it costs a small number of extra tracked FLOPs.

### Sort/Unsort Avoidance Probe

Tested a mask-grouping packed kernel that avoids chunk-wide `argsort`, sorted
`take(x_chunk, row_order)`, sorted `take(mask_chunk, row_order)`, and final
full-chunk row-order restore. It instead builds each bucket with
`nonzero((nnz > lower) & (nnz <= limit))`, gathers only that bucket's rows, and
`put_along_axis` writes each bucket output back to original row positions.

Same fixed slice vs current wall-safe sort/unsort kernel:

| variant | adjusted | raw MSE | multiplier | wall | backend | residual | flops | calls | take | put | argsort | nonzero | einsum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wall-safe sort/unsort | `1.437e-7` | `5.452e-7` | `0.270845` | `44.130s` | `42.421s` | `0.203s` | `200.738G` | `27152` | `20.045s` | `3.558s` | `0.106s` | `0.002s` | `8.105s` |
| mask grouping | `1.450e-7` | `5.452e-7` | `0.272277` | `43.876s` | `42.164s` | `0.214s` | `200.758G` | `32495` | `19.197s` | `3.486s` | `0.000s` | `0.015s` | `9.078s` |

Readout: exact predictions were preserved, and wall/backend improved slightly
(`-0.58%` wall, `-0.61%` backend), but multiplier worsened `+0.53%` because
tracked FLOPs/call overhead rose slightly. The call count increased `+19.7%` and
`einsum` backend rose, partly offsetting the saved chunk-wide sort/restore. Do
not replace the current kernel with this version yet; it is useful evidence that
sort/unsort is not the dominant remaining wall-time problem at `16384` chunks.
The bigger target remains reducing gathered-weight/einsum memory volume.

## Remaining Packing Headroom

From `algorithm24_dtype_bitpacking.ipynb`, the remaining headroom in the packing
family appears to sit in two places:

- The row-sparse `einsum` path is still real-valued. Pairing rows there requires
  solving the union-of-supports problem, but if it can be made exact or nearly
  exact it is plausibly worth a few more percent.
- More than 2x quantized lanes, such as `c128+mantissa`, only beat exact c64 if
  the reconstruction residual falls below roughly `5e-8`. Current best residual
  is around `9e-8`, so this is not yet a win.

The floor-limited maximum for the whole packing family is estimated around
adjusted score `3.7e-8`.

### Row-Sparse Complex Pairing Probe

Support-overlap diagnostic, run by instrumenting the current packed sparse path
without changing predictions, on public-mini `seed=42`, `n=3`:

| pairing strategy | weighted union / current real support | ideal sparse-charge saving |
|---|---:|---:|
| adjacent rows in sorted bucket | `0.606467` | `39.35%` |
| first-half / second-half rows in sorted bucket | `0.609085` | `39.09%` |

Bucket detail for adjacent pairing:

| k | pairs | union / real support | overlap per pair |
|---:|---:|---:|---:|
| `16` | `206116` | `0.5376` | `5.57` |
| `32` | `724083` | `0.5809` | `16.38` |
| `48` | `858242` | `0.5910` | `25.18` |
| `64` | `328869` | `0.5822` | `33.86` |
| `80` | `23172` | `0.6082` | `38.34` |
| `112` | `6906` | `0.7017` | `62.30` |
| `128` | `154779` | `0.6690` | `73.93` |
| `144` | `149275` | `0.6366` | `85.16` |
| `160` | `6236` | `0.6082` | `99.95` |

This confirms real overlap: exact row-sparse complex pairing is theoretically
interesting. However, the first implementation was too expensive. It grouped
pairs by union size, built complex values on the union support, used complex
`einsum`, then unpacked real/imag lanes. Pairing every bucket from `k>=32`
timed out on one MLP.

High-`k`-only thresholds on one MLP:

| variant | adjusted | raw MSE | multiplier | wall | flops | einsum flops | take backend | astype backend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wall-safe baseline | `1.539e-7` | `6.762e-7` | `0.227583` | `13.502s` | `54.498G` | `31.678G` | `6.540s` | `0.046s` |
| pair `k>=128` | `1.474e-7` | `6.763e-7` | `0.217908` | `23.822s` | `50.889G` | `27.974G` | `4.423s` | `2.670s` |
| pair `k>=144` | `1.503e-7` | `6.763e-7` | `0.222272` | `19.098s` | `52.489G` | `29.621G` | `5.530s` | `1.723s` |
| pair `k>=160` | `1.517e-7` | `6.762e-7` | `0.224340` | `11.885s` | `54.345G` | `31.522G` | `4.936s` | `0.106s` |
| pair `k>=176` | `1.500e-7` | `6.762e-7` | `0.221783` | `12.317s` | `54.498G` | `31.678G` | `5.616s` | `0.048s` |

The one-MLP high-`k` result looked promising, but did not survive the n=3 fixed
slice:

| variant | adjusted | raw MSE | multiplier | wall | flops | einsum flops | take backend | astype backend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wall-safe baseline | `1.437e-7` | `5.452e-7` | `0.270845` | `44.130s` | `200.738G` | `115.978G` | `20.045s` | `0.189s` |
| pair `k>=160` | `1.455e-7` | `5.452e-7` | `0.272997` | `44.799s` | `200.373G` | `115.605G` | `18.763s` | `0.635s` |
| pair `k>=176` | `1.444e-7` | `5.452e-7` | `0.271105` | `45.794s` | `200.738G` | `115.978G` | `19.826s` | `0.193s` |

Decision: reject the naive union-support complex sparse implementation for now.
It proves the overlap signal exists and flops can drop, but union construction,
complex `astype`, extra pair grouping, and complex `einsum` backend time erase
the benefit on the fixed n=3 check. A future attempt needs a cheaper/fused union
builder or a very narrow policy learned from broader diagnostics; do not port
this wrapper into `estimator.py` as-is.

## Algorithm 14 Comparison

Same fixed public-mini slice (`seed=42`, `n=3`, subprocess, profile), with
Algorithm 14 run via a wrapper that loads the root `sobol_points.npz`:

| variant | adjusted | raw final MSE | multiplier | mean effective | wall | flops | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Algorithm 14 | `3.308501787537e-7` | `7.387683732910e-7` | `0.446825` | `121.536G` | `1.511s` | `360.368G` | `0/3` |
| Algorithm 25 wall-safe | `1.437164999749e-7` | `5.452320124277e-7` | `0.270845` | `73.670G` | `44.130s` | `200.738G` | `0/3` |

Readout: Algorithm 25 wall-safe is much better on the primary metric for this
slice: adjusted score `-56.56%`, raw final MSE `-26.20%`, mean effective compute
`-39.39%`, and tracked estimator FLOPs `-44.30%` versus Algorithm 14. Algorithm
14 is still dramatically faster in local wall time (`1.51s` vs `44.13s`) because
it uses simpler dense/flopscope operations, but it spends far more charged
compute and has worse final-layer accuracy. Algorithm 25's all-layer MSE is not
comparable to Algorithm 14 because current row-sparse submissions intentionally
return cheap analytical filler for intermediate layers; the leaderboard ranks
the adjusted final-layer score.

## Submission 316005 Result

Submitted `submission-algo25-rowdense-2026-07-12.tar.gz` as submission `316005`.
It packages `estimator.py`, `sobol_points.npz`, `requirements.txt`, and
`LICENSE`.

Public result versus Algorithm 24 / submission `315998`:

| submission | adjusted | raw final MSE | mean multiplier | mean effective | failures |
|---:|---:|---:|---:|---:|---:|
| `315998` | `1.006111002092e-7` | `3.724939691097e-7` | `0.2664105` | `72.464G` | `0/50` |
| `316005` | `1.002851105788e-7` | `3.724967319840e-7` | `0.2653999` | `72.189G` | `0/50` |

Readout: improved `-0.3240%` adjusted despite raw final MSE being effectively
unchanged/slightly worse (`+0.000742%`). The public gain came from the compute
multiplier (`-0.3793%`), matching the row-dense fallback hypothesis. Per-public
row adjusted wins were `28/50`.