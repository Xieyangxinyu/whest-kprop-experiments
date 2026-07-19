# Regenerated Sobol artifacts — 2026-07-19

Six fresh scrambled-Sobol realizations in the shipped-artifact format
(`points` key, float32, 30,720 half-samples x d=256), generated with:

```python
u = scipy.stats.qmc.Sobol(d=256, scramble=True, seed=SEED).random(30720)
points = scipy.special.ndtri(np.clip(u, 1e-7, 1 - 1e-7)).astype(np.float32)
```

scipy 1.15.3, numpy 2.2.6; seeds 2000–2005 (bit-reproducible from the
snippet above).

## Offline fleet MSE (12 He-init 256x32 nets, 61,440 antithetic samples, GT 2^21)

From `bench_logs/sobol_regeneration_probe_2026-07-19.{md,npz}`:

| Artifact | Fleet MSE |
|---|---|
| seed2005 | 4.4924e-7 |
| seed2003 | 4.5345e-7 |
| seed2002 | 4.6253e-7 |
| seed2004 | 4.9635e-7 |
| seed2001 | 5.3461e-7 |
| seed2000 | 5.7102e-7 |
| shipped `sobol_points.npz` | 6.9642e-7 (+40.8% vs fresh mean — but the entire gap is one proxy net: excluding net 4, shipped is +3%, inside noise) |

## Warning before shipping any of these

Offline realization rank does NOT transfer to the grader fleet:
submission 317412 shipped the offline-best of 8 realizations and graded
+13.6% raw WORSE than the shipped artifact. The shipped realization's
grader raw is leaderboard-validated twice (2.9772e-7 / 2.9758e-7).
These files are preserved for reproducibility and future controlled
tests, not as recommended replacements.
