# Submission Learnings - 2026-07-25

## PR150/v0.9.1 local experiment notebook (experiments_pr150.ipynb)

- flopscope v0.9.1 (merged #150+#151), installed package; budget 272G;
  10 mini nets [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]; FLOP-only multiplier, grader-true zeroing.

- algo34 ship bytes (f64, packed, 61440): F=414.9G mult=1.525 over=10 raw=3.788e-07 adj=1.236e+00
- + f32 cast: F=209.7G mult=0.771 over=0 raw=3.789e-07 adj=2.882e-07
- f32 dense-only 61440  «idea 1»: F=156.7G mult=0.576 over=0 raw=3.788e-07 adj=2.158e-07
- f32 dense-only 40960: F=104.7G mult=0.385 over=0 raw=4.295e-07 adj=1.647e-07
- f32 packed 40960 (318620+cast): F=139.9G mult=0.515 over=0 raw=4.295e-07 adj=2.198e-07
- f32 dense-only 30720: F=78.7G mult=0.289 over=0 raw=8.215e-07 adj=2.356e-07
- f32 dense-only 20480: F=52.7G mult=0.194 over=0 raw=1.476e-06 adj=2.789e-07
- f32 dense-only 16384: F=42.4G mult=0.156 over=0 raw=1.966e-06 adj=2.974e-07
- f32 dense-only 12288: F=32.6G mult=0.120 over=0 raw=2.754e-06 adj=3.227e-07

### Sweeps (6 nets)

**idea1 fixed-N sweep (10 nets)**
- dense N=61440: F=156.7G adj=2.158e-07 raw=3.788e-07
- dense N=40960: F=104.7G adj=1.647e-07 raw=4.295e-07
- dense N=30720: F=78.7G adj=2.356e-07 raw=8.215e-07
- dense N=20480: F=52.7G adj=2.789e-07 raw=1.476e-06
- dense N=16384: F=42.4G adj=2.974e-07 raw=1.966e-06
- dense N=12288: F=32.6G adj=3.227e-07 raw=2.754e-06

**idea5 routing**
- dense-only: F=158.5G adj=2.602e-07 raw=4.532e-07
- packed baseline (map,3/4): F=212.6G adj=3.476e-07 raw=4.533e-07
- packed MAX_K=1/3: F=170.3G adj=2.801e-07 raw=4.532e-07
- packed MAX_K=1/4: F=170.2G adj=2.797e-07 raw=4.533e-07
- packed fire=0.5 uniform: F=164.2G adj=2.705e-07 raw=4.533e-07
- packed L1 split (not full-pack): F=212.6G adj=3.477e-07 raw=4.533e-07
- packed 1/4 + L1split + fire.5: F=158.8G adj=2.614e-07 raw=4.533e-07

**idea2 fold/identity/threshold**
- fold on (baseline): F=158.5G adj=2.602e-07 raw=4.532e-07
- fold OFF: F=160.0G adj=2.631e-07 raw=4.530e-07
- identity from L=29: F=158.5G adj=2.616e-07 raw=4.556e-07
- identity from L=27: F=158.5G adj=2.633e-07 raw=4.586e-07
- identity from L=25: F=158.5G adj=2.648e-07 raw=4.611e-07
- on-thresh 2.5: F=157.9G adj=2.636e-07 raw=4.606e-07
- on-thresh 3.5: F=159.0G adj=2.603e-07 raw=4.521e-07

**idea3 pilot**
- staged 5%+20% (baseline): F=158.5G adj=2.602e-07 raw=4.532e-07
- single 20% stage: F=159.0G adj=2.611e-07 raw=4.532e-07

