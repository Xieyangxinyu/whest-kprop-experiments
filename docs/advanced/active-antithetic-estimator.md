# Active Antithetic Estimator

This note documents the current strongest estimator in `estimator.py`: a pathwise Monte Carlo estimator that uses a small pilot pass to identify a live subnetwork, then spends most FLOPs on antithetic samples through that subnetwork.

The method is not a pure analytic propagation method. It deliberately keeps input-sample path dependence through the ReLU gates, because block restarts and independent layer resampling lost too much joint structure in experiments.

## Goal

For a width-$m$, depth-$L$ bias-free ReLU MLP with weights $W_1, \ldots, W_L$, predict every post-ReLU layer mean

$$
\mathbb{E}_{x \sim \mathcal{N}(0, I_m)}\left[h_\ell(x)\right],
\qquad \ell = 1, \ldots, L.
$$

as a `(depth, width)` `flopscope.numpy` array under the WHest FLOP budget. The primary score is the adjusted final-layer MSE.

## Default Hyperparameters

| Parameter | Default | Role |
|---|---:|---|
| `WHEST_ACTIVE_PILOT_SAMPLES` | `512` | Full-network pilot samples for classification and fallback estimates. |
| `WHEST_ACTIVE_MAIN_SAMPLES` | `12288` | Main pathwise samples through the active subnetwork. |
| `WHEST_ACTIVE_USE_ANTITHETIC` | `1` | Use `x` and `-x` pairs. |
| `WHEST_ACTIVE_FIRE_THRESHOLD` | `0.02` | Deep-layer firing-rate keep threshold. |
| `WHEST_BLOCK_CLASS_START_LAYER` | `9` | Layers before this are kept conservative/full-width. |
| `WHEST_SPARSE_COV_CORR_THRESHOLD` | `0.06` | Positive pilot-correlation edge threshold. |
| `WHEST_SPARSE_COV_MIN_DEGREE` | `8` | Minimum positive-correlation degree for covariance-core rescue. |
| `WHEST_NLN_FALLBACK_BLEND` | `0.15` | Blend pilot mean toward fitted positive-slab lognormal mean for fallback rows. |
| `WHEST_NLN_KEEP_FRACTION` | `0.15` | Keep rare units whose fitted positive-slab mean is large enough relative to the layer. |
| `WHEST_NLN_MIN_POSITIVES` | `8` | Minimum positive pilot activations before trusting the slab fit. |

Most other knobs in `estimator.py` are experimental and default off. In particular, block restart, extra late sampling, importance sampling, magnitude-only core selection, identity simplification, and peripheral shrinkage were not enabled in the best default path.

## Key Ideas

### Exact First-Layer Mean

For the first layer, each pre-activation is Gaussian:

$$
z_j = x^\top w_j \sim \mathcal{N}\left(0, \lVert w_j \rVert_2^2\right).
$$

Therefore

$$
\mathbb{E}[\operatorname{ReLU}(z_j)] = \frac{\lVert w_j \rVert_2}{\sqrt{2\pi}}.
$$

The estimator returns this exact first-layer row instead of sampling it.

### Antithetic First-Layer Sharing

For an antithetic pair $x, -x$, the first pre-activation only needs one matrix multiply:

$$
z = x W_1,
\qquad
\operatorname{ReLU}(x W_1) = \operatorname{ReLU}(z),
\qquad
\operatorname{ReLU}((-x) W_1) = \operatorname{ReLU}(-z).
$$

This removes duplicate first-layer matmul work while preserving both antithetic branches after the first ReLU.

### Delayed Deep-Layer Classification

The four-block activation structure from the data visualization notebook is most reliable only after several layers. Early layers are therefore kept conservative:

```text
layers 1..8: keep every unit active
layers 9..L: classify units into active / fallback using pilot statistics
```

This avoids over-pruning before the dead/sparse/dense structure has stabilized.

### Zero-Inflated Positive-Slab Fallback

For each deep-layer unit, the pilot pass estimates:

$$
p_i = \mathbb{P}(h_i > 0),
\qquad
\mu^{\log}_i = \mathbb{E}[\log h_i \mid h_i > 0],
\qquad
(\sigma^{\log}_i)^2 = \operatorname{Var}(\log h_i \mid h_i > 0).
$$

The fitted zero-inflated lognormal mean is

$$
m_i^{\mathrm{zilog}}
= p_i \exp\left(\mu_i^{\log} + \frac{1}{2}(\sigma_i^{\log})^2\right).
$$

The fallback row blends the raw pilot mean toward this fitted slab mean when the slab fit is reliable. This is used for units not carried in the main sampled subnetwork.

### Positive Covariance Core Rescue

For deep layers, the pilot activations also define a same-layer covariance graph. The estimator computes a symmetric pilot correlation matrix and keeps positive edges only:

$$
\operatorname{edge}(i, j)
= \mathbf{1}\left\{\operatorname{corr}(h_i, h_j) \ge 0.06\right\},
\qquad
d_i = \sum_{j \ne i} \operatorname{edge}(i, j).
$$

A unit is rescued into the active subnetwork when

$$
d_i \ge 8.
$$

This uses the non-negative covariance prior and the observed dense-core structure. Negative pilot correlations are treated as finite-sample noise rather than evidence.

### Active Subnetwork Main Pass

After the pilot pass selects active indices $A_\ell$ for every layer, the main pass propagates antithetic samples through smaller matrices:

$$
h_\ell[:, A_\ell]
= \operatorname{ReLU}\left(
   h_{\ell-1}[:, A_{\ell-1}]\, W_\ell[A_{\ell-1}, A_\ell]
\right).
$$

The main matmul cost is therefore controlled by

$$
N_{\mathrm{main}}\, |A_{\ell-1}|\, |A_\ell|
$$

instead of

$$
N_{\mathrm{main}}\, m^2.
$$

Dropped units retain their fallback estimate and are treated as zero downstream in the sampled subnetwork.

### Pilot Reuse

For active units, the main sample mean is combined with the pilot mean:

$$
m_{\mathrm{active}}
= \frac{N_{\mathrm{main}} m_{\mathrm{main}} + N_{\mathrm{pilot}} m_{\mathrm{pilot}}}
   {N_{\mathrm{main}} + N_{\mathrm{pilot}}}.
$$

This gives a small variance reduction without extra matmul work.

### Functional Scatter

The AIcrowd grader uses immutable flopscope remote arrays. The estimator must not do indexed in-place writes like:

```python
row[idx] = active_mean
```

Instead, it builds updated rows functionally with `fnp.where` over an index mask. This keeps the same semantics while satisfying remote-array immutability.

## Algorithm

```text
Input: MLP weights W_1..W_L, budget
Output: matrix rows[1..L, 1..m] of post-ReLU means

1. Initialize RNG from mlp.seed.

2. Pilot pass with N_pilot antithetic samples:
   a. For layer 1:
      - compute first-layer pilot activations using one matmul per antithetic pair;
      - set returned row to the exact first-layer ReLU mean;
      - keep every first-layer unit active.
   b. For layers 2..8:
      - propagate the pilot samples pathwise;
      - keep every unit active;
      - store pilot means as fallback rows.
   c. For layers 9..L:
      - propagate the pilot samples pathwise;
      - estimate pilot mean and firing rate;
      - fit zero-inflated positive-slab lognormal fallback;
      - compute positive covariance degree;
      - keep units satisfying any keep rule:
          firing_rate >= 0.02
          or reliable lognormal-slab mean is large
          or positive covariance degree >= 8.

3. Main pass with N_main antithetic samples:
   a. For layer 1:
      - compute ReLU(z), ReLU(-z) from one first-layer matmul per pair;
      - output the exact first-layer mean row.
   b. For layers 2..L:
      - slice weights to previous/current active sets;
      - propagate only active units;
      - estimate active means;
      - combine active means with pilot means;
      - functionally scatter active estimates into the fallback row.

4. Return stack(rows).
```

## Empirical Status

On the public baked `mini` split with the subprocess runner, the current estimator was measured at approximately:

```text
adjusted_final_layer_score = 5.67e-07
final_layer_mse            = 3.90e-06
all_layers_mse             = 8.37e-06
n_failed_mlps              = 0
```

For comparison, `examples/03_covariance_propagation.py` on the same split had approximately:

```text
adjusted_final_layer_score = 8.37e-06
final_layer_mse            = 8.37e-05
all_layers_mse             = 5.57e-05
n_failed_mlps              = 0
```

Thus the active antithetic estimator is much more expensive than covariance propagation, but the final-layer MSE improvement more than pays for the compute multiplier.

## Negative Results That Shaped the Default

These variants remain in `estimator.py` as off-by-default knobs or were kept as separate experiments, but did not beat the default path in the fixed mini sweeps:

- Normal-lognormal corrections inside analytic mean/covariance propagation.
- Dense block restarts from a fitted boundary distribution.
- Bootstrap block restarts from pilot boundary samples.
- Extra late-layer samples without a better proposal distribution.
- Gradient-based importance sampling.
- Magnitude-only core selection as a full replacement for positive covariance degree.
- Peripheral shrinkage toward fallback means.
- Identity-block simplification applied too aggressively.

The main lesson is that the estimator should preserve pathwise input dependence, prune only after the deep sparse/dense structure stabilizes, and use positive covariance degree as a structural rescue signal rather than replacing it with a purely marginal statistic.