# Sobol Active-Set Estimator

This documents the best-scoring estimator in `estimator.py` (submission #313687, `adjusted_final_layer_score ≈ 3.31e-07`). It combines Sobol quasi-Monte Carlo sampling with analytical neuron classification and a two-layer fold at the network tail.

## Goal

For a width-$m$, depth-$L$ bias-free ReLU MLP with weights $W_1, \ldots, W_L$, predict every post-ReLU layer mean

$$
\mathbb{E}_{x \sim \mathcal{N}(0, I_m)}\left[h_\ell(x)\right],
\qquad \ell = 1, \ldots, L.
$$

as a `(depth, width)` `flopscope.numpy` array under a 272B FLOP budget. The primary score is:

$$
\text{score} = \text{final\_layer\_MSE} \times \max\!\left(0.1,\; \frac{\text{effective\_compute}}{\text{budget}}\right)
$$

## Hyperparameters

| Parameter | Value | Role |
|---|---:|---|
| `_MAIN_SAMPLES` | `16384` | 8192 Sobol half-samples × 2 (antithetic) |
| `_DEAD_THRESH` | `-2.5` | α below this → dead neuron (predict 0) |
| `_ON_THRESH` | `2.5` | α above this → always-on (predict W @ mean) |

No pilot pass, no tunable blend weights, no lognormal fallback. The method is deliberately simple.

## Key Ideas

### 1. Exact First-Layer Mean

For layer 0 with input $x \sim \mathcal{N}(0, I)$:

$$
z_j = x^\top w_j \sim \mathcal{N}(0, \lVert w_j \rVert^2)
\qquad\Rightarrow\qquad
\mathbb{E}[\operatorname{ReLU}(z_j)] = \frac{\lVert w_j \rVert}{\sqrt{2\pi}}
$$

This is exact — no sampling needed for the first row.

### 2. Analytical Classification (No Pilot)

Instead of spending ~2.7B FLOPs on a pilot pass to classify neurons, propagate diagonal mean/variance analytically through ReLU layers:

$$
\mu_{\text{pre},j}^{(\ell)} = w_j^\top \mu_{\text{post}}^{(\ell-1)}, \qquad
\sigma_{\text{pre},j}^{(\ell)} = \sqrt{\sum_i w_{ij}^2 \, v_{\text{post},i}^{(\ell-1)}}
$$

$$
\alpha_j = \mu_{\text{pre},j} / \sigma_{\text{pre},j}
$$

Classification rule:
- $\alpha < -2.5$ → **dead** (predict 0, exclude from active set)
- $\alpha > 2.5$ → **always-on** (ReLU is identity, use linear formula)
- otherwise → **kink** (needs per-sample ReLU)

Post-ReLU moments for the next layer:

$$
\mu_{\text{post}} = \mu_{\text{pre}} \Phi(\alpha) + \sigma_{\text{pre}} \phi(\alpha)
$$

$$
v_{\text{post}} = (\sigma_{\text{pre}}^2 + \mu_{\text{pre}}^2)\Phi(\alpha) + \mu_{\text{pre}}\sigma_{\text{pre}}\phi(\alpha) - \mu_{\text{post}}^2
$$

Cost: ~1B FLOPs total (negligible vs the 45B MC pass).

### 3. Sobol QMC Inputs

Pre-generate 8192 Sobol sequence points in $[0,1]^{256}$ offline (scrambled, `scipy.stats.qmc.Sobol`), transform via inverse-normal CDF, store as `sobol_points.npz`. Loaded in `setup()` at zero FLOP cost.

Sobol sequences fill the 256-dimensional unit cube more uniformly than pseudo-random samples, giving lower MC variance at the same $N$. Combined with antithetic pairing ($x$ and $-x$), effective sample size is 16384.

### 4. Active-Set Subnetwork Propagation

Only non-dead neurons are propagated. At each layer, slice the weight matrix to `W[prev_active, current_active]`:

$$
h_\ell[:, A_\ell] = \operatorname{ReLU}\!\left(h_{\ell-1}[:, A_{\ell-1}] \cdot W_\ell[A_{\ell-1}, A_\ell]\right)
$$

Cost per layer: $N \times |A_{\ell-1}| \times |A_\ell| \times 2$ FLOPs instead of $N \times m^2 \times 2$.

### 5. Layer Fold at Layers 30–31

For always-on neurons at layer 30, their expected activation is:

$$
\mathbb{E}[h_{30,j}] = w_j^\top \mathbb{E}[h_{29}] \quad \text{(exact, zero variance)}
$$

This saves propagating N=16384 samples through the on-neuron columns. At layer 31, the kink neurons receive contributions from BOTH:
- Layer-30 kink outputs (carried as samples in `x_main`)
- Layer-30 on outputs (not carried as samples — folded back)

The fold computes `W_30[:, on] @ W_31[on, kink]` as a single pre-multiplied matrix and applies it to the saved pre-fold samples. This recovers the correct pre-activation for layer-31 kink neurons without needing the full layer-30 sample matrix.

### 6. Functional Scatter

The grader uses immutable `flopscope` arrays. Instead of `row[idx] = values`, use:

```python
fnp.eye(width)[:, idx] @ values  # identity column read + matmul
```

This builds a width-length vector with values placed at the correct indices, at negligible cost.

## Algorithm

```text
Input: MLP weights W_1..W_L, shipped sobol_points.npz
Output: matrix rows[0..L-1, 0..m-1] of post-ReLU means

1. Stage 1 — Analytical classification:
   For each layer, propagate diagonal (μ, σ²) through ReLU.
   Classify each neuron by α = μ/σ into dead/on/kink.
   Store active_indices, kink_indices, on_indices per layer.

2. Stage 2 — Sobol MC through active subnetwork:
   x = concat(sobol_half, -sobol_half)  → (16384, 256)
   For layers 0..29:
     Slice W to [prev_active, current_active]
     x = ReLU(x @ W_sliced)
     row[layer] = scatter(mean(x), active_idx, width)
   For layer 30 (fold):
     x_kink = ReLU(x @ W[:, kink])
     on_mean = mean(x) @ W[:, on]        ← exact
     Save x_before_fold for layer 31
     row[30] = scatter(mean(x_kink), kink) + scatter(on_mean, on)
     x = x_kink; prev = kink_idx
   For layer 31 (fold from 30):
     pre_kink = x @ W[kink30, kink31]
     pre_on = x_before_fold @ (W30[:, on30] @ W31[on30, kink31])
     x_kink = ReLU(pre_kink + pre_on)
     on_mean = row[30][active30] @ W[active30, on31]
     row[31] = scatter(mean(x_kink), kink) + scatter(on_mean, on)

3. Stage 3 — Assemble output:
   row[0] = ||w_j|| / sqrt(2π)  (exact)
   Return stack(row[0], row[1], ..., row[L-1])
```

## Empirical Results

Best grader submission (#313687):

```text
adjusted_final_layer_score = 3.31e-07
final_layer_mse            ≈ 1.3e-06
effective_compute          ≈ 45B FLOPs (16.6% of budget)
compute_multiplier         = max(0.1, 0.166) = 0.166
n_failed_mlps              = 0
```

Local comparison (seed=0, 200k ground truth):

| Method | Final-layer MSE | Score estimate |
|--------|---:|---:|
| Diagonal mean propagation | 7.35e-04 | 1.22e-04 |
| Full covariance propagation | 4.20e-06 | 6.97e-07 |
| This estimator (N=16384 Sobol) | 1.96e-06 | 3.25e-07 |

## Why Not Go Further?

We explored several extensions that did NOT beat pure Sobol MC at N=16384:

| Approach | Result | Why |
|----------|--------|-----|
| **α-adaptive blend** (analytical + MC weighted by α) | ~same score | Diagonal-cov analytical has high bias; blend helps on seed=0 but hurts on grader ensemble |
| **Control variate** (analytical as CV baseline) | 1000× worse | g_exact (diagonal propagation) is too biased after 32 layers to serve as a CV |
| **More samples** (N=24576) | ~same 3.3e-07 | Score = MSE × compute_mult; more N pushes mult above 0.1 floor, canceling MSE gain |
| **Fewer samples** (N=8192) | slightly worse | MSE increases faster than mult decreases |
| **Full covariance propagation** | 4.2e-06 MSE | Pure analytical can't match MC due to non-Gaussianity after 32 ReLU layers |
| **Sparse covariance** | Promising structure | Block-diagonal pattern exists but propagation cost is prohibitive under budget |

The ~3.3e-07 ceiling is the fundamental MSE×mult tradeoff at the 0.1 compute floor. Breaking it requires either a fundamentally better estimator (not MC-based) or a way to get MC-quality estimates at <10% budget utilization.

## Files

- `estimator.py` — the submission code
- `sobol_points.npz` — pre-computed (8192, 256) float32 Sobol N(0,1) points
- `examples/13_sobol_active_set_fold.py` — clean standalone reimplementation
