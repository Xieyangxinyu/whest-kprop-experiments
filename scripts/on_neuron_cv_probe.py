"""Probe: control variate for FINAL-layer ON-neuron estimates.

Motivation: cv_rho_gate (2026-07-08) measured rho^2=0.06 for KINK neurons,
but kink carries only ~19% of final sampling variance -> on-neurons carry
~81% and were never CV-tested. On-neuron final estimate is LINEAR in the
penultimate activations: t_j = w_j . mean(x_pen). Antithetic kills odd
surrogates, so candidates are EVEN with known mean:
  F1  |u . m_j|, m_j = M @ w_j  (per-neuron expected-gate direction;
      E|u.m| = ||m|| sqrt(2/pi))
  F2  shared top-8 SVD dirs of M: |u . U_i| and (u . U_i)^2 (E known)
where M = prod_l diag-gate-linearized map input->penult.

Measure per-on-neuron R^2 of pair-averaged target vs features over 8192
antithetic pairs, 10 nets; variance-weight to a net-level potential cut.
Also re-measure the on/kink/dead share of final pair-variance on the
current classification (_ON_THRESH=3).

Bar: deployable needs variance-weighted R^2 >~ 0.10 on the on-share
(-> ~8% raw cut) to beat the ~1% flop cost of computing features.
"""
import math

import numpy as np
import whestbench as wb

_erf = np.frompyfunc(math.erf, 1, 1)

N_NETS = 10
N_PAIRS = 8192
TOPM = 8

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")


def diag_propagate(weights):
    """Return per-layer (alpha, Phi(alpha)) from diagonal Gaussian propagation."""
    out = []
    mu_post = None
    var_post = None
    for li, w in enumerate(weights):
        if li == 0:
            mu_pre = np.zeros(w.shape[1])
            var_pre = np.sum(w * w, axis=0)
        else:
            mu_pre = w.T @ mu_post
            var_pre = np.sum(w * w * var_post[:, None], axis=0)
        var_pre = np.maximum(var_pre, 1e-12)
        sigma = np.sqrt(var_pre)
        a = mu_pre / sigma
        phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1.0 + _erf(a / np.sqrt(2)).astype(np.float64))
        out.append((a, Phi))
        mu_post = mu_pre * Phi + sigma * phi
        var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma * phi - mu_post * mu_post
        var_post = np.maximum(var_post, 1e-12)
    return out


share_on, share_kink = [], []
r2_f1, r2_f2, r2_both = [], [], []

for i in range(N_NETS):
    mlp = wb.mlp_at(ds, i)
    weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    stats = diag_propagate(weights)
    alpha_fin = stats[31][0]
    on_fin = alpha_fin > 3.0
    kink_fin = (alpha_fin >= -3.0) & ~on_fin

    # expected-gate linearized map input -> penult post-relu (layers 0..30)
    M = None
    for li in range(31):
        step = weights[li] * stats[li][1][None, :]  # W_l diag(Phi_l)
        M = step if M is None else M @ step

    rng = np.random.default_rng(7)
    u = rng.standard_normal((N_PAIRS, weights[0].shape[0]))

    # exact forward for +u and -u, collect penult activations
    def penult(x):
        for li in range(31):
            x = np.maximum(x @ weights[li], 0.0)
        return x

    xp = penult(u)
    xn = penult(-u)
    pen_even = 0.5 * (xp + xn)                      # (pairs, 256)

    w31 = weights[31]
    t_all = 0.5 * (np.maximum(xp @ w31, 0) + np.maximum(xn @ w31, 0))  # pair-avg final
    var_all = t_all.var(axis=0)
    share_on.append(var_all[on_fin].sum() / var_all.sum())
    share_kink.append(var_all[kink_fin].sum() / var_all.sum())

    # ON-neuron linear targets (identity: relu(z)=z), pair-averaged
    t_on = pen_even @ w31[:, on_fin]                # (pairs, n_on)
    n_on = t_on.shape[1]
    if n_on == 0:
        continue

    # F1 per-neuron folded feature
    m_dirs = M @ w31[:, on_fin]                     # (d_in, n_on)
    f1 = np.abs(u @ m_dirs)                         # (pairs, n_on)

    # F2 shared top-m SVD features
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    proj = u @ U[:, :TOPM]                          # (pairs, m)
    F2 = np.concatenate([np.abs(proj), proj ** 2], axis=1)  # (pairs, 2m)

    def r2_multi(y, X):
        Xc = X - X.mean(0)
        yc = y - y.mean(0)
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        res = yc - Xc @ beta
        return 1.0 - res.var(0) / np.maximum(yc.var(0), 1e-30)

    v_on = t_on.var(0)
    wgt = v_on / v_on.sum()
    # F1: single feature per neuron
    r1 = np.empty(n_on)
    for j in range(n_on):
        c = np.corrcoef(t_on[:, j], f1[:, j])[0, 1]
        r1[j] = c * c
    r2_f1.append(float((r1 * wgt).sum()))
    r2_f2.append(float((r2_multi(t_on, F2) * wgt).sum()))
    both = np.concatenate([F2], axis=1)
    r2b = np.empty(n_on)
    for j in range(n_on):
        X = np.concatenate([F2, f1[:, j:j + 1]], axis=1)
        r2b[j] = r2_multi(t_on[:, j:j + 1], X)[0]
    r2_both.append(float((r2b * wgt).sum()))
    print(f"net {i}: n_on={n_on} share_on={share_on[-1]:.3f} "
          f"R2 f1={r2_f1[-1]:.4f} f2={r2_f2[-1]:.4f} both={r2_both[-1]:.4f}")

print("\n=== summary (variance-weighted, 10 nets) ===")
print(f"final pair-variance share: on {np.mean(share_on):.3f}, kink {np.mean(share_kink):.3f}")
print(f"R2 per-neuron folded |u.m_j|   : mean {np.mean(r2_f1):.4f}")
print(f"R2 shared top-{TOPM} |u.U|,(u.U)^2: mean {np.mean(r2_f2):.4f}")
print(f"R2 combined                    : mean {np.mean(r2_both):.4f}")
print(f"implied raw-variance cut on on-share: {np.mean(share_on) * np.mean(r2_both) * 100:.2f}%")
