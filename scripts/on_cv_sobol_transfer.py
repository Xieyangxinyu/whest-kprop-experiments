"""Probe 3: does the on-neuron CV survive the Sobol block?

Across R scrambled-Sobol antithetic realizations, compare Var of the
variance-weighted on-neuron total estimate, corrected vs uncorrected:
  T(r)      = sum_j v_j * (pair-avg w_j . x_pen)      (v_j: var weights)
  T_cv(r)   = T(r) - beta . (Fbar(r) - E[F])
Arms:
  A  scrambled Sobol (deployment regime)
  B  fresh Gaussian (positive control; should show ~4-5% cut)
Null floor: beta refit with realization-shuffled features -> ratio ~1.
beta: fit once per net on pooled pair-level Gaussian data (deployable:
pilot-block fit). E[F] analytic: E|u.a| = ||a|| sqrt(2/pi), E[(u.a)^2] = ||a||^2.
"""
import math

import numpy as np
import whestbench as wb
from scipy.special import ndtri
from scipy.stats import qmc

_erf = np.frompyfunc(math.erf, 1, 1)

N_NETS = 6
N_PAIRS = 4096
R = 24
TOPM = 8
D = 256

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")


def diag_propagate(weights):
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


def penult(x, weights):
    for li in range(31):
        x = np.maximum(x @ weights[li], 0.0)
    return x


ratios_sobol, ratios_gauss, ratios_null = [], [], []

for i in range(N_NETS):
    mlp = wb.mlp_at(ds, i)
    weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    stats = diag_propagate(weights)
    on_fin = stats[31][0] > 3.0
    w_on = weights[31][:, on_fin]

    M = None
    for li in range(31):
        step = weights[li] * stats[li][1][None, :]
        M = step if M is None else M @ step
    U = np.linalg.svd(M, full_matrices=False)[0][:, :TOPM]
    EF = np.concatenate([np.sqrt(2 / np.pi) * np.ones(TOPM), np.ones(TOPM)])  # ||U_i||=1

    # ---- fit beta on pooled Gaussian pair-level data (deployable analogue) ----
    rng = np.random.default_rng(50_000 + i)
    u_fit = rng.standard_normal((8192, D))
    pe = 0.5 * (penult(u_fit, weights) + penult(-u_fit, weights))
    t_on_fit = pe @ w_on
    v = t_on_fit.var(0)
    vw = v / v.sum()
    y_fit = t_on_fit @ vw                       # variance-weighted total
    proj = u_fit @ U
    X_fit = np.concatenate([np.abs(proj), proj ** 2], axis=1)
    Xc = X_fit - X_fit.mean(0)
    beta, *_ = np.linalg.lstsq(Xc, y_fit - y_fit.mean(), rcond=None)

    def realization(u):
        pe = 0.5 * (penult(u, weights) + penult(-u, weights))
        T = (pe @ w_on) @ vw
        proj = u @ U
        F = np.concatenate([np.abs(proj), proj ** 2], axis=1)
        return T.mean(), F.mean(0)

    T_s, F_s, T_g, F_g = [], [], [], []
    for r in range(R):
        eng = qmc.Sobol(d=D, scramble=True, seed=9000 + 100 * i + r)
        u_s = ndtri(np.clip(eng.random(N_PAIRS), 1e-12, 1 - 1e-12))
        t, f = realization(u_s)
        T_s.append(t); F_s.append(f)
        rg = np.random.default_rng(70_000 + 100 * i + r)
        u_g = rg.standard_normal((N_PAIRS, D))
        t, f = realization(u_g)
        T_g.append(t); F_g.append(f)

    T_s = np.array(T_s); F_s = np.array(F_s)
    T_g = np.array(T_g); F_g = np.array(F_g)

    def ratio(T, F):
        corr = T - (F - EF[None, :]) @ beta
        return corr.var(ddof=1) / T.var(ddof=1)

    rs = ratio(T_s, F_s)
    rgs = ratio(T_g, F_g)
    # null: shuffle feature rows so any apparent cut is fit/noise artifact
    perm = np.random.default_rng(123 + i).permutation(R)
    rnull = ratio(T_g, F_g[perm])
    ratios_sobol.append(rs); ratios_gauss.append(rgs); ratios_null.append(rnull)

    # feature integration-error shrink: Sobol vs Gaussian feature-mean variance
    shrink = np.median(F_s.var(0, ddof=1) / np.maximum(F_g.var(0, ddof=1), 1e-30))
    print(f"net {i}: var-ratio sobol {rs:.4f} gauss {rgs:.4f} null {rnull:.4f}  "
          f"feature-mean var shrink (sobol/gauss) {shrink:.3f}")

print("\n=== summary (corrected/uncorrected variance ratio; <1 = CV helps) ===")
print(f"sobol : mean {np.mean(ratios_sobol):.4f}  (n={N_NETS} nets x {R} reps)")
print(f"gauss : mean {np.mean(ratios_gauss):.4f}")
print(f"null  : mean {np.mean(ratios_null):.4f}")
