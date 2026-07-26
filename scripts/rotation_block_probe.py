"""Rotation-block probe: can orthogonal rotations of the shipped Sobol
artifact act as quasi-independent extra sample blocks (Gaussian isotropy:
x ~ N(0,I) => xQ ~ N(0,I))?

Motivation: the on-neuron error = sampling error of the penultimate mean;
all in-block variance-reduction handles are dead, and the artifact cannot be
extended (scipy stream change). Rotations re-roll the point-set/net
ALIGNMENT without re-rolling the digital scramble — if rotated-block errors
are ~uncorrelated with the original block's, averaging cuts the wobble+MC
share like independent draws, from the same artifact bytes.

Configs (real shipped artifact, plain exact forward, final row vs stored
final_means, 8 mini nets):
  A  orig block, N=61,440 (current)
  B  rotated block (fixed Q from seed 7), N=61,440
  B2 rotated block (Q from seed 8), N=61,440
  C  average of A and B (effective 2N; billing +8G for the rotation)
  D  fixed-N mix: 15,360 orig halves + 15,360 rotated halves (N=61,440
     total; billing +4G) — the budget-relevant variant
Also: correlation of final-row error vectors between A and B/B2.

Deployment billing context: rotating a full block costs one (N,256)@(256,256)
matmul ~ 8.05G (~5.6% of 143G); half block ~4G (~2.8%).
"""
import numpy as np
import whestbench as wb

N_NETS = 8
ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
pts = np.load("sobol_points.npz")["points"].astype(np.float32)  # (30720, 256)
H = pts.shape[0]


def rot(seed):
    g = np.random.default_rng(seed).standard_normal((256, 256))
    q, r = np.linalg.qr(g)
    return (q * np.sign(np.diag(r))).astype(np.float32)


def final_mean(weights, half_block):
    x = np.concatenate([half_block, -half_block], axis=0)
    for li, w in enumerate(weights):
        z = x @ w
        x = np.maximum(z, 0.0)
        if li == len(weights) - 1:
            return x.mean(axis=0)
    return None


Q1, Q2 = rot(7), rot(8)
mse = {k: [] for k in ["A", "B", "B2", "C", "D"]}
corrs_ab, corrs_ab2 = [], []
for i in range(N_NETS):
    mlp = wb.mlp_at(ds, i)
    weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
    gt = np.asarray(ds[i]["final_means"], dtype=np.float64)
    fa = final_mean(weights, pts)
    fb = final_mean(weights, pts @ Q1)
    fb2 = final_mean(weights, pts @ Q2)
    fd = final_mean(weights, np.concatenate([pts[: H // 2], (pts @ Q1)[: H // 2]], axis=0))
    ea, eb, eb2 = fa - gt, fb - gt, fb2 - gt
    mse["A"].append(np.mean(ea**2))
    mse["B"].append(np.mean(eb**2))
    mse["B2"].append(np.mean(eb2**2))
    mse["C"].append(np.mean(((fa + fb) / 2 - gt) ** 2))
    mse["D"].append(np.mean((fd - gt) ** 2))
    corrs_ab.append(np.corrcoef(ea, eb)[0, 1])
    corrs_ab2.append(np.corrcoef(ea, eb2)[0, 1])

print(f"=== rotation-block probe (n={N_NETS} nets, shipped artifact) ===")
for k, label in [("A", "orig N=61440"), ("B", "rot(Q7) N=61440"), ("B2", "rot(Q8) N=61440"),
                 ("C", "avg(orig,rotQ7) eff 2N"), ("D", "mix 50/50 fixed N=61440")]:
    arr = np.array(mse[k])
    print(f"{label:<26} mean MSE {arr.mean():.4e}  (vs A {arr.mean()/np.mean(mse['A'])-1:+.1%})")
print(f"error corr(A,B) per net: mean {np.mean(corrs_ab):+.3f}  (A,B2): {np.mean(corrs_ab2):+.3f}")
print("independence would give corr~0 and C ~ -50% vs A on the variance share;")
print("deployment bar: C must beat A by >~5.6% adj (full extra block billing),")
print("D must beat A by >~2.8% (half-block rotation billing).")
