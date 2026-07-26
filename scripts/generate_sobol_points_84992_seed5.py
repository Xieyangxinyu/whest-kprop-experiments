"""Generate sobol_points_84992_seed5.npz -- the algo63 sampling artifact.

Construction (scipy 1.15.3 -- the scramble realization is version-dependent, so
regenerating under another SciPy will NOT reproduce these bytes):

    scrambled Sobol, d=256, seed=5, one 2^16 draw; keep the first 42,496 rows;
    map through the standard normal inverse CDF; store float32.

Rows are antithetic HALF-samples: a block of N effective samples is built as
x = [P; -P], so 42,496 rows = 84,992 effective samples (pilot 10,240 +
continuation 74,752 in the algo63 estimator). Because one draw is taken and a
prefix kept, the first 40,960 rows are byte-identical to the 81,920-sample
artifact used by the algo59/algo61-era experiments (same sequence, longer keep).
"""
import numpy as np
from scipy.stats import norm, qmc

ROWS = 42_496  # 84,992 effective antithetic samples

eng = qmc.Sobol(d=256, scramble=True, seed=5)
u = eng.random_base2(16)  # 65,536 rows in one draw (prefix property holds)
points = norm.ppf(u)[:ROWS].astype(np.float32)
np.savez_compressed("sobol_points_84992_seed5.npz", points=points)
print(f"wrote sobol_points_84992_seed5.npz {points.shape} {points.dtype}")
