"""Algorithm 15: staged active-set Sobol with smooth sample allocation.

This example names the current best submission family from `estimator.py`:

- staged active/dead/on classification refinement from a 30,720-sample prefix
- first-layer antithetic sign-flip shortcut
- variance-free active-set forward pass
- smooth analytical-final-variance allocation
  `N_i = clip(49152 * sqrt(V_i / 0.02143), 30720, 61440)`

The implementation is re-exported from the root estimator so the curriculum
example cannot silently drift from the submitted Algorithm 15 surface. The
wrapper only adapts `setup()` so this file can be validated directly from the
`examples/` directory while loading the root `sobol_points.npz` artifact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import flopscope.numpy as fnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimator import Estimator as _Algorithm15Estimator  # noqa: E402
from whestbench import SetupContext  # noqa: E402


class Estimator(_Algorithm15Estimator):
    """Algorithm 15 with example-friendly Sobol artifact resolution."""

    def setup(self, ctx: SetupContext) -> None:
        fnp.random.default_rng(ctx.seed)
        env_path = os.environ.get("SOBOL_POINTS_PATH")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend(
            [
                Path(ctx.submission_dir) / "sobol_points.npz",
                Path(__file__).resolve().parent / "sobol_points.npz",
                Path(__file__).resolve().parent.parent / "sobol_points.npz",
            ]
        )
        for sobol_path in candidates:
            if sobol_path.exists():
                data = fnp.load(str(sobol_path))
                self._sobol_points = data["points"]
                return
        raise FileNotFoundError("Could not find sobol_points.npz for Algorithm 15 example.")


if __name__ == "__main__":
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=256, depth=32, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, estimator_budget=272_000_000_000)