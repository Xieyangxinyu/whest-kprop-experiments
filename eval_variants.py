"""Conservative local ablations at 16384 effective Sobol samples.

This keeps the estimator's sampled propagation path intact:
- non-dead neurons are sampled normally
- the existing layer-30 fold is preserved
- optional variants allocate extra final-layer samples to high-variance kink neurons
- optional variants use a late-layer pilot to refine always-on decisions
- optional variants use a late-layer pilot to refine dead and always-on decisions
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy import special

import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp, monte_carlo_layer_means


WIDTH = 256
DEPTH = 32
BUDGET = 272_000_000_000
N_SAMPLES = int(os.environ.get("N_SAMPLES", "16384"))
GT_SAMPLES = int(os.environ.get("GT_SAMPLES", "300000"))
SEEDS = tuple(int(seed) for seed in os.environ.get("SEEDS", "0,1,2,3,4").split(",") if seed)
# Comma-separated substrings; when set, only variants whose name contains one of them run.
VARIANT_FILTER = tuple(s for s in os.environ.get("VARIANT_FILTER", "").split(",") if s)
SOBOL_POINTS_PATH = os.environ.get("SOBOL_POINTS_PATH", str(Path(__file__).parent / "sobol_points.npz"))


@dataclass(frozen=True)
class Variant:
    name: str
    dead_thresh: float = -3.0
    on_thresh: float = 2.5
    sample_mode: str = "normal"
    allocation_top_k: int = 0
    allocation_base_fraction: float = 1.0
    refine_layer29_dead: bool = False
    refine_dead_layers: tuple[int, ...] = ()
    refine_layer30_on: bool = False
    refine_layer30_dead: bool = False
    refine_layer31_on: bool = False
    refine_pilot_fraction: float = 0.25
    refine_on_thresh: float = 3.0
    refine_dead_thresh: float = -2.5
    refine_borderline_only: bool = False
    paired_pilot: bool = False
    pair_tail_refine: bool = False
    pair_on_neg_tail: float = 4.0e-4
    pair_dead_pos_tail: float = 2.0e-3
    refine_on_probe_max: float = 4.0
    refine_dead_probe_min: float = -4.0
    dead_scale: float = 1.0
    l30_on_clip_scale: float = 0.0
    final_on_thresh: float | None = None
    final_on_blend: float = 0.0
    final_on_blend_mode: str = ""
    final_analytical_blend: float = 0.0
    final_on_signed_scale: float = 0.0
    final_on_signed_scale_mode: str = ""
    final_mirror_mode: str = ""
    final_sign_bootstrap_rows: int = 0
    final_sign_bootstrap_blend: float = 0.0
    final_sign_bootstrap_center: str = "zero"
    final_raw_numpy: bool = False
    final_tail_quant_keep_top: int = 0
    final_tail_quant_bins: int = 0
    skip_on_relu_from_layer: int = 0
    skip_on_relu_alpha: float = 0.0
    final_on_sample_correction_mode: str = ""
    final_on_sample_correction_keep: float = 1.0
    final_on_sample_correction_blend: float = 0.0
    dynamic_sample_mode: str = ""
    dynamic_easy_samples: int = 40960
    dynamic_mid_samples: int = 40960
    dynamic_hard_samples: int = 40960
    dynamic_anchor_samples: int = 40960
    dynamic_var_ref: float = 0.0218483
    rotate_w0: bool = False
    rotation_mode: str = ""
    rotation_max_samples: int = 0
    rotation_var_max: float = 0.0


VARIANTS = (
    Variant("normal baseline"),
    Variant("normal fold-on 3.00", on_thresh=3.00),
    Variant("plain sobol fold-on 3.00", on_thresh=3.00, sample_mode="plain_sobol"),
    Variant("pilot refine l30 5%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00),
    Variant("pilot refine l30 10%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.10, refine_on_thresh=3.00),
    Variant("pilot refine l30 both 5%/-2.5", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50),
    Variant("pilot refine l30 both 5%/-2.0", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.00),
    Variant("pilot on borderline 4", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_borderline_only=True, refine_on_probe_max=4.00),
    Variant("pilot dead borderline -4", on_thresh=3.00, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot both borderline 4/-4", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("pilot both borderline 3.75/-3.75", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=3.75, refine_dead_probe_min=-3.75),
    Variant("pilot both borderline 3.5/-3.5", on_thresh=3.00, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=3.50, refine_dead_probe_min=-3.50),
    Variant("pilot l29 dead borderline -4", on_thresh=3.00, refine_layer29_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot l29+30 borderline 4/-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("plain sobol l29+30 4/-4", on_thresh=3.00, sample_mode="plain_sobol", refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 fixed rot-w0", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotation_mode="w0"),
    Variant("l29+30 fixed sens30", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotation_mode="sens30"),
    Variant("l29+30 fixed sens31", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotation_mode="sens31"),
    Variant("l29+30 sens30 var<=0309", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotation_mode="sens30", rotation_var_max=0.0308592),
    Variant("l29+30 w0 var<=0309", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotation_mode="w0", rotation_var_max=0.0308592),
    Variant("l29+30 paired pilot 4/-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, paired_pilot=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 pairtail 04/20", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_borderline_only=True, paired_pilot=True, pair_tail_refine=True, pair_on_neg_tail=4.0e-4, pair_dead_pos_tail=2.0e-3, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 pairtail 08/20", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_borderline_only=True, paired_pilot=True, pair_tail_refine=True, pair_on_neg_tail=8.0e-4, pair_dead_pos_tail=2.0e-3, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 pairtail 04/10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_borderline_only=True, paired_pilot=True, pair_tail_refine=True, pair_on_neg_tail=4.0e-4, pair_dead_pos_tail=1.0e-3, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 pairtail 02/10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_borderline_only=True, paired_pilot=True, pair_tail_refine=True, pair_on_neg_tail=2.0e-4, pair_dead_pos_tail=1.0e-3, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 pairtail 02/05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_borderline_only=True, paired_pilot=True, pair_tail_refine=True, pair_on_neg_tail=2.0e-4, pair_dead_pos_tail=5.0e-4, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 pairtail 01/05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_borderline_only=True, paired_pilot=True, pair_tail_refine=True, pair_on_neg_tail=1.0e-4, pair_dead_pos_tail=5.0e-4, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 l30 on clip 0.25", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, l30_on_clip_scale=0.25),
    Variant("l29+30 l30 on clip 0.50", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, l30_on_clip_scale=0.50),
    Variant("l29+30 l30 on clip 1.00", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, l30_on_clip_scale=1.00),
    Variant("l29+30 on-alpha6 blend 0.02", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_blend=0.02, final_on_blend_mode="alpha6"),
    Variant("l29+30 on-alpha6 blend 0.05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_blend=0.05, final_on_blend_mode="alpha6"),
    Variant("l29+30 on-alpha6 blend 0.10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_blend=0.10, final_on_blend_mode="alpha6"),
    Variant("l29+30 on-pred90 blend 0.02", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_blend=0.02, final_on_blend_mode="pred90"),
    Variant("l29+30 on-pred90 blend 0.05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_blend=0.05, final_on_blend_mode="pred90"),
    Variant("l29+30 on-pred90 blend 0.10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_blend=0.10, final_on_blend_mode="pred90"),
    Variant("dyn var easy20 mid40 hard40", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_q", dynamic_easy_samples=20480, dynamic_mid_samples=40960, dynamic_hard_samples=40960),
    Variant("dyn var easy30 mid40 hard40", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_q", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=40960),
    Variant("dyn var easy30 mid40 hard49", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_q", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=49152),
    Variant("dyn sqrtvar 16-82 refmed", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=16384, dynamic_mid_samples=40960, dynamic_hard_samples=81920, dynamic_var_ref=0.0218483),
    Variant("dyn sqrtvar 65k ref2143", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=16384, dynamic_mid_samples=40960, dynamic_hard_samples=81920, dynamic_anchor_samples=65536, dynamic_var_ref=0.02143),
    Variant("dyn sqrt30-61 ref2143", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143),
    Variant("dyn sqrt30-61 rot-w0 low32", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="w0", rotation_max_samples=32768),
    Variant("dyn sqrt30-61 sens30 low32", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="sens30", rotation_max_samples=32768),
    Variant("dyn sqrt30-61 sens31 low32", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="sens31", rotation_max_samples=32768),
    Variant("dyn sqrt30-61 sens31 low40", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="sens31", rotation_max_samples=40960),
    Variant("dyn sqrt30-61 sens31 low49", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="sens31", rotation_max_samples=49152),
    Variant("dyn sqrt30-61 sens30 low49", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="sens30", rotation_max_samples=49152),
    Variant("dyn sqrt30-61 rot-w0 low49", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="w0", rotation_max_samples=49152),
    Variant("dyn sqrt30-61 sens31 all", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, dynamic_sample_mode="anal_var_sqrt", dynamic_easy_samples=30720, dynamic_mid_samples=40960, dynamic_hard_samples=61440, dynamic_anchor_samples=40960, dynamic_var_ref=0.02143, rotation_mode="sens31"),
    Variant("l29+30 final-on thresh 3.25", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_thresh=3.25),
    Variant("l29+30 final-on thresh 3.50", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_thresh=3.50),
    Variant("l29+30 final-on thresh 4.00", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_thresh=4.00),
    Variant("pilot dead l28-30 border -4", on_thresh=3.00, refine_dead_layers=(28, 29), refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot dead l27-30 border -4", on_thresh=3.00, refine_dead_layers=(27, 28, 29), refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot dead l24-30 border -4", on_thresh=3.00, refine_dead_layers=(24, 25, 26, 27, 28, 29), refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_dead_probe_min=-4.00),
    Variant("pilot l28-30 + on30", on_thresh=3.00, refine_dead_layers=(28, 29), refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("pilot l27-30 + on30", on_thresh=3.00, refine_dead_layers=(27, 28, 29), refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 border 8% pilot", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.08, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 border 10% pilot", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.10, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00),
    Variant("l29+30 border + blend 0.05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_analytical_blend=0.05),
    Variant("l29+30 border + blend 0.10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_analytical_blend=0.10),
    Variant("l29+30 border + blend 0.20", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_analytical_blend=0.20),
    Variant("l29+30 final-on signed scale 5e-5", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_signed_scale=5.0e-5, final_on_signed_scale_mode="pred_minus_anal"),
    Variant("l29+30 final-on signed scale 1e-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_signed_scale=1.0e-4, final_on_signed_scale_mode="pred_minus_anal"),
    Variant("l29+30 final-on signed scale 2e-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_signed_scale=2.0e-4, final_on_signed_scale_mode="pred_minus_anal"),
    Variant("l29+30 final-on signed rev 5e-5", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_signed_scale=5.0e-5, final_on_signed_scale_mode="anal_minus_pred"),
    Variant("l29+30 final-on signed rev 1e-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_signed_scale=1.0e-4, final_on_signed_scale_mode="anal_minus_pred"),
    Variant("l29+30 final mirror sample", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_mirror_mode="sample"),
    Variant("l29+30 final mirror anal", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_mirror_mode="analytical"),
    Variant("l29+30 final signboot b25", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_sign_bootstrap_rows=16384, final_sign_bootstrap_blend=0.25),
    Variant("l29+30 final signboot b50", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_sign_bootstrap_rows=16384, final_sign_bootstrap_blend=0.50),
    Variant("l29+30 final signboot b75", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_sign_bootstrap_rows=16384, final_sign_bootstrap_blend=0.75),
    Variant("l29+30 final signboot mean b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_sign_bootstrap_rows=16384, final_sign_bootstrap_blend=0.10, final_sign_bootstrap_center="meanclip"),
    Variant("l29+30 final signboot mean b25", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_sign_bootstrap_rows=16384, final_sign_bootstrap_blend=0.25, final_sign_bootstrap_center="meanclip"),
    Variant("l29+30 final signboot mean b50", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_sign_bootstrap_rows=16384, final_sign_bootstrap_blend=0.50, final_sign_bootstrap_center="meanclip"),
    Variant("l29+30 final rawnp exact", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_raw_numpy=True),
    Variant("l29+30 final qtail k64 b8", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_tail_quant_keep_top=64, final_tail_quant_bins=8),
    Variant("l29+30 final qtail k96 b8", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_tail_quant_keep_top=96, final_tail_quant_bins=8),
    Variant("l29+30 final qtail k128 b8", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_tail_quant_keep_top=128, final_tail_quant_bins=8),
    Variant("l24-29 on-id alpha4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, skip_on_relu_from_layer=24, skip_on_relu_alpha=4.00),
    Variant("l24-29 on-id alpha5", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, skip_on_relu_from_layer=24, skip_on_relu_alpha=5.00),
    Variant("l27-29 on-id alpha4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, skip_on_relu_from_layer=27, skip_on_relu_alpha=4.00),
    Variant("l31 on sample all b05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="all", final_on_sample_correction_blend=0.05),
    Variant("l31 on sample all b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="all", final_on_sample_correction_blend=0.10),
    Variant("l31 on sample all b20", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="all", final_on_sample_correction_blend=0.20),
    Variant("l31 on sample alpha40 b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_alpha", final_on_sample_correction_keep=0.40, final_on_sample_correction_blend=0.10),
    Variant("l31 on sample adiff40 b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_anal_diff", final_on_sample_correction_keep=0.40, final_on_sample_correction_blend=0.10),
    Variant("l31 on sample alpha20 b05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_alpha", final_on_sample_correction_keep=0.20, final_on_sample_correction_blend=0.05),
    Variant("l31 on sample alpha20 b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_alpha", final_on_sample_correction_keep=0.20, final_on_sample_correction_blend=0.10),
    Variant("l31 on sample adiff20 b05", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_anal_diff", final_on_sample_correction_keep=0.20, final_on_sample_correction_blend=0.05),
    Variant("l31 on sample adiff20 b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_anal_diff", final_on_sample_correction_keep=0.20, final_on_sample_correction_blend=0.10),
    Variant("l31 on sample active20 b10", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, final_on_sample_correction_mode="top_active", final_on_sample_correction_keep=0.20, final_on_sample_correction_blend=0.10),
    Variant("rot-w0 fold-on 3.00", on_thresh=3.00, rotate_w0=True),
    Variant("rot-w0 l29+30 borderline 4/-4", on_thresh=3.00, refine_layer29_dead=True, refine_layer30_on=True, refine_layer30_dead=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00, refine_dead_thresh=-2.50, refine_borderline_only=True, refine_on_probe_max=4.00, refine_dead_probe_min=-4.00, rotate_w0=True),
    Variant("pilot refine l31 5%/3.0", on_thresh=3.00, refine_layer31_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00),
    Variant("pilot refine l30+31 5%", on_thresh=3.00, refine_layer30_on=True, refine_layer31_on=True, refine_pilot_fraction=0.05, refine_on_thresh=3.00),
    Variant("pilot refine l30 25%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.25, refine_on_thresh=3.00),
    Variant("pilot refine l30 25%/3.5", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.25, refine_on_thresh=3.50),
    Variant("pilot refine l30 50%/3.0", on_thresh=3.00, refine_layer30_on=True, refine_pilot_fraction=0.50, refine_on_thresh=3.00),
)


def _chi_radius_mean(dim: int) -> float:
    return math.exp(0.5 * math.log(2.0) + math.lgamma((dim + 1) / 2.0) - math.lgamma(dim / 2.0))


def _scatter(values, idx, width):
    scatter_mat = fnp.eye(width, dtype=fnp.float32)[:, idx]
    return scatter_mat @ values


def _pilot_samples(x, n_samples: int, pilot_fraction: float, paired: bool):
    pilot_rows = max(2, min(n_samples, int(n_samples * pilot_fraction)))
    if not paired:
        return x[:pilot_rows, :]

    half_samples = n_samples // 2
    pilot_pairs = max(1, min(half_samples, pilot_rows // 2))
    return fnp.concatenate([x[:pilot_pairs, :], x[half_samples : half_samples + pilot_pairs, :]], axis=0)


def _rotation_basis(mlp, alpha_rows, mode: str, stop_layer: int | None = None):
    if mode == "w0":
        u_rot, _, _ = np.linalg.svd(np.asarray(mlp.weights[0], dtype=np.float64), full_matrices=False)
        return u_rot.astype(np.float32)

    if mode not in {"sens30", "sens31"}:
        raise ValueError(f"unknown rotation_mode={mode!r}")

    if stop_layer is None:
        stop_layer = 30 if mode == "sens30" else 31

    effective = np.asarray(mlp.weights[0], dtype=np.float64).copy()
    gate_prob = special.ndtr(np.asarray(alpha_rows[0], dtype=np.float64))
    effective *= gate_prob[None, :]
    for layer_idx in range(1, stop_layer + 1):
        effective = effective @ np.asarray(mlp.weights[layer_idx], dtype=np.float64)
        gate_prob = special.ndtr(np.asarray(alpha_rows[layer_idx], dtype=np.float64))
        effective *= gate_prob[None, :]

    u_rot, _, _ = np.linalg.svd(effective, full_matrices=False)
    return u_rot.astype(np.float32)


def _selected_final_update(
    mlp,
    sobol_points,
    start_pair: int,
    extra_pairs: int,
    active_indices,
    kink_indices,
    base_final_row,
    base_final_var,
    base_samples: int,
    top_k: int,
    width: int,
):
    if extra_pairs <= 0 or top_k <= 0:
        return base_final_row

    final_kink_idx = np.asarray(kink_indices[-1], dtype=np.int64)
    if final_kink_idx.size == 0:
        return base_final_row

    final_var = np.asarray(base_final_var)
    selected = final_kink_idx[np.argsort(final_var[final_kink_idx])[-top_k:]]
    selected.sort()
    selected_idx = fnp.array(selected.astype(np.int64))

    half = fnp.array(sobol_points[start_pair : start_pair + extra_pairs, :width])
    x = fnp.concatenate([half, -half], axis=0)

    prev_idx = None
    for layer_idx, w in enumerate(mlp.weights[:-1]):
        idx = active_indices[layer_idx]
        if len(idx) == 0:
            return base_final_row
        if prev_idx is None:
            w_active = w[:, idx]
        else:
            w_active = w[prev_idx, :][:, idx]
        x = fnp.maximum(x @ w_active, 0.0)
        prev_idx = idx

    w_selected = mlp.weights[-1][prev_idx, :][:, selected_idx]
    extra_final = fnp.maximum(x @ w_selected, 0.0)
    extra_mean = fnp.mean(extra_final, axis=0)
    extra_samples = extra_pairs * 2

    base_selected = base_final_row[selected_idx]
    combined = (base_selected * base_samples + extra_mean * extra_samples) / (base_samples + extra_samples)
    return base_final_row + _scatter(combined - base_selected, selected_idx, width)


def _sign_bootstrap_relu_mean(input_blocks, weight_blocks, n_rows: int, seed: int, center_mode: str):
    if n_rows <= 0:
        return None

    inputs = [np.asarray(block, dtype=np.float32) for block in input_blocks if block.shape[1] > 0]
    weights = [np.asarray(block, dtype=np.float32) for block in weight_blocks if block.shape[0] > 0]
    if not inputs or not weights:
        return None

    input_np = inputs[0] if len(inputs) == 1 else np.concatenate(inputs, axis=1)
    weight_np = weights[0] if len(weights) == 1 else np.vstack(weights)
    if input_np.shape[0] == 0 or input_np.shape[1] == 0 or weight_np.shape[1] == 0:
        return None

    rows = int(n_rows)
    take_rows = np.arange(rows, dtype=np.int64) % input_np.shape[0]
    rng = np.random.default_rng(seed + input_np.shape[1] * 1009 + weight_np.shape[1] * 9176 + rows * 37)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(rows, input_np.shape[1]))
    if center_mode == "zero":
        signed_input = np.abs(input_np[take_rows]) * signs
    elif center_mode in {"mean", "meanclip"}:
        center = np.mean(input_np, axis=0, keepdims=True)
        signed_input = center + np.abs(input_np[take_rows] - center) * signs
        if center_mode == "meanclip":
            signed_input = np.maximum(signed_input, 0.0)
    else:
        raise ValueError(f"unknown final_sign_bootstrap_center={center_mode!r}")
    pre = signed_input @ weight_np
    return np.maximum(pre, 0.0).mean(axis=0).astype(np.float32)


def _tail_quantized_relu_stats(input_blocks, weight_blocks, keep_top: int, n_tail_bins: int):
    inputs = [np.asarray(block, dtype=np.float32) for block in input_blocks if block.shape[1] > 0]
    weights = [np.asarray(block, dtype=np.float32) for block in weight_blocks if block.shape[0] > 0]
    if not inputs or not weights:
        return None

    input_np = inputs[0] if len(inputs) == 1 else np.concatenate(inputs, axis=1)
    weight_np = weights[0] if len(weights) == 1 else np.vstack(weights)
    if input_np.shape[0] == 0 or input_np.shape[1] == 0 or weight_np.shape[1] == 0:
        return None

    qx = np.zeros_like(input_np, dtype=np.float32)
    codes = np.zeros(input_np.shape, dtype=np.int16)
    positive = input_np > 0.0
    top_count = min(max(int(keep_top), 0), input_np.shape[1])
    if top_count > 0:
        top_cols = np.argpartition(input_np, -top_count, axis=1)[:, -top_count:]
        keep = np.zeros(input_np.shape, dtype=bool)
        row_ids = np.arange(input_np.shape[0])[:, None]
        keep[row_ids, top_cols] = input_np[row_ids, top_cols] > 0.0
    else:
        keep = np.zeros(input_np.shape, dtype=bool)
    qx[keep] = input_np[keep]

    tail = positive & ~keep
    values = input_np[tail]
    if values.size > 0 and n_tail_bins > 0:
        if n_tail_bins > 1:
            edges = np.quantile(values, np.linspace(0.0, 1.0, n_tail_bins + 1)[1:-1])
            tail_codes = np.searchsorted(edges, values, side="right") + 1
        else:
            tail_codes = np.ones(values.shape, dtype=np.int16)
        centers = np.zeros(n_tail_bins + 1, dtype=np.float32)
        for bin_id in range(1, n_tail_bins + 1):
            selected = values[tail_codes == bin_id]
            centers[bin_id] = float(selected.mean()) if selected.size else 0.0
        qx[tail] = centers[tail_codes]
        codes[tail] = tail_codes
    codes[keep] = -1

    pre = qx @ weight_np
    post = np.maximum(pre, 0.0)
    occupied_tail_bins = np.array([np.unique(row[row > 0]).size for row in codes], dtype=np.float32)
    units_per_row = keep.sum(axis=1).astype(np.float32) + occupied_tail_bins
    return (
        post.mean(axis=0).astype(np.float32),
        post.var(axis=0).astype(np.float32),
        float(np.mean(units_per_row)),
        int(input_np.shape[1]),
    )


def _raw_numpy_relu_stats(input_blocks, weight_blocks):
    inputs = [np.asarray(block, dtype=np.float32) for block in input_blocks if block.shape[1] > 0]
    weights = [np.asarray(block, dtype=np.float32) for block in weight_blocks if block.shape[0] > 0]
    if not inputs or not weights:
        return None

    input_np = inputs[0] if len(inputs) == 1 else np.concatenate(inputs, axis=1)
    weight_np = weights[0] if len(weights) == 1 else np.vstack(weights)
    if input_np.shape[0] == 0 or input_np.shape[1] == 0 or weight_np.shape[1] == 0:
        return None

    post = np.maximum(input_np @ weight_np, 0.0)
    return post.mean(axis=0).astype(np.float32), post.var(axis=0).astype(np.float32)


def predict_variant(mlp, sobol_points, variant: Variant):
    width = mlp.width
    if variant.allocation_top_k > 0:
        n_pairs = max(1, int((N_SAMPLES // 2) * variant.allocation_base_fraction))
    else:
        n_pairs = N_SAMPLES // 2
    n_samples = n_pairs * 2
    extra_pairs = N_SAMPLES // 2 - n_pairs

    active_indices = []
    kink_indices = []
    on_indices = []
    dead_indices = []
    dead_corrections = []
    analytical_pre_rows = []
    analytical_rows = []
    alpha_rows = []
    stats = {
        "pre30_dead_probed": 0,
        "pre30_dead_promoted": 0,
        "l30_on_probed": 0,
        "l30_on_demoted": 0,
        "l30_on_clip_cols": 0,
        "l30_dead_probed": 0,
        "l30_dead_promoted": 0,
        "l31_on_demoted": 0,
        "final_on_blend_cols": 0,
        "final_sign_bootstrap_wall_s": 0.0,
        "final_raw_numpy_wall_s": 0.0,
        "final_tail_quant_wall_s": 0.0,
        "final_tail_quant_units": 0.0,
        "final_tail_quant_dense": 0,
        "skip_on_relu_cols": 0,
        "final_on_sample_cols": 0,
        "n_samples": n_samples,
        "dynamic_bucket": 1,
        "dynamic_score": 0.0,
    }
    anal_mu_post = fnp.zeros(width)
    anal_var_post = fnp.zeros(width)

    for layer_idx, w in enumerate(mlp.weights):
        if layer_idx == 0:
            var_pre = fnp.sum(w * w, axis=0)
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            mu_pre = fnp.zeros(width)
            alpha = mu_pre / sigma_pre
        else:
            mu_pre = w.T @ anal_mu_post
            var_pre = fnp.sum(w * w * anal_var_post[:, None], axis=0)
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre

        analytical_pre_rows.append(mu_pre)
        dead_mask = alpha < variant.dead_thresh
        on_mask = alpha > variant.on_thresh
        kink_mask = (~dead_mask) & (~on_mask)

        dead_idx = fnp.nonzero(dead_mask)[0]
        alpha_rows.append(alpha)
        dead_indices.append(dead_idx)
        active_indices.append(fnp.nonzero(~dead_mask)[0])
        kink_indices.append(fnp.nonzero(kink_mask)[0])
        on_indices.append(fnp.nonzero(on_mask)[0])

        phi = flops.stats.norm.pdf(alpha)
        Phi = flops.stats.norm.cdf(alpha)
        anal_mu_post = mu_pre * Phi + sigma_pre * phi
        anal_var_post = (var_pre + mu_pre * mu_pre) * Phi + mu_pre * sigma_pre * phi - anal_mu_post * anal_mu_post
        anal_var_post = fnp.maximum(anal_var_post, 1e-12)
        analytical_rows.append(anal_mu_post)

        if len(dead_idx) > 0:
            dead_corrections.append(_scatter(anal_mu_post[dead_idx], dead_idx, width) * variant.dead_scale)
        else:
            dead_corrections.append(fnp.zeros(width))

    if variant.dynamic_sample_mode:
        if variant.dynamic_sample_mode == "anal_var_q":
            dynamic_score = float(np.mean(np.asarray(anal_var_post)))
            if dynamic_score < 0.0158787:
                n_samples = min(N_SAMPLES, variant.dynamic_easy_samples)
                stats["dynamic_bucket"] = 0
            elif dynamic_score >= 0.0299710:
                n_samples = min(N_SAMPLES, variant.dynamic_hard_samples)
                stats["dynamic_bucket"] = 2
            else:
                n_samples = min(N_SAMPLES, variant.dynamic_mid_samples)
                stats["dynamic_bucket"] = 1
            if n_samples % 2 != 0:
                n_samples -= 1
            n_samples = max(2, n_samples)
            n_pairs = n_samples // 2
            extra_pairs = N_SAMPLES // 2 - n_pairs
            stats["dynamic_score"] = dynamic_score
            stats["n_samples"] = n_samples
        elif variant.dynamic_sample_mode == "anal_var_sqrt":
            dynamic_score = float(np.mean(np.asarray(anal_var_post)))
            scaled = float(variant.dynamic_anchor_samples) * (dynamic_score / max(variant.dynamic_var_ref, 1e-30)) ** 0.5
            n_samples = int(round(scaled / 2.0) * 2)
            n_samples = min(N_SAMPLES, max(variant.dynamic_easy_samples, min(variant.dynamic_hard_samples, n_samples)))
            if n_samples % 2 != 0:
                n_samples -= 1
            n_samples = max(2, n_samples)
            n_pairs = n_samples // 2
            extra_pairs = N_SAMPLES // 2 - n_pairs
            stats["dynamic_score"] = dynamic_score
            stats["n_samples"] = n_samples
            if n_samples < 40960:
                stats["dynamic_bucket"] = 0
            elif n_samples > 40960:
                stats["dynamic_bucket"] = 2
            else:
                stats["dynamic_bucket"] = 1
        else:
            raise ValueError(f"unknown dynamic_sample_mode={variant.dynamic_sample_mode!r}")

    if variant.sample_mode == "plain_sobol":
        if sobol_points.shape[0] < n_samples:
            raise ValueError(f"plain_sobol needs {n_samples} points, got {sobol_points.shape[0]}")
        half = fnp.array(sobol_points[:n_samples, :width])
    else:
        half = fnp.array(sobol_points[:n_pairs, :width])
    if variant.rotate_w0:
        # Gaussian rotation invariance: align the first (highest-quality) Sobol
        # coordinates with W0's top left-singular directions. SVD runs outside
        # flopscope on raw NumPy; in a submission this cost would need tracking.
        u_rot, _, _ = np.linalg.svd(np.asarray(mlp.weights[0], dtype=np.float64))
        half = half @ fnp.array(u_rot.T.astype(np.float32))
    should_rotate = bool(variant.rotation_mode)
    if variant.rotation_max_samples > 0 and n_samples > variant.rotation_max_samples:
        should_rotate = False
    if variant.rotation_var_max > 0.0 and float(np.mean(np.asarray(anal_var_post))) > variant.rotation_var_max:
        should_rotate = False
    if should_rotate:
        rot = _rotation_basis(mlp, alpha_rows, variant.rotation_mode)
        half = half @ fnp.array(rot.T.astype(np.float32))
        stats["rotation_applied"] = 1
    else:
        stats["rotation_applied"] = 0
    sample_scale = fnp.float32(1.0)
    if variant.sample_mode == "sphere":
        norm = fnp.sqrt(fnp.maximum(fnp.sum(half * half, axis=1, keepdims=True), 1e-12))
        half = half / norm
        sample_scale = fnp.float32(_chi_radius_mean(width))
    elif variant.sample_mode not in {"normal", "plain_sobol"}:
        raise ValueError(f"unknown sample_mode={variant.sample_mode!r}")

    if variant.sample_mode == "plain_sobol":
        x_main = half
    else:
        x_main = fnp.concatenate([half, -half], axis=0)
    mc_rows = []
    mc_vars = []
    mc_var_of_mean = []

    prev_idx = None
    x_before_fold = None

    for layer_idx, w in enumerate(mlp.weights):
        idx = active_indices[layer_idx]
        kink_idx = kink_indices[layer_idx]
        on_idx = on_indices[layer_idx]
        k_active = len(idx)
        k_on = len(on_idx)

        if k_active == 0:
            mc_rows.append(fnp.zeros(width))
            mc_vars.append(fnp.zeros(width))
            mc_var_of_mean.append(fnp.ones(width) * 1e-12)
            x_main = fnp.zeros((n_samples, 0))
            prev_idx = idx
            x_before_fold = None
            continue

        if layer_idx == 30 and k_on > 0 and prev_idx is not None:
            x_before_fold = x_main
            on_clip_np = np.zeros(0, dtype=np.int64)
            on_clip_values = fnp.zeros(0)
            if variant.refine_layer30_on or variant.refine_layer30_dead:
                pilot_x = _pilot_samples(
                    x_main,
                    n_samples,
                    variant.refine_pilot_fraction,
                    variant.paired_pilot or variant.pair_tail_refine,
                )

            if variant.refine_layer30_on:
                on_np = np.asarray(on_idx, dtype=np.int64)
                kink_np = np.asarray(kink_idx, dtype=np.int64)
                if variant.refine_borderline_only:
                    alpha_np = np.asarray(alpha_rows[layer_idx])
                    probe_mask = alpha_np[on_np] <= variant.refine_on_probe_max
                else:
                    probe_mask = np.ones(on_np.shape, dtype=bool)

                probe_on_np = on_np[probe_mask]
                trusted_on_np = on_np[~probe_mask]
                stats["l30_on_probed"] = int(probe_on_np.size)

                if probe_on_np.size > 0:
                    probe_on_idx = fnp.array(probe_on_np.astype(np.int64))
                    w_on_probe = w[prev_idx, :][:, probe_on_idx]
                    pre_on_pilot = pilot_x @ w_on_probe
                    pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                    pilot_var = fnp.var(pre_on_pilot, axis=0)
                    pilot_scale = fnp.sqrt(fnp.maximum(pilot_var, 1e-12))
                    if variant.l30_on_clip_scale > 0.0:
                        pilot_clip = fnp.mean(fnp.maximum(-pre_on_pilot, 0.0), axis=0)

                    if variant.pair_tail_refine:
                        neg_tail = fnp.mean(fnp.maximum(-pre_on_pilot, 0.0), axis=0) / pilot_scale
                        keep_probe_on = np.asarray(neg_tail) <= variant.pair_on_neg_tail
                    else:
                        pilot_alpha = pilot_mean / pilot_scale
                        keep_probe_on = np.asarray(pilot_alpha) > variant.refine_on_thresh
                    demoted_np = probe_on_np[~keep_probe_on]
                    kept_probe_on_np = probe_on_np[keep_probe_on]
                    if variant.l30_on_clip_scale > 0.0:
                        on_clip_np = kept_probe_on_np
                        on_clip_values = pilot_clip[keep_probe_on]
                else:
                    demoted_np = np.zeros(0, dtype=np.int64)
                    kept_probe_on_np = np.zeros(0, dtype=np.int64)

                stats["l30_on_demoted"] = int(demoted_np.size)
                stats["l30_on_clip_cols"] = int(on_clip_np.size)
                on_idx = fnp.array(np.sort(np.concatenate([trusted_on_np, kept_probe_on_np])).astype(np.int64))
                kink_idx = fnp.array(np.sort(np.concatenate([kink_np, demoted_np])).astype(np.int64))
                on_indices[layer_idx] = on_idx
                kink_indices[layer_idx] = kink_idx
                k_on = len(on_idx)

            if variant.refine_layer30_dead:
                dead_idx = dead_indices[layer_idx]
                if len(dead_idx) > 0:
                    dead_np = np.asarray(dead_idx, dtype=np.int64)
                    kink_np = np.asarray(kink_idx, dtype=np.int64)
                    on_np = np.asarray(on_idx, dtype=np.int64)
                    if variant.refine_borderline_only:
                        alpha_np = np.asarray(alpha_rows[layer_idx])
                        probe_mask = alpha_np[dead_np] >= variant.refine_dead_probe_min
                    else:
                        probe_mask = np.ones(dead_np.shape, dtype=bool)

                    probe_dead_np = dead_np[probe_mask]
                    trusted_dead_np = dead_np[~probe_mask]
                    stats["l30_dead_probed"] = int(probe_dead_np.size)

                    if probe_dead_np.size > 0:
                        probe_dead_idx = fnp.array(probe_dead_np.astype(np.int64))
                        w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                        pre_dead_pilot = pilot_x @ w_dead_probe
                        pilot_mean = fnp.mean(pre_dead_pilot, axis=0)
                        pilot_var = fnp.var(pre_dead_pilot, axis=0)
                        pilot_scale = fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                        if variant.pair_tail_refine:
                            pos_tail = fnp.mean(fnp.maximum(pre_dead_pilot, 0.0), axis=0) / pilot_scale
                            promote_dead = np.asarray(pos_tail) >= variant.pair_dead_pos_tail
                        else:
                            pilot_alpha = pilot_mean / pilot_scale
                            promote_dead = np.asarray(pilot_alpha) > variant.refine_dead_thresh
                        promoted_np = probe_dead_np[promote_dead]
                        remaining_probe_dead_np = probe_dead_np[~promote_dead]
                    else:
                        promoted_np = np.zeros(0, dtype=np.int64)
                        remaining_probe_dead_np = np.zeros(0, dtype=np.int64)

                    remaining_dead_np = np.sort(np.concatenate([trusted_dead_np, remaining_probe_dead_np]))
                    stats["l30_dead_promoted"] = int(promoted_np.size)
                    if promoted_np.size > 0:
                        promoted_idx = fnp.array(promoted_np.astype(np.int64))
                        dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                            analytical_rows[layer_idx][promoted_idx] * variant.dead_scale,
                            promoted_idx,
                            width,
                        )
                    dead_idx = fnp.array(remaining_dead_np.astype(np.int64))
                    kink_idx = fnp.array(np.sort(np.concatenate([kink_np, promoted_np])).astype(np.int64))
                    idx = fnp.array(np.sort(np.concatenate([np.asarray(kink_idx, dtype=np.int64), on_np])).astype(np.int64))
                    dead_indices[layer_idx] = dead_idx
                    kink_indices[layer_idx] = kink_idx
                    active_indices[layer_idx] = idx
                    k_active = len(idx)

            w_kink = w[prev_idx, :][:, kink_idx]
            x_kink = fnp.maximum(x_main @ w_kink, 0.0)

            kink_mean = fnp.mean(x_kink, axis=0)
            kink_var = fnp.var(x_kink, axis=0)
            mean_prev = fnp.mean(x_main, axis=0)
            var_prev_mc = fnp.var(x_main, axis=0)
            w_on = w[prev_idx, :][:, on_idx]
            on_mean = mean_prev @ w_on
            on_var = fnp.sum(w_on * w_on * var_prev_mc[:, None], axis=0)

            row = _scatter(kink_mean, kink_idx, width) + _scatter(on_mean, on_idx, width)
            if variant.l30_on_clip_scale > 0.0 and on_clip_np.size > 0:
                on_clip_idx = fnp.array(on_clip_np.astype(np.int64))
                row = row + _scatter(on_clip_values * fnp.float32(variant.l30_on_clip_scale), on_clip_idx, width)
            mc_rows.append(row)

            full_var = _scatter(kink_var, kink_idx, width) + _scatter(on_var, on_idx, width)
            mc_vars.append(full_var)

            full_vom = _scatter(kink_var / n_samples, kink_idx, width) + _scatter(on_var / n_samples, on_idx, width)
            mc_var_of_mean.append(full_vom)

            x_main = x_kink
            prev_idx = kink_idx
            continue

        if layer_idx == 31 and x_before_fold is not None and len(on_indices[30]) > 0:
            fold_on_idx = on_indices[30]
            fold_prev_idx = active_indices[29]

            this_kink_idx = kink_idx
            this_on_idx = on_idx

            if variant.final_on_thresh is not None and len(this_on_idx) > 0:
                on_np = np.asarray(this_on_idx, dtype=np.int64)
                kink_np = np.asarray(this_kink_idx, dtype=np.int64)
                alpha_np = np.asarray(alpha_rows[layer_idx])
                keep_on = alpha_np[on_np] > variant.final_on_thresh
                demoted_np = on_np[~keep_on]
                this_on_idx = fnp.array(on_np[keep_on].astype(np.int64))
                this_kink_idx = fnp.array(np.sort(np.concatenate([kink_np, demoted_np])).astype(np.int64))
                active_indices[layer_idx] = fnp.array(
                    np.sort(
                        np.concatenate(
                            [np.asarray(this_kink_idx, dtype=np.int64), np.asarray(this_on_idx, dtype=np.int64)]
                        )
                    ).astype(np.int64)
                )
                on_indices[layer_idx] = this_on_idx
                kink_indices[layer_idx] = this_kink_idx
                stats["l31_on_demoted"] = int(demoted_np.size)

            if variant.refine_layer31_on and len(this_on_idx) > 0:
                pilot_x = _pilot_samples(x_main, n_samples, variant.refine_pilot_fraction, variant.paired_pilot)
                pilot_x_before_fold = _pilot_samples(
                    x_before_fold, n_samples, variant.refine_pilot_fraction, variant.paired_pilot
                )
                w_fold_layer = mlp.weights[30]
                w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]

                w_from_kink_on = w[prev_idx, :][:, this_on_idx]
                pre_from_kink_on = pilot_x @ w_from_kink_on
                w_this_from_on_on = w[fold_on_idx, :][:, this_on_idx]
                w_folded_on = w_fold_on @ w_this_from_on_on
                pre_from_on_on = pilot_x_before_fold @ w_folded_on
                pre_on_pilot = pre_from_kink_on + pre_from_on_on

                pilot_mean = fnp.mean(pre_on_pilot, axis=0)
                pilot_var = fnp.var(pre_on_pilot, axis=0)
                pilot_alpha = pilot_mean / fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                on_np = np.asarray(this_on_idx, dtype=np.int64)
                kink_np = np.asarray(this_kink_idx, dtype=np.int64)
                keep_on = np.asarray(pilot_alpha) > variant.refine_on_thresh
                demoted_np = on_np[~keep_on]
                this_on_idx = fnp.array(on_np[keep_on].astype(np.int64))
                this_kink_idx = fnp.array(np.sort(np.concatenate([kink_np, demoted_np])).astype(np.int64))
                on_indices[layer_idx] = this_on_idx
                kink_indices[layer_idx] = this_kink_idx

            w_from_kink = w[prev_idx, :][:, this_kink_idx]

            w_fold_layer = mlp.weights[30]
            w_fold_on = w_fold_layer[fold_prev_idx, :][:, fold_on_idx]
            w_this_from_on = w[fold_on_idx, :][:, this_kink_idx]
            w_folded = w_fold_on @ w_this_from_on

            if variant.final_raw_numpy:
                raw_start = time.perf_counter()
                raw_stats = _raw_numpy_relu_stats((x_main, x_before_fold), (w_from_kink, w_folded))
                stats["final_raw_numpy_wall_s"] += time.perf_counter() - raw_start
                if raw_stats is None:
                    x_kink_this = None
                    kink_mean = fnp.zeros(len(this_kink_idx))
                    kink_var = fnp.zeros(len(this_kink_idx))
                else:
                    raw_mean, raw_var = raw_stats
                    x_kink_this = None
                    kink_mean = fnp.array(raw_mean)
                    kink_var = fnp.array(raw_var)
            elif variant.final_tail_quant_keep_top > 0 and variant.final_tail_quant_bins > 0:
                quant_start = time.perf_counter()
                quant_stats = _tail_quantized_relu_stats(
                    (x_main, x_before_fold),
                    (w_from_kink, w_folded),
                    variant.final_tail_quant_keep_top,
                    variant.final_tail_quant_bins,
                )
                stats["final_tail_quant_wall_s"] += time.perf_counter() - quant_start
                if quant_stats is None:
                    x_kink_this = None
                    kink_mean = fnp.zeros(len(this_kink_idx))
                    kink_var = fnp.zeros(len(this_kink_idx))
                else:
                    quant_mean, quant_var, quant_units, quant_dense = quant_stats
                    x_kink_this = None
                    kink_mean = fnp.array(quant_mean)
                    kink_var = fnp.array(quant_var)
                    stats["final_tail_quant_units"] = quant_units
                    stats["final_tail_quant_dense"] = quant_dense
            else:
                pre_from_kink = x_main @ w_from_kink
                pre_from_on = x_before_fold @ w_folded

                pre_kink_this = pre_from_kink + pre_from_on
                if variant.final_mirror_mode:
                    if variant.final_mirror_mode == "sample":
                        mirror_center = fnp.mean(pre_kink_this, axis=0)
                    elif variant.final_mirror_mode == "analytical":
                        mirror_center = analytical_pre_rows[layer_idx][this_kink_idx]
                    else:
                        raise ValueError(f"unknown final_mirror_mode={variant.final_mirror_mode!r}")
                    mirror_pre = fnp.float32(2.0) * mirror_center - pre_kink_this
                    x_kink_this = fnp.float32(0.5) * (
                        fnp.maximum(pre_kink_this, 0.0) + fnp.maximum(mirror_pre, 0.0)
                    )
                else:
                    x_kink_this = fnp.maximum(pre_kink_this, 0.0)
                kink_mean = fnp.mean(x_kink_this, axis=0)
                kink_var = fnp.var(x_kink_this, axis=0)

            if variant.final_sign_bootstrap_rows > 0 and variant.final_sign_bootstrap_blend > 0.0:
                boot_start = time.perf_counter()
                boot_mean = _sign_bootstrap_relu_mean(
                    (x_main, x_before_fold),
                    (w_from_kink, w_folded),
                    variant.final_sign_bootstrap_rows,
                    seed=31,
                    center_mode=variant.final_sign_bootstrap_center,
                )
                stats["final_sign_bootstrap_wall_s"] += time.perf_counter() - boot_start
                if boot_mean is not None:
                    blend = fnp.float32(variant.final_sign_bootstrap_blend)
                    kink_mean = (fnp.float32(1.0) - blend) * kink_mean + blend * fnp.array(boot_mean)

            prev_layer_mean = mc_rows[30]
            fold_active_idx = active_indices[30]
            w_to_on = w[fold_active_idx, :][:, this_on_idx]
            on_mean = prev_layer_mean[fold_active_idx] @ w_to_on
            prev_layer_var = mc_vars[30]
            on_var = fnp.sum(w_to_on * w_to_on * prev_layer_var[fold_active_idx, None], axis=0)

            if variant.final_on_sample_correction_mode and variant.final_on_sample_correction_blend > 0.0 and len(this_on_idx) > 0:
                on_np = np.asarray(this_on_idx, dtype=np.int64)
                if variant.final_on_sample_correction_mode == "all":
                    selected_np = on_np
                elif variant.final_on_sample_correction_mode == "top_alpha":
                    alpha_np = np.asarray(alpha_rows[layer_idx], dtype=np.float64)
                    count = max(1, int(round(variant.final_on_sample_correction_keep * on_np.size)))
                    selected_np = on_np[np.argpartition(alpha_np[on_np], -count)[-count:]]
                elif variant.final_on_sample_correction_mode == "top_anal_diff":
                    current_np = np.asarray(on_mean, dtype=np.float64)
                    analytical_np = np.asarray(analytical_rows[layer_idx], dtype=np.float64)
                    score = np.abs(current_np - analytical_np[on_np])
                    count = max(1, int(round(variant.final_on_sample_correction_keep * on_np.size)))
                    selected_np = on_np[np.argpartition(score, -count)[-count:]]
                elif variant.final_on_sample_correction_mode == "top_active":
                    current_np = np.asarray(on_mean, dtype=np.float64)
                    count = max(1, int(round(variant.final_on_sample_correction_keep * on_np.size)))
                    selected_np = on_np[np.argpartition(np.abs(current_np), -count)[-count:]]
                else:
                    raise ValueError(f"unknown final_on_sample_correction_mode={variant.final_on_sample_correction_mode!r}")

                if selected_np.size > 0:
                    selected_pos_np = np.flatnonzero(np.isin(on_np, selected_np)).astype(np.int64)
                    selected_np = on_np[selected_pos_np]
                    selected_idx = fnp.array(selected_np.astype(np.int64))
                    selected_pos = fnp.array(selected_pos_np)
                    w_from_kink_on = w[prev_idx, :][:, selected_idx]
                    pre_from_kink_on = x_main @ w_from_kink_on
                    w_this_from_on_on = w[fold_on_idx, :][:, selected_idx]
                    w_folded_on = w_fold_on @ w_this_from_on_on
                    pre_from_on_on = x_before_fold @ w_folded_on
                    sampled_on_mean = fnp.mean(fnp.maximum(pre_from_kink_on + pre_from_on_on, 0.0), axis=0)
                    blend = fnp.float32(variant.final_on_sample_correction_blend)
                    on_delta = (sampled_on_mean - on_mean[selected_pos]) * blend
                    on_mean = on_mean + _scatter(on_delta, selected_pos, len(on_np))
                    stats["final_on_sample_cols"] = int(selected_np.size)

            row = _scatter(kink_mean, this_kink_idx, width) + _scatter(on_mean, this_on_idx, width)
            mc_rows.append(row)

            full_var = _scatter(kink_var, this_kink_idx, width) + _scatter(on_var, this_on_idx, width)
            mc_vars.append(full_var)

            full_vom = _scatter(kink_var / n_samples, this_kink_idx, width) + _scatter(on_var / n_samples, this_on_idx, width)
            mc_var_of_mean.append(full_vom)

            prev_idx = idx
            x_before_fold = None
            continue

        if layer_idx < 30 and ((layer_idx == 29 and variant.refine_layer29_dead) or layer_idx in variant.refine_dead_layers) and prev_idx is not None:
            dead_idx = dead_indices[layer_idx]
            if len(dead_idx) > 0:
                pilot_x = _pilot_samples(
                    x_main,
                    n_samples,
                    variant.refine_pilot_fraction,
                    variant.paired_pilot or variant.pair_tail_refine,
                )
                dead_np = np.asarray(dead_idx, dtype=np.int64)
                kink_np = np.asarray(kink_idx, dtype=np.int64)
                on_np = np.asarray(on_idx, dtype=np.int64)
                if variant.refine_borderline_only:
                    alpha_np = np.asarray(alpha_rows[layer_idx])
                    probe_mask = alpha_np[dead_np] >= variant.refine_dead_probe_min
                else:
                    probe_mask = np.ones(dead_np.shape, dtype=bool)

                probe_dead_np = dead_np[probe_mask]
                trusted_dead_np = dead_np[~probe_mask]
                stats["pre30_dead_probed"] += int(probe_dead_np.size)

                if probe_dead_np.size > 0:
                    probe_dead_idx = fnp.array(probe_dead_np.astype(np.int64))
                    w_dead_probe = w[prev_idx, :][:, probe_dead_idx]
                    pre_dead_pilot = pilot_x @ w_dead_probe
                    pilot_mean = fnp.mean(pre_dead_pilot, axis=0)
                    pilot_var = fnp.var(pre_dead_pilot, axis=0)
                    pilot_scale = fnp.sqrt(fnp.maximum(pilot_var, 1e-12))

                    if variant.pair_tail_refine:
                        pos_tail = fnp.mean(fnp.maximum(pre_dead_pilot, 0.0), axis=0) / pilot_scale
                        promote_dead = np.asarray(pos_tail) >= variant.pair_dead_pos_tail
                    else:
                        pilot_alpha = pilot_mean / pilot_scale
                        promote_dead = np.asarray(pilot_alpha) > variant.refine_dead_thresh
                    promoted_np = probe_dead_np[promote_dead]
                    remaining_probe_dead_np = probe_dead_np[~promote_dead]
                else:
                    promoted_np = np.zeros(0, dtype=np.int64)
                    remaining_probe_dead_np = np.zeros(0, dtype=np.int64)

                remaining_dead_np = np.sort(np.concatenate([trusted_dead_np, remaining_probe_dead_np]))
                stats["pre30_dead_promoted"] += int(promoted_np.size)
                if promoted_np.size > 0:
                    promoted_idx = fnp.array(promoted_np.astype(np.int64))
                    dead_corrections[layer_idx] = dead_corrections[layer_idx] - _scatter(
                        analytical_rows[layer_idx][promoted_idx] * variant.dead_scale,
                        promoted_idx,
                        width,
                    )
                dead_idx = fnp.array(remaining_dead_np.astype(np.int64))
                kink_idx = fnp.array(np.sort(np.concatenate([kink_np, promoted_np])).astype(np.int64))
                idx = fnp.array(np.sort(np.concatenate([np.asarray(kink_idx, dtype=np.int64), on_np])).astype(np.int64))
                dead_indices[layer_idx] = dead_idx
                kink_indices[layer_idx] = kink_idx
                active_indices[layer_idx] = idx
                k_active = len(idx)

        if prev_idx is None:
            w_active = w[:, idx]
        else:
            w_active = w[prev_idx, :][:, idx]

        skip_on_np = np.zeros(0, dtype=np.int64)
        if (
            variant.skip_on_relu_alpha > 0.0
            and prev_idx is not None
            and variant.skip_on_relu_from_layer <= layer_idx < 30
            and len(on_idx) > 0
        ):
            on_np = np.asarray(on_idx, dtype=np.int64)
            alpha_np = np.asarray(alpha_rows[layer_idx])
            skip_on_np = on_np[alpha_np[on_np] >= variant.skip_on_relu_alpha]

        if skip_on_np.size > 0:
            active_np = np.asarray(idx, dtype=np.int64)
            relu_np = active_np[~np.isin(active_np, skip_on_np)]
            skip_on_idx = fnp.array(skip_on_np.astype(np.int64))
            stats["skip_on_relu_cols"] += int(skip_on_np.size)

            if prev_idx is None:
                w_skip = w[:, skip_on_idx]
            else:
                w_skip = w[prev_idx, :][:, skip_on_idx]
            x_skip = x_main @ w_skip

            if relu_np.size > 0:
                relu_idx = fnp.array(relu_np.astype(np.int64))
                if prev_idx is None:
                    w_relu = w[:, relu_idx]
                else:
                    w_relu = w[prev_idx, :][:, relu_idx]
                x_relu = fnp.maximum(x_main @ w_relu, 0.0)
                x_main = fnp.concatenate([x_relu, x_skip], axis=1)
                idx = fnp.concatenate([relu_idx, skip_on_idx])
            else:
                x_main = x_skip
                idx = skip_on_idx
            active_indices[layer_idx] = idx
            active_mean = fnp.mean(x_main, axis=0)
        else:
            pre_active = x_main @ w_active
            if variant.final_mirror_mode and layer_idx == mlp.depth - 1:
                if variant.final_mirror_mode == "sample":
                    mirror_center = fnp.mean(pre_active, axis=0)
                elif variant.final_mirror_mode == "analytical":
                    mirror_center = analytical_pre_rows[layer_idx][idx]
                else:
                    raise ValueError(f"unknown final_mirror_mode={variant.final_mirror_mode!r}")
                mirror_pre = fnp.float32(2.0) * mirror_center - pre_active
                x_main = fnp.float32(0.5) * (fnp.maximum(pre_active, 0.0) + fnp.maximum(mirror_pre, 0.0))
            else:
                x_main = fnp.maximum(pre_active, 0.0)
            active_mean = fnp.mean(x_main, axis=0)
        active_var = fnp.var(x_main, axis=0)

        if layer_idx == mlp.depth - 1 and variant.final_sign_bootstrap_rows > 0 and variant.final_sign_bootstrap_blend > 0.0:
            boot_start = time.perf_counter()
            boot_mean = _sign_bootstrap_relu_mean(
                (x_main,),
                (w_active,),
                variant.final_sign_bootstrap_rows,
                seed=layer_idx,
                center_mode=variant.final_sign_bootstrap_center,
            )
            stats["final_sign_bootstrap_wall_s"] += time.perf_counter() - boot_start
            if boot_mean is not None:
                blend = fnp.float32(variant.final_sign_bootstrap_blend)
                active_mean = (fnp.float32(1.0) - blend) * active_mean + blend * fnp.array(boot_mean)

        mc_rows.append(_scatter(active_mean, idx, width))
        mc_vars.append(_scatter(active_var, idx, width))
        mc_var_of_mean.append(_scatter(active_var / n_samples, idx, width))

        prev_idx = idx

    w0 = mlp.weights[0]
    var_pre_0 = fnp.sum(w0 * w0, axis=0)
    sigma_pre_0 = fnp.sqrt(fnp.maximum(var_pre_0, 1e-12))
    row0 = sigma_pre_0 * fnp.float32(0.3989422804014327)

    if variant.sample_mode == "sphere":
        rows = [row0 + dead_corrections[0]] + [sample_scale * mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))]
    else:
        rows = [row0 + dead_corrections[0]] + [mc_rows[layer] + dead_corrections[layer] for layer in range(1, len(mc_rows))]

    if variant.allocation_top_k > 0:
        rows[-1] = _selected_final_update(
            mlp,
            sobol_points,
            n_pairs,
            extra_pairs,
            active_indices,
            kink_indices,
            rows[-1],
            mc_vars[-1],
            n_samples,
            variant.allocation_top_k,
            width,
        )

    if variant.final_on_blend > 0.0:
        final_on_np = np.asarray(on_indices[-1], dtype=np.int64)
        selected_np = np.zeros(0, dtype=np.int64)
        if final_on_np.size > 0:
            if variant.final_on_blend_mode == "alpha6":
                final_alpha = np.asarray(alpha_rows[-1])
                selected_np = final_on_np[final_alpha[final_on_np] >= 6.0]
            elif variant.final_on_blend_mode == "pred90":
                final_row = np.asarray(rows[-1])
                cutoff = np.quantile(final_row[final_on_np], 0.90)
                selected_np = final_on_np[final_row[final_on_np] >= cutoff]
            else:
                raise ValueError(f"unknown final_on_blend_mode={variant.final_on_blend_mode!r}")

        stats["final_on_blend_cols"] = int(selected_np.size)
        if selected_np.size > 0:
            selected_idx = fnp.array(selected_np.astype(np.int64))
            blend = fnp.float32(variant.final_on_blend)
            rows[-1] = rows[-1] + _scatter(
                (analytical_rows[-1][selected_idx] - rows[-1][selected_idx]) * blend,
                selected_idx,
                width,
            )

    if variant.final_analytical_blend > 0.0:
        blend = fnp.float32(variant.final_analytical_blend)
        rows[-1] = (1.0 - blend) * rows[-1] + blend * analytical_rows[-1]

    if variant.final_on_signed_scale > 0.0:
        final_on_np = np.asarray(on_indices[-1], dtype=np.int64)
        if final_on_np.size > 0:
            final_on_idx = fnp.array(final_on_np.astype(np.int64))
            if variant.final_on_signed_scale_mode == "pred_minus_anal":
                proxy = fnp.mean(rows[-1][final_on_idx] - analytical_rows[-1][final_on_idx])
            elif variant.final_on_signed_scale_mode == "anal_minus_pred":
                proxy = fnp.mean(analytical_rows[-1][final_on_idx] - rows[-1][final_on_idx])
            else:
                raise ValueError(f"unknown final_on_signed_scale_mode={variant.final_on_signed_scale_mode!r}")
            sign = fnp.where(proxy >= 0.0, fnp.float32(1.0), fnp.float32(-1.0))
            scale = fnp.float32(variant.final_on_signed_scale) * sign
            rows[-1] = rows[-1] + _scatter(rows[-1][final_on_idx] * scale, final_on_idx, width)

    final_row_np = np.asarray(rows[-1], dtype=np.float64)
    final_var_np = np.asarray(mc_vars[-1], dtype=np.float64)
    final_vom_np = np.asarray(mc_var_of_mean[-1], dtype=np.float64)
    final_analytical_np = np.asarray(analytical_rows[-1], dtype=np.float64)
    l30_row_np = np.asarray(rows[30], dtype=np.float64) if len(rows) > 30 else final_row_np
    l30_var_np = np.asarray(mc_vars[30], dtype=np.float64) if len(mc_vars) > 30 else final_var_np
    l30_vom_np = np.asarray(mc_var_of_mean[30], dtype=np.float64) if len(mc_var_of_mean) > 30 else final_vom_np
    l30_analytical_np = np.asarray(analytical_rows[30], dtype=np.float64) if len(analytical_rows) > 30 else final_analytical_np
    final_diff_np = np.abs(final_row_np - final_analytical_np)
    l30_diff_np = np.abs(l30_row_np - l30_analytical_np)
    final_on_np = np.asarray(on_indices[-1], dtype=np.int64)
    final_kink_np = np.asarray(kink_indices[-1], dtype=np.int64)
    stats.update(
        {
            "base_final_var_mean": float(np.mean(final_var_np)),
            "base_final_var_max": float(np.max(final_var_np)),
            "base_final_vom_mean": float(np.mean(final_vom_np)),
            "base_final_vom_max": float(np.max(final_vom_np)),
            "base_l30_var_mean": float(np.mean(l30_var_np)),
            "base_l30_var_max": float(np.max(l30_var_np)),
            "base_l30_vom_mean": float(np.mean(l30_vom_np)),
            "base_l30_vom_max": float(np.max(l30_vom_np)),
            "base_final_anal_diff_mean": float(np.mean(final_diff_np)),
            "base_final_anal_diff_max": float(np.max(final_diff_np)),
            "base_l30_anal_diff_mean": float(np.mean(l30_diff_np)),
            "base_l30_anal_diff_max": float(np.max(l30_diff_np)),
        }
    )
    if final_on_np.size > 0:
        stats["base_final_on_vom_mean"] = float(np.mean(final_vom_np[final_on_np]))
        stats["base_final_on_anal_diff_mean"] = float(np.mean(final_diff_np[final_on_np]))
    if final_kink_np.size > 0:
        stats["base_final_kink_vom_mean"] = float(np.mean(final_vom_np[final_kink_np]))
        stats["base_final_kink_anal_diff_mean"] = float(np.mean(final_diff_np[final_kink_np]))

    predict_variant.last_stats = stats
    return fnp.stack(rows, axis=0)


predict_variant.last_stats = {}


def evaluate():
    sobol_points = np.load(SOBOL_POINTS_PATH)["points"]
    if sobol_points.shape[0] < N_SAMPLES // 2:
        raise SystemExit(f"sobol_points.npz only has {sobol_points.shape[0]} half-samples")

    print(f"Conservative {N_SAMPLES}-sample ablation on seeds={SEEDS}, GT={GT_SAMPLES:,}")
    print(f"{'variant':<32} {'final_mse':>12} {'score':>12} {'flops':>12} {'util%':>7} {'samples':>8} {'dyn':>5} {'rot':>5} {'all_mse':>12} {'pre30d':>7} {'on->k':>7} {'l30d':>6} {'clip':>6} {'fblend':>7} {'l31on':>7} {'probe':>9}")
    print("-" * 172)

    cases = []
    for seed in SEEDS:
        mlp = build_mlp(width=WIDTH, depth=DEPTH, seed=seed)
        gt = np.asarray(monte_carlo_layer_means(mlp, GT_SAMPLES, seed=seed + 10_000))
        cases.append((seed, mlp, gt))

    variants = VARIANTS
    if VARIANT_FILTER:
        variants = tuple(v for v in VARIANTS if any(s in v.name for s in VARIANT_FILTER))

    results = []
    for variant in variants:
        adjusted_scores = []
        final_mses = []
        all_mses = []
        flops_used = []
        pre30_dead_promoted = []
        on_demoted = []
        dead_promoted = []
        on_clip_cols = []
        final_on_blend_cols = []
        final_on_demoted = []
        sample_counts = []
        dynamic_buckets = []
        rotation_applied = []
        probed_total = []

        for _seed, mlp, gt in cases:
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
                pred = np.asarray(predict_variant(mlp, sobol_points, variant))
            stats = predict_variant.last_stats

            err = pred - gt
            mlp_final_mse = float(np.mean(err[-1] * err[-1]))
            final_mses.append(mlp_final_mse)
            all_mses.append(float(np.mean(err * err)))
            flops_used.append(ctx.flops_used)
            adjusted_scores.append(mlp_final_mse * max(0.1, ctx.flops_used / BUDGET))
            pre30_dead_promoted.append(stats.get("pre30_dead_promoted", 0))
            on_demoted.append(stats.get("l30_on_demoted", 0))
            dead_promoted.append(stats.get("l30_dead_promoted", 0))
            on_clip_cols.append(stats.get("l30_on_clip_cols", 0))
            final_on_blend_cols.append(stats.get("final_on_blend_cols", 0))
            final_on_demoted.append(stats.get("l31_on_demoted", 0))
            sample_counts.append(stats.get("n_samples", N_SAMPLES))
            dynamic_buckets.append(stats.get("dynamic_bucket", 1))
            rotation_applied.append(stats.get("rotation_applied", 0))
            probed_total.append(stats.get("pre30_dead_probed", 0) + stats.get("l30_on_probed", 0) + stats.get("l30_dead_probed", 0))

        final_mse = float(np.mean(final_mses))
        all_mse = float(np.mean(all_mses))
        flops_mean = float(np.mean(flops_used))
        util = flops_mean / BUDGET
        score = float(np.mean(adjusted_scores))
        avg_pre30_dead_promoted = float(np.mean(pre30_dead_promoted))
        avg_on_demoted = float(np.mean(on_demoted))
        avg_dead_promoted = float(np.mean(dead_promoted))
        avg_on_clip_cols = float(np.mean(on_clip_cols))
        avg_final_on_blend_cols = float(np.mean(final_on_blend_cols))
        avg_final_on_demoted = float(np.mean(final_on_demoted))
        avg_sample_counts = float(np.mean(sample_counts))
        avg_dynamic_buckets = float(np.mean(dynamic_buckets))
        avg_rotation_applied = float(np.mean(rotation_applied))
        avg_probed_total = float(np.mean(probed_total))
        results.append((score, variant.name, final_mse, flops_mean, util, avg_sample_counts, avg_dynamic_buckets, avg_rotation_applied, all_mse, avg_pre30_dead_promoted, avg_on_demoted, avg_dead_promoted, avg_on_clip_cols, avg_final_on_blend_cols, avg_final_on_demoted, avg_probed_total))
        print(f"{variant.name:<32} {final_mse:12.3e} {score:12.3e} {flops_mean:12.2e} {util * 100:6.1f}% {avg_sample_counts:8.0f} {avg_dynamic_buckets:5.2f} {avg_rotation_applied:5.2f} {all_mse:12.3e} {avg_pre30_dead_promoted:7.1f} {avg_on_demoted:7.1f} {avg_dead_promoted:6.1f} {avg_on_clip_cols:6.1f} {avg_final_on_blend_cols:7.1f} {avg_final_on_demoted:7.1f} {avg_probed_total:9.1f}")

    print("\nRanked by adjusted score:")
    for rank, (score, name, final_mse, flops_mean, util, avg_sample_counts, avg_dynamic_buckets, avg_rotation_applied, all_mse, avg_pre30_dead_promoted, avg_on_demoted, avg_dead_promoted, avg_on_clip_cols, avg_final_on_blend_cols, avg_final_on_demoted, avg_probed_total) in enumerate(sorted(results), start=1):
        print(f"{rank:2d}. {name:<32} score={score:.3e} final_mse={final_mse:.3e} flops={flops_mean:.2e} util={util * 100:.1f}% samples={avg_sample_counts:.0f} dyn={avg_dynamic_buckets:.2f} rot={avg_rotation_applied:.2f} all_mse={all_mse:.3e} pre30d={avg_pre30_dead_promoted:.1f} on->k={avg_on_demoted:.1f} l30d={avg_dead_promoted:.1f} clip={avg_on_clip_cols:.1f} fblend={avg_final_on_blend_cols:.1f} l31on={avg_final_on_demoted:.1f} probed={avg_probed_total:.1f}")


if __name__ == "__main__":
    evaluate()
