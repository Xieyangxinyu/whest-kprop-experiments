"""Measure the billed-F headroom of the remaining exact micro-savings on the
shipped algo54 bytes (estimator.py = submissions 318978/319031).

Question: does the bundle (unassembled Strassen output + pilot-carve audit +
misc) clear the ~1% bar set by the measured grading-noise floor (~0.5%)?

Method: run the shipped estimator on mini nets 0,10,20,30,40,50 under
flopscope 0.9.1. Attribute every fnp.concatenate to its call-site by wrapping
the estimator's internal functions with a tag stack (strassen / carve /
antithetic / other). The strassen-tagged concat cost is the exact upper bound
on the "unassembled quadrants" saving; the carve-tagged share would need the
same row-block generalization downstream. Separately re-run with
_PILOT_COLD=False to get the pilot carve's net billed effect (it trades
smaller matmuls against extra scatter/cumsum ops).
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench as wb
from whestbench import SetupContext

NETS = [0, 10, 20, 30, 40, 50]
BUDGET = 272_000_000_000

spec = importlib.util.spec_from_file_location("est54probe", str(REPO / "estimator.py"))
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)

# ---- concat attribution via a call-site tag stack ----
TAG = ["other"]
CONCAT_ELEMS = {}

_orig_concat = fnp.concatenate


def _tagged_concat(arrs, axis=0):
    out = _orig_concat(arrs, axis=axis)
    CONCAT_ELEMS[TAG[-1]] = CONCAT_ELEMS.get(TAG[-1], 0) + int(np.prod(out.shape))
    return out


def _tag(fn, tag):
    def wrapped(*a, **k):
        TAG.append(tag)
        try:
            return fn(*a, **k)
        finally:
            TAG.pop()
    return wrapped


est.fnp.concatenate = _tagged_concat
est._strassen_impl = _tag(est._strassen_impl, "strassen")
est._cold_slice_relu_matmul = _tag(est._cold_slice_relu_matmul, "carve")
est._cold_slice_relu_matmul2 = _tag(est._cold_slice_relu_matmul2, "carve")
est._cold_slice_relu_matmul_restore = _tag(est._cold_slice_relu_matmul_restore, "pilot-carve")
est.Estimator._sample_block = _tag(est.Estimator._sample_block, "antithetic")


def run(tag, pilot_cold):
    est._PILOT_COLD = pilot_cold
    CONCAT_ELEMS.clear()
    F_tot, ops_tot = 0.0, {}
    ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
    for i in NETS:
        E = est.Estimator()
        E.setup(SetupContext(seed=0, width=256, depth=32, flop_budget=BUDGET,
                             api_version="v1", submission_dir=str(REPO)))
        m = wb.mlp_at(ds, i)
        flops.budget_reset()
        with flops.BudgetContext(flop_budget=10 * BUDGET) as b:
            E.predict(m, BUDGET)
        d = flops.budget_summary_dict(b)
        F_tot += float(d["flops_used"])
        for k, v in d["operations"].items():
            ops_tot[k] = ops_tot.get(k, 0.0) + v["flop_cost"]
    n = len(NETS)
    print(f"\n== {tag} (pilot_cold={pilot_cold}) ==")
    print(f"mean F = {F_tot/n/1e9:.3f}G   F-only mult = {F_tot/n/BUDGET:.4f}")
    for k, v in sorted(ops_tot.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {k:16s} {v/n/1e9:8.3f}G  {100*v/F_tot:5.2f}%")
    print("  concatenate by call-site (mean elems/net, billed 1/elem):")
    for k, v in sorted(CONCAT_ELEMS.items(), key=lambda kv: -kv[1]):
        print(f"    {k:12s} {v/n/1e9:8.4f}G  {100*(v/n)/(F_tot/n):5.3f}% of F")
    return F_tot / n


flops.configure(symmetry_warnings=False, callback_warnings=False)
F_base = run("shipped algo54", pilot_cold=True)
F_nopc = run("pilot carve OFF", pilot_cold=False)
print(f"\npilot-carve net billed effect: {100*(F_nopc-F_base)/F_base:+.3f}% "
      f"(positive = carve is saving F)")
