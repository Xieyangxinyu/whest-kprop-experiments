"""Micro-benchmark: billed cost of identity-skip (skip maximum on kh hot
columns) implementations vs plain maximum, under flopscope 0.9.1.

Shapes: N=61,440 rows, n=217 output columns, kh=20 hot columns (tail).
Each variant measured in its own BudgetContext; cost per variant printed
in elements (flops_used / 1).
"""
import flopscope as flops
import flopscope.numpy as fnp

N, n, kh = 61_440, 217, 20
BUD = int(1e12)


def fresh_pre():
    with flops.BudgetContext(flop_budget=BUD, quiet=True):
        rng = fnp.random.default_rng(0)
        return fnp.array(rng.standard_normal((N, n)).astype(fnp.float32))


def measure(name, fn):
    pre = fresh_pre()
    with flops.BudgetContext(flop_budget=BUD, quiet=True) as ctx:
        out = fn(pre)
        _ = out
    print(f"{name:<38} {ctx.flops_used:>14,}")
    return ctx.flops_used


base = measure("A baseline maximum(pre)", lambda pre: fnp.maximum(pre, 0.0))

measure(
    "B slab-max + column concat",
    lambda pre: fnp.concatenate(
        [fnp.maximum(pre[:, : n - kh], 0.0), pre[:, n - kh :]], axis=1
    ),
)


def variant_c(pre):
    act = fnp.maximum(pre, 0.0)
    idx = fnp.broadcast_to(fnp.arange(n - kh, n)[None, :], (N, kh))
    fnp.put_along_axis(act, idx, pre[:, n - kh :], axis=1)
    return act


measure("C full-max + scatter raw hot tail", variant_c)


def variant_d(pre):
    left = fnp.maximum(pre[:, : n - kh], 0.0)
    idx = fnp.broadcast_to(fnp.arange(0, n - kh)[None, :], (N, n - kh))
    fnp.put_along_axis(pre, idx, left, axis=1)
    return pre


measure("D slab-max + scatter left into pre", variant_d)


def variant_e(pre):
    # zeros buffer + two scatters (axis=1)
    buf = fnp.zeros((N, n), dtype=fnp.float32)
    li = fnp.broadcast_to(fnp.arange(0, n - kh)[None, :], (N, n - kh))
    hi = fnp.broadcast_to(fnp.arange(n - kh, n)[None, :], (N, kh))
    fnp.put_along_axis(buf, li, fnp.maximum(pre[:, : n - kh], 0.0), axis=1)
    fnp.put_along_axis(buf, hi, pre[:, n - kh :], axis=1)
    return buf


measure("E zeros + two column scatters", variant_e)

# reference points
measure("F zeros only", lambda pre: fnp.zeros((N, n), dtype=fnp.float32))
measure("G slice only pre[:, :n-kh]", lambda pre: pre[:, : n - kh])
measure("H slab-max only (no reassembly)", lambda pre: fnp.maximum(pre[:, : n - kh], 0.0))

print(f"\nbaseline = {base:,}; ideal saving would be kh*N = {kh*N:,}")
