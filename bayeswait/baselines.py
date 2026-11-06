"""Baseline check policies and the replay simulator."""
from __future__ import annotations
import math

BACKOFF_GRID = [(st, f) for st in (50, 100, 200, 500, 1000, 2000) for f in (1.25, 1.5, 2.0, 3.0)]
FIXED_GRID = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

def backoff_policy(start=100.0, factor=2.0, cap=10_000.0, horizon=900_000.0):
    """exponential backoff: gaps start, start*factor, ... capped at `cap`"""
    t, w, out = 0.0, float(start), []
    while t < horizon: t += w; out.append(t); w = min(cap, w * factor)
    return out

def fixed_policy(interval):
    return [interval * (j + 1) for j in range(int(900_000 / interval) + 1)]

def estimate_then_poll(est_ms, poll=100.0, horizon=600_000.0):
    """first check at a point estimate, then every `poll` ms"""
    return [est_ms + poll * j for j in range(int(horizon / poll) + 1)]

def simulate(times, actual):
    """replay a check schedule against a recorded runtime -> (detection delay ms, number of checks); past the
    last listed time the last gap repeats"""
    times = list(times) or [100.0]
    gap = (times[-1] - times[-2]) if len(times) > 1 else times[-1]
    for n, t in enumerate(times, 1):
        if t >= actual: return t - actual, n
    t = times[-1]; k = math.ceil((actual - t) / gap)
    return t + k * gap - actual, len(times) + k
