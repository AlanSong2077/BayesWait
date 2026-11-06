"""Robustness of Bayes + Keller (same model and hyper-parameters as run_offline.py, results/<mode>.json).

  python3 experiments/robustness.py {leet|local|swe|tracelab} [calls.jsonl]

  1. misspecified check cost: plan with c_plan = c / 10 ... 10 c, pay the true c
  2. resampling: 5 different random scheduling samples (seeds), improvement vs tuned backoff
  3. overhead: wall time of one predict + update and of one Keller schedule
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import run_offline as RO
from bayeswait.bayes import HB, keller_profile, keller_checks
from bayeswait.baselines import backoff_policy, simulate, BACKOFF_GRID

def main():
    mode = RO.MODE; calls = RO.suite_data() if mode in ("leet", "local") else RO.trace_data()
    sp = RO.splits(calls); tune, test, sched, end = sp[:4]; sched_tune = sp[4] if len(sp) > 4 else tune
    scaled = any(c["scale"] for c in calls); h = json.load(open(os.path.join(RO.OUT, f"{mode}.json")))["hyper"]
    rng = np.random.default_rng(7)
    pools = [sorted(rng.choice(test, min(len(test), 4000), replace=False).tolist()) for _ in range(5)] if mode in ("swe", "tracelab") else [test]
    want = set(sched) | set(sched_tune) | set().union(*map(set, pools))
    t0 = time.time(); res = RO.run_pass(calls, h, want, None, scaled); tp = time.time() - t0
    n_ev = sum(1 for c in calls if c.get("learn", True)) + len(want)
    T = {i: calls[i]["ms"] for i in res}; L = [f"== {mode}: robustness of Bayes + Keller"]
    pred = lambda i: res[i].get("fused", res[i]["pred"]) if res[i]["llm"] else res[i]["pred"]
    prof = {i: keller_profile(*pred(i)) for i in want if i in res}
    run = lambda f, ids: np.array([simulate(f(i), T[i]) for i in ids])
    obj = lambda sm, c: float((sm[:, 0] + c * sm[:, 1]).mean())
    # 1. misspecified c
    L.append("\n   1. planning with a wrong check cost (objective reduction vs tuned backoff at the true c)")
    L.append("      true c     " + "".join(f"{'c_plan=' + f:>14s}" for f in ("c/10", "c/3", "c", "3c", "10c")))
    for c in (200, 1000, 5000):
        g = min(BACKOFF_GRID, key=lambda g: obj(run(lambda i: backoff_policy(*g), sched_tune), c))
        b = obj(run(lambda i: backoff_policy(*g), sched), c)
        r = [1 - obj(run(lambda i: keller_checks(*pred(i), c * k, prof[i]), sched), c) / b for k in (0.1, 1 / 3, 1, 3, 10)]
        L.append(f"      {c:5d} ms  " + "".join(f"{x:+14.1%}" for x in r))
    # 2. resampling
    if len(pools) > 1:
        L.append("\n   2. five independent scheduling samples of 4,000 test calls: reduction vs tuned backoff")
        for c in (200, 1000):
            g = min(BACKOFF_GRID, key=lambda g: obj(run(lambda i: backoff_policy(*g), sched_tune), c))
            r = [1 - obj(run(lambda i: keller_checks(*pred(i), c, prof[i]), p), c) / obj(run(lambda i: backoff_policy(*g), p), c) for p in pools]
            L.append(f"      c = {c:5d} ms: " + "  ".join(f"{x:+.1%}" for x in r) + f"   mean {np.mean(r):+.1%}, sd {np.std(r):.1%}")
    # 3. overhead
    ids = sorted(want)[:300]; t1 = time.time()
    for i in ids: keller_checks(*pred(i), 1000.0)
    tk = (time.time() - t1) / len(ids)
    L.append(f"\n   3. overhead: model predict + update {1000 * tp / n_ev:.3f} ms per call event (pure Python); "
             f"Keller schedule {1000 * tk:.2f} ms per call (6,000-point grid)")
    print("\n".join(L)); open(os.path.join(RO.OUT, f"robustness_{mode}.txt"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
