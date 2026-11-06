"""Where do the gains come from? Per runtime bucket: delay, checks and cost of Bayes + Keller vs tuned backoff and
tuned fixed interval (same model, splits, samples and tuning as run_offline.py).

  python3 experiments/buckets.py {leet|swe|tracelab} [calls.jsonl]
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import run_offline as RO
from bayeswait.bayes import keller_profile, keller_checks
from bayeswait.baselines import backoff_policy, fixed_policy, simulate, BACKOFF_GRID, FIXED_GRID

BUCKETS = [(0, 1000, "< 1 s"), (1000, 10_000, "1-10 s"), (10_000, 1e12, ">= 10 s")]

def main():
    mode = RO.MODE; calls = RO.suite_data() if mode in ("leet", "local") else RO.trace_data()
    sp = RO.splits(calls); tune, test, sched, end = sp[:4]; sched_tune = sp[4] if len(sp) > 4 else tune
    scaled = any(c["scale"] for c in calls); h = json.load(open(os.path.join(RO.OUT, f"{mode}.json")))["hyper"]
    res = RO.run_pass(calls, h, set(sched) | set(sched_tune), None, scaled)
    T = {i: calls[i]["ms"] for i in res}
    pred = lambda i: res[i].get("fused", res[i]["pred"]) if False else res[i]["pred"]
    prof = {i: keller_profile(*pred(i)) for i in sched}
    run = lambda f, ids: np.array([simulate(f(i), T[i]) for i in ids])
    obj = lambda sm, c: float((sm[:, 0] + c * sm[:, 1]).mean())
    L = [f"== {mode}: gains by runtime bucket (n={len(sched)})"]
    for c in RO.COSTS:
        g = min(BACKOFF_GRID, key=lambda g: obj(run(lambda i: backoff_policy(*g), sched_tune), c))
        d = min(FIXED_GRID, key=lambda d: obj(run(lambda i: fixed_policy(d), sched_tune), c))
        R = {"ours": run(lambda i: keller_checks(*pred(i), c, prof[i]), sched),
             "backoff": run(lambda i: backoff_policy(*g), sched), "fixed": run(lambda i: fixed_policy(d), sched)}
        tt = np.array([T[i] for i in sched])
        L.append(f"\n   c = {c} ms   (cells: mean delay ms / mean checks / mean cost; share of total cost)")
        for lo, hi, name in BUCKETS:
            m = (tt >= lo) & (tt < hi)
            if m.sum() == 0: continue
            cells = []
            for p in ("ours", "backoff", "fixed"):
                sm = R[p][m]; cost = sm[:, 0] + c * sm[:, 1]
                tot = (R[p][:, 0] + c * R[p][:, 1]).sum()
                cells.append(f"{p} {sm[:,0].mean():7.0f} / {sm[:,1].mean():5.2f} / {cost.mean():8.0f} ({cost.sum()/tot:4.0%})")
            ob = R["ours"][m]; bb = R["backoff"][m]
            red = 1 - (ob[:, 0] + c * ob[:, 1]).mean() / (bb[:, 0] + c * bb[:, 1]).mean()
            L.append(f"   {name:8s} n={m.sum():5d} ({m.mean():4.0%})  " + "  |  ".join(cells) + f"  | ours vs backoff {red:+.1%}")
        # tail of the delay distribution
        L.append("   delay > 5 s: " + ", ".join(f"{p} {np.mean(R[p][:, 0] > 5000):.1%}" for p in R)
                 + ";  delay > 10 s: " + ", ".join(f"{p} {np.mean(R[p][:, 0] > 10000):.1%}" for p in R))
        ob, bb = R["ours"], R["backoff"]
        win = np.mean((ob[:, 0] + c * ob[:, 1]) < (bb[:, 0] + c * bb[:, 1]) - 1e-9); lose = np.mean((ob[:, 0] + c * ob[:, 1]) > (bb[:, 0] + c * bb[:, 1]) + 1e-9)
        L.append(f"   per call vs backoff: ours cheaper on {win:.0%}, costlier on {lose:.0%}, tie {1-win-lose:.0%}")
        print("\n".join(L[-6:]), flush=True)
    open(os.path.join(RO.OUT, f"buckets_{mode}_nollm.txt"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
