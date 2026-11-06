"""Summarise experiments/results/online_swe/episodes.jsonl -> experiments/results/online_swe/summary.txt

Per policy (tasks completed under both policies, paired):
  waiting delay   sum over commands of (time the agent saw the result - time the command finished)
  status rounds   extra LLM rounds spent on 'still running' checks, their tokens and dollars
  per command     mean delay, checks, and the offline objective delay + c x checks with the measured c
  task level      wall time, all LLM rounds, tokens, dollars, submitted rate
Paired bootstrap over tasks (2,000 resamples) for ours vs backoff.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("OUT_DIR") or os.path.join(os.path.dirname(HERE), "results", "online_swe")
E = [json.loads(l) for l in open(os.path.join(OUT, "episodes.jsonl"))]
by = {}
for e in E: by.setdefault(e["instance_id"], {})[e["policy"]] = e
pols = ["backoff", "ours"]; ids = sorted(i for i, v in by.items() if all(p in v and not v[p]["exit"].startswith("Error") for p in pols))
rng = np.random.default_rng(0); B = rng.integers(0, len(ids), (2000, len(ids))) if ids else None
cmds = {p: [c for i in ids for c in by[i][p]["commands"]] for p in pols}
c_meas = float(np.mean([c["c_plan_ms"] for p in pols for c in cmds[p]])) if ids else 0
def task(p, f): return np.array([f(by[i][p]) for i in ids], float)
M = {"waiting delay (s)": lambda e: sum(c["delay_ms"] for c in e["commands"]) / 1000,
     "status rounds": lambda e: e["status_rounds"],
     "status-round tokens": lambda e: sum(c["status_tokens"] for c in e["commands"]),
     "commands": lambda e: len(e["commands"]),
     "wall time (min)": lambda e: e["wall_s"] / 60,
     "LLM rounds (all)": lambda e: e["llm_rounds"],
     "tokens (all)": lambda e: e["tokens"],
     "cost ($)": lambda e: e["usd"],
     "submitted": lambda e: float(e["exit"] == "Submitted")}
L = [f"online SWE-bench Verified Mini: {len(ids)} tasks completed under both policies "
     f"({len(E)} episodes logged); mean planning check cost c = {c_meas/1000:.2f} s (measured LLM round latency)", "",
     f"   {'per task (mean)':24s} {'backoff':>10s} {'ours':>10s}   ours vs backoff [95% CI]"]
for name, f in M.items():
    a, b = task("backoff", f), task("ours", f)
    if not len(a): break
    if name == "submitted":
        L.append(f"   {name:24s} {a.mean():10.1%} {b.mean():10.1%}"); continue
    r = (b[B].mean(1) / np.maximum(a[B].mean(1), 1e-12) - 1) * 100
    L.append(f"   {name:24s} {a.mean():10.2f} {b.mean():10.2f}   {100*(b.mean()/max(a.mean(),1e-12)-1):+6.1f}% [{np.percentile(r,2.5):+.1f}%, {np.percentile(r,97.5):+.1f}%]")
L.append("")
for p in pols:
    c = cmds[p]
    if not c: continue
    d = np.array([x["delay_ms"] for x in c]); k = np.array([x["checks"] for x in c]); cp = np.array([x["c_plan_ms"] for x in c])
    L.append(f"   per command, {p:7s} n={len(c):5d}: delay mean {d.mean():7.0f} ms (P90 {np.percentile(d,90):7.0f}), checks {k.mean():5.2f}, "
             f"objective delay + c x checks {np.mean(d + cp * k)/1000:6.2f} s, runtime median {np.median([x['runtime_ms'] for x in c]):6.0f} ms")
txt = "\n".join(L); print(txt); open(os.path.join(OUT, "summary.txt"), "w").write(txt + "\n")
for p in pols:      # predictions for the official SWE-bench evaluation harness
    json.dump({i: dict(instance_id=i, model_name_or_path=f"bayeswait-{p}", model_patch=by[i][p]["submission"])
               for i in by if p in by[i]}, open(os.path.join(OUT, f"preds_{p}.json"), "w"), indent=1)
