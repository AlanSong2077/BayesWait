"""Component ablations of the Bayesian model: refit each reduced model on the tuning part and evaluate.

  python3 experiments/components.py {leet|swe|tracelab} [calls.jsonl]
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); OUT = os.path.join(HERE, "results")
mode = sys.argv[1]; extra = sys.argv[2:]
depth = 3 if mode in ("leet", "local") else 5
S = {"FIT_SHARED_DELTA": "1"}          # the final model: one discount shared by all levels
V = {"final model": dict(S),
     "per-level discounts": {},
     "no discount (delta = 1)": dict(S, FIT_FIX=json.dumps({"d0": 1.0})),
     "Gaussian noise (no t)": dict(S, FIT_FIX=json.dumps({"nu_obs": 1e9})),
     "no hierarchy (root + exact)": dict(S, CHAIN_KEEP=f"0,{depth-1}")}
if mode in ("leet", "local"): V["no curvature (power law)"] = dict(S, FIT_FIX=json.dumps({"kappa_c": 1e9}))
rows = []
for name, env in V.items():
    tag = "_comp_" + name.split()[0] + str(abs(hash(name)) % 1000)
    e = dict(os.environ, TAG=tag, COSTS="200,1000", **env)
    e.update(FIT_WORKERS=os.environ.get("FIT_WORKERS", "1"))
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_offline.py"), mode, *extra], env=e, check=True,
                   stdout=open(f"/tmp/comp_{mode}{tag}.log", "w"), stderr=subprocess.STDOUT)
    d = json.load(open(os.path.join(OUT, f"{mode}{tag}.json")))["results"]
    r = (name, d["fit"]["test_lpd"], d["fit"]["coverage80"], d["pred|ours: Bayes"]["mdape"],
         d["c=200|ours: Bayes + Keller"]["vs_backoff"], d["c=1000|ours: Bayes + Keller"]["vs_backoff"])
    rows.append(r); print(f"{r[0]:30s} test LPD {r[1]:.3f}  cov80 {r[2]:.1%}  MdAPE {r[3]:.1f}%  sched c=200 {r[4]:+.1%}  c=1s {r[5]:+.1%}", flush=True)
    os.makedirs(os.path.join(OUT, "components_raw"), exist_ok=True)
    for f in (f"{mode}{tag}.json", f"{mode}{tag}.txt"): os.replace(os.path.join(OUT, f), os.path.join(OUT, "components_raw", f))
with open(os.path.join(OUT, f"components_{mode}.txt"), "w") as f:
    f.write(f"== {mode}: component ablations (each model refit on the tuning part)\n")
    f.write(f"   {'model':30s} {'test LPD':>9s} {'cov80':>7s} {'MdAPE':>7s} {'vs backoff c=200ms':>19s} {'c=1s':>8s}\n")
    for r in rows: f.write(f"   {r[0]:30s} {r[1]:9.3f} {r[2]:7.1%} {r[3]:6.1f}% {r[4]:+18.1%} {r[5]:+8.1%}\n")
