"""Offline evaluation: hierarchical Bayesian runtime model + Keller optimal checks vs. the baselines.

  python3 experiments/run_offline.py leet                       # LeetCode suite (data/leetcode_trials.jsonl)
  python3 experiments/run_offline.py local                      # local-tool suite (data/local_trials.jsonl)
  python3 experiments/run_offline.py swe      data/swe_calls.jsonl
  python3 experiments/run_offline.py tracelab data/tracelab_calls.jsonl
  env: COSTS (default 50,200,1000,5000 ms per check), LLM_PRIOR (SWE: data/swe_llm_prior.json),
       HYPER_FROM (reuse hyper-parameters from a results/*.json), TAG (output suffix)

Counterfactual replay of recorded runtimes. Everything is causal: a runtime enters the model only after its
call finished. Split: LeetCode / local suites - first 30% of warm runs (runs of a task that has run before) tune, the rest test;
traces - targets (successful calls >= 100 ms) in 10 time blocks, blocks 1-2 history only, block 3 tunes,
blocks 4-10 test (scheduling on a random sample of 4000 test targets). Tuning: the model's hyper-parameters
maximise the log predictive density on the tuning part; every baseline's parameters minimise the objective
on the tuning part (the local suite, 34 tuning runs, reuses the LeetCode hyper-parameters). Objective = mean detection delay + c x mean number of checks. 95% CIs: bootstrap over
tasks (suites) / sessions (traces).

Prediction baselines: LLM runtime estimate (where available); last run of the same command; median of the
most specific history level (3Sigma-style empirical history).
Check policies
  exponential backoff (tuned)   start s, factor f, cap 10 s
  fixed interval (tuned)        check every d ms
  LLM estimate + 100 ms         (where available) first check at the LLM's runtime estimate, then every 100 ms
  ours: Bayes + Keller          hierarchical Bayesian predictive -> Keller checks n(t) = sqrt(h(t)/2c)
"""
import collections, hashlib, heapq, json, math, os, sys, time
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from bayeswait.bayes import HB, fuse, keller_profile, keller_checks
from bayeswait.baselines import backoff_policy, fixed_policy, estimate_then_poll, simulate, BACKOFF_GRID, FIXED_GRID
from bayeswait.families import describe_swe, describe_skeleton, clean

MODE = sys.argv[1]; CALLS = sys.argv[2] if len(sys.argv) > 2 else None
COSTS = tuple(int(x) for x in os.environ.get("COSTS", "50,200,1000,5000").split(","))
SCHED_N = int(os.environ.get("SCHED_N", 4000)); MIN_MS = 100.0; DEFAULT_MS = 1000.0
LLM_PRIOR = os.environ.get("LLM_PRIOR")
OUT = os.path.join(ROOT, "experiments", "results"); lines = []; summary = {}
def out(s=""): print(s, flush=True); lines.append(s)

# ======================================================================= data -> causal event stream
def suite_data():
    """-> list of calls in execution order with chain, x, family history key, LLM prior; warm flag"""
    if MODE == "local":
        R = [json.loads(l) for l in open(os.path.join(ROOT, "data", "local_trials.jsonl"))]; sk = "predicted_input_size"
    else:
        L = [json.loads(l) for l in open(os.path.join(ROOT, "data", "leetcode_trials.jsonl"))]
        keep = set(sorted({r["question_id"] for r in L})[:543]); R = [r for r in L if r["question_id"] in keep and r["success"]]; sk = "k"
    calls, seen = [], set()
    for r in R:
        s = r[sk]; tid = r["task_id"]
        calls.append(dict(ms=r["actual_runtime_ms"], group=tid, fam=tid, scale=s, x=math.log(s) if s else 0.0,
                          chain=(("root",), ("task", tid), ("exact", tid, s)), llm=r["llm_prior_ms"],
                          target=tid in seen))
        seen.add(tid)
    return calls

def trace_data():
    rows = []
    for l in open(CALLS):
        r = json.loads(l)
        if r["dur_s"] <= 0: continue
        if MODE == "swe":
            repo = r["iid"].rsplit("-", 1)[0].replace("__", "/"); k = describe_swe(r["cmd"], repo, r["iid"]); r["repo"] = repo
        else: k = describe_skeleton(r["cmd"], r["repo"], r["iid"])
        r["ms"] = r["dur_s"] * 1000.0; r["ok"] = r["rc"] != -1; r["k"] = k; rows.append(r)
    rows.sort(key=lambda r: r["t_start"])
    prior = json.load(open(LLM_PRIOR)) if LLM_PRIOR else {}
    pk = lambda r: hashlib.sha1((r["repo"] + "\n" + clean(r["cmd"])).encode()).hexdigest()[:16]
    calls = []
    for r in rows:
        k = r["k"]; s = k.get("scale")
        calls.append(dict(ms=r["ms"], group=r["iid"], fam=k["exact"], scale=s, x=math.log(s) if s else 0.0,
                          chain=(("root",), ("glob", k["glob"]), ("l3", k["l3"]), ("l2", k["l2"]), ("exact", k["exact"])),
                          llm=prior.get(pk(r)) if prior else None, target=r["ok"] and r["ms"] >= MIN_MS,
                          t_start=r["t_start"], t_end=r["t_end"], learn=r["ok"] and r["ms"] >= MIN_MS))
    return calls

KEEP = [int(x) for x in os.environ["CHAIN_KEEP"].split(",")] if os.environ.get("CHAIN_KEEP") else None
def prune(calls):
    """ablation: keep only the listed hierarchy levels (e.g. 0,4 = root + exact command)"""
    if KEEP:
        for c in calls: c["chain"] = tuple(c["chain"][i] for i in KEEP if i < len(c["chain"]))
    return calls

def splits(calls):
    tg = [i for i, c in enumerate(calls) if c["target"]]
    if MODE in ("leet", "local"):
        k = int(0.3 * len(tg)); return tg[:k], tg[k:], tg[k:], tg[k - 1]
    rng = np.random.default_rng(0); blocks = np.array_split(np.array(tg), 10)     # same samples as llm_prior_swe.py
    calib = sorted(rng.choice(blocks[2], min(len(blocks[2]), 1500), replace=False).tolist())
    test_all = np.concatenate(blocks[3:]); sub = sorted(rng.choice(test_all, min(len(test_all), SCHED_N), replace=False).tolist())
    return [int(i) for i in blocks[2]], [int(i) for i in test_all], sub, int(blocks[2][-1]), calib

def events(calls, upto=None):
    """yield ('predict', i) / ('learn', i) in causal order (traces: a call is learned when it has finished)"""
    n = len(calls) if upto is None else upto + 1
    if MODE in ("leet", "local"):
        for i in range(n): yield "predict", i; yield "learn", i
        return
    pend = []
    for i in range(n):
        c = calls[i]
        while pend and pend[0][0] < c["t_start"]: yield "learn", heapq.heappop(pend)[1]
        yield "predict", i
        heapq.heappush(pend, (c["t_end"], i))

# ======================================================================= one causal pass
def run_pass(calls, h, want, upto=None, scaled=True):
    """h: model hyper-parameters. want: set of target ids to record. -> {i: dict(pred, llm, last, med, fused)}"""
    m = HB(scaled=scaled, **h); fam = {}; res = {}; node = collections.defaultdict(lambda: collections.deque(maxlen=20))
    lp = [1.0, math.log(4.0)]           # pooled log squared error of log(LLM prior) (geometric estimator)
    for kind, i in events(calls, upto):
        c = calls[i]
        if kind == "predict":
            if i in want:
                p = m.predict(c["chain"], c["x"]); fb = c["llm"] or DEFAULT_MS
                deep = next((node[n] for n in reversed(c["chain"]) if node[n]), None)
                lst = fam.get(c["fam"])
                r = dict(pred=p, llm=c["llm"], last=lst[0] if lst else fb, med=math.exp(float(np.median(deep))) if deep else fb,
                         scaled=lst[0] * c["scale"] / lst[1] if lst and lst[1] and c["scale"] else (lst[0] if lst else fb))
                if c["llm"]: r["fused"] = fuse(p, math.log(c["llm"]), math.exp(lp[1] / lp[0] + 1.2704))
                res[i] = r
        elif c.get("learn", True):
            y = math.log(c["ms"]); m.update(c["chain"], c["x"], y); fam[c["fam"]] = (c["ms"], c["scale"])
            for n in c["chain"]: node[n].append(y)
            if c["llm"]: lp[0] = 0.999 * lp[0] + 1; lp[1] = 0.999 * lp[1] + math.log(max((y - math.log(c["llm"])) ** 2, 1e-12))
    return res

# ======================================================================= empirical-Bayes fit (coordinate ascent)
KG = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; DG = [1.0, 0.95, 0.85, 0.7, 0.5, 0.3]
GRID = dict(kappa_b=[0.001, 0.01, 0.1, 1, 10], kappa_c=[0.1, 1, 10, 100, 1e4], nu0=[1, 2, 4, 8, 16, 32, 64], nu_obs=[1, 2, 4, 10, 1e9])

_CTX = {}
def _lpd(h):
    calls, tset, tune, end, scaled = _CTX["c"], _CTX["s"], _CTX["t"], _CTX["e"], _CTX["sc"]
    res = run_pass(calls, h, tset, end, scaled)
    return float(np.mean([HB.logpdf(math.log(calls[i]["ms"]), *res[i]["pred"]) for i in tune]))

def fit(calls, tune, end, depth, scaled):
    shared = bool(os.environ.get("FIT_SHARED_DELTA")); pin = json.loads(os.environ.get("FIT_FIX", "{}"))
    def hp(f):
        return dict(kappa=[f[f"k{i}"] for i in range(depth)], delta=[f["d0" if shared else f"d{i}"] for i in range(depth)],
                    **{k: f[k] for k in GRID})
    grid = {**{f"k{i}": KG for i in range(depth)}, **{f"d{i}": DG for i in range(1 if shared else depth)}, **GRID}
    if not scaled: grid.pop("kappa_b"); grid.pop("kappa_c")
    for k in pin: grid.pop(k, None)
    best = {**{f"k{i}": 1 for i in range(depth)}, **{f"d{i}": 1.0 for i in range(depth)}, "kappa_b": 1, "kappa_c": 10, "nu0": 4, "nu_obs": 4}
    best.update(pin)
    if os.environ.get("FIT_START"):    # start the search from hyper-parameters fitted elsewhere
        st = json.load(open(os.environ["FIT_START"]))["hyper"]
        for i in range(depth):
            j = min(KEEP[i] if KEEP else i, len(st["kappa"]) - 1)
            best[f"k{i}"] = min(KG, key=lambda v: abs(math.log(v / st["kappa"][j])))
            best[f"d{i}"] = min(DG, key=lambda v: abs(v - st["delta"][j]))
        for k in GRID: best[k] = st.get(k, best[k])
        best.update(pin)
    rounds = int(os.environ.get("FIT_ROUNDS", 3)); workers = int(os.environ.get("FIT_WORKERS", 1))
    _CTX.update(c=calls, s=set(tune), t=tune, e=end, sc=scaled); cache = {}; t0 = time.time()
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    def evaluate(cands):
        todo = [x for x in cands if tuple(sorted(x.items())) not in cache]
        vals = (pool.map(_lpd, [hp(x) for x in todo]) if pool else [_lpd(hp(x)) for x in todo])
        for x, v in zip(todo, vals):
            cache[tuple(sorted(x.items()))] = v
            print(f"   [fit {len(cache):3d} | {time.time()-t0:5.0f}s] " + " ".join(f"{k}={x[k]:g}" for k in sorted(x)) + f"  LPD {v:.4f}", flush=True)
        return [cache[tuple(sorted(x.items()))] for x in cands]
    for r in range(rounds):
        old = dict(best)
        for p, vals in grid.items():
            cands = [dict(best, **{p: v}) for v in vals]; sc = evaluate(cands); best[p] = vals[int(np.argmax(sc))]
        print(f"   [fit] round {r+1} done: best LPD {cache[tuple(sorted(best.items()))]:.4f}", flush=True)
        if best == old: break
    if pool: pool.close()
    f = lambda x: cache[tuple(sorted(x.items()))]
    return hp(best), f(best), len(cache)

# ======================================================================= evaluation
def main():
    t0 = time.time()
    calls = prune(suite_data() if MODE in ("leet", "local") else trace_data())
    sp = splits(calls); tune, test, sched, end = sp[:4]; sched_tune = sp[4] if len(sp) > 4 else tune
    depth = len(calls[0]["chain"]); scaled = any(c["scale"] for c in calls)
    if os.environ.get("HYPER_FROM"):          # transfer hyper-parameters fitted on another dataset
        h = json.load(open(os.environ["HYPER_FROM"]))["hyper"]; nfit = 0
        r0 = run_pass(calls, h, set(tune), end, scaled); lpd = float(np.mean([HB.logpdf(math.log(calls[i]["ms"]), *r0[i]["pred"]) for i in tune]))
    else: h, lpd, nfit = fit(calls, tune, end, depth, scaled)
    out(f"== {MODE}: {len(calls)} calls; tune {len(tune)}, test {len(test)}, scheduling sample {len(sched)}")
    out(f"   model fitted on the tuning part ({nfit} settings, {time.time()-t0:.0f}s): {json.dumps(h)}  tune LPD {lpd:.3f}")
    res = run_pass(calls, h, set(tune) | set(test) | set(sched_tune) | set(sched), None, scaled)
    T = {i: calls[i]["ms"] for i in res}; ape = lambda p, t: abs(p - t) / t * 100
    has_llm = any(res[i]["llm"] for i in test) and not os.environ.get("NO_LLM")   # NO_LLM=1: history-only predictive
    ours_pred = lambda i: res[i].get("fused", res[i]["pred"]) if has_llm else res[i]["pred"]
    # ---------------- prediction
    out(f"\n   prediction on the test part (n={len(test)}): MdAPE / P90 APE of the point estimate (median)")
    rows = [("last run, same command", lambda i: res[i]["last"]), ("median, deepest history level", lambda i: res[i]["med"]),
            ("ours: Bayes", lambda i: math.exp(res[i]["pred"][0]))]
    if scaled: rows.insert(1, ("last run x size ratio", lambda i: res[i]["scaled"]))
    if has_llm: rows = [("LLM prior", lambda i: res[i]["llm"] or DEFAULT_MS)] + rows + [("ours: Bayes + LLM prior", lambda i: math.exp(res[i]["fused"][0]) if "fused" in res[i] else math.exp(res[i]["pred"][0]))]
    for name, f in rows:
        a = [ape(f(i), T[i]) for i in test]
        out(f"   {name:30s} MdAPE {np.median(a):7.1f}%   P90 {np.percentile(a, 90):8.0f}%")
        summary[f"pred|{name}"] = dict(mdape=float(np.median(a)), p90=float(np.percentile(a, 90)))
    yv = [math.log(T[i]) for i in test]; pv = [ours_pred(i) for i in test]
    cov = np.mean([stats.t.ppf(.1, d, l, s) <= y <= stats.t.ppf(.9, d, l, s) for (l, s, d), y in zip(pv, yv)])
    tlpd = float(np.mean([HB.logpdf(y, *p) for y, p in zip(yv, pv)]))
    out(f"   ours: test log predictive density {tlpd:.3f}, 80% interval coverage {cov:.1%}")
    summary["fit"] = dict(tune_lpd=lpd, test_lpd=tlpd, coverage80=float(cov))
    by_group_tables(calls, res, test, T, ape)
    # ---------------- scheduling
    groups = np.array([calls[i]["group"] for i in sched], object); ug = np.unique(groups); gi = {u: np.where(groups == u)[0] for u in ug}
    rng = np.random.default_rng(1); boots = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))]) for _ in range(1000)]
    run = lambda f, ids: np.array([simulate(f(i), T[i]) for i in ids])
    obj = lambda sm, c: float((sm[:, 0] + c * sm[:, 1]).mean())
    prof = {i: keller_profile(*ours_pred(i)) for i in sched}
    for c in COSTS:
        out(f"\n   scheduling, c = {c} ms per check (n={len(sched)}; objective = mean delay + c x mean checks)")
        d = min(FIXED_GRID, key=lambda d: obj(run(lambda i: fixed_policy(d), sched_tune), c))
        g = min(BACKOFF_GRID, key=lambda g: obj(run(lambda i: backoff_policy(*g), sched_tune), c))
        R = {f"exponential backoff {g[0]:g} ms x{g[1]:g} (tuned)": run(lambda i: backoff_policy(*g), sched),
             f"fixed interval {d} ms (tuned)": run(lambda i: fixed_policy(d), sched)}
        if has_llm: R["LLM estimate + 100 ms"] = run(lambda i: estimate_then_poll(res[i]["llm"] or DEFAULT_MS), sched)
        R["ours: Bayes + Keller"] = run(lambda i: keller_checks(*ours_pred(i), c, prof[i]), sched)
        base = next(iter(R)); bv = R[base][:, 0] + c * R[base][:, 1]
        for name, sm in R.items():
            v = sm[:, 0] + c * sm[:, 1]; dd = [1 - v[b].mean() / bv[b].mean() for b in boots]
            tag = "" if name == base else f"  vs backoff {1 - v.mean()/bv.mean():+7.1%} [{np.percentile(dd, 2.5):+.1%}, {np.percentile(dd, 97.5):+.1%}]"
            out(f"   {name:38s} delay {sm[:,0].mean():8.0f} ms  P90 {np.percentile(sm[:,0], 90):8.0f}  checks {sm[:,1].mean():7.2f}  objective {v.mean():9.0f}{tag}")
            summary[f"c={c}|{name}"] = dict(delay=float(sm[:, 0].mean()), checks=float(sm[:, 1].mean()), objective=float(v.mean()),
                                            vs_backoff=float(1 - v.mean() / bv.mean()), ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
    tag = os.environ.get("TAG", "")
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, f"{MODE}{tag}.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(hyper=h, results=summary), open(os.path.join(OUT, f"{MODE}{tag}.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")

def by_group_tables(calls, res, test, T, ape):
    if MODE in ("leet", "local"):
        hist = collections.defaultdict(list); grp = {}
        for i, c in enumerate(calls):
            hs = hist[c["fam"]]
            if i in res:
                if MODE == "leet": grp[i] = f"earlier runs = {min(len(hs), 4)}{'+' if len(hs) >= 4 else ''}"
                else:
                    sz = [s for _, s in hs]; x = c["scale"]
                    grp[i] = ("same-scale repeat" if x in sz else "interpolation" if min(sz) < x < max(sz)
                              else "upward extrapolation" if x > max(sz) else "downward extrapolation")
            hs.append((c["ms"], c["scale"]))
    else:   # most specific level with earlier finished history
        seen = set(); grp = {}
        for kind, i in events(calls):
            c = calls[i]
            if kind == "predict" and i in res:
                grp[i] = next((n[0] for n in reversed(c["chain"]) if n in seen), "cold")
            elif kind == "learn" and c.get("learn", True): seen.update(c["chain"])
    key = "scaled" if any(calls[i]["scale"] for i in test) else "last"
    out(f"   MdAPE by group ({'last run x size ratio' if key == 'scaled' else 'last run of the same command'} / ours):")
    for g in sorted({grp[i] for i in test}):
        ids = [i for i in test if grp[i] == g]
        if len(ids) < 5: continue
        out(f"     {g:26s} n={len(ids):6d}  {np.median([ape(res[i][key], T[i]) for i in ids]):7.1f}  {np.median([ape(math.exp(res[i]['pred'][0]), T[i]) for i in ids]):7.1f}")

if __name__ == "__main__":
    main()
