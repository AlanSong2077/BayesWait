"""Ablations on the scheduling objective (same data, splits, samples and hyper-parameters as run_offline.py).

  python3 experiments/ablate.py {leet|local|swe|tracelab} [calls.jsonl]       (uses experiments/results/<mode>.json)

  A  Bayes + Keller                    the method
  B  history median + empirical residuals + Keller
                                       point estimate = median of the deepest history level (3Sigma-style),
                                       distribution = causal pool of log(T / median) of finished calls with the
                                       same history level (last 1000), Gaussian-kernel smoothed
  C  Bayes + exact optimal checks (DP) Barlow-Hunter-Proschan dynamic programme on the same predictive
  D  median + residuals + DP
A vs B: does the Bayesian predictive matter?   A vs C: how far is Keller's closed form from the optimum?
"""
import collections, json, math, os, sys, time
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import run_offline as RO
from bayeswait.bayes import HB, keller_profile, keller_checks
from bayeswait.baselines import backoff_policy, simulate, BACKOFF_GRID

TG = np.geomspace(5.0, 6e8, 4000)     # time grid for KDE hazards (ms)

def kde_profile(mu, resid):
    """(t, N_1) for log T ~ mu + KDE(resid)"""
    r = np.asarray(resid); bw = 1.06 * max(r.std(), 1e-3) * len(r) ** -0.2
    z = (np.log(TG)[:, None] - mu - r[None, :]) / bw
    f = stats.norm.pdf(z).mean(1) / (bw * TG); S = stats.norm.sf(z).mean(1)
    n = np.sqrt(np.maximum(f / np.maximum(S, 1e-300), 0) / 2.0)
    return TG, np.concatenate([[0.0], np.cumsum(0.5 * (n[1:] + n[:-1]) * np.diff(TG))])

DG = np.geomspace(5.0, 6e8, 300)
def dp_checks(samples, c):
    """exact optimal check times on grid DG for the empirical distribution `samples` (ms)"""
    G = DG; s = np.sort(np.clip(samples, G[0], G[-1])); n = len(s); m = len(G)
    idx = np.searchsorted(s, G, side="right"); cum = np.concatenate([[0.0], np.cumsum(s)])
    S = 1.0 - idx / n; A = cum[idx] / n; V = np.zeros(m); nxt = np.full(m, -1)
    for i in range(m - 2, -1, -1):
        if S[i] <= 0: continue
        j = np.arange(i + 1, m)
        tot = c * S[i] + G[j] * (idx[j] - idx[i]) / n - (A[j] - A[i]) + V[j]
        k = int(np.argmin(tot)); V[i] = tot[k]; nxt[i] = j[k]
    tot = c + G * idx / n - A + V
    k = int(np.argmin(tot)); out, i = [float(G[k])], k
    while nxt[i] >= 0 and S[i] > 1e-12: i = int(nxt[i]); out.append(float(G[i]))
    return out

Q = (np.arange(400) + 0.5) / 400

def main():
    mode = RO.MODE; t0 = time.time()
    calls = RO.suite_data() if mode in ("leet", "local") else RO.trace_data()
    sp = RO.splits(calls); tune, test, sched, end = sp[:4]; sched_tune = sp[4] if len(sp) > 4 else tune
    scaled = any(c["scale"] for c in calls); h = json.load(open(os.path.join(RO.OUT, f"{mode}.json")))["hyper"]
    keep = set(sched) | set(sched_tune)
    # causal pass: Bayes predictive + median of deepest level + residual pool per level
    m = HB(scaled=scaled, **h); node = collections.defaultdict(lambda: collections.deque(maxlen=20))
    pool = collections.defaultdict(lambda: collections.deque(maxlen=1000)); info = {}; R = {}
    for kind, i in RO.events(calls):
        c = calls[i]
        if kind == "predict" and c["target"]:
            lv = next((d for d in range(len(c["chain"]) - 1, -1, -1) if node[c["chain"][d]]), -1)
            med = float(np.median(node[c["chain"][lv]])) if lv >= 0 else math.log(RO.DEFAULT_MS)
            info[i] = (lv, med)
            if i in keep:
                R[i] = dict(pred=m.predict(c["chain"], c["x"]), med=med, resid=np.array(pool[lv]) if len(pool[lv]) >= 20 else np.random.default_rng(0).normal(0, 1, 200))
        elif kind == "learn" and c.get("learn", True):
            y = math.log(c["ms"]); m.update(c["chain"], c["x"], y)
            if i in info: lv, med = info[i]; pool[lv].append(y - med)
            for n in c["chain"]: node[n].append(y)
    print(f"[{mode}] pass done {time.time()-t0:.0f}s", flush=True)
    T = {i: calls[i]["ms"] for i in keep}
    groups = np.array([calls[i]["group"] for i in sched], object); ug = np.unique(groups); gi = {u: np.where(groups == u)[0] for u in ug}
    rng = np.random.default_rng(1); boots = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))]) for _ in range(1000)]
    run = lambda f, ids: np.array([simulate(f(i), T[i]) for i in ids])
    obj = lambda sm, c: float((sm[:, 0] + c * sm[:, 1]).mean())
    profA = {i: keller_profile(*R[i]["pred"]) for i in sched}; profB = {i: kde_profile(R[i]["med"], R[i]["resid"]) for i in sched}
    sampA = {i: np.exp(np.clip(R[i]["pred"][0] + R[i]["pred"][1] * stats.t.ppf(Q, R[i]["pred"][2]), 1.0, 21.0)) for i in sched}
    sampB = {i: np.exp(R[i]["med"] + np.random.default_rng(2).choice(R[i]["resid"], 400)) for i in sched}
    kel = lambda prof, c: (lambda i: (lambda t, N: (lambda ks: list(np.interp(ks, N, t)) if len(ks) else [t[np.searchsorted(N, 0.5 * N[-1])]])(np.arange(0.5, N[-1], 1.0)))(prof[i][0], prof[i][1] / math.sqrt(c)))
    lines = [f"== {mode}: scheduling ablations (n={len(sched)}), objective reduction vs tuned exponential backoff"]
    summ = {}
    for c in RO.COSTS:
        g = min(BACKOFF_GRID, key=lambda g: obj(run(lambda i: backoff_policy(*g), sched_tune), c))
        res = {"backoff (tuned)": run(lambda i: backoff_policy(*g), sched),
               "A  Bayes + Keller": run(kel(profA, c), sched),
               "B  median + residuals + Keller": run(kel(profB, c), sched),
               "C  Bayes + exact DP": run(lambda i: dp_checks(sampA[i], c), sched),
               "D  median + residuals + exact DP": run(lambda i: dp_checks(sampB[i], c), sched)}
        bv = res["backoff (tuned)"][:, 0] + c * res["backoff (tuned)"][:, 1]
        av = res["A  Bayes + Keller"][:, 0] + c * res["A  Bayes + Keller"][:, 1]
        lines.append(f"\n   c = {c} ms")
        for name, sm in res.items():
            v = sm[:, 0] + c * sm[:, 1]
            d = [1 - v[b].mean() / bv[b].mean() for b in boots]; dA = [v[b].mean() / av[b].mean() - 1 for b in boots]
            lines.append(f"   {name:34s} delay {sm[:,0].mean():8.0f}  checks {sm[:,1].mean():6.2f}  objective {v.mean():9.0f}"
                         + ("" if name.startswith("backoff") else f"   vs backoff {1 - v.mean()/bv.mean():+6.1%} [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]")
                         + ("" if name.startswith(("backoff", "A ")) else f"   vs A {v.mean()/av.mean()-1:+6.1%} [{np.percentile(dA,2.5):+.1%}, {np.percentile(dA,97.5):+.1%}]"))
            summ[f"c={c}|{name}"] = dict(objective=float(v.mean()), vs_backoff=float(1 - v.mean() / bv.mean()))
        print("\n".join(lines[-6:]), flush=True)
    # in-model optimality gap: expected objective under the predictive itself (fresh Monte-Carlo draws)
    sub = sorted(sched)[:: max(1, len(sched) // 400)]; mc = np.random.default_rng(3)
    lines.append("\n   in-model check (expected objective under the model's own predictive, 400 targets): Keller / exact DP - 1")
    for c in RO.COSTS:
        gaps = []
        for i in sub:
            loc, sc, dof = R[i]["pred"]; d = np.exp(np.clip(loc + sc * mc.standard_t(dof, 4000), 1.0, 21.0))
            sk, sd = kel(profA, c)(i), dp_checks(d[1000:], c)      # Keller vs DP fitted on independent draws
            cost = lambda sch: np.mean([(lambda r: r[0] + c * r[1])(simulate(sch, t)) for t in d[:1000]])
            ek, ed = cost(sk), cost(sd)
            gaps.append(ek / ed - 1)
        lines.append(f"   c = {c:5d} ms: median gap {np.median(gaps):+.1%}, mean {np.mean(gaps):+.1%}, P90 {np.percentile(gaps, 90):+.1%}")
        print(lines[-1], flush=True)
    open(os.path.join(RO.OUT, f"ablate_{mode}.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(summ, open(os.path.join(RO.OUT, f"ablate_{mode}.json"), "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
