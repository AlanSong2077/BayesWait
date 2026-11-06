"""Paper-style LLM runtime priors for the SWE-bench trace (run on your machine; needs the DeepSeek key).

Queries one prior per unique (repo, command) among exactly the targets that run_offline.py uses for
calibration (block 3 sample) and scheduling (test sample), same seeds. Cached and resumable:
data/swe_llm_prior.json.  Budget: MAX_USD (default 3).

  python3 experiments/llm_prior_swe.py
  LLM_PRIOR=data/swe_llm_prior.json TAG=_llm python3 experiments/run_offline.py swe data/swe_calls.jsonl
"""
import concurrent.futures as cf, hashlib, json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from bayeswait.families import clean
import deepseek as D

DATA = os.path.join(os.path.dirname(HERE), "data"); CALLS = os.path.join(DATA, "swe_calls.jsonl"); OUT = os.path.join(DATA, "swe_llm_prior.json")
SCHED_N = int(os.environ.get("SCHED_N", 4000))

def prior_key(repo, cmd): return hashlib.sha1((repo + "\n" + clean(cmd)).encode()).hexdigest()[:16]

def subset():
    rows = [json.loads(l) for l in open(CALLS)]; rows = [r for r in rows if r["dur_s"] > 0]; rows.sort(key=lambda r: r["t_start"])
    tgt = [i for i, r in enumerate(rows) if r["rc"] != -1 and r["dur_s"] * 1000 >= 100]
    rng = np.random.default_rng(0); blocks = np.array_split(np.array(tgt), 10)
    calib = rng.choice(blocks[2], min(len(blocks[2]), 1500), replace=False)
    test_all = np.concatenate(blocks[3:]); sched = rng.choice(test_all, min(len(test_all), SCHED_N), replace=False)
    need = {}
    for i in np.concatenate([calib, sched]):
        r = rows[int(i)]; repo = r["iid"].rsplit("-", 1)[0].replace("__", "/")
        need[prior_key(repo, r["cmd"])] = (repo, clean(r["cmd"]))
    return need

def main():
    need = subset(); cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [(k, v) for k, v in need.items() if k not in cache]
    print(f"{len(need)} unique commands needed, {len(todo)} to query (model {D.MODEL}, budget ${D.MAX_USD})", flush=True)
    lock = threading.Lock(); t0 = time.time()
    def one(item):
        k, (repo, cmd) = item
        ctx = f"Command type: shell command run by a coding agent inside the Docker container of a SWE-bench task\nRepository: {repo}"
        return k, D.runtime_prior(ctx, cmd)
    done = 0
    with cf.ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in cf.as_completed(futs):
            try: k, v = f.result()
            except D.OutOfBudget as e:
                print("stopped:", e); [g.cancel() for g in futs]; break
            with lock:
                cache[k] = v; done += 1
                if done % 100 == 0 or done == len(todo):
                    json.dump(cache, open(OUT, "w"))
                    print(f"  {done}/{len(todo)}  ${D.spent():.2f}  {time.time()-t0:.0f}s", flush=True)
    json.dump(cache, open(OUT, "w")); print(f"saved {len(cache)} priors -> {OUT}; spent ${D.spent():.2f}")

if __name__ == "__main__":
    main()
