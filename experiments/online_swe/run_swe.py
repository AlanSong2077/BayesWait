"""Online experiment: a real LLM agent solves SWE-bench Verified Mini tasks while its shell commands run
asynchronously; the waiting policy decides when the agent gets to see each command's result.

  python experiments/online_swe/run_swe.py                 # 50 tasks x {backoff, ours}, resumable
  python experiments/online_swe/run_swe.py --limit 2       # pilot on 2 tasks
  python experiments/online_swe/analyze_swe.py

Agent: the mini-swe-agent (v2.4.6) SWE-bench prompts and text action format (mini_swe_agent_swebench.yaml), one
bash command per step, executed with `docker exec` in the task's official SWE-bench image (cwd /testbed,
60 s command timeout), DeepSeek as the model (DS_MODEL, temperature 0), at most STEP_LIMIT steps.

Waiting. Every command runs in the background. The harness looks at it at the policy's check times; if it is
still running the agent is told so and must answer (one real LLM round: that is the cost of a check), after which
the harness waits for the next check time. When a check finds the command finished, its output becomes the
next observation.
  backoff  exponential backoff with the parameters that are optimal on the offline SWE-bench trace for the
           current check cost (backoff_table.json, tuned offline on the tuning block)
  ours     Bayes + Keller: hierarchical Bayesian predictive of the runtime (hyper-parameters fitted offline,
           history warm-started with the offline SWE-bench calls of the other tasks and updated with every finished live command)
           -> Keller checks n(t) = sqrt(h(t) / 2c)
The check cost c used for planning is the mean LLM round latency measured so far in this run (both policies).
The two policies of one task run at the same time in two containers, so they see the same machine load.
"""
import argparse, json, math, os, random, re, subprocess, sys, threading, time, urllib.request, uuid
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE); ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT); sys.path.insert(0, EXP)
os.environ.setdefault("MAX_USD", "40")
import numpy as np, yaml
from jinja2 import StrictUndefined, Template
import deepseek as D
from bayeswait.bayes import HB, keller_checks
from bayeswait.baselines import backoff_policy
from bayeswait.families import describe_swe

OUT = os.path.join(EXP, "results", "online_swe"); os.makedirs(OUT, exist_ok=True)
EPIS = os.path.join(OUT, "episodes.jsonl"); DATA = os.path.join(ROOT, "data", "swe_verified_mini.json")
CFG = yaml.safe_load(open(os.path.join(HERE, "mini_swe_agent_swebench.yaml")))
STEP_LIMIT = int(os.environ.get("STEP_LIMIT", 60)); CMD_TIMEOUT = 60; MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 4000))
DOCKER = os.environ.get("DOCKER", "docker"); PLATFORM = os.environ.get("PLATFORM", "linux/amd64")
ACTION_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.S)
STATUS = ("<status>The command is still running ({elapsed:.1f} s so far). Its output will be shown when it finishes. "
          "Reply with a short THOUGHT and the command `wait` in the usual format.</status>")
LOCK = threading.Lock(); C_ROUNDS = []        # measured LLM round latencies (s), shared by both policies

# ------------------------------------------------------------------ data
def instances():
    if not os.path.exists(DATA):
        rows, off = [], 0
        while True:
            u = ("https://datasets-server.huggingface.co/rows?dataset=MariusHobbhahn/swe-bench-verified-mini"
                 f"&config=default&split=test&offset={off}&length=100")
            d = json.loads(urllib.request.urlopen(u, timeout=60).read()); rows += [r["row"] for r in d["rows"]]
            if len(rows) >= d["num_rows_total"]: break
            off += 100
        json.dump(rows, open(DATA, "w"))
    return json.load(open(DATA))

def image(iid): return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()

# ------------------------------------------------------------------ the Bayesian model, shared and causal
class Model:
    def __init__(self, exclude=()):
        self.h = json.load(open(os.path.join(EXP, "results", "swe.json")))["hyper"]; self.m = HB(**self.h); self.lock = threading.Lock()
        n = 0; exclude = set(exclude)     # the online tasks themselves are left out of the warm start
        for l in open(os.path.join(ROOT, "data", "swe_calls.jsonl")):          # warm start with the offline trace
            r = json.loads(l)
            if r["dur_s"] <= 0 or r["rc"] == -1 or r["dur_s"] < 0.1 or r["iid"] in exclude: continue
            repo = r["iid"].rsplit("-", 1)[0].replace("__", "/"); k = describe_swe(r["cmd"], repo, r["iid"])
            self.m.update(self.chain(k), self.x(k), math.log(r["dur_s"] * 1000)); n += 1
        print(f"model warm-started with {n} offline calls", flush=True)
    chain = staticmethod(lambda k: (("root",), ("glob", k["glob"]), ("l3", k["l3"]), ("l2", k["l2"]), ("exact", k["exact"])))
    x = staticmethod(lambda k: math.log(k["scale"]) if k.get("scale") else 0.0)
    def predict(self, k):
        with self.lock: return self.m.predict(self.chain(k), self.x(k))
    def learn(self, k, ms):
        if ms < 100: return                                                    # same targets as offline
        with self.lock: self.m.update(self.chain(k), self.x(k), math.log(ms))

BACKOFF = {int(k): v for k, v in json.load(open(os.path.join(HERE, "backoff_table.json"))).items()}
def c_ms():
    with LOCK: return 1000.0 * (float(np.mean(C_ROUNDS[-200:])) if C_ROUNDS else 3.0)
def plan(policy, model, k):
    c = c_ms()
    if policy == "backoff":
        st, f = BACKOFF[min(BACKOFF, key=lambda x: abs(math.log(x / c)))]; return backoff_policy(st, f, 10_000.0), c
    return keller_checks(*model.predict(k), c), c

# ------------------------------------------------------------------ one episode
def render(t, **kw): return Template(t, undefined=StrictUndefined).render(**kw)

def run_episode(inst, policy, model, log):
    iid = inst["instance_id"]; repo = inst["repo"]; name = f"bw-{policy}-{uuid.uuid4().hex[:6]}"
    env = sum([["-e", f"{a}={b}"] for a, b in CFG["environment"]["env"].items()], [])
    cid = subprocess.run([DOCKER, "run", "-d", "--platform", PLATFORM, "--name", name, "-w", "/testbed", "--rm", image(iid), "sleep", "3h"],
                         capture_output=True, text=True, check=True).stdout.strip()
    msgs = [{"role": "system", "content": CFG["agent"]["system_template"]},
            {"role": "user", "content": render(CFG["agent"]["instance_template"], task=inst["problem_statement"])}]
    ep = dict(instance_id=iid, policy=policy, commands=[], llm_rounds=0, status_rounds=0, tokens=0, prompt_tokens=0,
              usd=0.0, exit="StepLimit", submission="", t_start=time.time())
    usd0 = D.spent()
    def llm():
        text, u, dt = D.chat(msgs, max_tokens=MAX_TOKENS); ep["llm_rounds"] += 1
        ep["tokens"] += u.get("total_tokens", 0); ep["prompt_tokens"] += u.get("prompt_tokens", 0)
        with LOCK: C_ROUNDS.append(dt)
        msgs.append({"role": "assistant", "content": text}); return text
    try:
        for step in range(STEP_LIMIT):
            text = llm(); acts = ACTION_RE.findall(text)
            if len(acts) != 1:
                msgs.append({"role": "user", "content": render(CFG["model"]["format_error_template"], error="", actions=acts)}); continue
            cmd = acts[0]; k = describe_swe(cmd, repo, iid)
            sched, c = plan(policy, model, k)
            t0 = time.time(); st = {}
            p = subprocess.Popen([DOCKER, "exec", "-w", "/testbed", *env, cid, "bash", "-c", cmd], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            def waiter():
                try: out, _ = p.communicate(timeout=CMD_TIMEOUT); st.update(out=out, rc=p.returncode, exc="")
                except subprocess.TimeoutExpired:
                    p.kill(); out, _ = p.communicate(); st.update(out=out or "", rc=-1, exc=f"Command timed out after {CMD_TIMEOUT} s")
                st["t_end"] = time.time()
            th = threading.Thread(target=waiter, daemon=True); th.start()
            j, checks, srounds, stok = 0, 0, 0, 0
            while True:
                target = t0 + (sched[j] / 1000.0 if j < len(sched) else 0.0)
                if target > time.time(): time.sleep(target - time.time())
                checks += 1; j += 1
                if "t_end" in st: break
                tk0 = ep["tokens"]
                msgs.append({"role": "user", "content": STATUS.format(elapsed=time.time() - t0)}); llm()
                srounds += 1; stok += ep["tokens"] - tk0; ep["status_rounds"] += 1
            t_seen = time.time(); ms = (st["t_end"] - t0) * 1000.0
            if st["rc"] != -1: model.learn(k, ms)
            ep["commands"].append(dict(cmd=cmd[:300], runner=k["runner"], runtime_ms=ms, delay_ms=(t_seen - st["t_end"]) * 1000.0,
                                       checks=checks, status_rounds=srounds, status_tokens=stok, rc=st["rc"], c_plan_ms=c,
                                       first_check_ms=sched[0] if sched else None))
            out = dict(output=st["out"], returncode=st["rc"], exception_info=st["exc"])
            lines = st["out"].lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and st["rc"] == 0:
                ep["exit"] = "Submitted"; ep["submission"] = "".join(lines[1:]); break
            msgs.append({"role": "user", "content": render(CFG["model"]["observation_template"], output=out)})
    except D.OutOfBudget:
        ep["exit"] = "OutOfBudget"; raise
    except Exception as e:
        ep["exit"] = f"Error: {type(e).__name__}: {str(e)[:200]}"
    finally:
        ep["wall_s"] = time.time() - ep["t_start"]; ep["usd"] = D.spent() - usd0
        subprocess.Popen(f"{DOCKER} rm -f {cid} >/dev/null 2>&1", shell=True)
        json.dump(dict(ep, messages=msgs), open(os.path.join(OUT, "traj", f"{iid}.{policy}.json"), "w"))
        with LOCK: open(EPIS, "a").write(json.dumps(ep) + "\n")
        cm = ep["commands"]
        log(f"{iid:32s} {policy:7s} {ep['exit']:10s} steps {ep['llm_rounds'] - ep['status_rounds']:3d}  status rounds {ep['status_rounds']:3d}  "
            f"wait delay {sum(c['delay_ms'] for c in cm)/1000:6.1f} s  wall {ep['wall_s']/60:5.1f} min  ${ep['usd']:.2f}  (total ${D.spent():.2f})")
    return ep

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--policies", default="backoff,ours")
    a = ap.parse_args(); os.makedirs(os.path.join(OUT, "traj"), exist_ok=True)
    insts = instances(); random.Random(0).shuffle(insts)
    if a.limit: insts = insts[: a.limit]
    pols = a.policies.split(",")
    done = {(e["instance_id"], e["policy"]) for e in map(json.loads, open(EPIS))} if os.path.exists(EPIS) else set()
    log = lambda s: print(time.strftime("%H:%M:%S ") + s, flush=True)
    log(f"{len(insts)} tasks x {pols}; {len(done)} episodes already done; model {D.MODEL}; budget ${D.MAX_USD}, spent ${D.spent():.2f}")
    model = Model(exclude=[i["instance_id"] for i in instances()])
    for n, inst in enumerate(insts, 1):
        todo = [p for p in pols if (inst["instance_id"], p) not in done]
        if not todo: continue
        img = image(inst["instance_id"]); log(f"[{n}/{len(insts)}] pulling {img}")
        try: pr = subprocess.run([DOCKER, "pull", "--platform", PLATFORM, img], capture_output=True, text=True)
        except FileNotFoundError:
            log(f"   '{DOCKER}' not found: install / start Docker Desktop (or set DOCKER=/path/to/docker)"); return
        if pr.returncode != 0:
            err = (pr.stderr or pr.stdout).strip().splitlines()[-1:] or [""]
            log(f"   could not pull {img}: {err[0][:300]}")
            if "daemon" in err[0].lower(): log("   Docker is not running: start Docker Desktop and rerun"); return
            continue
        errs = []
        def go(p):
            try: run_episode(inst, p, model, log)
            except D.OutOfBudget as e: errs.append(e)
        ths = [threading.Thread(target=go, args=(p,)) for p in todo]
        [t.start() for t in ths]; [t.join() for t in ths]
        if os.environ.get("RM_IMAGES"): subprocess.run([DOCKER, "rmi", img], capture_output=True)   # save disk space
        if errs: log(f"stopped: {errs[0]}"); return
    log("all tasks done")

if __name__ == "__main__":
    main()
