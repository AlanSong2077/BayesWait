"""Minimal DeepSeek client for the experiments that need an LLM (run on your own machine).

The API key is read locally (env DEEPSEEK_API_KEY, else ENV_FILE, else ~/Desktop/ClaudeFile/.env) and is never
printed or written anywhere. Spending is tracked in experiments/results/deepseek_ledger.json and capped by
MAX_USD (default 3). Prices (USD / 1M tokens) are env-overridable: P_IN_MISS, P_IN_HIT, P_OUT.
"""
import json, os, threading, time, urllib.error, urllib.request

BASE = os.environ.get("DS_BASE", "https://api.deepseek.com")
MODEL = os.environ.get("DS_MODEL", "deepseek-flash")
P_IN_MISS = float(os.environ.get("P_IN_MISS", 0.56)); P_IN_HIT = float(os.environ.get("P_IN_HIT", 0.07)); P_OUT = float(os.environ.get("P_OUT", 1.68))
MAX_USD = float(os.environ.get("MAX_USD", 3))
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "deepseek_ledger.json")
_lock = threading.Lock(); _key = None

class OutOfBudget(Exception): pass

def key():
    global _key
    if _key: return _key
    if os.environ.get("DEEPSEEK_API_KEY"): _key = os.environ["DEEPSEEK_API_KEY"].strip(); return _key
    for p in (os.environ.get("ENV_FILE"), os.path.expanduser("~/Desktop/ClaudeFile/.env")):
        if p and os.path.exists(p):
            lines = [l.strip() for l in open(p) if l.strip() and not l.strip().startswith("#")]
            for l in lines:
                k, _, v = l.partition("=")
                if k.replace("export ", "").strip() in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "API_KEY") and v.strip():
                    _key = v.strip().strip("'\""); return _key
            sk = [l.strip("'\" ") for l in lines if l.strip("'\" ").startswith("sk-")]
            if len(sk) == 1: _key = sk[0]; return _key
    raise RuntimeError("DeepSeek API key not found (DEEPSEEK_API_KEY or ~/Desktop/ClaudeFile/.env)")

def spent():
    try: return json.load(open(LEDGER))["usd"]
    except Exception: return 0.0

def _charge(u):
    hit = u.get("prompt_cache_hit_tokens", 0); miss = u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0) - hit)
    usd = (hit * P_IN_HIT + miss * P_IN_MISS + u.get("completion_tokens", 0) * P_OUT) / 1e6
    with _lock:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        s = spent() + usd; json.dump({"usd": s}, open(LEDGER, "w"))

def chat(messages, max_tokens=400, temperature=0.0):
    """-> (text, usage, latency_s). Retries empty replies of reasoning models with a larger budget."""
    if spent() >= MAX_USD: raise OutOfBudget(f"budget {MAX_USD} USD reached")
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}).encode()
    for attempt in range(5):
        t0 = time.time()
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r: d = json.loads(r.read())
            u = d.get("usage", {}); _charge(u); ch = d["choices"][0]; text = ch["message"].get("content") or ""
            if not text.strip() and ch.get("finish_reason") == "length" and max_tokens < 3200:
                return chat(messages, max_tokens * 4, temperature)
            return text, u, time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code == 402: raise OutOfBudget("DeepSeek account balance exhausted (HTTP 402)") from None
            if e.code in (429, 500, 502, 503) and attempt < 4: time.sleep(2 ** attempt); continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4: time.sleep(2 ** attempt); continue
            raise

PRIOR_PROMPT = ("Estimate the wall-clock runtime in milliseconds for the following shell command.\n"
                "Output ONLY compact JSON: {\"estimated_ms\": <number>}.\n")   # same prompt as the paper

def runtime_prior(context, command, fallback_ms=2000.0):
    """paper-style LLM runtime prior -> ms (fallback on any failure)."""
    try:
        text, _, _ = chat([{"role": "user", "content": PRIOR_PROMPT + context + "\nCommand:\n" + command[:4000]}], max_tokens=400)
        a, b = text.find("{"), text.rfind("}")
        v = float(json.loads(text[a:b + 1])["estimated_ms"])
        return v if v > 0 else fallback_ms
    except OutOfBudget:
        raise
    except Exception:
        return fallback_ms
