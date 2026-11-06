"""What does one status check cost when the agent itself performs it (one extra LLM round)?
Measured on TraceLab: rounds that start from a tool result, latency from the tool result to the model's last
output event, and input/output tokens of that round.   Usage: python3 experiments/check_cost.py syfi_coding_trace.jsonl.gz"""
import collections, datetime as dt, gzip, json, sys
import numpy as np
ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
lat, tok = collections.defaultdict(list), collections.defaultdict(list)
for line in gzip.open(sys.argv[1], "rt"):
    r = json.loads(line); E = r["timing_events"]
    if not E or r.get("first_input_event_type") != "tool_result": continue
    ins = [ts(e["timestamp"]) for e in E if e["event_type"] in ("tool_result", "user_message")]
    outs = [ts(e["timestamp"]) for e in E if e["event_type"] in ("text", "tool_call", "reasoning")]
    if not ins or not outs or max(outs) <= max(ins): continue
    lat[r["provider"]].append(max(outs) - max(ins))
    tok[r["provider"]].append((r["input_tokens_total"] or 0, r["prefix_tokens"] or 0, r["output_tokens"] or 0))
for p in lat:
    a = np.array(lat[p]); t = np.array(tok[p])
    print(f"{p:6s} rounds={len(a):7d}  latency s: median {np.median(a):.1f} (IQR {np.percentile(a,25):.1f}-{np.percentile(a,75):.1f})  "
          f"input tokens median {np.median(t[:,0]):.0f} (cached prefix {np.median(t[:,1]):.0f})  output {np.median(t[:,2]):.0f}")
