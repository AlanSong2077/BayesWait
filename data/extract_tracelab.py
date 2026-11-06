"""TraceLab (UW SyFI) real Claude Code / Codex sessions -> shell tool calls with wall latency."""
import gzip, json, sys, datetime as dt
src, dst = sys.argv[1], sys.argv[2]
ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
n = 0
with gzip.open(src, "rt") as f, open(dst, "w") as o:
    for line in f:
        r = json.loads(line)
        for t in r["tools"]:
            if t.get("tool_name") not in ("Bash", "exec_command", "shell_command", "shell") or not t.get("command_skeleton"): continue
            lat = t.get("tool_internal_latency_ms") or t.get("tool_wall_latency_ms")
            if lat is None or not t.get("emitted_at"): continue
            st = ts(t["emitted_at"]); o.write(json.dumps(dict(
                iid=r["session_id"], repo=r.get("project") or r.get("user"), provider=r["provider"], cmd=t["command_skeleton"],
                in_chars=t.get("input_chars"), dur_s=lat / 1000.0, rc=(1 if t.get("is_error") else 0),
                t_start=st, t_end=st + lat / 1000.0, wall_ms=t.get("tool_wall_latency_ms"), internal_ms=t.get("tool_internal_latency_ms"))) + "\n")
            n += 1
print(n, "shell calls")
