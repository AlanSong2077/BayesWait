# Data

| file | source | rows |
|---|---|---|
| `swe_calls.jsonl` (6.6 MB, shipped) | SWE-bench Verified, mini-SWE-agent v2.4.2 + Gemini 3.5 Flash public trajectories: https://github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash (`trajs/`, 441 tasks) | 19,073 tool calls |
| `tracelab_calls.jsonl` (110 MB, **not shipped**) | TraceLab, UW SyFI: https://github.com/uw-syfi/TraceLab, release file `syfi_coding_trace.jsonl.gz` (real Claude Code / Codex sessions) | 415,279 shell calls (6,216 sessions, 365 projects) |
| `leetcode_trials.jsonl` (shipped) | LeetCode suite: a DeepSeek agent executed solution programs of LeetCode problems, 5 repetitions per problem at input size k, on the author's laptop (collected for the earlier version of this work) | 3,042 trials (used: first 543 ids, successful runs: 2,622) |
| `local_trials.jsonl` (shipped) | local-tool suite: csv / jsonl / gzip / matrix workloads at several input sizes, 3 repetitions each, same laptop | 120 trials |

Regenerate the trace files:

    git clone https://github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash
    python3 data/extract_swe.py 20260901_mini-v2.4.2_gemini-3-5-flash/trajs data/swe_calls.jsonl

    # download syfi_coding_trace.jsonl.gz from the TraceLab GitHub release, then
    python3 data/extract_tracelab.py syfi_coding_trace.jsonl.gz data/tracelab_calls.jsonl

Fields: `iid` task / session, `repo` (TraceLab: project), `cmd` command (TraceLab: anonymised skeleton of
executables), `dur_s` runtime, `rc` return code (-1 = killed by the 60 s mini-SWE-agent timeout, excluded),
`t_start`, `t_end` wall-clock seconds.

Runtime definitions: SWE-bench = observation timestamp - assistant timestamp in the trajectory (command run in
the task's Docker container); TraceLab = the runner's internal command latency when reported (Codex), else
tool wall latency (Claude Code).

Suite fields: `task_id`, `k` / `predicted_input_size` (workload scale), `actual_runtime_ms`, `llm_prior_ms`
(DeepSeek's runtime estimate for the command, asked before the run), `success`.
