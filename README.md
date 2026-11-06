# BayesWait: Bayesian runtime estimation + optimal status checks for agent tool polling

An agent that starts a long tool call (a test run, a build) must decide when to look at its status. Every look
costs one more LLM round (median 5-7 s and ~120-140k input tokens on real Claude Code / Codex sessions,
`experiments/results/check_cost.txt`); looking too late leaves the agent idle. BayesWait chooses the check
times in two steps.

1. **Estimate** (`bayeswait/bayes.py`, class `HB`). A hierarchical Bayesian model of the call's runtime T.
   With y = log T and x = log(workload scale): y = b0 + b1 x + b2 x^2 + Student-t noise; nested levels
   root -> runner -> repo/runner -> template -> exact command; each node is a conjugate Normal-Inverse-Gamma
   update whose prior is its parent's posterior (credibility weighting); old runs are discounted per level.
   Output: a Student-t predictive distribution of log T. Mean / last-value / EMA estimates, linear and
   quadratic growth in the input size, and per-family model choice are all special cases (discount 1 / 0 /
   in between; slope 1 / 2; shrinkage of slope and curvature). Optional: an LLM runtime guess is fused by
   precision weighting, its error variance learned from finished calls (`fuse`).
2. **Decide** (`keller_checks`). Minimising E[detection delay] + c E[number of checks] gives the optimal
   checking density n(t) = sqrt(h(t) / 2c) (Keller 1974), h = hazard rate of the predictive distribution:
   checks are dense where the job is likely to finish and thin out in the tail. Closed form, nothing to tune.

```
bayeswait/
  bayes.py               hierarchical Bayesian model, LLM-estimate fusion, Keller checks
  families.py            command -> hierarchy keys + workload scale (SWE-bench commands, TraceLab skeletons)
  baselines.py           exponential backoff, fixed interval, estimate-then-poll; replay simulator
data/                    call tables of the four benchmarks + extraction scripts (data/README.md)
experiments/
  run_offline.py         all offline results: leet | local | swe | tracelab
  check_cost.py          cost of one status check (LLM round latency / tokens on TraceLab)
  deepseek.py            minimal DeepSeek client (key from env or ~/Desktop/ClaudeFile/.env, never printed;
                         spend ledger results/deepseek_ledger.json, hard cap MAX_USD, default 3)
  llm_prior_swe.py       LLM runtime estimates for the SWE-bench calls (API)
  ablate.py              ablations: Bayes vs median + residuals, Keller vs exact DP, in-model optimality gap
  components.py          component ablations of the model (hierarchy, discount, t-noise, curvature)
  robustness.py          misspecified check cost, resampling, overhead
  online_swe/            online experiment: SWE-bench Verified Mini, backoff vs ours (API + Docker; README inside)
  results/               *.txt human-readable, *.json numbers + fitted hyper-parameters
```

## Reproduce
CPU only (numpy, scipy):
```
python3 experiments/run_offline.py leet                                            # ~1 min
HYPER_FROM=experiments/results/leet.json python3 experiments/run_offline.py local   # seconds
python3 experiments/run_offline.py swe data/swe_calls.jsonl                         # ~10 min
FIT_START=experiments/results/swe.json FIT_ROUNDS=2 FIT_WORKERS=2 \
    python3 experiments/run_offline.py tracelab data/tracelab_calls.jsonl               # ~10 min (needs the extracted file)
```
With a DeepSeek key (on a machine that can reach api.deepseek.com):
```
python3 experiments/llm_prior_swe.py              # ~4.7k estimate queries, < $3, cached / resumable
LLM_PRIOR=data/swe_llm_prior.json TAG=_llm python3 experiments/run_offline.py swe data/swe_calls.jsonl
# online experiment: see experiments/online_swe/README.md
```

## Protocol
Counterfactual replay of recorded runtimes; strictly causal (a runtime is used only after its call
finished). LeetCode / local suites: first 30% of warm runs (runs of a task that ran before) tune, the rest
test. Traces: targets (successful calls >= 100 ms) in 10 time blocks; blocks 1-2 history only, block 3 tunes,
blocks 4-10 test; scheduling on 4,000 random test targets. The model's hyper-parameters maximise the log
predictive density on the tuning part (empirical Bayes); each baseline's parameters (backoff start / factor,
fixed interval) minimise the objective on the tuning part. The local suite has only 34 tuning runs, so it
reuses the LeetCode hyper-parameters (same harness and machine). Objective = mean detection delay + c x mean
number of checks, c = 50 ms / 200 ms / 1 s / 5 s. TraceLab: the hyper-parameter search starts from the SWE-bench
values and runs 2 coordinate rounds on 2 cores (`FIT_START`, `FIT_ROUNDS`, `FIT_WORKERS`). 95% CIs: bootstrap over tasks / sessions.

## Results

### Prediction, test part (MdAPE of the point estimate; P90 APE in brackets)
| data | LLM estimate | last run (x size ratio) | median, deepest history level | ours |
|---|---|---|---|---|
| LeetCode (1,456 warm runs) | 91.7% (2442%) | 5.4% (27%) | 29.6% (92%) | **3.8% (18%)** |
| local tools (82) | 92.5% (364%) | 9.3% (88%) | 11.7% (95%) | **7.7% (34%)** |
| SWE-bench Verified (13,256 calls) | (API run pending) | 542.8%* | **5.2%** (55%) | 5.7% (55%) |
| TraceLab (195,902 calls, Claude Code / Codex) | - | 51.6% (549%) | **29.4%** (192%) | 30.1% (**177%**) |

\* only 8% of SWE-bench test calls have an identical earlier command; without one this baseline falls back to 1 s.
Ours also gives calibrated intervals: 80% predictive intervals cover 88% (LeetCode), 87% (SWE-bench) and 81% (TraceLab)
of test runs; on the traces a plain history median is as accurate a point estimate, the gain is the distribution.

### Status checking: objective reduction vs tuned exponential backoff (positive = better; 95% CI)
| data | c = 50 ms | c = 200 ms | c = 1 s | c = 5 s |
|---|---|---|---|---|
| LeetCode | **+57.0%** [53.6, 59.5] | **+48.1%** [44.8, 50.8] | **+32.7%** [30.0, 35.3] | **+24.0%** [20.3, 27.4] |
| local tools | **+39.9%** [14.2, 51.0] | **+34.4%** [1.3, 47.6] | **+41.5%** [37.8, 43.8] | **+22.3%** [16.7, 26.8] |
| SWE-bench | **+8.3%** [4.2, 12.2] | **+12.3%** [10.5, 14.3] | **+16.1%** [14.0, 18.2] | **+7.7%** [6.7, 8.8] |
| TraceLab | **+21.6%** [17.1, 26.0] | **+9.4%** [5.5, 13.3] | **+19.5%** [15.2, 23.6] | **+21.6%** [17.1, 26.0]00 |

Other baselines at c = 200 ms: tuned fixed interval -78% (LeetCode), -26% (local), -64% (SWE-bench),
-116% (TraceLab); LLM estimate + 100 ms polling -987% (LeetCode), -113% (local). Full tables:
`experiments/results/{leet,local,swe,tracelab}.txt`.

### Ablations and robustness (experiments/results/ablate_*, components_*, robustness_*)
* Bayesian predictive vs history median + empirical residuals (both with Keller): ours lower objective in all
  16 dataset x cost settings, significantly in 12 (LeetCode +36-91%, local +6-20%, SWE-bench 0-9%, TraceLab 5-50%).
* Keller vs the exact optimum (DP) of the same predictive: in-model gap +1-3% at c >= 1 s, up to +20% at c = 50 ms;
  on real data the DP is worse than Keller everywhere (it trusts the predictive tail; Keller degrades gracefully).
* Components: the hierarchy is essential (SWE-bench c = 200 ms: +11.3% -> -0.6% without it), so is the discount
  on traces and the t-noise for accuracy; the curvature term matters on workloads with a start-up cost (local).
  One discount shared by all levels is as good as one per level.
* Resampling 5 x 4,000 test calls: SWE-bench +11.6% +- 0.8% (c = 200 ms), +14.5% +- 0.8% (1 s); TraceLab
  +9.0% +- 2.0%, +17.3% +- 3.9%. Planning with c off by 3x keeps most of the gain; off by 10x loses to backoff on
  the traces, so c is measured online (mean LLM round latency).
* Overhead: 0.02-0.36 ms per model update / prediction, 1.6-3.1 ms per Keller schedule.
* Theory (docs/theory.md): Keller density, scale equivariance, optimal gaps grow like sqrt(t) up to log factors.

## Related work
Optimal checking: Barlow, Hunter & Proschan (1963); Keller, "Optimum checking schedules for systems subject
to random failure", Management Science 21(3), 1974. Bayesian / history-based runtime prediction: Lotaru
(Bayesian linear regression on input size), Kalman-filter HPC runtime prediction, 3Sigma. Agent serving:
Continuum (tool-duration CDF for KV-cache TTL), "Ask the Tool" (tool progress signals). Industry polling:
US 8,392,923 (polling intervals from historical completion times).
