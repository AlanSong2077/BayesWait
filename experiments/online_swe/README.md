# Online experiment: SWE-bench Verified Mini, exponential backoff vs Bayes + Keller

50 tasks (MariusHobbhahn/swe-bench-verified-mini) x 2 policies, paired; a DeepSeek agent with the mini-swe-agent
SWE-bench prompts; every shell command runs in the background and the policy decides when the agent sees its
result (details: docstring of `run_swe.py`).

## Setup (once, on your Mac)
1. Docker Desktop running; in Settings -> General enable "Use Rosetta for x86_64/amd64 emulation on Apple
   Silicon" (the official SWE-bench images are x86_64). Disk: ~10-20 GB for the images (or set `RM_IMAGES=1` to
   delete each image after its task).
2. In the project folder:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install numpy scipy jinja2 pyyaml
   ```

## Run
```
source .venv/bin/activate
python experiments/online_swe/run_swe.py --limit 2      # pilot: 2 tasks, check time and cost per task
python experiments/online_swe/run_swe.py                # all 50 tasks (resumable: rerun to continue)
python experiments/online_swe/analyze_swe.py            # -> experiments/results/online_swe/summary.txt
```
Budget: `MAX_USD` (default 40 here) is a hard cap on DeepSeek spend (ledger in experiments/results/deepseek_ledger.json).
Other knobs: `STEP_LIMIT` (60 agent steps per task), `DS_MODEL`, `PLATFORM` (linux/amd64).

## Optional: resolve rates (official SWE-bench harness)
```
pip install swebench
python -m swebench.harness.run_evaluation --dataset_name MariusHobbhahn/swe-bench-verified-mini --split test \
    --predictions_path experiments/results/online_swe/preds_ours.json --run_id bw_ours --max_workers 2
python -m swebench.harness.run_evaluation --dataset_name MariusHobbhahn/swe-bench-verified-mini --split test \
    --predictions_path experiments/results/online_swe/preds_backoff.json --run_id bw_backoff --max_workers 2
```
