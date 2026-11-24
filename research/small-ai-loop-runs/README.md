# Feedback-loop runs with local models: launch scripts

This folder contains the scripts that started the feedback-loop runs of the small AI harness (`src/ai4data/small_ai`)
in November 2026. Training session 02 (`training/session-02-small-ai-synthetic-data`) uses the outputs of three of
these runs: one on ISCO-08 occupation titles and two on report topics. The run folders are not stored in the
repository.

## Scripts

| Script | Purpose |
|---|---|
| `serve_local.sh` | Starts one vLLM server: `qwen` (Qwen3.8-27B, GPU 0, port 8100) or `gemma` (Gemma 4 31B, GPU 1, port 8102). |
| `env.sh` | Settings shared by the other scripts: the writer and orchestrator on the Qwen server, the labeller on the Gemma server, and the run folder. |
| `run_loop.sh` | Runs one experiment: round 0, the targeted arm with the orchestrator, the control arm, the final training of both arms with seeds 42, 1 and 2, and the large models asked directly. |
| `baselines.sh` | Asks Qwen3.8-27B, Gemini 3.8 Flash and Claude Sonnet 5.5 for labels on the report set of a task and scores the replies. |
| `compare.py` | Paired bootstrap over the report set for the difference between two run folders. |

`run_loop.sh` and `baselines.sh` skip a step when its output exists, so an interrupted run continues from the
last finished step.

## Settings

| Variable | Meaning |
|---|---|
| `PY` | Python with the harness dependencies. The default is `python`. |
| `SMALL_AI_RUNS` | Folder of the run folders, logs and baseline replies. The default is `runs/` in the repository. |
| `OPENROUTER_API_KEY` | Key for Gemini 3.8 Flash and Claude Sonnet 5.5 in `baselines.sh`. The scripts read it from the environment and do not print it. `run_loop.sh` removes it from the environment of the loop, because the loop uses the local servers only. |
| `VLLM`, `HF_HUB_CACHE` | The `vllm` command and the model cache for `serve_local.sh`. |
| `TOL`, `GEN0` | Options of `run_loop.sh`, described at the top of the script. |

The orchestrator needs `pi` and the packages in `src/ai4data/small_ai/orchestrator` (see the harness README).

## Commands of the recorded runs

The two servers ran in a tmux session, one window per model:

```bash
tmux new -d -s small-ai-llm -n qwen 'research/small-ai-loop-runs/serve_local.sh qwen'
tmux new-window -t small-ai-llm -n gemma 'research/small-ai-loop-runs/serve_local.sh gemma'
```

The occupation run wrote its own round 0 (900 requests):

```bash
research/small-ai-loop-runs/run_loop.sh isco-l isco 42 900 150 > "$SMALL_AI_RUNS/isco-l.log" 2>&1
```

The two topic runs started from one set of 20,296 sentences that Qwen3.8-27B had written earlier in 1,000 requests.
The session distributes this set as `assets/topics/sentences.jsonl.gz`. `GEN0` pointed to the folder with the
`examples.jsonl` and `raw.jsonl` files of that generation, which is not part of the repository.

```bash
GEN0=<folder> research/small-ai-loop-runs/run_loop.sh topics-l  wb_topics 42 1000 150 > "$SMALL_AI_RUNS/topics-l.log" 2>&1
GEN0=<folder> research/small-ai-loop-runs/run_loop.sh topics2-l wb_topics 7  1000 150 > "$SMALL_AI_RUNS/topics2-l.log" 2>&1
```

The differences between arms came from `compare.py`:

```bash
python research/small-ai-loop-runs/compare.py "$SMALL_AI_RUNS/isco-l-targeted" "$SMALL_AI_RUNS/isco-l-control"
python research/small-ai-loop-runs/compare.py "$SMALL_AI_RUNS/isco-l-targeted" "$SMALL_AI_RUNS/isco-l-targeted" final final_r00
```

The scripts in `research/small-ai-survey-style` continue from the folder `isco-l-targeted`.

## Differences from the scripts as run

The recorded runs used copies with fixed paths on the machine of the runs and read the OpenRouter key from a
local file. These copies take the paths and the key from the environment. Three further changes were made:

- The two settings files of the runs are merged into `env.sh`.
- The baseline step, which existed once inside the loop script and once per task as a separate script, is the
  single script `baselines.sh`.
- The run folders are created in `SMALL_AI_RUNS` directly. The runs used links from a working folder to a data disk.

The copies were checked on the recorded run folders without a model or a graphics card. `run_loop.sh` found every
step finished for `isco-l` and `topics2-l` and started no generation or training. Its evaluation step wrote
score files identical to the recorded ones for the three large models on both tasks. `compare.py` printed the
recorded differences (occupation, targeted minus control: 0.012, interval -0.006 to 0.029). `serve_local.sh` and
the steps of `run_loop.sh` that generate or train were not run again.

Scripts of earlier runs are not included: the October runs with a hosted writer, a probe on the COICOP task,
and the download of labeller candidates.
