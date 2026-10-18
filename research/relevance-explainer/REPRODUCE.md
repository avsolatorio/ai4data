# Reproducing the claim-carrying model

This file gives the lineage of the model served as "Claims + DPO (verified references)" (`runs/final-C-dpo`, shipped
as a bundle by `relevance_explainer.inference`): the environment, every step from the pretraining corpus to the
bundle with its command, inputs, outputs and the code commit it ran at, and the checks that compare a rerun with the
released files. `artifacts.json` holds the SHA-256, size and row count of each file named here.

## What can be reproduced, and how exactly

| Level | Starting point | Expected agreement | How it is checked |
|---|---|---|---|
| Serving and evaluation | the bundle and the test sets | identical answers on CPU with the same thread count; identical GPU evaluation outputs | `tests/test_inference.py`, step 4 of `scripts/verify_reproduction.sh` |
| Training | the released datasets and the base model | GPU kernels are not bit-deterministic, so weights differ; test accuracy within run-to-run variation | step 5 of `scripts/verify_reproduction.sh` |
| Data | catalogs and the teacher model | the procedure is scripted, but teacher outputs depend on sampling and on the served model, and catalog contents change over time, so regenerated data differs in detail | not comparable row by row |

The deterministic data step, building format C from the round G data and the teacher's tl;dr and grade, gives
byte-identical files under different Python hash seeds (step 1 of `scripts/verify_reproduction.sh`).

## Environment

- Python 3.10. Project environment: `requirements/relexp-lock.txt` (torch 2.5.1 with CUDA 12.4). Teacher serving
  environment: `requirements/vllm-lock.txt` (vllm 0.30.0). Hardware used: two NVIDIA H100 NVL (95 GB), driver 595.58.
- Install for research work:
  ```
  python -m venv ~/venvs/relexp && ~/venvs/relexp/bin/pip install -r requirements/relexp-lock.txt
  ~/venvs/relexp/bin/pip install --no-deps -e .
  ```
  Inference alone needs only `pip install .` (torch, tokenizers, numpy).
- Machine paths: copy `scripts/local.env.example` to `scripts/local.env` and set the interpreter, the vllm executable
  and a log directory on a large disk. Every shell script reads them through `scripts/env.sh`.
- Third-party files: `scripts/fetch_resources.sh` downloads fastText language identification and checks its checksum.
  The gazetteer, lexicon and word lists are in `src/relevance_explainer/resources`.
- Teacher servers: `scripts/serve_vllm.sh` serves Qwen3.8-27B (FP8) on ports 8000 and 8001, and Gemma 4 31B (FP8) on
  port 8002; `scripts/keepalive.sh` keeps them running and `scripts/teachers.sh` starts and stops them around the steps
  that need them. Thinking mode is off for every teacher call.
- Further judges: `synth/models.py` lists every model with its weights, server settings and request fields, and
  `scripts/serve_judge.sh <name>` serves one (gpt-oss-120b on one GPU, Mistral Small 4 on both). Mistral Small 4
  needs the transformers 5.16.1 overlay of `scripts/vllm_overlay.sh` and its Mistral-format weights. Set
  `RELEXP_HF_CACHE` and `RELEXP_VLLM_OVERLAY` in `scripts/local.env`.

## Lineage

Commits are those current when each output was written (from file times on the training machine).

| Step | Output | Command | Inputs | Commit |
|---|---|---|---|---|
| 1. Task data, round G | `data/v2_g_itemfirst_nosim` (977,538 / 119,837 / 110,566 rows) | `scripts/round_g.sh` (adds the round G slices to the earlier rounds; RECIPE.md, "Round G data") | earlier rounds' data, catalog record pools, teacher and Gemma 4 | df14467 |
| 2. Pretraining corpus and 32k tokenizer | `data/corpus_multi_final` | `scripts/fw2_sizes_from_hub.py`, then `scripts/multibase_final_data.sh` | FineWeb-2, FineWeb-Edu, Wikipedia | 790e933 |
| 3a. Base, 1,024-token sequences | `runs/base-final/state-457764.pt` (60B tokens, stable phase) | `scripts/base_final.sh` | step 2; embeddings initialised from Qwen3-0.6B | 7b287cf |
| 3b. Base, 2,048-token sequences | `runs/base-2k/model.pt` (decay from 60B to 80B tokens) | `scripts/base_2k.sh` | step 3a | 6d85a81 |
| 4. Teacher tl;dr and grade | `data/fmt_s/teacher.jsonl`, `data/fmt/teacher_rest.jsonl` | `scripts/fmt_pipeline.sh`, steps 4 and 5 (`synth/tldr_grade.py`: temperature 0.3, 0.7 on retries) | step 1 | 233a0e7 |
| 5. Format C data | `data/final/C` (959,446 / 19,634 / 2,934 rows) | `scripts/build_format_arms.py data/v2_g_itemfirst_nosim data/fmt_s/teacher.jsonl,data/fmt/teacher_rest.jsonl data/final --workers 96` | steps 1 and 4 | 233a0e7 |
| 6. Fine-tune | `runs/final-C` | `scripts/fmt_train_one.sh data/final/C runs/final-C 0 --special-tokens` (lr 1e-3, 4 epochs, batch 32, dropout 0.1, verdict weight 5, 2,048 tokens; the epoch with the best validation verdict accuracy is kept) | steps 3b and 5 | 233a0e7 |
| 7. DPO | `runs/final-C-dpo` | `scripts/dpo_round_c.sh`: 15,000 training inputs (seed 0), 8 samples each at temperature 0.8; verifier on all; teacher judge on the 2 best and the worst per input; pairs with the verdict fixed; DPO beta 0.3, lr 2e-5, 2 epochs, batch 16, SFT weight 0.5 | step 6 | c372b27 |
| 8. INT8 export | `runs/final-C-dpo/explainer-int8.pt` | `python -m relevance_explainer.tiny.export --run runs/final-C-dpo --out runs/final-C-dpo/explainer-int8.pt` | step 7 | 1ce3a63 |
| 9. Calibration | `runs/final-C-dpo/calibration.json` | `scripts/calibrate_c.py runs/final-C-dpo data/final/C/val.jsonl data/eval_real/itemfirst_nosim_v4/test.jsonl` (3,000 validation rows, seed 0) | step 7 | 1ce3a63 |
| 10. Faithfulness scorer | `runs/final-C-dpo/faith_scorer.json` | `scripts/faith_refit.sh runs/final-C-dpo 8000 1` (8,000 training inputs, seed 1, 8 samples each, 24,000 teacher judgements) | step 7 | 0f20ab2 |
| 11. Bundle | a directory with the files of steps 8 to 10 and `bundle.json` | `relevance-explainer bundle --run runs/final-C-dpo --out dist/relevance-explainer-c-dpo` | steps 8 to 10 | |

The scripts behind step 1's earlier rounds (realistic search chunks, paraphrase, translation, item types, challenge
phenomena, hard negatives) are listed under "Data generation" in `scripts/README.md`.

## Evaluation

| Check | Command |
|---|---|
| Verdict accuracy on the realistic, paraphrase, item-type and challenge test sets | `RUN_DIR=runs/final-C-dpo SUF=_v4 scripts/eval_sets.sh` |
| Verification with and without the repeat ban, verifier-selected retries | `scripts/fmt_decode_test.py runs/final-C-dpo data/eval_real/itemfirst_nosim_v4/test.jsonl --k 6` |
| Teacher judge and Gemma 4 judge of 800 realistic-test answers | `scripts/judge_decode.sh runs/final-C-dpo`, `scripts/judge_gemma.sh runs/final-C runs/final-C-dpo` |
| The app's CPU path on the realistic test | `scripts/cpu_parity_c.py runs/final-C-dpo data/eval_real/itemfirst_nosim_v4/test.jsonl` |
| Faithfulness scorer against both judges | `scripts/faith_scorer_eval.py runs/final-C-dpo [--suffix _gemma4]`, `scripts/faith_cpu_check.py runs/final-C-dpo` |
| Served render version 5 against version 4 on the evaluation sets | `python -m relevance_explainer.synth.eval_v3 --version 5`, `RUN_DIR=runs/final-C-dpo SUF=_v5 scripts/eval_sets.sh`, `scripts/render_change_relabel.py label` and `score` |
| Records from catalogs absent from training | `scripts/unseen_catalogs.py run --bundle <bundle>` and `report` (`evalsets/unseen_catalogs`) |
| Open-weights judges on the challenge, realistic and unseen sets | `scripts/judge_panel.py label <name>` for each model, then `report` |

Released model on the realistic test (1,495 searches; CPU INT8, the app's answer path, inputs at render version 4):

| Measure | Value |
|---|---|
| Verdict accuracy | 86.6% |
| Answers with every claim and reference verified, first answer / after one round of 3 retries | 97.3% / 99.3% |
| 90% conformal verdict set: coverage, mean size | 91.6%, 1.13 |
| Faithfulness score: AUROC against the teacher judge / Gemma 4 (800 answers) | 0.856 / 0.782 |
| Seconds per answer, 4 threads: median, 90th percentile | 2.64, 3.49 |

## Checks of a rerun

- `scripts/verify_reproduction.sh` reruns steps 5, 7 (training from the released pairs), 9 and 10 and the realistic
  evaluation, writing under `data/repro` and `runs/repro-*`, and reports which outputs are identical.
  Result on the training machine, 2026-10-16 (commit 29eb2f3):

  | Rerun | Result |
  |---|---|
  | Format C data, Python hash seeds 1 and 2 | train, validation and test files byte-identical to the released ones |
  | Calibration | identical file |
  | Faithfulness scorer, features recomputed on GPU | identical file (AUROC 0.8812) |
  | Realistic-test generations of the released model | identical file |
  | DPO from the released pairs, same seed | different weights; realistic-test verdict accuracy 86.62% against 86.82% (3 of 1,495 searches) |
- `python scripts/manifest.py --check` compares the files present with `artifacts.json`.
- `pytest` runs the unit tests, the repository checks and, when the released model is present, the inference tests.

## Limits

- Teacher outputs (steps 1, 4, 7 and 10) cannot be regenerated identically. Exact reproduction of training needs the
  released datasets of steps 1, 4 and 5 and the DPO pairs.
- Catalog records were fetched in October 2026 from public APIs (Data360, the Microdata Library, Policy Research
  Working Papers, HDX, Statistics Canada, ILOSTAT, Eurostat); later fetches return different records.
- The base runs used two GPUs with data parallelism and resumed from saved states after interruptions.
