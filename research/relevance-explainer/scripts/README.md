# Scripts

Command-line steps of the project, run from `research/relevance-explainer`. Shell scripts read machine paths from
`env.sh` (see `local.env.example`). Each script's header gives its arguments; `REPRODUCE.md` gives the order of the
steps behind the released model. Earlier models and experiments are kept because the project's results were
produced with them.

## Setup and operations

| Script | Purpose |
|---|---|
| `env.sh` | Shared settings read by every shell script: the Python interpreter, the vllm executable and the log directory, from the environment, then local.env (not committed), then defaults. |
| `local.env.example` | Template of local.env, the paths of one machine. |
| `fetch_resources.sh` | Download the third-party files the data pipeline and the live app read, and check them against the checksums of the copies used to build the released data: data/geo/lid.176.ftz fastText language identification… |
| `serve_vllm.sh` | Serve the 8-bit Qwen3.8-27B teacher with vLLM (OpenAI-compatible API on port 8000). |
| `serve_all_llms.sh` | Start the three model servers used for data generation and judging, detached, one after another: teacher on GPU 0 (port 8000), teacher on GPU 1 (port 8001), then Gemma 4 on GPU 0 (port 8002) once GPU 0's teacher is up. |
| `serve_app.sh` | Serve the relevance-explainer web app on CPU (default port 3410, all interfaces so a containerized proxy can reach it). |
| `keepalive.sh` | Keep the model servers, the web app and a long job running: every minute, start whatever is down. |
| `run_keepalive.sh` | Run scripts/keepalive.sh with its log appended to the shared log directory. |
| `run_logged.sh` | Run a script with its output appended to <log_dir>/<script name>.log, for use as a tmux window command: tmux new-window -t relexp -n roundf scripts/run_logged.sh scripts/round_f.sh. |
| `teachers.sh` | Start the teacher servers only for the steps that need them, through the keepalive's pause files. |
| `serve_judge.sh` | Serve one model of synth/models.py (the teacher, Gemma 4 or the judges added on 2026-10-18: gpt-oss-120b, Nemotron 3.5 Lightning, Mistral Small 4) with vLLM, for a tmux window. |
| `vllm_overlay.sh` | Build an environment that reuses the vllm environment's packages with another transformers release, for a model the locked transformers cannot load (Mistral Small 4 needs transformers 5.16.1 with vllm 0.30.0). |
| `py.sh` | Run a project module with the training environment. |
| `manifest.py` | Manifest of the artifacts behind the released claim-carrying model (REPRODUCE.md): for each file, its SHA-256, size and, for JSONL, the number of rows. |
| `verify_reproduction.sh` | Rerun the deterministic steps behind runs/final-C-dpo and compare the outputs with the released artifacts (artifacts.json, REPRODUCE.md). |

## Released model (REPRODUCE.md)

| Script | Purpose |
|---|---|
| `round_g.sh` | Round G: verdict rules (NARRATE_V3) and render version 4 for the targeted slices, lexical hard negatives, more exclusion / look-alike / proxy / group / period searches, and Eurostat tables (one half for training… |
| `strip_similarity.py` | Copy a data directory with the "Retrieval similarity" line removed from every input (train, val, test). |
| `fw2_sizes_from_hub.py` | Per-config train sizes of FineWeb-2 from the repository file listing (the dataset viewer's size endpoint fails for some configs). |
| `multibase_final_data.sh` | Multilingual base, data for the final run: about 80B training tokens (176B characters at the 32k tokenizer's 2.2 characters per token), English at a fixed 10% (Wikipedia and FineWeb-Edu), the other languages by… |
| `base_final.sh` | Multilingual base, final run: 32k vocabulary, width 448, 8 layers, 7 heads, per-head attention gate (31.8M parameters), 80B tokens on both GPUs, warmup-stable-decay at 3e-3 (decay over the last 20%). |
| `base_2k.sh` | Multilingual base with 2,048-token context: resume the final run's 60B state (step 457,764, stable phase) and train the rest of the run on 2,048-token sequences with 64 sequences per step, so a step holds the same… |
| `fmt_pipeline.sh` | Output-format test on the 2,048-token base, in two parts that run at the same time. |
| `sample_unique_rows.py` | A random sample of unique (input, output) pairs of a training file, written with all their repeats so the sample keeps the file's weighting, and the mix of origins and languages it keeps compared with the full file. |
| `tldr_pilot_report.py` | Report on a tl;dr and grade run (synth/tldr_grade.py) and gate the full run on it: pass rate, attempts, failure reasons, the grade distribution per verdict and examples per language. |
| `build_format_arms.py` | Build the three output-format arms from one dataset and the teacher's tl;dr and grade (synth/tldr_grade.py). |
| `align_arm.py` | Keep the rows of a rebuilt arm that are also in a reference arm, so a rebuilt arm holds the rows of the arms it is compared with. |
| `final_pipeline.sh` | Final model in format C (verdict, grade, claims, tl;dr, explanation with references) on the full round G data. |
| `fmt_train_one.sh` | Train and evaluate one format-test run with the settings of scripts/fmt_pipeline.sh (used to rerun a run that failed). |
| `dpo_round_c.sh` | The DPO round for full-data C with the judge of full-data B in the same teacher window: sample (teachers off), teachers up, judge runs/final-B, then scripts/dpo_c.sh (scores, takes the teachers down for training and… |
| `dpo_c.sh` | One on-policy DPO round for a format-C run (distill/onpolicy_c.py), then evaluation: test sets (eval_sets.sh), decoding test with retries, structure report and the judge of the realistic-test answers. |
| `calibrate_c.py` | Calibration file for serving a tagged run: the verdict temperature and the split-conformal thresholds, fitted on rows of the training-distribution validation set and checked on the realistic test (CPU, int8, as in… |
| `faith_refit.sh` | Fit the live faithfulness scorer for a format-C run on its own samples: sample (GPU 0, teachers off), verifier and teacher judge (scripts/teachers.sh up/down around it), fit (scripts/faith_scorer.py, GPU 1), then… |
| `faith_scorer.py` | Live faithfulness scorer for format C: the probability that the teacher judge would rate an answer's explanation faithful 4 or 5, from signals available at inference without a teacher. |

## Evaluation

| Script | Purpose |
|---|---|
| `eval_sets.sh` | Score a no-score run on the realistic, paraphrase, nine-type and challenge test sets. |
| `fmt_decode_test.py` | Decoding test for the format arms: does the repeat ban hurt structured outputs, and does resampling with the verifier as the selector raise the share of fully verified answers?. |
| `fmt_structure_report.py` | Structure and verification of format-test outputs, per run and test set. |
| `fmt_table.py` | Verdict accuracy (int8) of format-test runs on the version-4 test sets, the Eurostat tables and the held-out test. |
| `paired_verdicts.py` | Paired comparison of two runs' verdicts on the same test set: accuracy of each, rows where only one is right, and an exact two-sided McNemar p-value. |
| `paired_verified.py` | Paired comparison of two format-C runs' decoding tests (scripts/fmt_decode_test.py) on the same rows: share of answers with every mention and reference verified, for one column, with McNemar's exact test. |
| `judge.sh` | Judge generated narratives with the vLLM-served teacher. |
| `judge_decode.sh` | Judge runs' realistic-test answers from the decode test (greedy, no repeat ban) with the teacher: the explanation (tiny/judge.py, 800 examples, the same for every run) and, for tagged runs, the tl;dr… |
| `judge_prep_decode.py` | Write a run's decode-test answers (scripts/fmt_decode_test.py) as a generations file the judge reads (tiny/judge.py: input, generated, reference), with the explanation in words only and the raw answer kept. |
| `judge_tldr.py` | LLM-judge scores for a tagged run's tl;dr, against the run's own explanation and the input. |
| `judge_gemma.sh` | Held-out judge: Gemma 4 judges the realistic-test answers that the teacher judged (tiny/judge.py, same 800 examples, <run>/realnb_generations_int8.jsonl from scripts/judge_prep_decode.py), then paired comparisons. |
| `judge_paired.py` | Paired comparison of two runs' judge scores on the same examples (tiny/judge.py items, same seed and n): mean faithfulness difference with a bootstrap 95% interval, and McNemar on "faithful 4 or 5". |
| `judge_agreement.py` | Agreement between the teacher judge and Gemma 4 on the same student answers (tiny/judge.py files judge_<tag>_int8.json and judge_<tag>_int8_gemma4.json): agreement on "faithful 4 or 5", counts below 4 for each judge… |
| `cpu_parity_c.py` | The app's answer path (answer_c.answer: int8 file on CPU, greedy without the repeat ban, one round of verifier- selected retries, calibration, expected grade, faithfulness score) on a test set, compared with the GPU… |
| `faith_scorer_eval.py` | Evaluate a fitted faithfulness scorer (<scorer run>/faith_scorer.json) on a run's judged realistic-test answers: the greedy answers of the decoding test (scripts/judge_prep_decode.py) with the teacher judge's scores… |
| `faith_cpu_check.py` | The faithfulness scores the app computed on CPU (scripts/cpu_parity_c.py) against the judges' labels, on the rows where the CPU answer is identical to the judged GPU greedy answer (judge labels exist only for that text). |
| `unseen_catalogs.py` | Test set of records from catalogs absent from training (evalsets/unseen_catalogs): rebuild the records from Our World in Data, open.canada.ca, data.gov.au and Zenodo, explain every labelled case with a model bundle at a chosen render version, and report verdict agreement, trust states and faithfulness bands. |
| `render_change_relabel.py` | Effect of a render-version change on the evaluation sets: the teacher and Gemma 4 label every row whose input changes on both versions' inputs, and the run's verdicts on each version are scored against the stored and the new labels (McNemar's exact test). |
| `judge_panel.py` | Open-weights models as judges of the challenge, realistic and unseen-catalog sets: each model's verdict on the same inputs (NARRATE_V3, temperature 0, reasoning off), agreement with each set's references, Cohen's kappa between models, agreement by language and the models' majority. |
| `uncertainty_check.py` | Calibration, conformal verdict sets and the expected grade for a tagged run on the realistic test (CPU, int8). |
| `grade_vs_probs.py` | Does the model's grade rank results better than its verdict probabilities?. |
| `teacher_grade_repeat.py` | How reliable is the teacher's 0-10 grade? Re-grade a sample of realistic-test rows with the same prompt and checks (synth/tldr_grade.one) and compare with the grade written for the training data… |
| `queue_grade_repeat.sh` | Re-grade 600 realistic-test rows with the teacher (scripts/teacher_grade_repeat.py) once the teachers are free of the remaining-rows run and the arms' judge (scripts/final_pipeline.sh step 2). |
| `bench_formats_cpu.py` | CPU time per answer for format runs, with the app's settings (int8, 4 threads, KV-cache decoding): one greedy answer, and the greedy answer plus 3 samples written as one batch (the retry round). |
| `bench_parallel_samples.py` | Time N explanations written after the same verdict line on CPU (the app's settings: int8 model, 4 threads), one at a time (as the app's retry does now) and as one batch (generate_batch), for N = 1, 3, 6, 9. |
| `app_answers.py` | Run the app's answer pipeline (webapp.server._events: greedy decoding, number check, answer confidence, retry) on a test set on CPU and save each final answer with its score and attempts. |
| `judge_app_answers.py` | Judge the app's final answers (scripts/app_answers.py) with the teacher, and for answers that were retried also the first answer, so the retry's effect is measured on the deployed pipeline. |
| `rating_sample.py` | Draw the human rating sample of docs/rating_protocol.md from the app's answers (scripts/app_answers.py). |
| `pick_best.py` | Print the run directory with the highest best-epoch validation verdict accuracy (metrics.json), and each run's value on stderr. |
| `eval_arm.sh` | Evaluate one run on the standard test set (data/f2, INT8) and judge 800 generations with the teacher and with Gemma 4. |
| `judge_real.sh` | Judge a run's realistic-test narratives with the teacher and Gemma 4 (800 examples). |
| `judge_tag.sh` | Judge a run's generations for a tag with the teacher and Gemma 4 (800 examples). |
| `numguard_eval.sh` | Evaluate a run with the number guard on the version-3 test sets (tags <set>_v3_rs) and judge the realistic test. |
| `pairwise.sh` | Pairwise usefulness preference between two runs with Gemma 4. |
| `label_compare.py` | Compare two labellings of the same challenge searches: teacher-Gemma 4 agreement per phenomenon, label distributions and how the teacher's labels moved. |

## Data generation (earlier rounds, test sets, record pools)

| Script | Purpose |
|---|---|
| `run_pipeline.sh` | Run the synthetic-data pipeline against a local Ollama server. |
| `run_v3.sh` | Third data round: 1,500 theme and 1,500 anchored (theme, source) combinations with a flatter language mix, against the vLLM teacher on port 8000. |
| `run_round.sh` | A further data round with the flat language mix against the vLLM teacher on port 8000. |
| `realistic_pilot.sh` | Realistic-search pilot: spec-driven searches, anchored searches, retrieval, rendering, narration. |
| `realistic_scale.sh` | Generate realistic-search data in resumable chunks: each chunk runs the full pipeline with its own seed. |
| `realistic_long.sh` | Long realistic-data run after scripts/realistic_scale.sh: chunks FIRST..LAST (coverage-steered through data/real/coverage.json), and after every second chunk a translation set of about 20% of those two chunks. |
| `after_scale.sh` | Wait for the first 10 chunks to finish, translate chunks 1-10 (20%), then start the long run. |
| `after_backfill.sh` | Wait until a log reports BACKFILL-DONE, then run the given command. |
| `coverage_watch.sh` | Recompute coverage weights (data/real/coverage.json) whenever another realistic chunk finishes. |
| `translate_backfill.sh` | Create missing translation sets tK for even K whose chunks cK-1 and cK exist (20% of the two chunks). |
| `meta_translate.sh` | Metadata-translation set from realistic chunks. |
| `meta_long.sh` | Metadata-translation sets m1..mN, each 3,000 pairs drawn from five realistic chunks (c41 onward, after the anchored language fix took effect from about c43). |
| `paraphrase_pilot.sh` | Pilot of paraphrased fully-answering searches: anchored stage only, every 'full' search paraphrased. |
| `para_long.sh` | Paraphrase-focused anchored chunks: fully-answering searches in everyday words, and hard cases with the same wording and a wrong place, time or group, or a related concept. |
| `para_test.sh` | Paraphrase test set (anchored searches only, every fully-answering one paraphrased) and the current models' scores. |
| `types_long.sh` | Realistic chunks for the metadata-schemas types from the local record pools (tables, map layers, scripts). |
| `build_pools.sh` | Build the record pools for tables, map layers and scripts. |
| `fetch_nada.py` | Fetch every catalog entry (search row plus full metadata) from a NADA instance into a JSONL file. |
| `schemas_demo.py` | Convert every NADA demo entry (data/nada_demo.jsonl) to a record and print one model input per schema type. |
| `schemas_eval.sh` | Build the metadata-schemas test set and evaluate a run on it. |
| `themes_more.sh` | Expand search themes with Qwen and Gemma 4 (see synth/themes_more.py). |
| `challenge.sh` | Challenge suite: write, label, build, then score the current realistic models per phenomenon. |
| `phen_train.sh` | Round E training slices for the challenge phenomena: training writers, records outside the held-out suite, rows kept where the teacher and Gemma 4 agree. |
| `policy_check.sh` | Label a sample of realistic test pairs twice, with the version-3 setup (NARRATE_V2, render 3) and with the verdict rules (NARRATE_V3, render 4), to measure how often the rules change ordinary labels. |
| `policy_sample.py` | Sample realistic test pairs as challenge searches, to measure how the verdict rules change ordinary labels. |
| `policy_relabel.sh` | Relabel the challenge suite at render version 4 with the verdict rules (NARRATE_V3), to compare the teacher-Gemma 4 agreement with the version-3 labels. |
| `realistic_test.sh` | Unseen realistic test set: searches on the reserved fresh themes (never used for training data), labelled by the teacher from the full format, plus the teacher's thinking mode and Gemma 4 for consensus labels. |
| `build_fresh.sh` | Build a fresh evaluation set (searches and items absent from every data round) with the vLLM teacher and Gemma 4. |
| `render_chunk.sh` | Re-run the render stage of one chunk (for example after a failure). |
| `render_dry.sh` | Dry run of the render stage on the first 20 pairs of a chunk. |
| `round_d.sh` | Round D: all realistic chunks plus the targeted sets (paraphrase x2, new item types x2, translated metadata), no score line; continue the no-score model, evaluate on every test set, then on-policy DPO and evaluate again. |
| `round_e.sh` | Round E: render version 3 (thresholds, units, exclusions, named periods) and the concept rules. |
| `round_f.sh` | Round F: round E's data without narratives that state numbers absent from the input; one epoch from round E's supervised model; DPO whose pairs prefer explanations without invented numbers; evaluations with and… |
| `round_when_ready.sh` | After a data round finishes narrating, assemble the combined set, fine-tune from the base, run the on-policy preference pipeline, and evaluate. |
| `filter_numbers.py` | Copy a data directory without rows whose narrative states a number absent from the input (train, val, test). |
| `sim_variants.sh` | Build the retrieval-score variants of the realistic test set and score a run on each. |
| `sim_shifted.sh` | Score a run on the simulated retriever-scale variants after scripts/sim_variants.sh. |
| `sim_shifted.py` | Add simulated retriever-scale variants to data/sim_variants: shifted up (a general-purpose embedding model whose cosines sit higher), compressed, and both. |
| `train_sample.py` | Write a test set drawn from the training split: up to N rows per relation for the given origin, in the evaluation schema, so the model's fit on its own training rows can be compared with its accuracy on the held-out… |
| `claims_annotate.py` | Convert explanations to the claim-carrying format (relevance_explainer/claims.py) and report what the conversion found: claims and references per example, the share of examples with at least one reference, whether… |

## Earlier models and experiments

| Script | Purpose |
|---|---|
| `pretrain.sh` | Detached-friendly pretraining launcher. |
| `finetune_when_ready.sh` | Wait for a pretraining run to reach its final step, then fine-tune on the task data with the standard recipe. |
| `continue_train.sh` | Continue training an existing model on new-format data, then score on the realistic test set and judge narratives. |
| `milestone_v2.sh` | Train and compare the three input formats once enough realistic data exists. |
| `milestone_cont.sh` | Milestone by continued training: at <min_chunks> chunks, assemble data, continue training <init> on the item-first variant, score and judge on the realistic test set, then on-policy DPO. |
| `milestone_full.sh` | Models at a data milestone: each listed format trained and scored on the realistic test set; on-policy DPO for the first format. |
| `arm_nosim.sh` | No-score arm: milestone C's recipe (continue runs/f6-dpo on data/v2_c_itemfirst, lr 3e-4, 3 epochs) with the retrieval-similarity line removed from every training and test input. |
| `nosim_followup.sh` | No-score model: on-policy DPO, realistic-test scoring and judging without the similarity line, then the retrieval-score sensitivity variants. |
| `dapt.sh` | Domain pretraining experiment, after round G. |
| `dapt_followup.sh` | After scripts/dapt.sh: can one fine-tuning stage plus DPO replace the eight-stage chain? 1. |
| `single_stage.sh` | Single-stage recipe: one fine-tune of a pretrained base on round G's data, then DPO, then all evaluations and judges. |
| `dpo_run.sh` | On-policy DPO for a trained run on its data, then realistic-test scoring and judging. |
| `dpo_round2.sh` | A second on-policy DPO round from an already preference-tuned model, with the same number of teacher judge calls as a GRPO run (N inputs x K samples). |
| `grpo.sh` | GRPO on the narratives with the teacher as the judge. |
| `sweep_copy_bias.sh` | Evaluate a task model at several copy-bias strengths. |
| `bytetrain.sh` | Detached-friendly launcher for the ByteFlow task fine-tune. |
| `multibase.sh` | Multilingual base model, data side: plan per-language budgets (FineWeb-2 languages with >= 10 MB, plus English Wikipedia), fetch them, then train tokenizers of several sizes and compare bytes per token by language. |
| `multibase_shapes.sh` | Multilingual base model: which vocabulary and shape fit best in about 20.6M parameters (the 20 MB INT8 budget)? Each candidate "name vocab d_model layers heads" is tokenized with its vocabulary (shards per vocabulary… |
| `multibase_shapes30.sh` | Multilingual base model, 30 MB budget: the same comparison as scripts/multibase_shapes.sh (1B tokens each, held-out bits per byte) for shapes that keep 7 or 8 layers at width 448 with a larger vocabulary. |
| `multibase_attention.sh` | Multilingual base, attention variants on the 32k, width 448 shape (same 1B-token comparison as scripts/multibase_shapes30.sh, baseline runs/mb-v32k-d448-l7): gqa: 8 layers with one key/value head shared by the 7… |
| `multibase_lr.sh` | Multilingual base: learning rate for the gated 32k, 8-layer shape. |
| `multibase_rerun.sh` | Rerun 20 MB shapes from scripts/multibase_shapes.sh that failed (out of memory next to the teacher server on GPU 1), with 32 sequences per forward pass. |
| `multibase_compare.py` | Compare multilingual base candidates on held-out bits per byte (last evaluation of each run's log.json). |
| `fw2_inventory.py` | Inventory of FineWeb-2 language configs by size, for choosing the multilingual base's languages. |
| `corpus_breakdown.py` | Size and breakdown of the multilingual corpus: raw characters and bytes per language (from raw/*.jsonl), tokens with a tokenized vocabulary, planned training share, and groups by script and by size. |
| `eval_base.py` | Full held-out evaluation of a pretrained base: bits per byte per language on every held-out sequence (the last --holdout tokens of each language, as in training), and the same measure on text the model trained on… |
| `eval_base_position.py` | Bits per byte of a base model by position in each language's shard: the first 8 held-out sequences (what training evaluated), the rest of the held-out tail, and windows of training text at the start, middle and just… |
| `base_trajectory.py` | Held-out bits per byte at every evaluation of a pretraining run: all languages, the explainer's 11, the smallest third of the run's corpus plan, and English. |
| `quant_embedding_check.py` | Held-out bits per byte of a pretrained base under weight quantization schemes, to decide whether the embedding table (also the output layer, since it is tied) can be stored at 4 bits. |
| `selfcheck.sh` | Correctness head experiment (tiny/selfcheck.py) on the deployed model. |
| `selfcheck_ood.sh` | Answer-confidence calibration beyond the realistic test: greedy outputs of the deployed model on the challenge suite, the held-out Eurostat tables and the paraphrase set, judged by the teacher, scored by the saved… |
| `selfcheck_checks.py` | Checks on the correctness-head result (tiny/selfcheck.py): 1. |
| `save_score_head.py` | Fit the 9-parameter answer-confidence head (target: verdict right and explanation good; inputs: the 8 uncertainty scalars of tiny/selfcheck.py) on the selfcheck training features and save it as JSON for the app. |
| `calibration_report.py` | Calibration of the saved answer-confidence head (selfcheck/score_head.json) on greedy outputs of several sets. |
| `check_answer_score.py` | Check that the app's answer confidence (features from decoding-time step statistics, CPU, INT8 export) matches the teacher-forced features of tiny/selfcheck.py on the same realistic-test inputs, and time the scoring… |
| `claims_vs_judge.py` | Does the mechanical check agree with the judge? Converts the deployed model's judged app answers to the claim format and compares the judge's faithfulness for answers with and without unverified numbers or places,… |

## Checks of the code

| Script | Purpose |
|---|---|
| `check_ddp.sh` | Checks of two-GPU data-parallel pretraining (torchrun) in tiny/pretrain.py: 1. |
| `check_ddp_seeds.sh` | Seed spread for the 300-step comparison of scripts/check_ddp.sh: one GPU and two GPUs, seeds 1 and 2. |
| `check_wsd_resume.sh` | Check of the warmup-stable-decay schedule and resumable states in tiny/pretrain.py: a 40-step run straight through and the same run resumed from its step-20 state must end with the same held-out score and learning rates. |
| `check_attention_variants.py` | Checks for grouped-query attention and the attention gate in tiny/model.py. |
| `check_retry_app.py` | Send each realistic app example through the app's own /api/generate route (in process, CPU), as the examples page does, and report for low-confidence answers whether the retry ran and how many attempts came back. |
| `check_retry_stream.py` | Run the app's generation stream (webapp.server._events) on realistic-test inputs and print the events after the tokens: answer confidence, retry, attempts. |
