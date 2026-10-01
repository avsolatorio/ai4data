# Lab notebook: Relevance Explainer

Append-only. Entry headers carry a UTC date and time taken from a network source. Each entry names the git commit its results were produced against.

## 2026-09-30 22:36 UTC — Project scoping

Commit: 40ee179 (branch `relevance-explainer`, no code yet)

**Question/Context.** Can a model of about 20 MB explain why a catalog result matches a query, if the retriever already performs the semantic matching?

**What was done.** Design discussion only. No experiments run. Proposed a hybrid pipeline: retriever, then match decomposition into structured evidence (matched concepts, matched fields, missing constraints, similarity), then a small encoder-decoder (10-30M parameters, INT8/INT4) that verbalizes the evidence. Training data would come from distilling a larger teacher model over synthetic (query, record, diagnostics) examples.

**Result.** None yet. The size and latency targets (about 20 MB, sub-100 ms startup) are untested assumptions.

**Implication.** The design decision to keep reasoning out of the small model (precompute relations at indexing time) needs empirical support. First steps: define the diagnostics schema, build a small synthetic set across the twelve case types listed in the README, and establish a baseline (template-based explanations) for comparison.

## 2026-10-01 00:23 UTC — Teacher model selection and first synthetic pilot

Commit: b4877ae plus uncommitted `synth/` pipeline (committed immediately after this entry).

**Question/Context.** The task definition changed from structured diagnostics to this: input is the query, applied filters and result metadata; output is a short narrative judging the result as relevant, partly relevant or irrelevant. Which locally runnable teacher model produces usable narratives, and does the pipeline work end to end on real Data360 records?

**What was done.**
- OpenRouter returned HTTP 402 on every call (account has 0 credits; the $250 figure is only a key cap), so a local teacher was used.
- Pilot 1: `llama3.1:70b` (Q4, already installed, Ollama 0.23.4) on 41 generated queries; 24 narratives written.
- Installed Ollama 0.35.0 user-local (the system 0.23.4 cannot pull Qwen3.8; no root used). Pulled `qwen3.8:27b-q4_K_M` (18 GB) and served it on GPU 1 with `OLLAMA_NUM_PARALLEL=16`, `reasoning_effort=none`.
- Pilot 2: Qwen3.8 27B on 48 multilingual queries (12 themes x 4), 192 retrieved (query, result) pairs from the Data360 search API, 60 narratives written.
- DeepSeek V4.1 Flash and GLM-5.3 Flash are cloud-only on Ollama and were not usable locally.

**Result.**
- Llama 3.1 70B narratives were generic ("This result measures ...") and in one case unsupported (GDP per capita growth marked partly relevant without naming the difference). Judged by reading 6 examples; no automatic metric.
- Qwen3.8 27B narratives were grounded in the filter checks and metadata and named concrete mismatches (national versus regional coverage, annual versus quarterly). Narratives in Chinese, Indonesian, French, Spanish, Arabic and Portuguese read naturally to the reviewer for the examples inspected (not checked by a native speaker). Median length 39 words.
- Throughput 60 examples in 83 s (0.72/s, about 2,600/hour per GPU, 16 parallel requests).
- Verdict balance was poor: 35 irrelevant, 22 partly relevant, 3 relevant (n=60). Language mix in this batch: 24 English, 12 French, 8 Spanish, 4 each Chinese, Indonesian, Arabic, Portuguese.

**Implication.** Qwen3.8 27B Q4 replaces Llama 3.1 70B as the teacher. Theme-based queries retrieve few direct matches, so the training set needs queries anchored to a sampled record to supply relevant and partly relevant examples, plus verdict-balanced sampling. The pilot is a single run of 60 examples from 12 themes, so the proportions are indicative only. Next: record-anchored query generation, Microdata Library and document-passage context types, then a larger run.

## 2026-10-01 01:01 UTC — Multi-source pilots, defects found, and class balance

Commit: 0cf25fd plus uncommitted multi-source changes (committed immediately after this entry). Teacher: `qwen3.8:27b-q4_K_M`, thinking off.

**Question/Context.** Extend the data from Data360 indicators to Microdata Library studies, World Bank Policy Research Working Papers (with a passage cut from the full text) and HDX datasets, with multilingual queries. Are the narratives and labels trustworthy, and what does the verdict distribution look like?

**What was done.** Pilot 3 (240 narratives) and pilot 4 (515 narratives) over 16 theme-by-source combinations for theme queries and 16 for record-anchored queries (one query the item answers fully, one it answers partly). 11 query languages.

**Result.**
- Pilot 3 had two defects that make its study and document labels unreliable. (1) The Microdata fetcher read the wrong JSON level, so all 68 studies had empty abstracts and topics; the teacher then invented content ("contains variables likely to address malnutrition"). (2) Country matching failed on catalog spellings ("Egypt" vs "Egypt, Arab Republic of"), producing wrong "not covered" checks. Pilot 3 data is discarded.
- Pilot 4 fixed those and added alias matching, a sparse-metadata rule in the teacher prompt, and canonical language labels. Reading pilot 4 found three more defects, fixed afterwards and not yet re-run: the World Bank Documents `count` field is messy ("Arab Republic of,Egypt, Arab Republic of") and failed the country check; a study whose coverage list shows one country while its abstract names four was reported as not covering the other three; study geography text was unclipped (inputs up to 3,245 characters). Pilot 4 narratives that depend on those checks may be mislabelled.
- Pilot 4 verdicts (n=515): 338 irrelevant (66%), 135 partly relevant (26%), 42 relevant (8%). By source, studies were 191 of 213 irrelevant, because Microdata Library search ranks loosely.
- Anchor items (n=64): full-match anchors judged relevant 30 of 32, partly relevant 1, irrelevant 1; partial anchors judged partly relevant 22 of 32, irrelevant 10, relevant 0. The teacher agrees with the intended label on 52 of 64 anchors (81%); the 12 disagreements were not inspected individually.
- Input length: median 1,000 to 1,730 characters by source, maximum 1,915 excluding the unclipped studies. Narrative median 41 words, maximum 72.
- No narrative called an item relevant when a country filter check said "not covered" (0 of 515).
- Balancing to at most 1.5x the rarest class leaves 144 of 515 examples in train (36 relevant, 54 partly relevant, 54 irrelevant), because relevant examples are scarce.

**Implication.** Relevant examples come almost only from anchored full-match queries, so scale runs need many more of those; theme queries mostly supply irrelevant and partly relevant examples. The pilot 4 counts are one run of 515 examples and carry the three unrerun defects, so they are indicative. Teacher agreement on anchors (81%) is a sanity check on label quality, not a measure of narrative quality; narrative quality has only been read by eye on a few examples, and non-English narratives have not been checked by native speakers. Next: rerun with fixes, then a larger run weighted toward anchored queries.

## 2026-10-01 05:13 UTC — Smoke-test fine-tune of Qwen3-0.6B on the partial v1 set

Commit: 3f438bd plus uncommitted `train/finetune.py` and `scripts/run_pipeline.sh` (committed immediately after this entry).

**Question/Context.** Can a small model learn the (query, filters, item metadata) -> (verdict, narrative) mapping from the teacher-written data, and how well does it do on held-out themes?

**What was done.** The large generation run (`data/v1`) was stopped by the 2-hour background limit with 3,241 of 4,133 pairs narrated; the 3,241 were assembled as-is (train 1,107 balanced to at most 1.5x the rarest class, val 263, test 267, split by theme). Full fine-tune (no LoRA) of `Qwen/Qwen3-0.6B` (596M parameters), fp32 weights with bf16 autocast, AdamW lr 2e-5, effective batch 16, 3 epochs, one seed, no hyperparameter search, on one H100. Greedy decoding on the test split, one evaluation.

**Result.**
- Training took about 1 minute (3 epochs). Validation loss 0.906, 0.856, 0.915 for epochs 1 to 3 while training loss fell 1.32, 0.64, 0.34, so overfitting begins after epoch 2. The epoch-3 weights were evaluated.
- Test verdict accuracy 0.861 (230 of 267) against 0.573 for always predicting "irrelevant". Macro-F1 0.854 (relevant 0.862, partly relevant 0.802, irrelevant 0.896). No invalid verdict lines. Test class counts: 153 irrelevant, 88 partly relevant, 26 relevant (unbalanced, so the relevant F1 rests on 26 examples).
- By source: study 0.98 (n=64), indicator 0.84 (140), dataset 0.80 (55), document 0.75 (8). By language: English 0.92 (109), Arabic 0.91 (23), Chinese 0.93 (15), French 0.85 (48), Spanish 0.76 (42), Swahili 0.60 (5), Russian 0.45 (11), Turkish 1.0 (14). Hindi has no test examples. Most groups are small.
- Narratives are fluent and in the right script (Arabic 97%, Chinese 99%, Russian 97% of letters in the expected script). Median 40 words, matching the teacher.
- Reading generations showed faithfulness errors that the verdict metric does not capture: a narrative that says the item does not match while the verdict line says "relevant"; claims about item content that the metadata does not state (Russian example); a nonsensical phrase in French ("le bruitage"). No automatic measure of narrative faithfulness has been run.

**Implication.** The task is learnable at this scale and the verdict is mostly right on unseen themes, but the test labels are the teacher's, not human judgments, and part of the accuracy may come from the deterministic filter-check line. Narrative faithfulness is the open problem: next steps are an LLM-judge faithfulness score on test generations, an ablation without the filter-check line, an early-stopped (epoch 2) run, and a smaller model closer to the 20 MB target. This is one seed on 267 test examples, so differences of a few points between runs are not interpretable.

## 2026-10-01 06:17 UTC — 20 MB model: from-scratch baseline, Qwen embedding initialization, shape ablation

Commit: a916c0e plus uncommitted `tiny/` package, corpus scripts and vLLM scripts (committed immediately after this entry).

**Question/Context.** The target is a model of about 20 MB. Can a ~20M-parameter decoder learn the task, and what helps it?

**What was done.**
1. From-scratch model: 20.77M parameters (10 layers, d=384, SwiGLU, RoPE, tied embeddings), 8,000-token BPE trained on the 1,107 task examples, 60 epochs, best epoch by validation loss. Exported with per-row symmetric INT8 and INT4 weights.
2. Pretraining corpus: Wikipedia (wikimedia/wikipedia 20231101) in the 11 query languages, 1.62B tokens with a new 12,000-token BPE trained on 25M characters per language plus the task text. English 411M, Chinese 271M, French 160M, Russian 149M, Spanish 152M, Arabic 129M, Hindi 103M, Portuguese 87M, Turkish 74M, Indonesian 66M, Swahili 20M tokens.
3. Pretraining ablations at 100M tokens each (762 steps, batch 128x1024 tokens, lr 2e-3, one seed each): random embedding initialization versus embeddings initialized from Qwen3-0.6B (each token string re-tokenized with Qwen's tokenizer, vectors averaged, PCA to d_model). Then three shapes with Qwen initialization.
4. Task fine-tune of the 100M-token Qwen-initialized model (lr 3e-4, 15 epochs, dropout 0.1, best epoch by validation loss).

**Result.**
- From scratch: best validation loss 4.06 at epoch 9, then memorization (train loss 0.003, validation 6.69). Test verdict accuracy 56.2% fp32, 55.4% INT8, 55.1% INT4 against a 57.3% majority baseline. Exported sizes 83.1, 20.9 and 10.5 MB. Narratives had the right shape and wrong content.
- Qwen embedding initialization, held-out perplexity at 100M tokens (random -> Qwen init): English 36.0 -> 21.3, French 36.4 -> 21.7, Arabic 58.0 -> 31.1, Chinese 119 -> 51.9, Russian 35.7 -> 19.3, Indonesian 54.1 -> 30.2, Hindi 10.3 -> 6.4, Turkish 41.3 -> 23.3, Swahili 50.6 -> 27.6. Portuguese held-out perplexity (1.9 to 2.8) is not meaningful; the slice appears to be repetitive stub articles. Held-out slices are the last 1M tokens of each language file and are small.
- Shape ablation with Qwen initialization (English perplexity): 7 layers d=448, 21.8M parameters 20.7; 10 layers d=384, 22.3M parameters 21.3; 14 layers d=320, 21.6M parameters 22.6. The wider and shallower model is best on every language, by small margins; one seed each, so the gaps between the wide and baseline models are within plausible noise.
- Task fine-tune from the 100M-token checkpoint: validation loss 1.88 at epoch 4 (from scratch: 4.06; fine-tuned Qwen3-0.6B: 0.86). Test verdict accuracy 60.3% fp32 (macro-F1 0.526), 60.7% INT8 (22.4 MB), 59.2% INT4 (11.3 MB). Generations were fluent but repeated phrases and echoed query words without grounding; most verdicts were "partly_relevant".

**Implication.** Qwen embedding initialization is a large gain for language modelling and is kept. The wide 7-layer shape is adopted. The judgment (verdict) is not learned from 1,107 examples even with pretraining: the task needs far more labelled data, and the teacher speed (0.7 examples per second with Ollama) limits that. A 4-gram repeat block was added to decoding. Next: serve the teacher with vLLM, generate tens of thousands of examples, pretrain on the full 1.6B-token corpus, and fine-tune.

## 2026-10-01 07:41 UTC — Retrieval similarity as an input; judge scores for the 20 MB model

Commit: 903c62a plus uncommitted batched decoding, similarity signal, judge and faithfulness code (committed immediately after this entry). Data: partial v2 narration (15.5k of 26.9k pairs) plus v1, so these are interim numbers.

**Question/Context.** Fine-tuning the 20.6M-parameter model (7 layers, d=448, Qwen-initialized embeddings, 1.6B-token Wikipedia pretraining, held-out English perplexity 11.4) did not learn the verdict. Would the retriever's similarity score help?

**What was done.**
1. Task fine-tune on a natural-mix snapshot (11,494 train, test 1,080, 71.6% irrelevant), lr 3e-4, 6 epochs: test accuracy 73.7% (INT8 73.9%), relevant F1 0.00, partly relevant F1 0.36. Then lr 1e-3, 20 epochs, with verdict-token loss weight 1 and 20: teacher-forced validation verdict accuracy about 0.77 against a 0.759 majority rate throughout, training loss falling to 0.12 while validation loss rose after epoch 3; test accuracy 74.4% (weight 1) and 73.4% (weight 20) against 71.6%.
2. Predictive value of the retrieval similarity (cosine between the query and the item title plus description, from `ai4data/devdata-search-granite-97m-multilingual-cmnrl`): gradient-boosted classifiers on a handful of features, trained on 14,475 examples and tested on 1,350 (71.0% majority), split by theme. Similarity alone: accuracy 0.790, macro-F1 0.561. Rank alone: 0.716, 0.304. Similarity plus filter-check flags, rank and item type: 0.806, 0.641 (per-class F1 relevant 0.45, partly relevant 0.58, irrelevant 0.90). Mean similarity by teacher verdict: relevant 0.58, partly relevant 0.43, irrelevant 0.18.
3. Added a `Retrieval similarity: 0.61` line to the model input (narratives unchanged, the teacher did not see it) and fine-tuned (14,623 train, 2,155 val, 1,370 test; lr 1e-3, verdict weight 5, best epoch 4 by validation verdict accuracy).
4. Judged 300 test examples with the Qwen3.8-27B teacher (same model that wrote the references, so it favours the references); also an unsupported-number-and-name proxy.

**Result.**
- With similarity: test verdict accuracy 82.2% fp32, 82.3% INT8 (20.73 MB), 82.0% INT4 (10.41 MB), against a 70.7% majority baseline. Macro-F1 0.700, 0.699, 0.694; per-class F1 relevant 0.57, partly relevant 0.62, irrelevant 0.91. This is higher than the feature-based gradient-boosted model (0.806, 0.641) and than the same model without the line (about 74%, macro-F1 0.45).
- Judge (n=300, student narratives): faithfulness 2.98 of 5 (40.3% scored 4 or more), coherence 3.44, consistent with the verdict 97.0%, correct language 98.3%. By language, faithfulness: English 3.67 (n=141), Spanish 3.23 (35), French 2.70 (46), Arabic 1.71 (17). The teacher's own narratives scored 5.0 on faithfulness and coherence, which is self-preference and not a neutral ceiling.
- Unsupported-item proxy: 16.1% of student narratives against 14.5% of teacher narratives contain a number or capitalized name absent from the input. French reads 37% for both because translated place names do not match the English input; the proxy does not see invented claims or nonsense words, so it is not a faithfulness measure. Reading samples showed invented descriptions and nonsense words, mostly outside English.

**Implication.** The retrieval similarity is the signal that makes the verdict learnable at 20 MB; the explainer takes the retriever's evidence as input, as in the original design. Narrative faithfulness is the open problem (about 3 of 5 by the judge). Test set is 1,370 examples with 84 relevant, one seed, and a data snapshot; the class-wise numbers for "relevant" are noisy. Next: finish v2 narration, retry the 153 retrievals that returned HTTP 429, retrain on all data, try a second pretraining pass (running), and measure faithfulness again.

## 2026-10-01 09:00 UTC — Full-data 20 MB model, faithfulness, copy-bias, export

Commit: 875015a plus uncommitted export, copy-bias option and README (committed immediately after this entry). Data `data/f1`: v1 (Ollama, 4,155 pairs) plus v2 (vLLM, 27,469 narrated) merged, de-duplicated, with the retrieval-similarity input line: train 25,198 (2,176 relevant), val 3,797, test 2,437 (233 relevant, 657 partly relevant, 1,547 irrelevant; majority rate 63.5%).

**Question/Context.** Does more data and a second pretraining pass improve the 20.6M-parameter model, and can faithfulness be improved cheaply at decoding time?

**What was done.**
- v2 generation on vLLM serving Qwen3.8-27B-FP8 (two workarounds for the missing CUDA compiler: `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`). Narration ran at about 4.8 per second against 0.7 with Ollama (prefill-bound, 97% KV cache use).
- Second pretraining pass over the 1.62B-token corpus (lr 1e-3, resumed): held-out perplexity English 11.4 -> 10.7, French 12.3 -> 11.5, Arabic 17.3 -> 16.3, Chinese 25.3 -> 23.5.
- Task training on `data/f1` (lr 1e-3, 8 epochs, batch 32, verdict-token weight 5, best epoch by validation verdict accuracy) from the first-pass checkpoint (`f1-pre1`) and the second-pass checkpoint (`f1-pre2`).
- Judge scores (n=400, same teacher model as judge), then a logit bonus for tokens occurring in the input applied after the verdict line, strengths 1, 2 and 4.
- Exported INT8 file and CPU timing.

**Result.**
- `f1-pre2` test accuracy 82.6% at fp32, INT8 and INT4; macro-F1 0.775, 0.774, 0.776; per-class F1 at INT8 relevant 0.77, partly relevant 0.66, irrelevant 0.90. `f1-pre1`: accuracy 81.2% fp32, 81.1% INT8, 80.0% INT4; macro-F1 0.743, 0.742, 0.718. Best epoch 6 of 8 for both. One seed each, so the 1.4-point gap between pre2 and pre1 is suggestive and not established.
- Judge, `f1-pre2` INT8: faithfulness 3.49 of 5 (53.7% scored 4 or more), coherence 3.85, argues for its verdict 98.8%, correct language 98.0%. Previous model (11k training examples): 2.98, 3.44, 97.0%, 98.3% (n=300). By language faithfulness: English 4.07 (n=174), Spanish 3.37 (52), French 3.31 (48), Arabic 3.05 (22), Chinese 2.95 (22), Indonesian 2.45 (20), Russian 2.37 (19). Teacher narratives scored 4.99 (self-judged).
- Copy bonus: verdict accuracy unchanged (82.6%) at every strength. Faithfulness 3.50 / 3.43 / 2.78 and coherence 3.80 / 3.53 / 2.77 at strengths 1 / 2 / 4, against 3.49 and 3.85 without it. No benefit; not used.
- Export: one file of 21.59 MB (0.84 MB of it the tokenizer), 18.4 MB gzipped. Loading takes 0.31 s. A random 300-example sample scores 81.0% accuracy with the loaded file (full test 82.6%; sampling error about 2 points). Single CPU thread: 172 ms to first token for a 330-token prompt (98 ms with 4 threads), 11 ms per token, 0.77 s for a 71-token output.

**Implication.** The task data and pretraining are worth continuing: both raised faithfulness and verdict accuracy. Faithfulness is still about 3.5 of 5 and about half of the narratives carry an unsupported claim, so the model is suitable as a draft explanation and not for unreviewed use. Decoding-time copy bias is a negative result. The 21.6 MB file is slightly above a strict 20 MB budget; options are a 6-layer variant or fewer vocabulary entries. INT4 accuracy is simulated, with no packed export. Candidate next steps: more theme and source diversity, human review of a sample of non-English narratives, a judge that is not the teacher, and training for Russian, Indonesian and Chinese faithfulness.
