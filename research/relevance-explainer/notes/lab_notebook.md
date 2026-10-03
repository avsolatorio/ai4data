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

## 2026-10-01 16:59 UTC — Distilling the teacher better: label noise, soft labels, on-policy preference tuning

Commit: f14177d plus uncommitted `distill/` package, `tiny/dpo.py`, sampling in decoding and soft-label training (committed immediately after this entry). Baseline: `f1-pre2` (7 layers, d=448, 20.64M parameters), data `data/f1` (25,198 train, 3,797 val, 2,437 test).

**Question/Context.** The student only copies one teacher narrative per input. Can the teacher's information be used better?

**What was done.**
1. Label noise: 500 random training inputs relabelled with the teacher in thinking mode (498 parsed) and compared with the stored no-thinking verdicts.
2. Teacher verdict probabilities: probability of the first verdict token after the prefill `{"verdict": "` of the narration prompt, for 400 validation inputs and 5,433 training inputs (job stopped there to free the servers).
3. Seed variation of the baseline recipe (seeds 0, 1, 2).
4. On-policy preference data: the baseline student sampled 4 narratives (temperature 0.8, top-k 40) for 6,000 training inputs with the teacher's verdict as a fixed prefix; the teacher judged all 24,000 (faithfulness, coherence, consistency, language). Pairs: rejected = the lowest-scoring sample when its faithfulness is 3 or lower (or coherence 2 or lower); chosen = the best sample when it scores faithfulness 5, coherence 4 or more and is consistent, otherwise the teacher's own narrative. 4,397 pairs (2,489 chosen from the student, 1,908 from the teacher; 1,603 inputs gave no pair).
5. DPO from the baseline, four settings; rejection-sampling fine-tuning on the chosen narratives (lr 1e-4, 1 epoch selected by validation loss); soft-label training (5,433 covered examples, other examples hard labels) at 8 and 14 epochs. Judge on 800 test examples (400 for the first comparisons).

**Result.**
- Label noise: thinking-mode verdicts agree with the stored verdicts on 88.2% (440 of 498). Stored irrelevant -> thinking partly relevant 40 of 325; stored partly relevant -> thinking irrelevant 10 of 137; stored relevant -> thinking partly relevant 7 of 36; partly relevant -> relevant 2.
- Teacher verdict probabilities (400 validation inputs): the teacher's own stored verdict has mean probability 0.904 and the argmax equals it for 96.3%; 8.0% of inputs have a probability below 0.6 for the stored verdict (19 partly relevant, 13 irrelevant). A first run reported 0.962 because a parsing bug dropped most uncertain partly relevant cases ("partly" is tokenized as "part" + "ly"); fixed and recomputed.
- Seeds: test accuracy 82.6%, 82.6%, 82.7%; macro-F1 0.774, 0.771, 0.772.
- On-policy scores (24,000 samples): mean faithfulness 3.50; 42% scored 5; 69% of inputs had at least one sample scoring 5 and 74% at least one scoring 3 or lower.
- Judge, 800 examples, INT8: baseline faithfulness 3.498 (54.5% at 4 or more), coherence 3.853; DPO beta 0.1, lr 1e-5, 2 epochs, sft 0.2: 3.580 (57.1%), 3.894; DPO beta 0.3, lr 2e-5, 2 epochs, sft 0.5: 3.647 (58.6%), 3.876, consistent 99.1%, language 97.8%; rejection-sampling fine-tuning: 3.596 (57.3%), 3.929. With 400 examples: DPO beta 0.1, lr 5e-5, 3 epochs: 3.37, coherence 3.39, language 90.7%; DPO beta 0.05, lr 3e-5, 3 epochs, sft 0.1: 2.72, 2.57, 78.5% (over-optimization). Verdict accuracy of the selected DPO model: 82.4% fp32, 82.5% INT8, 82.6% INT4, macro-F1 0.773, 0.775, 0.773 (baseline 82.6%); the over-optimized settings gave 82.7% and 81.5%.
- Per-language faithfulness, baseline -> selected DPO model (n=800 total): English 4.13 -> 4.28 (334), French 3.46 -> 3.64 (98), Spanish 3.39 -> 3.45 (93), Arabic 2.94 -> 3.09 (54), Chinese 3.21 -> 3.42 (48), Russian 2.25 -> 2.52 (44), Turkish 2.62 -> 2.79 (29), Hindi 2.25 -> 2.68 (28), Indonesian 2.52 -> 2.44 (27), Portuguese 3.77 -> 3.38 (26), Swahili 2.58 -> 3.00 (19).
- Soft labels: the first run (soft term weighted by the verdict weight 5) trained slowly (train loss 6.9 at epoch 1; gradient clipping at 1.0 shrinks the whole update); with its own weight of 1: test accuracy 81.0% (8 epochs) and 80.3%, macro-F1 0.741 and 0.736 (14 epochs), against 82.6% and 0.774 for the baseline. No benefit observed.
- Export of the selected model: 21.59 MB.

**Implication.** Preference tuning on the student's own scored samples gives a small, consistent faithfulness gain (+0.15 of 5, about 3 standard errors at n=800) without changing the verdict, but only with conservative settings; stronger settings degrade the text. Fine-tuning on the preferred narratives gets two thirds of that gain more simply. The teacher is close to one-hot, so verdict probabilities carry little extra, and the 12% label disagreement at the irrelevant and partly relevant boundary limits the achievable verdict accuracy. Caveats: the judge is the same model as the teacher and also selected the pairs, so part of the gain may be judge preference; the soft-label test covered 21% of the training inputs; per-language groups are small. Not tried: a second on-policy round, full-coverage soft labels, distribution matching on narrative tokens (needs the student to share Qwen's tokenizer), and a larger model at INT4 (a 40M-parameter model fits the same file size).

## 2026-10-01 19:53 UTC — Layer sharing (recursion) and adaptive depth at 20M parameters

Commit: cb179d5 plus uncommitted layer-sharing option, latency benchmark and tests (committed immediately after this entry).

**Question/Context.** Would weight-tied layers (looped depth) save space without losing quality, and would they or adaptive depth make generation faster? Prompted by Mixture-of-Recursions (arXiv 2507.10524), which reports parity with standard models from 360M parameters, worse results at 135M, and up to 2.06x to 2.18x inference throughput from continuous depth-wise batching of many requests; it tests nothing below 135M.

**What was done.**
1. Added `n_unique` and `share` ("repeat": each block applied consecutively; "cycle": blocks reused in rotation) to the model; KV caches are kept per application. Decoding equivalence between batched and single-sequence paths was tested.
2. Pretrained five shapes for 100M tokens each (762 steps, Qwen-initialized embeddings, d=448, 7 heads, feed-forward 1024, one seed each) and compared held-out perplexity.
3. CPU latency of the same shapes (random weights, fp32, one thread, AMD EPYC 9654 with 768 MB of L3, 330-token prompt, 64 new tokens, best of 5).
4. Early-exit probe on `f1-dpo-d` (7 passes): the output-layer guess after each layer, teacher-forced on the reference outputs of 300 test examples (300 verdict tokens, 24,376 narrative tokens); no early-exit training.

**Result.**
- Held-out perplexity (English / French / Arabic / Chinese / Russian / Hindi): unshared 8 blocks (22.8M parameters) 21.1 / 21.8 / 31.4 / 51.1 / 18.7 / 6.3; plain 4 layers (14.1M) 22.8 / 23.6 / 32.9 / 55.0 / 20.2 / 6.7; 8 passes over 4 shared blocks, cycle (14.1M) 22.9 / 23.7 / 33.8 / 55.6 / 20.5 / 7.0; the same with repeat 23.3 / 23.7 / 34.2 / 55.8 / 20.8 / 7.0; 12 passes over 4 shared blocks, repeat (14.1M) 23.1 / 23.6 / 33.5 / 54.7 / 20.5 / 6.7. At equal stored size, looping gave no gain over the plain 4-layer model, and 12 passes did not beat 8. Against the unshared 8-block model, sharing lost about 8% to 10% in English perplexity for 8.7M fewer parameters.
- Latency (first token / per token): 8 passes unshared 192 ms / 11.2 ms; 8 passes shared 181 ms / 11.0 ms; 4 passes 94 ms / 6.1 ms; 12 passes shared 282 ms / 12.8 ms; 7 passes (current model) 161 ms / 9.6 ms. Time follows the number of passes and not the number of distinct blocks. The CPU's cache holds every model, so the memory-bandwidth saving that sharing could give on a small device is not measured here.
- Early-exit probe: the verdict token matches the final answer 0% after layer 3, 8% after 4, 71% after 5, 91% after 6. Narrative tokens match the final answer 17%, 23%, 30%, 39%, 52%, 79%, 100% after layers 1 to 7. Exiting at the first layer with confidence at least 0.9 gives a mean depth of 5.85 of 7 with 94.8% of narrative tokens unchanged (verdict: 6.33, 99.7%); at 0.99, 6.53 and 99.6%. Models trained for early exit do better than this untrained read-out, so it is a pessimistic estimate.

**Implication.** Weight sharing saves bytes and not time, and at 20M parameters with 100M tokens it also does not preserve quality at equal stored size. Speed comes from fewer layer passes (4 passes: 6.1 ms per token against 9.6 ms for 7). Adaptive depth has little room in a 7-layer model and the verdict token needs nearly full depth. Mixture-of-Recursions is not a good fit at this size, in line with the paper's own 135M result. Caveats: 100M tokens of pretraining, one seed per shape, perplexity only (no task fine-tune), and recursion may need longer training to pay off. Untested: a looped model fine-tuned on the task, a model with fewer passes (4 or 5 layers) on the task, and packed INT4 kernels where memory bandwidth matters.

## 2026-10-01 21:54 UTC — Layer sharing re-checked at tuned learning rates

Commit: 2c3fb63 (no new code; results only).

**Question/Context.** The layer-sharing study above used one learning rate (2e-3) for every shape. Tied-weight models can need a different rate, so the "sharing gives no gain at equal stored size" result could have been an artefact of that choice. Separately, the user asked whether the implementation was correct.

**What was done.**
- Implementation audit: shared blocks are the same objects at every application (repeat order [0,0,1,1,2,2,3,3], cycle order [0,1,2,3,0,1,2,3]); every block receives a non-zero gradient; parameter counts match the stored blocks; batched KV-cache decoding for shared models matches single-sequence decoding (unit test); training-loss curves of all five shapes are smooth (3.53 to 3.64 at step 400, 3.25 to 3.35 at step 762).
- Learning-rate grid, 100M tokens each (762 steps), same data and settings as before, one seed per cell: 5e-4, 1e-3, 2e-3 and 4e-3 for the unshared 8-block model and the shared cycle; 1e-3, 2e-3 and 4e-3 for plain 4 layers and the shared repeat.

**Result.** English held-out perplexity at step 762:

| Shape | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|
| 8 blocks, unshared (22.8M) | 32.7 | 23.8 | 21.1 | 21.3 |
| 4 layers, plain (14.1M) | n/a | 28.4 | 22.8 | 22.3 |
| 8 passes over 4 blocks, cycle (14.1M) | 39.3 | 29.3 | 22.9 | 22.5 |
| 8 passes over 4 blocks, repeat (14.1M) | n/a | 26.1 | 23.3 | 22.7 |

At each shape's best tested rate: unshared 8 blocks 21.1, plain 4 layers 22.3, shared cycle 22.5, shared repeat 22.7. Lower rates were worse for every shape. At 1e-3 the shared repeat (26.1) beat plain 4 layers (28.4), but at their best rates plain 4 layers was ahead by 0.2 to 0.4. The best rate for plain 4 layers and both shared shapes is the largest one tested (4e-3), so a higher rate (not run) could move them further; the unshared model peaks at 2e-3 to 4e-3.

**Implication.** The earlier conclusion holds under a tuned learning rate: at equal stored size, looping blocks to double the depth gives no measurable gain over a plain 4-layer model (differences of 0.2 to 0.4 perplexity, one seed, 100M tokens), and the time per token is that of the extra passes. The implementation checks passed. Untested: rates above 4e-3, longer training, and task fine-tuning of looped models.

## 2026-10-01 22:48 UTC — Error types, and a hybrid output (one sentence plus separate filter display)

Commit: b174620 plus uncommitted `distill/short_format.py`, `tiny/judge_issues.py` and `scripts/pretrain.sh` (committed immediately after this entry). Judge and sentence writer: Qwen3.8-27B-FP8 (the model that also wrote the training narratives).

**Question/Context.** About 41% of the current best model's narratives score 3 or lower on faithfulness. Which problems dominate, and would a hybrid output reduce them: deterministic display of the filter results next to a short model-written sentence?

**What was done.**
1. Error types: the teacher named the single most damaging problem in each of 800 narratives of `f1-dpo-d` (INT8), from a fixed list of categories (single judge pass, one label per narrative).
2. Hybrid format: for 10,000 random training inputs the teacher wrote one sentence of at most 25 words supporting the already stored verdict, told not to mention filters, years, countries or coverage, and allowed to quote up to 8 words of the item's own text. 9,788 inputs gave a valid sentence (no empty, multi-line or over 40-word output, and every quoted phrase occurs in the input); median 19 words. Two models were trained from the same pretrained checkpoint with the same recipe (lr 1e-3, 8 epochs, verdict weight 5, best epoch by validation verdict accuracy) on the same 9,788 inputs: `h-old10k` with the original narratives, `h-short10k` with the sentences. Both were judged at fp32 on the same 800 test examples with the issue judge; for the sentence model the judge was told that filter results are displayed separately and must not be penalized when absent.

**Result.**
- Error types in the 405 narratives of `f1-dpo-d` scoring 3 or lower (this judge grades harder than the earlier one): item content 51.9%, garbled text 25.4%, misstated filter status 12.1%, numbers or names 6.2%, query relation 4.0%, contradicts its verdict 0.5%. English only (109): item content 45.9%, garbled 19.3%, filter status 19.3%. Other languages (296): item content 54.1%, garbled 27.7%, filter status 9.5%.
- Verdict accuracy, `h-old10k` / `h-short10k` (test 2,437, majority 63.5%): fp32 80.7% / 80.3%, INT8 80.8% / 80.3%, INT4 80.9% / 79.6%; macro-F1 at INT8 0.737 / 0.728. Median output length 37 / 18 words.
- Faithfulness (800 judged examples): mean 2.83 / 3.07; share scoring 3 or lower 62.9% / 49.6%. Main issue as a share of all 800: none 35.4% / 49.7%, item content 34.0% / 26.5%, misstated filter status 11.0% / 2.0%, garbled 9.9% / 12.4%, numbers or names 4.8% / 5.5%, query relation 5.0% / 3.9%. Share with a problem, English only: 46.4% / 34.7%; other languages: 77.7% / 61.4%.

**Implication.** The hybrid removes almost all filter-status errors and lowers the share of explanations with a faithfulness problem by 13 points (63% to 50%, about a fifth fewer), at a cost of about 0.4 points of verdict accuracy (within the usual seed variation of about 0.1 points plus a possible small real effect; not separated). Item-content errors remain the largest category (about half of the remaining problems), and garbled text does not fall, so non-English free text is still the weak part. Caveats: only 9,788 training examples (the 25k-example model has a lower problem rate under the earlier judge, and the two judge prompts are not comparable), one seed each, the sentence writer and judge are the same model, the deterministic filter display is assumed correct and was not judged, and no user study was done on whether the structured layout is preferred. Not tried: all 25k examples in the new format, copying the item's own definition text for the content part, a shorter sentence, and a native-speaker check of non-English output.

## 2026-10-02 01:06 UTC — Per-language models and adapters: capacity per language is not the fine-tuning bottleneck

Commit: 8a4d911 plus uncommitted `tiny/lora.py`, `distill/split_language.py`, a `--language` filter in the judge and `--lora-rank` in task training (committed immediately after this entry).

**Question/Context.** Non-English narratives are the weak part (faithfulness 2.4 to 3.4 against 4.1 to 4.3 for English). If the 20M model lacks capacity per language, a model fine-tuned on one language, or a shared base with a per-language adapter, should do better on that language. The full per-language fine-tune is the upper bound for any adapter.

**What was done.**
1. Monolingual fine-tunes from the shared pretrained base (`pre-wide-full2`), same recipe as the multilingual model (lr 1e-3, batch 32, verdict weight 5, best epoch by validation verdict accuracy): English only (9,890 train, 8 epochs), French only (3,168, 12 epochs), Russian only (1,002, 20 epochs).
2. A rank-16 adapter (alpha 32, on qkv, proj, gate, up and down of every block; 0.8M parameters, about 0.8 MB at INT8) trained on French data on top of the multilingual task model `f1-pre2`, base frozen, 12 epochs at lr 1e-3; best epoch 1.
3. Judge (same prompt and model as before) on the same test examples per language: the multilingual `f1-pre2` filtered to the language, and each per-language model on its language's test split. English 400 of 977 sampled with a fixed seed, French all 325, Russian all 134. INT8 weights.

**Result.**
- English: multilingual faithfulness 4.25 (75.7% at 4 or more), verdict accuracy 85.5%; English-only 3.96 (68.0%), 83.7%.
- French: multilingual 3.28 (47.7%), 80.3%; French-only 2.90 (39.1%), 75.7%; French adapter on the multilingual model 3.31 (49.8%), 80.6%.
- Russian: multilingual 2.21 (18.7%), 79.9%; Russian-only 1.84 (14.2%), 70.9%.
- Coherence followed the same pattern (English 4.38 against 4.26; French 3.79 against 3.71; Russian 2.63 against 2.87, the one exception).

**Implication.** Restricting the task fine-tune to one language made every language worse, including English with 9,890 examples. The task knowledge (what to say about an item and how to argue a verdict) transfers across languages, and each language alone has too little task data. The adapter on top of the multilingual model changed nothing measurable (+0.03 faithfulness, +0.3 accuracy, one rank, one language). So the weak non-English output is not explained by competition for capacity in the fine-tune. What remains as explanation: the pretrained representation of the smaller languages (Swahili 20M tokens, Russian 149M, against 411M for English) and the overall model size. Caveats: one seed per model, the judge is the teacher, Russian test has 134 examples, and the adapter was tried at one rank and learning rate. Not tried: adapters trained on a language's Wikipedia slice (continued pretraining per language), and mixing in all languages while up-weighting one.

## 2026-10-02 01:26 UTC — 40M parameters at INT4 against 20M at INT8

Commit: 4caf133 (results only; `scripts/pretrain.sh` committed earlier).

**Question/Context.** INT4 cost the 20M model nothing, so a model with twice the parameters at INT4 fits the same file size. Does it generate better narratives?

**What was done.** Pretrained a 38.77M-parameter model (8 layers, d=576, 9 heads, feed-forward 1536, same tokenizer and Qwen-initialized embeddings) on the 1.62B-token corpus, one pass, lr 2e-3: held-out perplexity English 9.85, French 10.61, Arabic 15.14, Chinese 21.83, Russian 9.25 (20M after two passes: 10.7, 11.5, 16.3, 23.5, 9.9). Task fine-tune on `data/f1`. A first run with lr 1e-3 and best epoch by validation verdict accuracy selected epoch 2 (validation loss 1.171 while loss kept falling to 1.061 at epoch 5) and scored badly on narratives (faithfulness 2.92 at INT8, verdict accuracy 81.2%); it is discarded as an under-trained checkpoint. The rerun used lr 5e-4 and best epoch by validation loss (epoch 6, loss 0.999 against 1.073 for the 20M model). Judge on the same 800 test examples as the 20M results.

**Result** (verdict accuracy on 2,437 test examples; faithfulness and coherence on 800 judged):

| Model | File | Accuracy | Macro-F1 | Faithfulness | Coherence |
|---|---|---|---|---|---|
| 20M, INT8 (`f1-pre2`) | 20.7 MB | 82.6% | 0.774 | 3.50 | 3.85 |
| 20M, INT8, after DPO (`f1-dpo-d`) | 20.7 MB | 82.5% | 0.775 | 3.65 | 3.88 |
| 40M, INT4 (`f1-40m-b`) | 19.5 MB | 82.1% | 0.767 | 3.46 | 3.84 |
| 40M, INT8 | 38.9 MB | 82.5% | 0.770 | 3.59 | 4.02 |
| 40M, fp32 | 155 MB | 82.5% | 0.770 | 3.57 | 4.02 |

**Implication.** At equal file size, 40M parameters at INT4 match the 20M model at INT8 and do not beat it (differences of 0.04 faithfulness and 0.5 accuracy points, within noise). INT4 costs the 40M model about 0.1 faithfulness and 0.17 coherence against its own INT8, which removes the gain from the extra parameters. At twice the file size (INT8), the 40M model adds 0.09 faithfulness and 0.16 coherence over the 20M model, less than preference tuning added (+0.15) at no size cost. Doubling the parameters in this range is a weak lever; the quality is bounded by something else, most plausibly the teacher data (12% label disagreement, one narrative per input) and the amount of pretraining per language. Caveats: one seed per model, one pass of pretraining for the 40M model against two for the 20M, judge is the teacher, and quantization-aware training was not tried (it might recover the INT4 loss).

## 2026-10-02 02:27 UTC — Deferred experiment: byte-level input (Byte Latent Transformer or ByteFlow)

Commit: 8c1f946 (decision only, no code).

**Question/Context.** The embedding table (12,000 byte-level BPE tokens x 448) is 26% of the 20M-parameter model, and the small BPE penalizes Arabic, Hindi and Chinese with more tokens per word. Byte-level architectures remove the table and the subword penalty. Byte Latent Transformer (arXiv 2412.09871) reports parity with BPE models from about 1B parameters with patching by next-byte entropy; its small models trail BPE. ByteFlow (ICLR 2026) is a newer adaptive-compression variant without independent replication.

**Decision.** Deferred, not rejected. Reasons: the measured bottleneck is the training signal and pretraining tokens per language, which a byte architecture does not change; the published gains appear above our scale; byte-level encoder and decoder layers run at byte rate, which is a harder CPU latency problem than the current 11 ms per token; and the test costs one to two days including bits-per-byte evaluation plumbing (perplexities across tokenizations are not comparable).

**Plan when picked up.** First a cheap probe with the existing code: pretrain 6,000, 12,000 and 24,000 BPE vocabularies for 100M tokens each and compare bits per byte per language. If vocabulary size barely matters, the structural benefit of bytes is unlikely to matter either; if it matters a lot, implement BLT-style entropy patching (or ByteFlow) at 20M parameters, pretrain on the same corpus, fine-tune on the same task data, and compare verdict accuracy, judged faithfulness, file size and CPU latency against the BPE model.

## 2026-10-02 06:09 UTC — More task data: the learning curve from 25k to 60k examples

Commit: 4a339c9 (data and runs are untracked). Teacher and judge: Qwen3.8-27B-FP8 on vLLM.

**Question/Context.** Is the narrative quality bounded by the amount of teacher-written task data? The verdict had saturated by about 15k examples, but faithfulness had kept rising (2.98 at 14.6k, 3.49 at 25k).

**What was done.**
- Third data round (`data/v3`): 1,500 theme and 1,500 anchored (theme, source) combinations with a flatter language mix (English 20%, Arabic, Chinese and Russian 10% each, Hindi and Indonesian 8%, Turkish 6%, Swahili 4%), seed 200; 11,301 queries, 44,725 retrieved pairs, 44,718 narrated in about 3 hours on one vLLM server.
- Combined set `data/f2` = v1 + v2 + v3, de-duplicated, with the retrieval-similarity line: train 60,157 (5,120 relevant, 15,235 partly relevant, 39,802 irrelevant), val 8,866, test 7,056 (majority rate 64.4%), split by theme as before. Per-language training counts rose most for the small languages: Russian 1,002 -> 4,610, Swahili about 500 -> 2,007, Hindi -> 4,093, Indonesian -> 3,539, Turkish -> 2,553.
- Three models on the same base (`pre-wide-full2`) and recipe (lr 1e-3, 8 epochs, batch 32, verdict weight 5, best epoch by validation verdict accuracy): the existing 25k model (`f1-pre2`, re-evaluated on the new test), a random half of the combined train set (30,078; `f2half-pre2`) and the full set (`f2-pre2`). Judge on the same 800 examples of the new test set (seeded sample); the 25k model at INT8, the other two at fp32 (INT8 has matched fp32 in every earlier run).

**Result.**
- Verdict accuracy on the 7,056-example test: 82.7% (25k, macro-F1 0.766), 83.2% (30k, 0.772), 83.5% (60k, 0.777); 60k at INT8 83.6% / 0.778, at INT4 83.3% / 0.773.
- Judge, same 800 examples: faithfulness 3.24 (25k), 3.57 (30k), 3.85 (60k); coherence 3.62, 3.96, 4.08; share scoring 4 or more 46.9%, 56.4%, 63.9%; consistent with the verdict 98.1%, 98.6%, 99.0%; correct language 97.4%, 99.0%, 99.3%.
- Faithfulness by language, 25k -> 30k -> 60k (n in the 800): English 4.21 -> 4.24 -> 4.47 (220), French 3.31 -> 3.64 -> 3.81 (74), Spanish 3.53 -> 3.58 -> 4.01 (88), Portuguese 3.55 -> 3.98 -> 4.38 (42), Arabic 2.68 -> 3.42 -> 3.48 (66), Chinese 2.78 -> 3.06 -> 3.42 (72), Russian 2.51 -> 3.25 -> 3.55 (77), Hindi 2.25 -> 3.02 -> 3.23 (48), Indonesian 2.50 -> 3.00 -> 3.31 (48), Swahili 2.83 -> 3.17 -> 3.65 (23), Turkish 2.31 -> 2.81 -> 3.12 (42).
- The 25k model scores lower on this test (3.24) than on the old one (3.50) because the new test has a larger non-English share; comparisons within this test are like for like.
- Export of the 60k model: 21.59 MB INT8 file.

**Implication.** Task data is the lever that was missing: 25k -> 60k examples gave +0.61 faithfulness and +17 points in the share of acceptable narratives, with the largest gains in the languages that had the least data (Russian +1.04, Hindi +0.98, Arabic +0.80, Swahili +0.82, Turkish +0.81). The curve is still rising between 30k and 60k (+0.28), so another round is likely to help, with diminishing returns expected as themes repeat. The verdict moved by 0.8 points, consistent with its label-noise ceiling. Caveats: one seed per point, the 30k point is a random half (same themes as the 60k set, so it measures data volume and not theme diversity), per-language groups of 23 to 88 judged examples, and the judge is the teacher. Preference tuning has not yet been applied to the 60k model; the earlier +0.15 should stack. The expanded pretraining corpus (first pass) did not help on 25k data (82.0% and 3.40 against 82.6% and 3.50); a second pass and its fine-tunes on 25k and 60k data are running.

## 2026-10-02 07:18 UTC — Expanded pretraining corpus (FineWeb-2 added): no gain

Commit: d208b94 (corpus code committed in eb79696 and 2f54e07).

**Question/Context.** The small languages had 20M to 150M pretraining tokens. Does more pretraining text for them improve the fine-tuned model, especially non-English faithfulness?

**What was done.** Added FineWeb-2 web text (ODC-By) for the ten non-English languages, streamed until a character budget per language: 250M to 450M characters each, 5.3 GB on disk, tokenized with the existing 12,000-token BPE into 1.62B tokens. Per-language totals went from (Wikipedia only) Swahili 20M, Indonesian 66M, Turkish 74M, Hindi 103M, Russian 149M to 127M, 197M, 226M, 404M, 341M; corpus 1.62B -> 3.24B tokens. Shards are concatenated ahead of the Wikipedia shard so the held-out Wikipedia tails are unchanged. Pretrained the 20.64M model (7 layers, d=448) on the expanded corpus for 3.2B tokens in two passes (lr 2e-3 then 1e-3), matching the old base's two passes over 1.6B tokens; language sampling followed the new sizes (alpha 0.5). Fine-tuned each base on the 25k set (`data/f1`) and on the 60k set (`data/f2`) with the standard recipe; judged on the standard 800 examples of each test set.

**Result.**
- Held-out Wikipedia perplexity, old base -> new base after two passes: English 10.7 -> 11.3, French 11.5 -> 11.7, Arabic 16.3 -> 15.8, Chinese 23.5 -> 24.6, Russian 9.9 -> 9.7, Indonesian 16.1 -> 16.5, Hindi 2.9 -> 4.2, Turkish 11.8 -> 12.5, Swahili 13.8 -> 13.2. After one pass the new base was worse on every language (English 12.0, Russian 10.3, Swahili 13.9).
- 25k task data: verdict 82.6% / macro-F1 0.774 (new base, fp32) against 82.6% / 0.775 (old); faithfulness 3.49 against 3.50, coherence 3.90 against 3.85; Russian faithfulness 2.22 against 2.21 (134 examples).
- 60k task data: verdict 83.4% / 0.773 (new base, INT8) against 83.6% / 0.778 (old); faithfulness 3.70 against 3.85, coherence 3.95 against 4.08, share at 4 or more 58.9% against 63.9% (same 800 examples).

**Implication.** Doubling the pretraining corpus with web text, and raising the small languages' pretraining tokens 2 to 6 times, gave no improvement on the task and a small loss on the 60k data (−0.15 faithfulness, about three standard errors at n=800, one seed). The old Wikipedia-only base stays. Possible reasons, not separated: the web text is a different register from the task text and the Wikipedia held-out slices; the language sampling moved share away from English; a single tokenizer trained on Wikipedia. Together with the earlier results, pretraining volume per language is not the lever at this model size, while task data is (previous entry). Not tried: a tokenizer retrained on the mixed corpus, web text for English too, or a third pass.

## 2026-10-02 08:12 UTC — Preference tuning on the 60k model

Commit: 74cf7b2 plus uncommitted `scripts/run_round.sh` (committed immediately after this entry).

**Question/Context.** Does the +0.15 faithfulness from on-policy preference tuning (measured on the 25k model) stack on top of the 60k model?

**What was done.** Same procedure as before: the 60k model (`f2-pre2`) sampled 4 narratives (temperature 0.8, top-k 40, verdict fixed) for 8,000 random training inputs of `data/f2`; the teacher scored all 32,000 (about 2 hours on one vLLM server, shared with judge runs); pairs built with the same rules gave 5,203 pairs (3,255 chosen from the student, 1,948 from the teacher narrative; 2,797 inputs without a pair). DPO from `f2-pre2` with the settings selected earlier (beta 0.3, lr 2e-5, 2 epochs, batch 16, supervised term 0.5): preference accuracy 0.99 and log-ratio margin 8.7 after epoch 2. Evaluated on the 7,056-example test; judged on the standard 800.

**Result.**
- Verdict: 83.6% / macro-F1 0.782 at INT8 (before DPO 83.6% / 0.778), 83.2% / 0.776 at INT4.
- Judge, INT8, same 800 examples: faithfulness 3.99 (before 3.85), share at 4 or more 70.0% (63.9%), coherence 4.19 (4.08), consistent with the verdict 99.4%, correct language 99.0%. At INT4: 3.94, 66.7%, 4.08.
- By language, 60k -> 60k+DPO (n): English 4.47 -> 4.52 (220), French 3.81 -> 3.91 (74), Spanish 4.01 -> 4.20 (88), Portuguese 4.38 -> 4.17 (42), Arabic 3.48 -> 3.70 (66), Chinese 3.42 -> 3.74 (72), Russian 3.55 -> 3.64 (77), Hindi 3.23 -> 3.62 (48), Indonesian 3.31 -> 3.54 (48), Swahili 3.65 -> 3.91 (23), Turkish 3.12 -> 3.31 (42).
- Export: 21.59 MB INT8 file (`runs/f2-dpo/explainer-int8.pt`).

**Implication.** Preference tuning stacks with the data gain: +0.14 faithfulness on top of the 60k model, the same size as on the 25k model, with the verdict unchanged. Over this session the 20 MB model went from 3.50 to 3.99 judged faithfulness (54.5% to 70.0% of narratives at 4 or more) through more task data and one round of preference tuning; the lowest language (Turkish) is now 3.31 against 2.25 for the lowest before. Caveats unchanged: one seed, the judge is the teacher and also selected the pairs, Portuguese (n=42) moved down. Next: a fourth data round (the learning curve is still rising), then a second preference round on the larger model.

## 2026-10-02 13:28 UTC — Information budget: is 20 MB enough for this task?

Commit: 8f6cd9d plus uncommitted `tiny/entropy.py` (committed immediately after this entry).

**Question/Context.** The user asked whether the 20 MB constraint is feasible in information-theoretic terms. The useful quantities are: how much information a narrative carries beyond its input (the irreducible part any imitator must produce), how far the student is from that, and how much of the model's capacity the task actually consumes.

**What was done.** On 600 random validation examples of `data/f2`: (1) teacher self-surprisal, the log-probability the teacher (Qwen3.8-27B-FP8, thinking off) assigns to its own narrative in the exact JSON form it produced, given the narration prompt, via echoed prompt log-probabilities; (2) teacher-forced cross-entropy of the same narrative under the 60k student (`f2-pre2`, fp32) given the student input and verdict line; (3) the share of narrative words (4+ letters) present in the input. Reported in bits per character so the two tokenizers are comparable. Also the gzip size of the exported INT8 model as an upper bound on the information in its weights.

**Result.**
- Teacher self-surprisal: 0.111 bits per character overall, 27 bits per narrative (English 0.080 and 22 bits, n=158; other languages 0.124 and 29 bits, n=442; Chinese 0.305 per character but 27 bits per narrative; Swahili 0.214 and 50 bits). The narratives were sampled at temperature 0.4, so this is the surprisal of sharpened samples under the temperature-1 model; a student that matched the teacher's distribution exactly would score about the same on these targets.
- Student cross-entropy: 0.466 bits per character, 115 bits per narrative (English 0.334 and 93; other languages 0.521 and 123; Chinese 1.949 and 171; Turkish 0.640 and 165; Swahili 0.636 and 148; Spanish 0.337 and 89). The student spends about 4.2 times the teacher's bits, an excess of about 88 bits per narrative overall and 71 for English.
- Copy share: 38% of narrative words occur in the input (English 63%, other languages 29%; the Chinese value is a regex artefact).
- Capacity: 20.64M parameters; INT4 (about 80M bits) loses nothing measurable; the INT8 export compresses from 21.6 MB to 18.3 MB with gzip. The task-specific information in the 60k training set, measured as teacher surprisal of the narratives given the inputs, is about 60,157 x 27 bits = 1.6M bits, or 0.2 MB.

**Implication.** The output is almost determined by the input: a narrative carries about 27 bits beyond what the input, the verdict and the teacher's style fix, roughly three to four bytes. The task's irreducible information is tiny relative to the model (0.2 MB of narrative information for the whole training set against 20 MB of weights), so storage is not what limits it. What the student lacks is the function that maps input to a faithful rendering in each language; it currently pays 88 extra bits per narrative, and those bits are where the unfaithful content and garbled words live. Three independent measurements say that capacity is not the binding constraint at this quality level: INT4 is free, doubling parameters (40M) gave +0.09 faithfulness, and 2.4 times the task data gave +0.61 with the curve still rising. A plausible floor for the size is set by language competence (about 1 bit per character on held-out Wikipedia English for this model; the narrow register of the narratives needs less), and TinyStories-scale results show fluent narrow-domain generation from 1 to 10M parameters per language, which is consistent with 11 languages in 15M non-embedding parameters being tight but workable. The constraint would bind if the task required knowledge absent from the input (it does not: the metadata is supplied) or teacher-level faithfulness in free prose across all 11 languages, where Chinese, Turkish and Swahili show the largest excess bits. Caveats: one student checkpoint, 600 examples, per-language groups of 24 to 158, and the teacher's own surprisal is a bound for imitating this teacher, not for the task in the abstract.

## 2026-10-02 15:57 UTC — Fourth data round: 100k examples, and preference tuning on top

Commit: c2a1656 (scripts `run_round.sh` and `round_when_ready.sh` committed in 6a19b4c and 8f6cd9d).

**Question/Context.** Is the data learning curve still rising past 60k examples?

**What was done.** Round 4 (`data/v4`, seed 300, same flat language mix): 11,292 queries, 45,205 pairs, 45,199 narrated. Narration ran at about 5 per second with the teacher at 90% of GPU 0 and 3.2 per second after it was restarted at 45% (about 43 GB, 14 concurrent requests) to free GPU memory. Combined set `data/f3` = v1 to v4: train 95,405 (8,152 relevant, 24,345 partly relevant, 62,908 irrelevant), val 14,469, test 11,326 (majority 64.5%). The unattended chain then fine-tuned from `pre-wide-full2` (standard recipe; best epoch 7 of 8), sampled 8,000 x 4 narratives, scored them (80 minutes at the smaller server size), built 4,986 pairs (3,149 chosen from the student, 1,837 from the teacher) and ran DPO with the usual settings. For the learning curve, both models were also evaluated on the third-round test set (7,056 examples) and judged on the same 800 examples as all earlier points.

**Result.**
- Verdict on the 7,056-example test (INT8): 100k SFT 84.6% / macro-F1 0.792; 100k + DPO 84.5% / 0.791; previous 60k + DPO 83.6% / 0.782. On its own 11,326-example test: 84.8% / INT4 84.6% (SFT), 84.5% / 0.792 and INT4 84.3% / 0.790 (DPO).
- Judge, same 800 examples, INT8: SFT faithfulness 4.04 (69.6% at 4 or more), coherence 4.28; with DPO 4.24 (76.4%), coherence 4.34, consistent 99.9%, correct language 99.3%. Learning curve of SFT models: 25k 3.24, 30k 3.57, 60k 3.85, 100k 4.04. With DPO: 60k 3.99, 100k 4.24.
- By language, 60k+DPO -> 100k+DPO (n): English 4.52 -> 4.65 (220), Spanish 4.20 -> 4.39 (88), Portuguese 4.17 -> 4.36 (42), French 3.91 -> 4.27 (74), Chinese 3.74 -> 4.15 (72), Swahili 3.91 -> 4.04 (23), Russian 3.64 -> 4.03 (77), Turkish 3.31 -> 3.98 (42), Arabic 3.70 -> 3.92 (66), Indonesian 3.54 -> 3.73 (48), Hindi 3.62 -> 3.71 (48).
- Export: 21.59 MB INT8 (`runs/f3-dpo/explainer-int8.pt`).

**Implication.** The curve is still rising but flattening: +0.28 from 30k to 60k, +0.19 from 60k to 100k for the SFT models; the DPO gain stays about +0.2. Every language improved, and the floor rose from 3.31 to 3.71; Hindi and Indonesian gain least. The verdict reached 84.5% (from 82.5% at the start of the data work), with the label-noise ceiling near 88 to 90%. A fifth round (seed 400) and its chain are running; if the SFT gain drops below about 0.1 for 45k more examples, further rounds should give way to judge-filtering and to the hybrid output. Caveats unchanged: one seed per point, teacher as judge, per-language groups of 23 to 220.

## 2026-10-02 19:28 UTC — Vocabulary-size probe (first step of the deferred byte-level question)

Commit: c5c8ed6.

**Question/Context.** The embedding table is 26% of the 20M model. A byte-level architecture would remove it; the cheap proxy is to vary the BPE size and see whether freeing or spending embedding parameters changes language modelling quality at equal training budget.

**What was done.** Three byte-level BPE tokenizers (6,000, 12,000 and 24,000 merges) trained the same way on 25M characters per language of Wikipedia plus the task text; the Wikipedia corpus (1.62B tokens at 12k) tokenized with each; one 7-layer d=448 model per vocabulary pretrained for 100M tokens (762 steps, lr 2e-3, Qwen-initialized embeddings, one seed) on GPU 1 while it also served a teacher instance. Quality compared as bits per UTF-8 byte of the same held-out Wikipedia text (per-token perplexity is not comparable across vocabularies). The bits-per-byte formula in `pretrain.py` undercounted by a factor of 64 (bytes of 64 sequences against the loss of one) in these runs; the values below are corrected by that factor and the code is fixed in c5c8ed6.

**Result.** Bits per byte (parameters): 6k 1.484 (17.9M), 12k 1.442 (20.6M), 24k 1.412 (26.0M), mean over ten languages. By language, 6k / 12k / 24k: English 1.562 / 1.523 / 1.498, French 1.530 / 1.510 / 1.485, Spanish 1.523 / 1.453 / 1.485, Arabic 1.222 / 1.210 / 1.171, Chinese 2.208 / 1.939 / 1.926, Russian 1.082 / 0.992 / 0.992, Indonesian 1.645 / 1.869 / 1.677, Hindi 0.723 / 0.749 / 0.742, Turkish 1.645 / 1.549 / 1.606, Swahili 1.702 / 1.626 / 1.536. The 12k model beats 6k on 8 of 10 languages; 24k beats 12k on 7 of 10 but costs 5.4M more parameters (26 MB at INT8, over budget).

**Implication.** Vocabulary size is a weak lever: 3% between 6k and 12k and 2% between 12k and 24k in bits per byte, with mixed per-language signs and one seed each (Indonesian and Hindi move against the trend). Freeing embedding parameters for the layers (6k) does not improve the model, which removes the main structural argument for a byte-level architecture at this size. The 12k choice stands. The deferred byte-level experiment (Byte Latent Transformer or ByteFlow) would have to be justified by something other than the embedding budget, such as robustness or the patching mechanism itself; this probe lowers its priority. Caveats: 100M tokens is a short budget, the Qwen-initialized embeddings favour vocabularies whose tokens re-tokenize well under Qwen, and the task fine-tune was not run for 6k or 24k.

## 2026-10-02 21:59 UTC — ByteFlow Net implemented; pretraining against the BPE model at matched text

Commit: 7f6a1df (`tiny/byteflow.py`, `tiny/bytecorpus.py`, `tiny/bytepretrain.py`, `tiny/bytetrain.py`, tests).

**Question/Context.** The user asked for ByteFlow Net (Deng et al., ICLR 2026, arXiv 2603.03583) to be implemented and tested rather than argued about. The implementation follows section 3 of the paper: byte embedding (258 symbols), local encoder of pre-norm blocks with sliding-window attention and Canon layers, the exact lossy coding rate R = 1/2 log det(I + (d/eps^2) h h^T) computed incrementally by the matrix determinant lemma and Sherman-Morrison (epsilon^2 = d on unit-RMS rows), Top-K selection of marginal rates with position 0 fixed, a 7-layer global transformer, the upsampler with bin-shared projections and a residual, a symmetric local decoder, and next-byte softmax. Two choices the paper leaves open: upsampler bins index the position within the chunk (the paper bins absolute position by sequence length, undefined while generating), and boundaries after the prompt during generation use a threshold equal to the smallest marginal rate in the prompt's own Top-K selection (the paper evaluates teacher-forced only). Tests: parameter count, selection constraints, the incremental coding rate against a direct log-det computation, decoder causality with fixed boundaries, and agreement between the cached generation path and teacher forcing.

**What was done.** Configuration at 18.6M parameters: d_local 192, 2 encoder and 2 decoder blocks (3 heads, feed-forward 512, window 128), d_global 448 with 7 global layers (the same shape as the BPE model's transformer, 15.3M parameters), compression ratio 3 bytes per global position, 16 upsampler bins, exact log-det rate. Pretrained on the Wikipedia byte shards for 290M bytes (the number of bytes the 12k BPE model saw in its 100M-token run), 2,950 steps of 32 x 3072 bytes, lr 2e-3, one seed, 96 minutes on a shared GPU (50k bytes per second; the sequential rate update dominates). Held-out bits per byte on exactly the text used for the BPE models' bits-per-byte figures.

**Result.** Mean held-out bits per byte 1.695 (ByteFlow) against 1.442 (12k BPE, same text budget, same global transformer shape), 1.484 (6k BPE) and 1.412 (24k BPE). By language, ByteFlow / 12k BPE: English 1.95 / 1.52, French 1.86 / 1.51, Spanish 1.77 / 1.45, Arabic 1.48 / 1.21, Chinese 2.61 / 1.94, Russian 1.28 / 0.99, Indonesian 2.27 / 1.87, Hindi 0.90 / 0.75, Turkish 1.92 / 1.55, Swahili 2.00 / 1.63. ByteFlow's curve was still falling at the end (1.724 at 2,500 steps, 1.695 at 2,950), so it is not converged at this budget. Training throughput 50k bytes per second against about 1.4M bytes per second for the BPE model.

**Implication.** At 20M parameters and 290M bytes, ByteFlow Net trails the BPE model by 18% in bits per byte on every language, consistent with the paper's own finding that byte-level models need scale and data to catch up (its 600M models overtake BPE only after about 25B bytes). The implementation is not tuned (one seed, one learning rate, the paper's 600M hyperparameters were not re-fitted), and the sequential coding-rate loop makes it about 25 times slower to train here. The task fine-tune and generation test on the standard 800 examples are running; that result decides whether the byte model's robustness or segmentation advantages show up on the task despite the worse language modelling.

## 2026-10-02 22:10 UTC — Fifth data round: 131k examples

Commit: 48abca1.

**Question/Context.** Continue the data learning curve past 100k examples.

**What was done.** Round 5 (`data/v5`, seed 400, flat language mix): 11,364 queries, 44,844 pairs, 44,838 narrated, the second half across two teacher servers (GPU 0 at 45% memory and GPU 1 at 50%) at a combined 4.2 narrations per second. Combined set `data/f4` = v1 to v5: train 130,813 (11,214 relevant, 33,490 partly relevant, 86,109 irrelevant), val 19,987, test 15,159. The unattended chain fine-tuned from `pre-wide-full2` (standard recipe), sampled 8,000 x 4 narratives, scored them, built 5,091 pairs (3,287 chosen from the student) and ran DPO with the usual settings. Both models evaluated on the third-round test set (7,056) and judged on the same 800 examples as the rest of the curve.

**Result.**
- Verdict on the 7,056-example test (INT8): 131k SFT 84.7% / macro-F1 0.794; 131k + DPO 84.5% / 0.791 (100k + DPO: 84.5% / 0.791). On its own 15,159-example test: SFT 84.6% (INT4 83.9%), DPO 84.5% (INT4 83.5%).
- Judge, same 800 examples: SFT faithfulness 4.18 (74.3% at 4 or more), coherence 4.30; with DPO 4.33 (80.1%), coherence 4.41, consistent 99.8%, correct language 99.4%. SFT curve: 25k 3.24, 30k 3.57, 60k 3.85, 100k 4.04, 131k 4.18. With DPO: 60k 3.99, 100k 4.24, 131k 4.33.
- By language, 100k+DPO -> 131k+DPO (n): English 4.65 -> 4.71 (220), Spanish 4.39 -> 4.49 (88), Portuguese 4.36 -> 4.40 (42), Chinese 4.15 -> 4.25 (72), Russian 4.03 -> 4.21 (77), French 4.27 -> 4.16 (74), Indonesian 3.73 -> 4.08 (48), Hindi 3.71 -> 4.06 (48), Turkish 3.98 -> 4.05 (42), Swahili 4.04 -> 4.04 (23), Arabic 3.92 -> 3.89 (66).
- Export: `runs/f4-dpo/explainer-int8.pt`, 21.59 MB.

**Implication.** The SFT gain per round is now +0.14 (from +0.19 and +0.28 before) and the verdict is flat at 84.5 to 84.7%, so the data curve is flattening but has not stopped; every language is now at or above 3.9 on the judge, and 80% of narratives score 4 or more after DPO. A sixth round (seed 500) and its chain are running; if its SFT gain is below 0.1, further rounds stop. Caveats unchanged: one seed per point, teacher as judge, per-language groups of 23 to 220, French and Arabic moved down within noise.

## 2026-10-03 02:57 UTC — ByteFlow Net on the task: teacher forcing works, generation does not

Commit: 3a90796.

**Question/Context.** After pretraining (1.695 against 1.442 bits per byte for the BPE model), does ByteFlow Net do the task?

**What was done.** Fine-tuned the pretrained ByteFlow Net on the 60k set (`data/f2`) as bytes: BOS + input + separator, target = output + EOS, verdict bytes weighted 5, lr 5e-4, batch 16, 3 epochs (38 minutes each; a first 4-epoch attempt was killed at the 2-hour background limit before saving). Validation loss per byte 0.559, 0.499, 0.477 nats. Generation for the same seeded 800 test examples as the judge uses, about 1 second per example, with three boundary rules after the prompt: (a) the prompt's K-th rate as threshold; (b) a forced boundary every 3 bytes (the training density); (c) an adaptive rule (boundary when the byte's rate is in the top third of the last 96 rates, capped at 6 bytes per chunk). Checked with a traced run that the cached generation path reproduces teacher-forced logits exactly for the same boundaries (max difference 0.000 on six real prompts).

**Result.**
- Verdict accuracy 81.0% on the 800 (BPE model with the same data: 83.6%; majority 63.5%), macro-F1 0.743, no invalid verdict lines.
- Rule (a) created zero boundaries in 200 generated bytes on all six probed prompts: marginal coding rates fall as the context grows, so the prompt's threshold is never reached again. Narratives were fluent but described a different item from the input (judge faithfulness 1.04 of 5, 0.3% at 4 or more, coherence 2.60; 11% contained invalid UTF-8).
- Rule (b) restored conditioning on the input in the probe (item names and years from the prompt appear) but produced invalid UTF-8 in 74% of outputs; judge faithfulness 1.00, coherence 1.23.
- Rule (c): invalid UTF-8 in 41%, faithfulness 1.01, coherence 1.83.
- For reference, the teacher-forced task cross-entropy is about twice the BPE model's (0.477 nats per byte against about 0.36 nats per byte for the BPE student on the same narratives, converting its 0.466 bits per character at roughly 1.3 bytes per character).

**Implication.** The paper's chunking is non-causal at training time (Top-K over the full sequence, which it acknowledges) and it never generates text; the decoder learns to rely on a chunking pattern that no causal streaming rule reproduces, so generation collapses whichever rule is used. Making ByteFlow generate would require training with a causal boundary rule, which changes the method. Combined with the pretraining gap (18% more bits per byte) and the 25-times slower training from the sequential coding-rate update, ByteFlow Net is not a candidate for this task at 20M parameters. The experiment is closed with the implementation and tests kept (`tiny/byteflow.py`). Caveats: one seed, no hyperparameter search for the byte model, 290M bytes of pretraining, and the paper's own gains appear at 600M parameters and tens of billions of bytes.

## 2026-10-03 03:34 UTC — Sixth data round: 166k examples

Commit: d455f06.

**What was done.** Round 6 (`data/v6`, seed 500): 11,428 queries, 43,941 pairs, 43,933 narrated on two teacher servers. Combined set `data/f5` = v1 to v6: train 165,550 (14,353 relevant, 44,157 partly relevant, 107,040 irrelevant), val 25,232, test 19,003. Unattended chain as before; 4,499 preference pairs (3,136 chosen from the student). Evaluated on the third-round test set (7,056) and judged on the standard 800.

**Result.**
- Verdict on the 7,056-example test (INT8): 166k SFT 85.6% / macro-F1 0.813; 166k + DPO 85.5% / 0.814 (131k + DPO: 84.5% / 0.791). On its own 19,003-example test: SFT 85.1% (INT4 84.9%), DPO 85.0% (INT4 84.7%).
- Judge, same 800: SFT faithfulness 4.30 (77.9% at 4 or more), coherence 4.44; with DPO 4.44 (81.8%), coherence 4.51, consistent 99.6%, correct language 99.8%. SFT curve: 25k 3.24, 30k 3.57, 60k 3.85, 100k 4.04, 131k 4.18, 166k 4.30. With DPO: 3.99, 4.24, 4.33, 4.44.
- By language, 131k+DPO -> 166k+DPO (n): English 4.71 -> 4.73 (220), Spanish 4.49 -> 4.58 (88), Chinese 4.25 -> 4.51 (72), Hindi 4.06 -> 4.42 (48), Russian 4.21 -> 4.29 (77), French 4.16 -> 4.26 (74), Arabic 3.89 -> 4.20 (66), Turkish 4.05 -> 4.21 (42), Swahili 4.04 -> 4.09 (23), Indonesian 4.08 -> 4.08 (48), Portuguese 4.40 -> 4.29 (42).
- Export: `runs/f5-dpo/explainer-int8.pt`, 21.59 MB.

**Implication.** The SFT faithfulness gain per round is +0.12 (after +0.14), and this round also moved the verdict by a full point (84.5% to 85.5%, macro-F1 0.791 to 0.814), the first verdict gain since 60k. Every language is now above 4.0 after DPO except Swahili at 4.09 and Indonesian at 4.08; the lowest is 4.08. A seventh round (seed 600) and its chain are running and will be the last unless its SFT gain stays above 0.1. Caveats unchanged: one seed per point, the teacher is the judge, per-language groups are 23 to 220.

## 2026-10-03 07:31 UTC — GRPO on the narratives, a second judge (Gemma 4), and a pairwise check that catches shorter outputs

Commit: fd3bfef. The first GRPO run used the same code without the informativeness and supervised terms (both default to off). Teacher judge: Qwen3.8-27B-FP8. Second judge: Gemma 4 31B instruct (RedHatAI FP8-dynamic build), served with vLLM.

**Question/Context.** Would GRPO improve the narratives beyond the current DPO model (`runs/f5-dpo`: verdict 85.5%, teacher-judged faithfulness 4.44)? GRPO optimizes the teacher judge's score directly, and the same judge grades the test set, so the result needs a judge from another model family.

**What was done.**
1. Reward check before training: on 32,000 scored DPO samples, narratives with no unsupported number or name (83%) average judge faithfulness 4.10 and those with one 3.95. The earlier error-type entry (2026-10-01 22:48) puts numbers and names at about 5% of faithfulness errors. The reward is therefore the teacher judge's score: 2 x faithfulness + 0.5 x coherence (both rescaled to 0..1), minus 1 if inconsistent with the verdict or in the wrong language, minus 0.25 per unsupported item (up to 4), and -2 for empty or truncated output.
2. GRPO (`tiny/grpo.py`, `runs/f5-grpo`): from `runs/f5-dpo`, 4,000 training inputs of `data/f5`, 16 inputs and 8 samples per step (temperature 0.9, top-40), teacher verdict forced as prefix, rewards standardized within each group, clipped surrogate (0.2) with 2 updates per batch, KL to the start model 0.04, lr 1e-5, 250 steps, 32,000 judge calls.
3. Control at the same judge budget: a second on-policy DPO round from `runs/f5-dpo` (8,000 new inputs x 4 samples, 4,154 pairs, same settings as the first round), `runs/f5-dpo2`.
4. Evaluation: verdict accuracy on the third-round test set (7,056); teacher and Gemma 4 judges on the standard seeded 800, with per-example scores and paired bootstrap intervals (`tiny/compare.py`); Gemma 4 pairwise preference for usefulness, order randomized (`tiny/pairwise.py`).

**Result.**
- Judge agreement on `f5-dpo`'s 800: Pearson 0.848 on faithfulness, identical score 81.2%. Narratives at 3 or lower: 114 flagged by both, 36 by the teacher only, 21 by Gemma 4 only. Gemma 4 scores `f5-dpo` 4.53 (teacher 4.43; the earlier 4.44 reproduces) and the teacher's own narratives 4.99.
- Second DPO round: verdict 85.6% (macro-F1 0.814). Faithfulness +0.044 [-0.019, +0.113] under the teacher and +0.040 [-0.019, +0.100] under Gemma 4. Not distinguishable from zero.
- GRPO: verdict 84.9% (macro-F1 0.808), 0.6 points below `f5-dpo`. Faithfulness 4.43 to 4.62 under the teacher (+0.185 [+0.116, +0.259]) and 4.53 to 4.65 under Gemma 4 (+0.115 [+0.055, +0.176]); share at 4 or more 88.1% / 88.0%. Against the second DPO round, Gemma 4 faithfulness +0.075 [+0.013, +0.138].
- GRPO narratives are shorter: median 31 words against 36 (mean 29.3 against 33.7), median 201 characters against 232. Share with an unsupported number or name 11.9% against 14.5%.
- Pairwise usefulness (Gemma 4, 800 examples): `f5-dpo` preferred over GRPO 398 times, GRPO 284, tie 83 (35 identical). Controls: `f5-dpo` against the second DPO round 289 / 303 / 121 (87 identical); `f5-dpo` against the teacher's narratives 82 / 658 / 60.

**Implication.** GRPO with a faithfulness reward raises faithfulness under both judges, but part of the gain comes from saying less: narratives are 14% shorter and a second judge finds them less useful. The gap between the two judges (+0.185 against +0.115) also suggests some fitting to the teacher judge's preferences. The verdict drop of 0.6 points exceeds the 0.1-point seed variation. A second GRPO run adds an informativeness score to the reward (weight 1.0) and a supervised term on the teacher's output (weight 0.2) to address both. A faithfulness score alone should not be used to select between narrative models; the pairwise usefulness check is now part of the evaluation.

## 2026-10-03 09:28 UTC — GRPO with an informativeness reward and a supervised anchor: verdict held, preferred for usefulness

Commit: fd3bfef (`tiny/grpo.py` with `--inf-weight 1.0 --sft 0.2`). Judges as in the previous entry.

**Question/Context.** The first GRPO run (previous entry) raised faithfulness by writing shorter narratives, lost the pairwise usefulness comparison (284 against 398), and lowered the verdict by 0.6 points. Does an informativeness term in the reward and a supervised term on the teacher's output fix both?

**What was done.** Same settings as the first GRPO run (from `runs/f5-dpo`, 4,000 inputs, 16 x 8 per step, 250 steps, KL 0.04, lr 1e-5, 32,000 judge calls), with two changes: the teacher judge also returns an informativeness score (1-5: what the item contains and how it relates to the query and each filter), added to the reward with weight 1.0 after rescaling to 0..1; and each update adds 0.2 x the mean token NLL of the teacher's full output (verdict and narrative) for the step's 16 inputs. Output `runs/f5-grpo2`. Evaluated like the other arms.

**Result.**
- Verdict accuracy 85.5% (macro-F1 0.811), equal to `f5-dpo` (85.5%, 0.814).
- Faithfulness, paired against `f5-dpo` on the same 800: teacher 4.43 to 4.50 (+0.066 [+0.001, +0.134]); Gemma 4 4.53 to 4.56 (+0.030 [-0.035, +0.092]). Coherence: teacher +0.071 [+0.029, +0.117]; Gemma 4 +0.020 [-0.011, +0.054].
- Length unchanged: median 35 words (mean 33.6) against 36 (33.7). Share with an unsupported number or name 13.8% against 14.5%.
- Pairwise usefulness (Gemma 4, random order): `f5-grpo2` preferred 316 times, `f5-dpo` 248, tie 143, identical 92, 1 failure. Sign test on the 564 decided pairs: 56.0%, z = 2.86 (two-sided p about 0.004).
- Training reward (10-step means) rose from 2.25 at step 1 to 2.58 at step 50 and then stayed between 2.44 and 2.58 to step 250; KL to the start model reached 0.012 per token.

**Implication.** With the two terms, GRPO keeps the verdict and the length, and a judge from another model family prefers its explanations more often than the current model's. The faithfulness gain under that judge is small and not distinguishable from zero. The first run's larger faithfulness gain came with shorter, less useful text. The reward stopped rising after about 50 steps, so more steps of this setting are unlikely to help; a larger KL budget or a stronger informativeness weight are the untested levers. The recipe will be applied to the round-7 model when its chain finishes, and judged against that model with the same two-judge and pairwise evaluation.

## 2026-10-03 10:40 UTC — Round 7 (201k examples): the data stopping rule is reached; a script constraint removes stray characters

Commit: 885dfcf (round-7 models trained by `scripts/round_when_ready.sh` at 7d1781a; the constraint is new in 885dfcf and does not change decoding when off). Judges: teacher (Qwen3.8-27B-FP8) and Gemma 4 31B, on the standard seeded 800 of the third-round test set (7,056).

**Question/Context.** Does a seventh round of task data (seed 600) still add more than 0.1 faithfulness to the fine-tuned model, the stopping rule set in the round-6 entry? Separately, the judges flagged 7 to 8 of 800 round-7 narratives as wrong-language. What are these?

**What was done.**
1. Round 7: 45,032 new pairs narrated; combined set `data/f6` with 201,401 training, 30,331 validation and 22,984 test examples. Same recipe as round 6 (SFT 8 epochs from `runs/pre-wide-full2`, on-policy DPO with 4,387 pairs). Both models evaluated as in the previous entries.
2. The flagged narratives were read. A script check counted narratives that contain letters of Cyrillic, Arabic, Devanagari or CJK script when neither the requested language nor the input uses that script.
3. A decoding constraint (`tiny/scriptguard.py`): every token is classed by the UTF-8 lead bytes it contains (D0-D3 Cyrillic, D8-DB Arabic, E0 Indic, E4-E9 CJK). At each step, tokens of a class that is neither the requested language's script nor present in the input get probability zero. `runs/f6-dpo` was evaluated again with it on (tag `f2testsg`).

**Result.**
- Round 7 SFT (`f6-sft`): verdict 85.9% (macro-F1 0.811); teacher faithfulness 4.36, against 4.30 for the round-6 SFT model: +0.06, below the 0.1 rule. Against `f5-dpo` (paired): teacher -0.069 [-0.152, +0.013], Gemma 4 -0.062 [-0.139, +0.011].
- Round 7 SFT + DPO (`f6-dpo`): verdict 86.1% (macro-F1 0.815). Against `f5-dpo`: teacher faithfulness 4.43 to 4.54 (+0.107 [+0.033, +0.186]); Gemma 4 4.53 to 4.62 (+0.094 [+0.024, +0.164]); share at 4 or more 85.4% (teacher) and 87.3% (Gemma 4). Pairwise usefulness (Gemma 4): `f6-dpo` 381, `f5-dpo` 323, tie 92, identical 4 (54.1% of the 704 decided pairs, z = 2.19, two-sided p about 0.03).
- The wrong-language flags are stray fragments of another script inside otherwise correct text, for example "luعتبر" in a Spanish sentence or "сдерживанииادолнительной" in Russian. Share of the 7,056 test narratives with such fragments: teacher 0.04%; student models 2.3% to 4.5% (`f4-dpo` 4.46%, `f5-sft` 3.44%, `f5-dpo` 3.59%, `f5-dpo2` 3.78%, `f5-grpo` 2.30%, `f5-grpo2` 3.40%, `f6-sft` 3.13%, `f6-dpo` 3.97%). The judges flag only a few of them as wrong-language (1 to 2 of 800 for `f5-dpo`, 7 to 8 for `f6-dpo`).
- With the constraint (`f6-dpo`): fragments 0 of 7,056; verdict 86.1% (unchanged, macro-F1 0.815); teacher faithfulness 4.55 (4.54 without), coherence 4.56 (4.55), correct language 100% (99.0%); Gemma 4 faithfulness 4.63 (4.62), coherence 4.87 (4.84), correct language 100% (99.1%). The blocked position is filled by a token of the right script, which is sometimes still a non-word ("rementara" in an Indonesian sentence).

**Implication.** Round 7 is the last data round: the fine-tuned model's faithfulness gain is +0.06, and the gains per round were +0.28, +0.19, +0.14, +0.12, +0.06. Its DPO model is the new best (verdict 86.1%, preferred for usefulness over `f5-dpo`), although its fine-tuned model alone is not better than `f5-dpo`, so much of the gain is from the DPO step on a better base. The stray-script fragments are a constant property of the 20M student at about 3 to 4% of narratives, and the judges undercount them. The constraint removes them at no cost to the verdict or faithfulness and is now the default in `export.explain`. GRPO with the informativeness reward is running from `f6-dpo`.

## 2026-10-03 12:35 UTC — Remaining errors, consensus decoding, the verdict ceiling, and GRPO on the round-7 model

Commit: 531bc97 (`tiny/mbr.py`, per-example thinking labels in `distill/teacher_checks.py`; the GRPO run used `tiny/grpo.py` from fd3bfef with `--inf-weight 1.0 --sft 0.2`). Judges: teacher (Qwen3.8-27B-FP8) and Gemma 4 31B; standard seeded 800 of the third-round test set; all generations INT8 with the script constraint unless stated.

**Question/Context.** What errors remain in the best model (`runs/f6-dpo`), and do three cheap levers help: consensus decoding, the verdict labels, and GRPO on the better base?

**What was done.**
1. Error types: the error-type judge (`tiny/judge_issues.py`, one primary issue per narrative) run with Gemma 4 on `f6-dpo`'s 800. Verbatim title use measured on 6,000 training narratives (does the narrative contain 3 consecutive words of the item's "Result:" title).
2. Consensus (minimum Bayes risk) decoding: greedy verdict, then 8 sampled narratives (temperature 0.7, top-40) with that verdict as prefix, plus the greedy narrative; keep the candidate with the highest mean chrF against the others.
3. The 800 judged test examples relabelled with the teacher's thinking mode; the model's verdicts compared with both labels.
4. GRPO from `f6-dpo` on `data/f6` (4,000 inputs, 250 steps, informativeness weight 1.0, supervised weight 0.2), evaluated without the script constraint against `f6-dpo` without it.

**Result.**
- Error types (Gemma 4): no problem 77.0% of 800; item content 13.5%, filter status 3.0%, query relation 2.9%, numbers or names 2.1%, garbled 1.5%. Among the 169 narratives at faithfulness 3 or lower: item content 59.2%, filter status 14.2%, query relation 13.6%, numbers or names 8.3%, garbled 4.7%. Item-content errors in 5.0% of English narratives and 16.7% of the others.
- Verbatim title spans in teacher narratives: English 55%, Spanish 16%, Portuguese 17%, French 13%, Russian 13%, Indonesian 11%, Swahili 10%, Turkish 7%, Hindi 6%, Arabic 1%, Chinese 1%.
- Verdict confusion of `f6-dpo` (7,056): partly relevant called irrelevant 474, irrelevant called partly relevant 248, relevant called partly relevant 83, partly relevant called relevant 142; other errors 36.
- Consensus decoding replaced the greedy narrative in 473 of 800 examples. Against greedy: teacher faithfulness 4.55 to 4.49 (-0.058 [-0.110, -0.005]), coherence -0.052 [-0.096, -0.011]; Gemma 4 faithfulness 4.63 to 4.60 (-0.030 [-0.077, +0.019]), coherence 4.87 to 4.81 (-0.058 [-0.089, -0.028]). Gemma 4 pairwise usefulness: consensus 234, greedy 169, tie 70, identical 327. Mean length 36.0 against 33.4 words.
- Thinking labels (795 parsed of 800): agree with the stored labels on 87.5%; the changes are mostly irrelevant to partly relevant (64) and partly relevant to irrelevant (20). `f6-dpo` agrees with the stored labels on 85.5% and with the thinking labels on 79.5%; on the 696 examples where both labels agree it is right on 87.2%. `f5-dpo`: 86.2%, 81.4% and 88.6%.
- GRPO on `f6-dpo`: verdict 86.4% (macro-F1 0.823) against 86.1% (0.815). Faithfulness: teacher -0.005 [-0.066, +0.055], Gemma 4 -0.026 [-0.085, +0.031]; coherence +0.041 [-0.003, +0.084] and +0.020 [-0.014, +0.054]. Pairwise usefulness: GRPO 303, `f6-dpo` 274, tie 130, identical 93 (z = 1.2).

**Implication.** Item-content errors in non-English narratives are the main remaining problem, and they coincide with the teacher translating item titles and technical names, which the student must then translate itself. Two experiments target this: re-narrating with the title quoted verbatim (data) and a pointer head that can copy input tokens (architecture); both are running. Consensus decoding is not adopted: the usefulness preference comes with longer text and lower judged faithfulness and coherence, at nine times the decoding compute. The verdict is not at the label-noise ceiling: the model misses 12.8% of the examples where the teacher's two modes agree. GRPO adds nothing measurable on the round-7 base, so the first GRPO gain over `f5-dpo` (previous entry) was within what another round of data and DPO provides. `f6-dpo` with the script constraint stays the best model.

## 2026-10-03 16:27 UTC — Pointer (copy) head and verbatim-title narratives

Commit: 531bc97 for both (`Config.copy`, `train.py --copy`, `distill/verbatim.py`). Judges: teacher (Qwen3.8-27B-FP8) and Gemma 4 31B, standard seeded 800 of the third-round test set, INT8 without the script constraint (so that runs compare with their earlier counterparts).

**Question/Context.** The previous entry located the main remaining error in non-English item descriptions and found that the teacher rarely keeps item titles verbatim outside English. Two hypotheses: a learned copy mechanism lets the student reproduce input terms more reliably (architecture), and narratives that quote the exact title are easier for the student to get right (data).

**What was done.**
1. Copy head: a gate g = sigmoid(w·h) mixes the tied-vocabulary distribution with a single-head attention distribution over all earlier tokens (query and key projections of dimension 64, scattered onto those tokens' ids); p = g·p_vocab + (1-g)·p_copy. 50,113 extra parameters; gate bias initialized to 2.2 (generation with probability 0.9). Fine-tuned from `runs/pre-wide-full2` on `data/f6` with the round recipe (`runs/f6-copy-sft`), then on-policy DPO with the round settings on 8,000 new samples (4,319 pairs, `runs/f6-copy-dpo`). Tests check that the mixture sums to one, can only point backwards, and that cached batched decoding equals full recomputation.
2. Verbatim titles: a 50,000-example sample of `data/f6` training data; the 38,926 non-English examples with a title were re-narrated by the teacher for the stored verdict with the exact "Result:" title in double quotes (34,693 passed validation; the other 4,233 keep the original narrative). Two models fine-tuned from the same base with the same recipe on the same inputs: original narratives (`runs/old50k-sft`) and re-narrated (`runs/vb50k-sft`). One seed each.

**Result.**
- Copy head, fine-tuned only, against `f6-sft`: validation loss 0.670 against 0.693; test verdict 86.2% (macro-F1 not compared) against 85.9%; faithfulness teacher 4.36 to 4.44 (+0.083 [+0.001, +0.163]), Gemma 4 4.47 to 4.55 (+0.086 [+0.010, +0.160]); coherence teacher +0.087 [+0.029, +0.147], Gemma 4 +0.050 [+0.007, +0.095]. Pairwise usefulness 360 to 353. Gemma 4 faithfulness by language (n): Arabic 4.05 to 4.45 (66), Hindi 4.21 to 4.56 (48), English 4.62 to 4.78 (220), Chinese 4.53 to 4.67 (72), Spanish 4.59 to 4.65 (88), Swahili 4.52 to 4.65 (23), Indonesian 4.17 to 4.17 (48), Russian 4.44 to 4.39 (77), French 4.51 to 4.41 (74), Portuguese 4.60 to 4.50 (42), Turkish 4.40 to 4.12 (42). INT8 file 20.79 MB (weights) against 20.73; GPU test decoding 213 s against 101 s for 7,056 examples.
- Copy head after DPO, against `f6-dpo`: verdict 86.0% (macro-F1 0.813) against 86.1% (0.815); faithfulness teacher +0.028 [-0.043, +0.099], Gemma 4 +0.006 [-0.056, +0.069]; coherence teacher +0.054 [-0.003, +0.110], Gemma 4 +0.048 [+0.009, +0.090]; pairwise usefulness 367 to 344. The DPO training curve (preference accuracy 0.72 then 0.99, margin 9.0) matches that of `f6-dpo` (0.71 then 0.99, margin 9.2).
- Verbatim titles, against the same inputs with original narratives: verdict 83.3% (macro-F1 0.764) against 83.9% (0.786); faithfulness teacher 3.92 to 3.71 (-0.205 [-0.304, -0.109]), Gemma 4 4.06 to 3.95 (-0.110 [-0.201, -0.021]); coherence Gemma 4 -0.087 [-0.154, -0.020]; pairwise usefulness 342 against 418. Gemma 4 faithfulness, English 4.38 to 4.24 (220) and other languages 3.94 to 3.84 (580); English training narratives were identical in both sets. The student quotes the exact title in 4,594 of 5,154 non-English test narratives; its errors remain in what the item measures, with code-switched words.

**Implication.** The copy head improves the fine-tuned model by about as much as the last data round (+0.09 under Gemma 4) for 0.06 MB, but the DPO step reaches the same faithfulness without it; after DPO only a small coherence gain remains, at about twice the GPU decoding time. It is not adopted for now; it may be worth revisiting if DPO is ever dropped from the pipeline. Quoting titles verbatim does not help and lowers all scores, including English, where the training text did not change; with one seed each the English drop indicates that differences of about 0.1 at 50k examples are within run-to-run variation, yet the non-English drop is no smaller. Translating titles is therefore not the cause of the item-content errors: the student copies titles correctly and still misdescribes what the item measures. `f6-dpo` with the script constraint stays the best model.
