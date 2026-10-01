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
