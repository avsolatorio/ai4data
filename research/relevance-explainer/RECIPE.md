# Training recipe

This file describes how the deployed relevance explainer is built: `runs/v2g-from-dapt-8ep-dpo/explainer-int8.pt`,
served by the web app as "Realistic, no retrieval score". The results behind each choice are in
`notes/lab_notebook.md` (entries of 2026-10-09 15:24, 2026-10-09 23:27 and 2026-10-10 20:43 UTC).

The model is built in four steps: a tokenizer, pretraining, one supervised fine-tune and one DPO step. Steps 3 and 4
run from one command:

```
scripts/single_stage.sh 8 runs/pre-dapt/model.pt v2g-from-dapt-8ep
```

It replaces the earlier eight-stage chain (`runs/v2g-nosim-dpo`), which it matches within noise on the test sets.

## Model

- Decoder-only transformer, 20.64M parameters: 7 layers, width 448, 7 heads, SwiGLU feed-forward of 1,024, RoPE,
  RMSNorm, tied input and output embeddings.
- Byte-level BPE vocabulary of 12,000; context of 1,536 tokens.
- Shipped as INT8 with a per-row scale (21.6 MB file), made by `relevance_explainer.tiny.export`.

## Input and output

The input lists the record's metadata first, then the search and how it was understood:

```
Item type / Result / Producer / Coverage / ...   (record metadata)
Respond in: <language>
Search: <query>
Facets: <interface filters, or none>
Understood: <place, time, group, threshold and exclusion constraints parsed from the search>
Checks: <each constraint's status against the record>
```

The Understood and Checks lines are rule-based (`synth/understand.py`, render version 4). There is no retrieval score.
The input is followed by `\n\n### Judgment\n`. The output is the verdict on one line (`relevant`, `partly_relevant` or
`irrelevant`), then a short explanation in the search's language.

## 1. Tokenizer

`relevance_explainer.tiny.corpus tokenizer`: byte-level BPE of 12,000 tokens, trained on 25 million characters per
language plus 10 repeats of the first-round task text (`data/corpus/tokenizer.json`).

## 2. Pretraining

| Run | Starts from | Data | Tokens | Learning rate |
|---|---|---|---|---|
| `pre-wide-full` | random, embeddings from Qwen3-0.6B | Wikipedia, 11 languages | 1.6B | 2e-3 |
| `pre-wide-full2` | `pre-wide-full` | Wikipedia, 11 languages | 1.6B | 1e-3 |
| `pre-dapt` | `pre-wide-full2` | Wikipedia and FineWeb-2, with 30% of sequences from the domain corpus | 600M | 3e-4 |

- **Wikipedia** (20231101 dump), 1.62 billion tokens: English 411M, Chinese 271M, French 160M, Spanish 152M, Russian
  149M, Arabic 129M, Hindi 103M, Portuguese 87M, Turkish 74M, Indonesian 66M, Swahili 20M. Languages are sampled in
  proportion to the square root of their size.
- **Embedding initialisation:** each token's embedding is the mean of the Qwen3-0.6B embeddings of its string, reduced
  to 448 dimensions by PCA (`--init-emb Qwen/Qwen3-0.6B`).
- **Shared settings:** sequences of 1,024 tokens, batch of 128, AdamW (betas 0.9/0.95, weight decay 0.1), warmup then
  cosine decay to 10%.
- **Domain corpus** (`synth/domain_corpus.py`, `data/corpus_domain`, 159M tokens): 910,452 documents of record metadata,
  search and explanation without the verdict line (including rows the labellers disagreed on), 78,978 searches and 8,870
  records. Test searches, validation and test themes, and held-out records are excluded.
- **Command for `pre-dapt`:** see step 2 of `scripts/dapt.sh`.

The domain corpus did not help: the same 600M tokens of general text only (`pre-cont`) gave the same validation
accuracy (83.5%) after fine-tuning. Either base can be used. `pre-dapt` is the one the deployed model was trained from.

## 3. Supervised fine-tune

| Setting | Value |
|---|---|
| Data | `data/v2_g_itemfirst_nosim` (round G), 977,538 training rows |
| Starts from | `runs/pre-dapt/model.pt` |
| Epochs | 8; the epoch kept is the one with the best validation verdict accuracy (here the last, 85.5%) |
| Learning rate | 1e-3, warmup then cosine decay to 5% |
| Batch | 32 |
| Dropout | 0.1 |
| Optimiser | AdamW, weight decay 0.1, gradient clipping at 1.0 |
| Verdict tokens | weighted 5x in the loss |
| Time | about 2.8 hours on one H100 NVL, shared with a vLLM server |

Validation verdict accuracy by epoch: 76.2, 78.2, 79.4, 79.8, 82.5, 83.8, 85.0, 85.5%. It was still rising at epoch 8,
so a longer run may help.

### Round G data

| Part | Rows | Contents |
|---|---|---|
| real | 490,182 | Realistic searches (from themes, or anchored to a record) with retrieved records |
| old | 200,402 | First-round data (rounds v1 to v7) re-rendered in the current format, with their original labels |
| translated | 94,077 | Translated searches |
| types | 79,766 | Tables, map layers and scripts, counted twice |
| para | 41,530 | Searches reworded with synonyms and everyday wording, counted twice |
| concepts | 20,216 | Concept inventory, counted twice |
| metatrans | 18,982 | Metadata in French and Spanish |
| phen | 16,438 | Phenomenon slices, counted twice |
| phen2 | 6,702 | Extra exclusion, look-alike, proxy, group and period searches, counted twice |
| hardneg | 5,223 | Catalog hard negatives (`synth/hardneg.py`) |
| estat | 4,020 | Eurostat table searches (training half of the tables), counted twice |

- **Teacher:** Qwen3.8-27B (FP8 on vLLM, thinking off) writes every verdict and explanation, at temperature 0.4.
- **Search writers:** Qwen3.8-27B (60%) and Gemma 4 31B (40%), at temperature 0.9.
- **Second labeller:** Gemma 4 31B. The concept, phenomenon, hard-negative and Eurostat rows keep only cases where both
  labellers agree. These rows were labelled with the verdict rules (prompt `NARRATE_V3` in `synth/realistic.py`).
- **Records:** Data360 indicators, Microdata Library studies, Policy Research Working Papers (with a passage of the full
  text), HDX datasets and map layers, Statistics Canada, ILOSTAT and Eurostat tables, and reproducibility packages.
- **Filtering:** rows whose explanation states a number absent from the input were dropped (6,494); rows over 1,280
  tokens were dropped (about 0.1%).
- **Verdict mix:** 60.2% irrelevant, 27.2% partly relevant, 12.6% relevant.
- **Languages:** English 17%; Arabic, Chinese, French, Hindi, Indonesian, Portuguese, Russian, Spanish, Swahili and
  Turkish between 7% and 9% each.
- **Splits:** train, validation and test are split by theme (a hash of the topic), so validation and test themes never
  appear in training.
- **How it was built:** `scripts/round_g.sh` adds the round G slices to the earlier rounds' data.

## 4. DPO

`scripts/dpo_run.sh` (called by `single_stage.sh`):

1. The fine-tuned model writes 4 explanations for each of 8,000 training inputs (temperature 0.8, top-k 40).
2. The teacher scores each for faithfulness, coherence, consistency with the verdict and language.
3. Pairs: the chosen side is the best student sample if it scores at least 5 for faithfulness and 4 for coherence,
   otherwise the teacher's explanation. The chosen side never states a number absent from the input. The rejected side
   is the worst sample, preferring one with an invented number. This run had 4,270 pairs (3,141 chosen from the student,
   354 rejected for an invented number).
4. Training: beta 0.3, learning rate 2e-5, 2 epochs, batch 16, plus a supervised term on the chosen explanation (weight
   0.5). The fine-tuned model is the frozen reference. The verdict is fixed in the prompt, so only the explanation is
   optimised.

## 5. Inference

- Greedy decoding.
- A script constraint blocks stray characters from other writing systems.
- If the explanation states a number absent from the input, up to 6 samples are drawn at temperature 0.7, keeping the
  greedy verdict.
- Year ranges written backwards are repaired.
- Verdict confidence is the probability of the chosen verdict at the first output step (`tiny/confidence.py`); the app
  marks verdicts below 70% as uncertain.

## Results

INT8, verdict accuracy against the teacher; in brackets, on the rows where the teacher and Gemma 4 agree. Faithfulness
is judged on 800 realistic-test explanations, on a 1 to 5 scale.

| Test set | Rows | Eight-stage chain | This recipe |
|---|---|---|---|
| Realistic | 1,495 | 84.4% (91.4%) | 85.3% (92.2%) |
| Paraphrase | 2,175 | 82.0% (88.7%) | 80.7% (87.1%) |
| Nine item types | 475 | 79.4% | 80.8% |
| Challenge suite | 2,244 | 73.0% (81.1%) | 73.6% (81.3%) |
| Eurostat tables, held out | 649 | 71.0% (81.8%) | 71.8% (83.0%) |
| Faithfulness, teacher / Gemma 4 | 800 | 4.56 / 4.70 | 4.60 / 4.73 |

Each arm is one training run (one seed).

## Reproducing from scratch

- `data/corpus/raw` now also holds FineWeb-2 shards (`*_fw2.jsonl`), which were added after `pre-wide-full` and
  `pre-wide-full2` were trained. Rerunning those two passes on the current directory would include FineWeb-2. To
  reproduce them exactly, tokenize the Wikipedia files only.
- The pretraining runs saved their model configuration but not their command-line arguments. The settings above come
  from `scripts/pretrain.sh` and the lab notebook.
- The teacher outputs depend on the served model and sampling, so regenerated data will differ in detail from the
  stored files.
