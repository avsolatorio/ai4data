# Relevance Explainer

A small model (about 20 MB at INT8) that reads a search query, the applied filters, the retriever's similarity score and the metadata of one catalog result, and writes a verdict (`relevant`, `partly_relevant` or `irrelevant`) followed by a short narrative that explains it. Queries and narratives can be in 11 languages; the catalog metadata is in English.

## Task

Input (one text block):

```
Respond in: French
Item type: survey microdata study
Query: données éducation filles Burkina Faso 2015
Filters: countries=Burkina Faso; years=2015; sex=female
Retrieval similarity: 0.40
Result: Threshold BRIGHT I 2007-2008
Producer: ...; Survey years: 2007-2008; Coverage: 1 country: Burkina Faso
Abstract: ...
Filter check: country Burkina Faso: covered; survey years 2015: not covered
```

Output: the verdict on the first line, then two or three sentences in the query's language that name the evidence (what the item contains, which filters pass or fail, what is missing).

The retrieval similarity and the filter check are computed outside the model. The similarity comes from the retriever. The filter check is a deterministic comparison of the filters with the item's country list, year range and frequency.

## Data

Synthetic, built from open catalogs and written by a larger teacher model.

- Sources: World Bank Data360 indicators, Microdata Library studies, World Bank Policy Research Working Papers (with a passage cut from the full text), and Humanitarian Data Exchange datasets.
- Queries: a teacher model writes searches with filters in 11 languages, for about 1,270 themes and for specific catalog items (one search the item answers fully and one it answers partly). Results come from each catalog's real search, sampled at several ranks.
- Labels: Qwen3.8-27B writes the verdict and narrative for each (query, filters, item) pair. It is served with vLLM (8-bit weights) for most of the data and with Ollama (4-bit weights) for the first 4,155 pairs.
- Final set: 25,198 training, 3,797 validation and 2,437 test examples after de-duplication. Splits are by theme, so test themes are unseen in training. Class mix in the test set: 63.5% irrelevant, 27.0% partly relevant, 9.6% relevant.
- Label noise: relabelling 498 training examples with the teacher's thinking mode on agrees with the stored verdicts on 88.2%. Most disagreements are at the irrelevant and partly relevant boundary.

## Model

- Decoder-only transformer with RMSNorm, rotary positions, SwiGLU and tied embeddings: 7 layers, width 448, 7 heads, feed-forward 1024, vocabulary 12,000. 20.64M parameters.
- Pretrained on Wikipedia in the 11 languages (1.62B tokens, two passes). The embeddings start from Qwen3-0.6B embeddings, matched by token string and reduced with PCA to the model width.
- Fine-tuned on the task data with the loss on the output only and a higher weight on the verdict tokens.
- Preference-tuned with DPO on 4,397 pairs. For each pair the student wrote four narratives for a training input with the teacher's verdict fixed, the teacher scored them for faithfulness and coherence, and the pair prefers a narrative scored 5 over one scored 3 or lower (`distill/`). Settings: beta 0.3, learning rate 2e-5, 2 epochs, plus a supervised term on the preferred narrative.
- Export: per-row INT8 weights with fp16 row scales in one file (`tiny/export.py`). The file is 21.6 MB including the 0.84 MB tokenizer, and 18.4 MB after gzip. INT4 weights (10.4 MB) were evaluated by simulated quantization; no packed INT4 file is written.

## Results

Test set of 2,437 examples, greedy decoding with a 4-gram repeat block. Verdict accuracy varies by about 0.1 points across training seeds (82.6%, 82.6%, 82.7% for the model before preference tuning).

| Weights | Size | Verdict accuracy | Macro-F1 |
|---|---|---|---|
| fp32 | 82.5 MB | 82.4% | 0.773 |
| INT8 | 20.7 MB (weights) | 82.5% | 0.775 |
| INT4 | 10.4 MB (weights) | 82.6% | 0.773 |

Always answering "irrelevant" scores 63.5%.

Narrative quality (INT8, 800 test examples, judged by the teacher model, which also wrote the training narratives and therefore favours them):

| | Before preference tuning | Final model | Teacher narratives |
|---|---|---|---|
| Faithfulness to the input (1 to 5) | 3.50 | 3.65 | 4.99 |
| Narratives scoring 4 or more | 54.5% | 58.6% | 99.5% |
| Coherence (1 to 5) | 3.85 | 3.88 | 5.00 |
| Argues for its own verdict | 98.4% | 99.1% | 100% |
| Written in the requested language | 98.0% | 97.8% | 100% |

Faithfulness by language, before and after preference tuning: English 4.13 to 4.28 (334 examples), French 3.46 to 3.64 (98), Spanish 3.39 to 3.45 (93), Arabic 2.94 to 3.09 (54), Chinese 3.21 to 3.42 (48), Russian 2.25 to 2.52 (44), Turkish 2.62 to 2.79 (29), Hindi 2.25 to 2.68 (28), Indonesian 2.52 to 2.44 (27), Portuguese 3.77 to 3.38 (26), Swahili 2.58 to 3.00 (19).

Inference on one CPU thread: 0.31 s to load, 172 ms to the first token for a 330-token input, 11 ms per generated token (0.77 s for a typical 71-token output).

## What mattered

- Pretraining with Qwen-initialized embeddings cut held-out perplexity by 40 to 56% at equal compute (English 36.0 to 21.3 at 100M tokens). A model trained from scratch on 1,107 examples memorized them (56% verdict accuracy, below the majority rate).
- The retrieval similarity is the largest single gain. Without it the verdict stayed near the majority rate even with 11,500 training examples (about 74%). With it, the same recipe reaches 82%.
- A 7-layer, wide model was slightly better than 10 and 14-layer models of the same size.
- Scaling the data from 11,000 to 25,000 examples and adding a second pretraining pass raised judged faithfulness from 2.98 to 3.49.
- Preference tuning on the student's own narratives scored by the teacher raised faithfulness from 3.50 to 3.65 without changing verdict accuracy. Settings matter: a learning rate of 5e-5 with 3 epochs lowered faithfulness to 3.37 (coherence 3.39) and 3e-5 with beta 0.05 lowered it to 2.72. Fine-tuning on the preferred narratives alone gave 3.60.
- Did not help: a logit bonus for input tokens at decoding time (faithfulness 3.50 at strength 1, 2.78 at strength 4), a higher loss weight on the verdict tokens (tested without the similarity input; the final recipe uses a weight of 5 without an ablation), and training on the teacher's verdict probabilities for the 21% of training examples that had them (80.3% accuracy after 14 epochs against 82.6%). The teacher's probabilities are close to one-hot (mean 0.90 on its own label).

## Limitations

- Labels and narratives come from one teacher model and were not reviewed by people. Non-English narratives were not checked by native speakers.
- 41% of the narratives score below 4 on faithfulness, and faithfulness is lowest in Russian, Hindi, Indonesian and Turkish.
- The judge and the teacher are the same model. Faithfulness numbers show the gap between student and teacher, and the absolute scale is uncertain. The preference pairs were also scored by this judge, so part of the gain may reflect its preferences.
- The test set has 233 relevant examples and a handful of languages have fewer than 30 judged examples, so per-language differences are noisy.
- The similarity is computed by a 97M-parameter search model that the retriever supplies. It is not part of the 20 MB.
- The year filter is not checked for documents, because a paper's publication year does not give the period its data covers.

## Reproduce

```
scripts/serve_vllm.sh                                   # teacher on one GPU (GPU=0 MEM=0.9 PORT=8000)
scripts/run_pipeline.sh data/v2 900 900                 # queries, anchored queries, retrieval, narratives
python -m relevance_explainer.synth.generate assemble --out data/f1 --also data/v1 --max-ratio 100 --similarity
python -m relevance_explainer.tiny.corpus fetch|tokenizer|tokenize --root data/corpus
python -m relevance_explainer.tiny.pretrain --root data/corpus --out runs/pre --tokens 1.6e9 \
    --init-emb Qwen/Qwen3-0.6B --layers 7 --d-model 448 --heads 7 --d-ff 1024
python -m relevance_explainer.tiny.train --data data/f1 --out runs/task --tokenizer data/corpus/tokenizer.json \
    --init runs/pre/model.pt --lr 1e-3 --epochs 8 --bs 32 --verdict-weight 5 --select val_verdict_acc
# preference tuning (TEACHER_URLS lists the vLLM servers)
python -m relevance_explainer.distill.onpolicy sample --run runs/task --data data/f1 --n 6000 --k 4
python -m relevance_explainer.distill.onpolicy score  --data data/f1
python -m relevance_explainer.distill.onpolicy pairs  --data data/f1
python -m relevance_explainer.tiny.dpo --run runs/task --data data/f1 --out runs/final --beta 0.3 --lr 2e-5 --epochs 2 --sft 0.5
python -m relevance_explainer.tiny.export --run runs/final --out runs/final/explainer-int8.pt
scripts/judge.sh runs/final int8 800
```

## Layout

- `src/relevance_explainer/synth/`: catalog sources, query and narrative generation, retrieval signals.
- `src/relevance_explainer/tiny/`: model, corpus, pretraining, task training, preference tuning, evaluation, export, judge.
- `src/relevance_explainer/distill/`: teacher-side tools (verdict probabilities, label-noise check) and on-policy sampling, scoring and pair building.
- `scripts/`: pipeline, vLLM serving, judge and sweep scripts.
- `notes/lab_notebook.md`: dated, append-only experiment log with all numbers.
- `data/` and `runs/` hold generated data and trained models and are not tracked by git.
