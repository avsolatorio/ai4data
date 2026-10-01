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

## Model

- Decoder-only transformer with RMSNorm, rotary positions, SwiGLU and tied embeddings: 7 layers, width 448, 7 heads, feed-forward 1024, vocabulary 12,000. 20.64M parameters.
- Pretrained on Wikipedia in the 11 languages (1.62B tokens, two passes). The embeddings start from Qwen3-0.6B embeddings, matched by token string and reduced with PCA to the model width.
- Fine-tuned on the task data with the loss on the output only and a higher weight on the verdict tokens.
- Export: per-row INT8 weights with fp16 row scales in one file (`tiny/export.py`). The file is 21.6 MB including the 0.84 MB tokenizer, and 18.4 MB after gzip. INT4 weights (10.4 MB) were evaluated by simulated quantization; no packed INT4 file is written.

## Results

Test set of 2,437 examples, one training run, greedy decoding with a 4-gram repeat block.

| Weights | Size | Verdict accuracy | Macro-F1 |
|---|---|---|---|
| fp32 | 82.5 MB | 82.6% | 0.775 |
| INT8 | 20.7 MB (weights) | 82.6% | 0.774 |
| INT4 | 10.4 MB (weights) | 82.6% | 0.776 |

Always answering "irrelevant" scores 63.5%. Per-class F1 at INT8: relevant 0.77, partly relevant 0.66, irrelevant 0.90.

Narrative quality (INT8, 400 test examples, judged by the teacher model, which also wrote the training narratives and therefore favours them):

| | Student | Teacher narratives |
|---|---|---|
| Faithfulness to the input (1 to 5) | 3.49 | 4.99 |
| Narratives scoring 4 or more | 54% | 99.5% |
| Coherence (1 to 5) | 3.85 | 5.00 |
| Argues for its own verdict | 98.8% | 100% |
| Written in the requested language | 98.0% | 100% |

Faithfulness by language: English 4.07, Spanish 3.37, French 3.31, Arabic 3.05, Chinese 2.95, Indonesian 2.45, Russian 2.37 (19 to 174 examples each).

Inference on one CPU thread: 0.31 s to load, 172 ms to the first token for a 330-token input, 11 ms per generated token (0.77 s for a typical 71-token output).

## What mattered

- Pretraining with Qwen-initialized embeddings cut held-out perplexity by 40 to 56% at equal compute (English 36.0 to 21.3 at 100M tokens). A model trained from scratch on 1,107 examples memorized them (56% verdict accuracy, below the majority rate).
- The retrieval similarity is the largest single gain. Without it the verdict stayed near the majority rate even with 11,500 training examples (about 74%). With it, the same recipe reaches 82%.
- A 7-layer, wide model was slightly better than 10 and 14-layer models of the same size.
- Scaling the data from 11,000 to 25,000 examples and adding a second pretraining pass raised judged faithfulness from 2.98 to 3.49.
- A logit bonus for input tokens at decoding time did not help (faithfulness 3.50 at strength 1, 2.78 at strength 4).

## Limitations

- Labels and narratives come from one teacher model and were not reviewed by people. Non-English narratives were not checked by native speakers.
- About half of the narratives contain a claim the input does not support, and faithfulness is lowest in Russian, Indonesian and Chinese.
- The judge and the teacher are the same model. Faithfulness numbers show the gap between student and teacher, and the absolute scale is uncertain.
- The test set has 233 relevant examples and one seed was run, so differences of a point or two between variants are within noise.
- The similarity is computed by a 97M-parameter search model that the retriever supplies. It is not part of the 20 MB.
- The year filter is not checked for documents, because a paper's publication year does not give the period its data covers.

## Reproduce

```
scripts/serve_vllm.sh                                   # teacher on one GPU
scripts/run_pipeline.sh data/v2 900 900                 # queries, anchored queries, retrieval, narratives
python -m relevance_explainer.synth.generate assemble --out data/f1 --also data/v1 --max-ratio 100 --similarity
python -m relevance_explainer.tiny.corpus fetch|tokenizer|tokenize --root data/corpus
python -m relevance_explainer.tiny.pretrain --root data/corpus --out runs/pre --tokens 1.6e9 \
    --init-emb Qwen/Qwen3-0.6B --layers 7 --d-model 448 --heads 7 --d-ff 1024
python -m relevance_explainer.tiny.train --data data/f1 --out runs/task --tokenizer data/corpus/tokenizer.json \
    --init runs/pre/model.pt --lr 1e-3 --epochs 8 --bs 32 --verdict-weight 5 --select val_verdict_acc
python -m relevance_explainer.tiny.export --run runs/task --out runs/task/explainer-int8.pt
scripts/judge.sh runs/task int8 400
```

## Layout

- `src/relevance_explainer/synth/`: catalog sources, query and narrative generation, retrieval signals.
- `src/relevance_explainer/tiny/`: model, corpus, pretraining, task training, evaluation, export, judge.
- `scripts/`: pipeline, vLLM serving, judge and sweep scripts.
- `notes/lab_notebook.md`: dated, append-only experiment log with all numbers.
- `data/` and `runs/` hold generated data and trained models and are not tracked by git.
