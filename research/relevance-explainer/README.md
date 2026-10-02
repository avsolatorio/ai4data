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
- Three rounds (v1 to v3); the third uses a flatter language mix so that Russian, Hindi, Indonesian, Turkish and Swahili each have 2,000 to 4,600 training examples. Final set: 60,157 training, 8,866 validation and 7,056 test examples after de-duplication. Splits are by theme, so test themes are unseen in training. Class mix in the test set: 64.4% irrelevant, 26.7% partly relevant, 8.9% relevant.
- Label noise: relabelling 498 training examples with the teacher's thinking mode on agrees with the stored verdicts on 88.2%. Most disagreements are at the irrelevant and partly relevant boundary.

## Model

- Decoder-only transformer with RMSNorm, rotary positions, SwiGLU and tied embeddings: 7 layers, width 448, 7 heads, feed-forward 1024, vocabulary 12,000. 20.64M parameters.
- Pretrained on Wikipedia in the 11 languages (1.62B tokens, two passes). The embeddings start from Qwen3-0.6B embeddings, matched by token string and reduced with PCA to the model width.
- Fine-tuned on the task data with the loss on the output only and a higher weight on the verdict tokens.
- Preference-tuned with DPO on 5,203 pairs. For each pair the student wrote four narratives for a training input with the teacher's verdict fixed, the teacher scored them for faithfulness and coherence, and the pair prefers a narrative scored 5 over one scored 3 or lower (`distill/`). Settings: beta 0.3, learning rate 2e-5, 2 epochs, plus a supervised term on the preferred narrative.
- Export: per-row INT8 weights with fp16 row scales in one file (`tiny/export.py`). The file is 21.6 MB including the 0.84 MB tokenizer, and 18.4 MB after gzip. INT4 weights (10.4 MB) were evaluated by simulated quantization; no packed INT4 file is written.

## Results

Test set of 7,056 examples, greedy decoding with a 4-gram repeat block. Verdict accuracy varies by about 0.1 points across training seeds (measured on the earlier 25k-example model).

| Weights | Size | Verdict accuracy | Macro-F1 |
|---|---|---|---|
| INT8 | 20.7 MB (weights) | 83.6% | 0.782 |
| INT4 | 10.4 MB (weights) | 83.2% | 0.776 |

Always answering "irrelevant" scores 64.4%.

Narrative quality (800 test examples, judged by the teacher model, which also wrote the training narratives and therefore favours them):

| | 25k examples | 60k examples | 60k + preference tuning (final, INT8) |
|---|---|---|---|
| Faithfulness to the input (1 to 5) | 3.24 | 3.85 | 3.99 |
| Narratives scoring 4 or more | 46.9% | 63.9% | 70.0% |
| Coherence (1 to 5) | 3.62 | 4.08 | 4.19 |
| Argues for its own verdict | 98.1% | 99.0% | 99.4% |
| Written in the requested language | 97.4% | 99.3% | 99.0% |

The teacher's own narratives score 4.99 on faithfulness under this judge (self-judged). At INT4 the final model scores 3.94 faithfulness and 66.7% at 4 or more.

Faithfulness of the final model by language (examples judged): English 4.52 (220), Spanish 4.20 (88), Portuguese 4.17 (42), French 3.91 (74), Swahili 3.91 (23), Chinese 3.74 (72), Arabic 3.70 (66), Russian 3.64 (77), Hindi 3.62 (48), Indonesian 3.54 (48), Turkish 3.31 (42). With 25k training examples the same languages ranged from 2.25 (Hindi) to 4.21 (English).

Inference on one CPU thread: 0.31 s to load, 172 ms to the first token for a 330-token input, 11 ms per generated token (0.77 s for a typical 71-token output).

## What mattered

- Pretraining with Qwen-initialized embeddings cut held-out perplexity by 40 to 56% at equal compute (English 36.0 to 21.3 at 100M tokens). A model trained from scratch on 1,107 examples memorized them (56% verdict accuracy, below the majority rate).
- The retrieval similarity is the largest single gain. Without it the verdict stayed near the majority rate even with 11,500 training examples (about 74%). With it, the same recipe reaches 82%.
- A 7-layer, wide model was slightly better than 10 and 14-layer models of the same size.
- Task data is the main lever for the narratives. On one test set, 25k, 30k and 60k training examples gave faithfulness 3.24, 3.57 and 3.85, with the largest gains in the languages that had the least data (Russian 2.51 to 3.55, Hindi 2.25 to 3.23). The verdict moved less (82.7% to 83.5%); it is bounded by the teacher's own label consistency (88% agreement between its thinking and non-thinking verdicts).
- Preference tuning on the student's own narratives scored by the teacher adds about +0.15 faithfulness at either data size (3.50 to 3.65 at 25k, 3.85 to 3.99 at 60k) without changing verdict accuracy. Settings matter: a learning rate of 5e-5 with 3 epochs lowered faithfulness to 3.37 (coherence 3.39) and 3e-5 with beta 0.05 lowered it to 2.72. Fine-tuning on the preferred narratives alone gave 3.60.
- Did not help: doubling the pretraining corpus with FineWeb-2 web text (2 to 6 times more tokens for the small languages; equal verdict and 0.15 lower faithfulness on the 60k data), per-language fine-tunes from the shared base (worse on every language tested), a per-language adapter on the multilingual model (no change), a 40M-parameter model at INT4 (equal to 20M at INT8 at the same file size), and weight-tied recursive layers (no quality gain at equal stored size, no speed gain).
- Did not help: a logit bonus for input tokens at decoding time (faithfulness 3.50 at strength 1, 2.78 at strength 4), a higher loss weight on the verdict tokens (tested without the similarity input; the final recipe uses a weight of 5 without an ablation), and training on the teacher's verdict probabilities for the 21% of training examples that had them (80.3% accuracy after 14 epochs against 82.6%). The teacher's probabilities are close to one-hot (mean 0.90 on its own label).

## Limitations

- Labels and narratives come from one teacher model and were not reviewed by people. Non-English narratives were not checked by native speakers.
- 30% of the narratives score below 4 on faithfulness, and faithfulness is lowest in Turkish, Indonesian, Hindi and Russian.
- The judge and the teacher are the same model. Faithfulness numbers show the gap between student and teacher, and the absolute scale is uncertain. The preference pairs were also scored by this judge, so part of the gain may reflect its preferences.
- The test set has 631 relevant examples, and the judged per-language groups have 23 to 220 examples, so per-language differences of a few tenths are noisy.
- The similarity is computed by a 97M-parameter search model that the retriever supplies. It is not part of the 20 MB.
- The year filter is not checked for documents, because a paper's publication year does not give the period its data covers.

## Reproduce

```
scripts/serve_vllm.sh                                   # teacher on one GPU (GPU=0 MEM=0.9 PORT=8000)
scripts/run_pipeline.sh data/v2 900 900                 # queries, anchored queries, retrieval, narratives
scripts/run_round.sh data/v3 200                        # a further round with the flatter language mix
python -m relevance_explainer.synth.generate assemble --out data/f2 --also data/v2 data/v1 --max-ratio 100 --similarity
python -m relevance_explainer.tiny.corpus fetch|tokenizer|tokenize --root data/corpus
python -m relevance_explainer.tiny.pretrain --root data/corpus --out runs/pre --tokens 1.6e9 \
    --init-emb Qwen/Qwen3-0.6B --layers 7 --d-model 448 --heads 7 --d-ff 1024
python -m relevance_explainer.tiny.train --data data/f2 --out runs/task --tokenizer data/corpus/tokenizer.json \
    --init runs/pre/model.pt --lr 1e-3 --epochs 8 --bs 32 --verdict-weight 5 --select val_verdict_acc
# preference tuning (TEACHER_URLS lists the vLLM servers)
python -m relevance_explainer.distill.onpolicy sample --run runs/task --data data/f2 --n 8000 --k 4
python -m relevance_explainer.distill.onpolicy score  --data data/f2
python -m relevance_explainer.distill.onpolicy pairs  --data data/f2
python -m relevance_explainer.tiny.dpo --run runs/task --data data/f2 --out runs/final --beta 0.3 --lr 2e-5 --epochs 2 --sft 0.5
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
