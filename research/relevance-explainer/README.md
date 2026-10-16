# Relevance explainer

A small multilingual model (34 MB at INT8, CPU) that reads one catalog record and a search, and writes whether the
record answers the search (`relevant`, `partly_relevant` or `irrelevant`) with a short explanation. The explanation's
facts are written as claims copied from the record, and a verifier outside the model checks each claim against the
record. This README covers the task, the use of the model from Python, the command line and the web app, the checks
attached to every answer, the results, the repository layout and the limitations. `REPRODUCE.md` gives the full
lineage of the released model.

## Task

Input: the record's metadata, then the search, its language, the facets set in the interface, and the constraints
parsed from the search with their status against the record (rule-based, `synth/understand.py`):

```
Item type: statistical indicator series
Result: Average sovereign recovery rates
Database: Global Emerging Markets Risk Database; Source: Global Emerging Markets Risk Database
Unit: Defaults, Percentage, Euros; Frequency: Annual; Years: 2024-2024
Coverage: not stated
Definition: Average recovery rates for sovereign and sovereign-guaranteed counterparts in 1984-2024
Respond in: English
Search: sovereign recovery rates census data 2024
Facets: none
Understood: time=2024 [query]; wants=microdata [query]
Checks: time 2024: covered (2024) | type microdata: item is an indicator series
```

Output (as the API returns it, from the realistic test set):

```
<verdict grade="5">partly_relevant</verdict>
<claims><c cid="3" field="time">2024</c><c cid="2" field="title">Average sovereign recovery rates</c></claims>
<tldr>The item matches the topic and year but is an aggregated indicator series, not the requested census microdata.</tldr>
<exp>The item provides <r cid="2">average sovereign recovery rates</r> for <r cid="3">2024</r>, matching the topic and
year. However, it is an aggregated indicator series rather than the census microdata requested in the search.</exp>
```

Each claim is a value from one field of the record; each reference in the text points to a claim. Searches and
explanations can be in English, Arabic, Chinese, French, Hindi, Indonesian, Portuguese, Russian, Spanish, Swahili and
Turkish.

## Use

```
pip install .                      # inference: torch, tokenizers, numpy
relevance-explainer bundle --run runs/final-C-dpo --out dist/relevance-explainer-c-dpo
```

```python
from relevance_explainer.inference import Explainer

ex = Explainer.from_bundle("dist/relevance-explainer-c-dpo", threads=4)
out = ex.explain(record, search="sovereign recovery rates census data 2024", language="English")
out["verdict"], out["explanation"], out["all_verified"], out["faithfulness"], out["verdict_set"]
```

`record` is a catalog record as built by `synth/sources.py`; `explain_text` takes an already rendered input. The
command line reads JSON lines: `relevance-explainer explain --bundle <dir> --input requests.jsonl`. A bundle holds the
INT8 model, the calibration and the faithfulness scorer with their checksums, and refuses to load if a file was
changed.

The web app (`scripts/serve_app.sh`, needs `pip install .[app]`) serves curated examples, live CPU generation and the
checks below for each answer; the earlier models remain selectable for comparison.

## Checks attached to every answer

| Check | What it is | Realistic test |
|---|---|---|
| Verified claims and references | each claim's value occurs in its record field, each reference matches its claim, and no number or place outside the record is stated (`claims.py`) | 97.3% of answers fully verified; 99.3% after one round of 3 retries chosen by the verifier |
| Calibrated verdict probabilities | temperature fitted on held-out training-distribution rows | expected calibration error 2.3% |
| Conformal verdict set | the verdicts whose calibrated probability passes the 90% threshold | contains the reference verdict for 91.6% of searches; one verdict for 86.9% |
| Expected grade | 0 to 10, from the grade token probabilities weighted by the verdict probabilities | concordance with the teacher's 0 to 10 grade 0.889, against 0.857 for the written grade (measured on the 250k-sample model) |
| Faithfulness score | probability that a judge rates the explanation faithful 4 or 5 of 5, from the verifier and the model's own uncertainty (`faith.py`) | AUROC 0.856 against the teacher judge and 0.782 against Gemma 4 (800 answers); at 0.8 or more (83% of answers) the judges rate 94% faithful |

## Results

Released model (`runs/final-C-dpo`), INT8, verdict accuracy against the teacher's labels:

| Test set | Searches | Accuracy |
|---|---|---|
| Realistic searches on themes unseen in training | 1,495 | 86.6% (CPU) / 86.8% (GPU) |
| Paraphrased searches | 2,175 | 80.9% |
| Nine item types | 482 | 81.3% |
| Challenge phrasings (negation, look-alike topics, proxies and others) | 2,244 | 74.7% |

These sets are rendered at render version 4, the format of the training data. The app and `inference.py` render
version 5, which drops places found in short everyday words or short alternate city names (Swahili "za" read as South
Africa, "spoken" as Spokane) and counts plural subnational words such as "provinces and territories". 293 of the
6,396 evaluation inputs change. On those rows, against teacher labels written on the version-5 inputs, the model is
right on 84.6% of version-5 inputs and 80.5% of version-4 inputs (18 rows better, 6 worse, McNemar p = 0.023; Gemma 4
labels: 19 better, 6 worse). Against the stored labels it loses 15 rows, because those labels were written on inputs
that named the false place.

On 45 searches over 29 records from catalogs absent from training (Our World in Data, open.canada.ca, data.gov.au and
Zenodo; `evalsets/unseen_catalogs`, with labels written before the model was run on them), the verdict agrees with
the label on 33 (73%). Five of the 12 disagreements are searches for a province or state of a record whose metadata
names only the country.

Explanations of 800 realistic-test answers judged faithful 4 or 5 of 5: 88.4% by the teacher (Qwen3.8-27B), 90.1% by
Gemma 4 31B. On CPU with 4 threads an answer takes 2.6 s (median) and 3.5 s at the 90th percentile.

## Layout

| Path | Contents |
|---|---|
| `src/relevance_explainer/inference.py`, `answer_c.py`, `claims.py`, `faith.py` | inference API, answer pipeline, verifier, faithfulness scorer |
| `src/relevance_explainer/tiny/` | model, pretraining, fine-tuning, DPO, evaluation, INT8 export, decoding guards |
| `src/relevance_explainer/synth/` | catalog sources, search generation, query understanding, rendering, teacher labelling |
| `src/relevance_explainer/distill/` | on-policy sampling, scoring and preference pairs |
| `src/relevance_explainer/webapp/` | web app |
| `src/relevance_explainer/resources/` | gazetteer, multilingual lexicon and word lists used by query understanding and the verifier |
| `scripts/` | command-line steps, indexed in `scripts/README.md` |
| `app/` | the app's curated examples |
| `requirements/` | exact versions of the research and teacher-serving environments |
| `tests/` | unit tests, repository checks and inference tests |
| `evalsets/` | committed test sets: records from catalogs absent from training, with labelled searches |
| `data/`, `runs/` | generated data and trained models, not tracked; `artifacts.json` lists their checksums |

## Documentation

- `REPRODUCE.md`: environment, lineage of the released model, checks of a rerun.
- `RECIPE.md`: the previous app model and the round G data.
- `docs/first_model.md`: the first model and the findings of rounds 1 to 7.
- `docs/design_realistic.md`: the realistic search design.
- `docs/rating_protocol.md`: the planned human rating study.

## Limitations

- Labels and explanations come from one teacher model; faithfulness has been judged by two LLMs and not yet by people.
- One verdict in eight is wrong on realistic searches, and about one in four on the challenge phrasings. The verdict
  suits explaining a result to a reader; it is not accurate enough to remove results automatically.
- The model has been evaluated on World Bank, HDX, ILOSTAT, Eurostat and Statistics Canada records with searches
  written by LLMs, and on a 45-search set from four other catalogs (73% agreement). A new catalog needs its own
  labelled test set before its accuracy is known.
- Searches in a language outside the eleven are withheld (`language.py`): their explanations were written in a third
  language with garbled words. The script and fastText checks miss short Latin-script keyword searches.
- Single answers can differ between CPU and GPU (41% of greedy texts on the realistic test), although the rates
  agree; a service should log the answer it served with its checks.
