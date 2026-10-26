---
language:
- en
license: cc-by-4.0
task_categories:
- text-classification
tags:
- synthetic
- multi-label
pretty_name: Synthetic sentences for World Bank topic classification (example)
size_categories:
- 10K<n<100K
---

# Synthetic sentences for World Bank topic classification

This card describes the training set generated for the running example of
the *Practical Guide to Small AI from Synthetic Data*. It is filled in as
an example; an organization replaces each value with its own.

## Summary

The set holds 20,074 English sentences, each labelled with one or more of
86 topics from the World Bank's topic list. Every sentence was written by
a large language model. Names of places, institutions, and people, and
every figure in the sentences, are invented. No sentence is a quotation
from a real document.

## Generation

| Item | Value |
|---|---|
| Generating model | openai/gpt-oss-120b, open weights, Apache License 2.0 |
| Access | OpenRouter, with the providers that enforce the JSON structure, fastest first |
| Dates | October 2026 |
| Prompt | Version v2-allow-names of `generate_examples.py`: sentences of the kind found in reports and working papers, using the words a report would use |
| Request plan | 1,000 requests of 8 labels each, drawn so that every label was offered 92 to 95 times (`plan_requests.py`, seed 42) |
| Output format | JSON Schema whose label values are limited to the labels of the request |
| Tokens and cost | 485,668 tokens read and 3,138,192 written; 1.96 US dollars at the listed price |

The raw replies, with the model, time, prompt version, token counts, and
finish reason of each, are kept in `raw.jsonl`.

## Fields

| Field | Content |
|---|---|
| `text` | The generated sentence. |
| `labels` | The topics the generating model assigned to it. |
| `request_id` | The request that produced it, which links to its reply in `raw.jsonl`. |
| `generator` | The model that wrote it. |
| `kind` | `topic` for these sentences. |

## Checks

The check report of `check_examples.py` against the label file of 86 labels:

| Check | Count |
|---|---|
| Sentences | 20,074 |
| Unknown label names | 0 |
| Repeated texts | 20 |
| Sentences that contain a label name | 15,588 |
| Clean sentences | 20,054 |
| Fewest and most sentences per label | 284 (Sexual Orientation and Gender Identity) and 517 (Climate Change) |

## Intended use

Training and testing a classifier that suggests World Bank topics for
English documents on economic and social development. The set is not a
source of facts about any country, institution, or program.

## Known issues

- Some sentences attribute invented figures to real institutions, for example "The 2025 World Bank report on debt relief recommends extending grace periods for low-income countries by up to five years."
- The labels are the generating model's choice and have not been reviewed one by one; a sample review measures the share of wrong labels (recipe 3.2 of the guide).
- Broad topics such as Environment overlap with their subtopics, and the labels do not always mark both.

## Licence

The sentences are released under CC BY 4.0. The generating model's
licence, Apache 2.0, places no conditions on its outputs; its usage
policy applies to the use of the model.

## Citation

World Bank. 2026. *Practical Guide to Small AI from Synthetic Data.* Version 0.1. AI for Data – Data for AI.
