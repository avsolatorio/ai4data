---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards, interfaces, and licences each chapter builds on, with the World Bank's use where there is one, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard, the guide uses that
one.

## Labels and crosswalks

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [SKOS](https://www.w3.org/TR/skos-reference/) (W3C) | Publishing a label set as a concept scheme, with broader topics and mappings to other schemes. | — | Chapter 1, the form of a published label file; chapter 5, mapping properties for the crosswalk. |

## Classifications

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [ISCO-08](https://isco.ilo.org/en/isco-08/) (ILO) | The International Standard Classification of Occupations, with definitions and example occupations for its 436 unit groups. | — | Chapter 8, the occupation coding task. |
| [ESCO](https://esco.ec.europa.eu/) (European Commission) | The European classification of occupations, whose codes extend ISCO-08. | — | Chapter 8, the mapping of the reference job titles to ISCO-08. |
| [COICOP](https://unstats.un.org/unsd/classifications/Econ/) (UN) | The Classification of Individual Consumption According to Purpose, 1999 and 2018 versions, with explanatory notes. | — | Chapter 8, the expenditure coding task. |

## Generation and data files

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [OpenAI chat completions interface](https://platform.openai.com/docs/api-reference/chat) | The request and reply format that hosted services and local servers such as vLLM and Ollama offer. | — | Chapters 2 and 5, generation and the large-model baseline. |
| [JSON Schema](https://json-schema.org/) | Declaring the shape of a reply, so that a label can only be a name from the label file. | — | Chapter 2, the constrained output of the generating model. |
| [JSON Lines](https://jsonlines.org/) | One JSON record per line, for examples, replies, and scores. | — | Every data file of the guide. |

## Models, cards, and datasets

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards) | The metadata and sections of a model card. | The [ai4data organization](https://huggingface.co/ai4data) publishes its models with cards. | Chapter 7, the model card template. |
| [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) | The metadata and sections of a dataset card. | — | Chapter 7, the dataset card template. |
| [Croissant](https://docs.mlcommons.org/croissant/) (MLCommons) | A machine-readable description of a machine-learning dataset. | Covered by the [ML-ready datasets cookbook](/cookbook/ml-ready-datasets/). | Chapter 7, an option for publishing the generated data. |
| [safetensors](https://huggingface.co/docs/safetensors/) | The file format of the model weights, which loads without running code. | — | Chapter 4, the saved model. |

## Evaluation

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [scikit-learn multilabel metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) | Definitions of micro and macro averaged precision, recall, and F1. | — | Chapters 4 and 5, `evaluate.py`. |

## Licences

| Licence | Use | World Bank use | In this guide |
|---|---|---|---|
| [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | The licence of gpt-oss-120b, the generating model of the example. | — | Chapter 2, the choice of generating model. |
| [MIT License](https://opensource.org/license/mit) | The licence of DeBERTa-v3-small, the 2024 doc-topics models, and the pi agent framework. | Code of the AI for Data program. | Chapters 4 and 7. |
| [Meta Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) | The licence of Llama 3, one generator of the public dataset, with terms on the use of outputs. | — | Chapter 2, recipe 2.1. |
| [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/) | The licence of World Bank publications, including the working papers whose abstracts form the real test set. | World Bank publications in the [Open Knowledge Repository](https://openknowledge.worldbank.org/). | Chapter 5, the real test set. |
| [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The licence of the job titles of the ISCO reference set (tabiya/occupation_titles_esco). | — | Chapter 8. |
| [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) | The licence of the ONS item descriptions of the COICOP reference set. | — | Chapter 8. |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
