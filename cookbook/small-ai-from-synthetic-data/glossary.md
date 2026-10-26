---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Calibration.** The comparison of model scores with reference scores or labels on a sample, to find systematic bias or to choose the threshold at which the model's scores are used.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Cost record.** One row per AI component and month with the provider, the number of requests, and the cost, summarized for the governance board.

**Croissant.** The MLCommons format, built on schema.org, for describing a machine-learning dataset so that tools can load it: its files with their checksums, the fields of each record, and the splits.

**Crosswalk.** A mapping from one classification to another, for example from a national occupation classification to ISCO.

**Dataset card.** The one document that tells a user what a dataset is, why it exists, how it was collected and labelled, whom it represents, what it is for and not for, its licence, its version, and how to cite it.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Document record.** The catalog record of a document in the World Bank document schema, with the series and surveys it draws on.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Encoder.** A pretrained model that turns a text into numbers representing its meaning, used as the base of a classifier; DeBERTa and BERT are encoders.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better or follows the organization's conventions.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the evaluation suite before and after the change.

**Generating model.** The large language model that writes synthetic training examples; its terms of use apply to the examples and can apply to the models trained on them.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Grouped split.** A draw of whole groups (households, documents) into the parts of a dataset, so that no group is divided between training and test.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Micro and macro F1.** Two ways to average the F1 score over labels: micro F1 pools every item-label decision, so frequent labels weigh more; macro F1 averages the F1 of each label, so every label weighs the same.

**ML-ready dataset.** A fixed, versioned set of examples with a documented structure, a stated purpose, a split for training and testing, a statement of whom the examples represent, and a licence that covers model training.

**Model card.** The document published with a model that states its task, its training, its scores with the data they were measured on, its known errors, and its licence.

**Multi-label classifier.** A classifier that gives a separate probability for each label, so that one text can receive several labels.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under a published licence that may be permissive or restricted.

**Precision.** The share of extracted mentions that are correct.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the source, the document and page, the method, and the version, carried with the value.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Recall.** The share of labelled mentions that the extractor found.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**SKOS and XKOS.** SKOS is the web standard for publishing controlled vocabularies (concepts with labels and identifiers); XKOS extends it for statistical classifications and their correspondences.

**Small model.** A model that runs on one GPU or on a CPU and is enough for narrow tasks: a language model of roughly one to fifteen billion parameters with the right context, or a model of tens to hundreds of millions of parameters trained for one task, such as a fine-tuned encoder.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Structured output.** A model's answer returned as data in a declared shape (a JSON object with named fields that conforms to a schema) so that a script can validate, read, and aggregate it.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Synthetic data.** Data generated by a model. For sharing microdata, records generated by a model fitted to real data so that they have the same shape and similar relationships while containing no real person's record; for training a model, labelled examples written by a large language model.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Variant.** A way of naming a dataset other than its canonical name: an acronym, an informal name, a translation, a misspelling.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.
