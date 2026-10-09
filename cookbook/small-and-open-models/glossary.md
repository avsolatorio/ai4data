---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Small and Open Models for Statistical Offices.
---

# Glossary

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Cost-quality frontier.** The candidates that no other candidate beats on both suite score and cost per query.

**Digest.** The SHA-256 hash of a model file, recorded in the lock file and verified before loading.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Fine-tuning.** Changing a model's weights by training it further on the organization's own labelled examples, so that it does one task better.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**GSBPM.** The Generic Statistical Business Process Model, the shared description of the steps of statistical production from specifying needs to disseminating and evaluating.

**Key-value cache.** The memory a model uses per token of context during generation; grows with context length and concurrency.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Lock file.** The record of a model's source, version, and file digests that the server verifies before loading.

**Manifest.** A file that lists the parts of a job or a release: for a job, the records and their status, so that a run can resume; for a release, every file with its checksum.

**Open-weight model.** A model whose weights can be downloaded and run by the organization, under a licence that may be permissive or restricted.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Provenance.** The record of where a value came from: the series, the release, the document and page, the method, and the version, carried with the value.

**Quantization.** Storing weights at fewer bits (8 or 4 in place of 16) to reduce memory, with a suite check for any loss.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Serving stack.** The software that loads a model and answers requests (a single-machine runtime or a batching server).

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.

**Structured output.** Model output returned as data in a declared shape (a JSON object with named fields) so that it can be checked and used by software.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices and throughput are counted in tokens.
