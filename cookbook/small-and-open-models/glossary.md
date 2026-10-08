---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Small and Open Models for Statistical Offices.
---

# Glossary

**Adapter.** A small set of trained parameters added to a model for a task, so that fine-tuning fits on modest hardware (parameter-efficient fine-tuning).

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Cost-quality frontier.** The candidates that no other candidate beats on both suite score and cost per query.

**Digest.** The SHA-256 hash of a model file, recorded in the lock file and verified before loading.

**Key-value cache.** The memory a model uses per token of context during generation; grows with context length and concurrency.

**Lock file.** The record of a model's source, version, and file digests that the server verifies before loading.

**Open-weight model.** A model whose weights can be downloaded and run by the organization, under a licence that may be permissive or restricted.

**Quantization.** Storing weights at fewer bits (8 or 4 in place of 16) to reduce memory, with a suite check for any loss.

**Required score.** The suite score a task's model has to reach, set from the cost of error before any candidate is tested.

**Serving stack.** The software that loads a model and answers requests (a single-machine runtime or a batching server).

**Small model.** A model of roughly one to fifteen billion parameters that runs on one GPU or on CPU, enough for narrow tasks with the right context.
