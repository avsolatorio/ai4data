---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Metadata Curation with Language Models.
---

# Glossary

**Acceptance rate.** The share of model suggestions that curators accept or edit, per field, task, and model.

**Agents manifest.** The file that defines the agents of the review pipeline (detectors, critic, categorizer, severity scorer) and their instructions.

**Calibration.** The comparison of model scores with curator scores on a sample, per dimension, to find systematic bias.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Draft status.** The mark on a model-written field that keeps it out of the published catalog until a decision.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Error taxonomy.** The coded reasons for rejections and edits: wrong flag, invented fact, wrong vocabulary term, style, and others.

**Evaluation set.** Decided records kept to test changes to prompts, manifests, rubrics, and models.

**Grounding.** The property of a draft that uses only what the record and its named sources support.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under its published licence.

**Quality dimensions.** Completeness, semantic alignment, specificity, and consistency, scored from 1 to 5 with a rubric.

**Review board.** The interface in which a curator sees the current and proposed values as a diff and decides.

**Rubric.** The written meaning of each score level per dimension, used by the model and the curators alike.

**Structured output.** Model output returned as JSON that conforms to a schema, so that it can be validated and aggregated.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.
