---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Language Models in Statistical Production.
---

# Glossary

**Automated share.** The share of responses coded at or above the confidence threshold without a coder; measured by a re-coded sample each round.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Confidence threshold.** The model confidence above which a code is accepted automatically, set from the measured accuracy curve.

**Decision point.** The step at which a named person decides on a model's output (a code, a proposal, an explanation, a draft).

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or reveals something about one; applied before any release.

**Edit rule.** A machine-readable condition a record must satisfy (range, consistency), owned by the organization and applied before and after any model proposal.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Fine-tuning.** Training an existing model further on the organization's own labelled data so that it follows the organization's conventions.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**GSBPM.** The Generic Statistical Business Process Model, the UNECE reference model of production phases and sub-processes.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Manifest.** A file that lists the parts of a job or a release: for a job, the records and their status, so that a run can resume; for a release, every file with its checksum.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Method of record.** The documented, reproducible imputation or estimation method the organization applies; a model proposes and explains beside it.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under its published licence.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Re-coded sample.** A random sample of the automated share coded blind by staff each round to measure the automated accuracy.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**Statement of model use.** The section of a release's quality report that says where models were used, which, how they performed, who decided, and what the effect was.

**Structured output.** Model output returned as data in a declared shape (a JSON object with named fields) so that it can be checked and used by software.

**Switch-off test.** The test that production proceeds with the model layer stopped, with the fallback per task exercised.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; model prices and context limits are counted in tokens.
