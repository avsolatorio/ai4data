---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Evaluation Suites for Statistical AI.
---

# Glossary

**Bootstrap interval.** A confidence interval obtained by resampling the questions with replacement many times; it shows how much a pass rate or a difference could move with a different draw of questions.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Error taxonomy.** The coded kinds of error a component makes (missed, spurious, boundary, wrong type), each pointing to a different fix.

**Gate.** The rule that decides whether a change ships, applied to a paired comparison of the suite before and after the change.

**Grader.** A language model, with a rubric, that scores answers the automatic checks cannot score; validated against human scores before use.

**Held-out part.** Questions kept out of development so that a change tuned on the rest is tested on questions it has not seen.

**Known-item question.** A question with one expected identifier, used to measure retrieval.

**Labelling guide.** The written rules labellers follow, maintained from their disagreements.

**MRR.** Mean reciprocal rank: the average of one over the rank of the first correct result.

**Paired comparison.** The same questions through two versions, so that differences are between versions.

**Prompt.** The instructions and context given to a language model for one task; kept as a versioned file when it is part of a workflow.

**Recall@k.** The share of questions whose expected item appears in the first k results.

**Report card.** A run's scores per slice with intervals, the set version, and the component versions, published with a release.

**Slice.** A subset of the questions (a language, a question type) scored on its own, so that a change's effect on it is visible.

**Structured output.** Model output returned as data in a declared shape (a JSON object with named fields) so that it can be checked and used by software.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; model prices and context limits are counted in tokens.
