---
id: evaluate
title: "6. Evaluate whether it works"
sidebar_label: "6. Evaluate"
sidebar_position: 6
description: Retrieval benchmarks, hallucination tests, answer accuracy, and multilingual evaluation.
---

# 6. Evaluate whether it works

**Question:** Can we tell whether our AI-ready steps improved anything?

## Why this matters

Each step in this guide is meant to improve a measurable outcome: more
questions find the right series, more answers carry the right number. Without
a fixed test set, changes to metadata, search, or models cannot be compared,
and problems appear first in front of users.

## What good looks like

- A small, versioned set of test questions with known correct answers.
- The same measures are run before and after each change.
- Results are reported by language, topic, and question type.
- Failures are reviewed and added to the test set.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Write 30 to 50 known-item questions. Run them by hand against the search page each quarter and record the rank of the correct result. |
| **AI-ready** | Automate the run. Report Recall@k and MRR for retrieval. Keep the question set in version control. Test each supported language separately. |
| **AI-native** | Add answer-level tests: numeric accuracy, citation validity, and correct refusals. Use a language model as a first-pass grader with human spot checks. Run the suite before every model or metadata change. |

## Implementation options

**Retrieval measures**

- **Recall@k:** the share of questions for which the correct item appears in the top k results.
- **MRR (mean reciprocal rank):** the average of 1 divided by the rank of the first correct result.
- **nDCG@k:** a graded measure that credits partially relevant results.

**Answer measures**

- **Numeric accuracy:** the share of numbers in answers that match the official value, with a stated tolerance for rounding.
- **Citation validity:** the share of citations that resolve and support the statement.
- **Refusal accuracy:** the share of unanswerable questions that the system declines.
- **Hallucination rate:** the share of answers that contain a claim with no support in the retrieved records.

**Building the question set**

- Draw questions from search logs and help-desk tickets.
- Write paraphrases of the official title, since users rarely use the exact words.
- Include questions with no answer, ambiguous questions, and questions about revised or discontinued series.
- Include each supported language, written by speakers of that language where possible. Machine translation is a starting point and needs review.

## World Bank examples

- The [PI-FT Pipeline Guide](/pift-toolkit/pipeline) includes an evaluation step that reports held-out Recall@k, MRR, and nDCG@k, and compares several embedding models on the same queries.
- The [PI-FT method](/pift-toolkit/method) describes graded evaluation with an LLM judge.

## How to test it

The evaluation suite is itself the test. To check the suite, confirm that a deliberately worse configuration, such as search with empty descriptions, scores lower.

## Checklist

- [ ] Versioned question set with known answers
- [ ] Recall@k and MRR reported for retrieval
- [ ] Results broken down by language and topic
- [ ] Unanswerable questions included
- [ ] Numeric accuracy and citation validity measured for generated answers
- [ ] Human review of a sample of graded results
- [ ] Suite run before every model, index, or metadata change
