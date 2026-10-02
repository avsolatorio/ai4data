---
id: monitor-use
title: "7. Monitor how your data are used"
sidebar_label: "7. Monitor use"
sidebar_position: 7
description: Dataset mention detection, citation tracking, and downstream-use monitoring for official statistics.
---

# 7. Monitor how your data are used

**Question:** Can we see how our data are used, including by AI systems?

## Why this matters

Download counts show only part of data use. Statistics are cited in
research papers, policy documents, news stories, and, increasingly, in the
answers of AI systems. Knowing where and how they appear helps an office
prioritize datasets, find misuse, and show the value of its work.

## What good looks like

- Datasets have consistent names and identifiers, which makes mentions easier to match.
- The office tracks mentions in publications and reports as well as downloads.
- It checks how AI systems describe its statistics, and corrects the source material when answers are wrong.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Publish a recommended citation and a persistent identifier for each dataset. Keep web and API logs, and separate known crawlers and automated agents from human traffic. |
| **AI-ready** | Search publication sources for dataset names and identifiers on a schedule. Keep a table of dataset name variants and acronyms. Review a sample of results by hand. |
| **AI-native** | Extract dataset mentions from documents with a named-entity model, harmonize variants to canonical identifiers, and report use by dataset, topic, and country. Periodically ask AI systems a fixed set of questions and record how they cite or misquote the office. |

## Implementation options

- **Citation design:** one canonical dataset name, a short acronym, and a persistent identifier reduce ambiguity.
- **Log analysis:** user-agent strings and request patterns identify many crawlers and automated clients. Record them in a separate category.
- **Mention extraction:** zero-shot named-entity models can find dataset mentions in PDF and web text. A matching step maps variants such as an acronym and a full title to one identifier.
- **AI visibility check:** a fixed set of questions run against several assistants, repeated on a schedule, shows whether the office is cited and whether the figures are right. Treat the results as a sample that changes over time.

## World Bank examples

- [Monitoring of Data Use](/docs/data_use/) documents dataset-mention extraction with GLiNER, harmonization of name variants, and structured output.

## How to test it

- **Extraction accuracy:** label a sample of documents by hand and measure precision and recall of the extraction.
- **Harmonization accuracy:** check a sample of matched variants against the canonical identifier.
- **Repeatability:** run the AI visibility check twice in a week and note how much the results differ before drawing conclusions.

## Checklist

- [ ] Canonical dataset names, acronyms, and identifiers listed
- [ ] Recommended citation published
- [ ] Automated clients separated in access logs
- [ ] Publication mentions searched on a schedule
- [ ] Extraction and matching accuracy measured on a labeled sample
- [ ] Fixed question set for checking how AI systems cite the office
