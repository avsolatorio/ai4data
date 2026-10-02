---
id: trust
title: "5. Make answers trustworthy"
sidebar_label: "5. Trust"
sidebar_position: 5
description: Provenance, citations, numeric verification, grounding, and freshness for answers that use official statistics.
---

# 5. Make answers trustworthy

**Question:** Can answers be trusted and traced?

## Why this matters

Official statistics carry the authority of the office that publishes them. When
an AI system quotes a number, readers assume it comes from that office. A
wrong or outdated value, shown with the office's name, damages trust in the
source. Users need to see where a number came from and check it.

## What good looks like

- Every number in an answer links to the series, release, and date it came from.
- Numbers shown to users are checked against the official record before display.
- Answers show the publication date and the latest release, so outdated values are visible.
- When the system is unsure, it says so.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Show the source, release date, and citation text on every dataset page. Provide a suggested citation in a consistent format. Publish a revisions log. |
| **AI-ready** | Return source, identifier, and release date with every API response. Publish citation metadata in machine-readable form. Mark superseded series and point to their replacements. |
| **AI-native** | Check each number in a generated answer against the official record and flag those that do not match. Attach provenance to the answer. Log answers with the retrieved records for audit. |

## Implementation options

- **Provenance vocabulary:** [W3C PROV](https://www.w3.org/TR/prov-overview/) describes the origin and derivation of data. A simpler form is a fixed set of fields: source, identifier, release, retrieval date.
- **Citation:** DOIs and a standard citation string make it easy for systems and people to cite the same way.
- **Grounding:** instruct the model to answer only from retrieved material, and return that material with the answer.
- **Numeric verification:** extract numbers from the generated text, match them to the retrieved values, and mark each one verified or unverified. This catches transcription errors and invented values.
- **Freshness:** include the date of the latest release in API responses, and have the answer interface mention when it is using an older release.

## World Bank examples

- [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) checks each number in a chatbot answer against the official record and marks it as verified or flagged. The home page shows an example.
- [Anomaly Detection and Explanation](/docs/anomaly-detection/) classifies unusual values in a series and cites the evidence, which supports review before publication.

## How to test it

- **Number match rate:** on a test set, count how many numbers in answers match the official values.
- **Citation check:** sample answers and confirm that each citation resolves and supports the statement.
- **Staleness test:** publish a revision in a test environment and confirm that answers reflect it.
- **Adversarial prompts:** ask for numbers the office does not publish and check that the system declines to invent them.

## Checklist

- [ ] Source, release, and citation text on every dataset page
- [ ] Source and release date in every API response
- [ ] Suggested citation published in a machine-readable form
- [ ] Revisions and superseded series documented
- [ ] Numbers in generated answers checked against the record
- [ ] Answers logged with the records they used
- [ ] Declining behavior tested
