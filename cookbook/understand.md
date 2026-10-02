---
id: understand
title: "3. Make the meaning of the numbers clear"
sidebar_label: "3. Understand"
sidebar_position: 3
description: Units, concepts, classifications, methodological notes, and the standards that make them machine-readable.
---

# 3. Make the meaning of the numbers clear

**Question:** Can AI understand what the numbers mean?

## Why this matters

A number carries meaning through its unit, reference period, population,
definition, and method. An AI system that retrieves "12.4" without these may
describe it as a percentage when it is a rate per 1,000, or combine two series
that use different definitions.

## What good looks like

- Each series states its unit, frequency, reference period, geographic coverage, and population.
- Definitions and methodological notes are written once, linked to the series, and available in structured form.
- Codes for countries, time periods, sex, age groups, and activities come from published code lists and classifications.
- Breaks in a series and revisions are recorded.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Write a plain-language definition, a unit, and a method note for each indicator. Use standard country and date codes. Add a "last revised" date. |
| **AI-ready** | Publish structure and code lists in SDMX (aggregate data) or DDI (surveys and microdata). Use standard classifications and publish the mapping from national codes. Record series breaks and revisions in fields as well as in footnotes. |
| **AI-native** | Link concepts to shared vocabularies or an ontology, so that systems can relate "poverty headcount" in one catalog to the same concept elsewhere. Use AI to propose concept links and variable groupings, with expert review. |

## Implementation options

- **SDMX** provides data structure definitions, code lists, and concept schemes for aggregate statistics.
- **DDI** documents variables, questions, and value labels for surveys and administrative microdata.
- **Classifications:** ISO 3166 for countries, ISO 8601 for dates, and the relevant international statistical classifications (for example ISIC, ISCO, COICOP, or ISCED) with crosswalks to national versions.
- **Controlled vocabularies and ontologies:** SKOS concept schemes can publish a thesaurus of statistical concepts with multilingual labels.
- **Documentation text:** a short, consistent template for definition, method, limitations, and comparability makes both human reading and machine extraction easier.

## World Bank examples

- [Metadata Augmentation](/docs/metadata-augmentation/) generates DDI-style variable groups from data dictionaries, with human review.
- [Generative AI for Metadata Quality](/docs/metadata-quality/generative-ai-for-metadata-quality) checks that an indicator's name and definition describe the same concept.
- The [AI-ready data framework](/docs/ai-ready-framework) lists the attributes, including standards and semantic knowledge, that make data interpretable.

## How to test it

- **Interpretation questions:** for 20 series, ask an assistant for the unit, the reference period, and the definition. Compare the answers with the official metadata.
- **Comparability questions:** ask whether two series can be compared, where you know the answer, for example because of a definition change.
- **Code audit:** list every code used in the data and confirm that each appears in a published code list.

## Checklist

- [ ] Unit, frequency, and reference period stated for each series
- [ ] Definition and method note for each indicator
- [ ] Standard codes for geography and time
- [ ] Code lists and classifications published, with national crosswalks
- [ ] Series breaks and revisions recorded as fields
- [ ] Structure published in SDMX or DDI
- [ ] Concept links to shared vocabularies considered
