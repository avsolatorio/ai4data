---
id: find
title: "1. Make your statistics findable"
sidebar_label: "1. Find"
sidebar_position: 1
description: Metadata completeness, machine-readable catalogs, and semantic search so that people and AI systems find the right statistics.
---

# 1. Make your statistics findable

**Question:** Can people and AI find our statistics?

## Why this matters

AI systems increasingly mediate how users discover data. A search assistant
looks at page titles, descriptions, and catalog records. A series with a vague
title, an empty description, or no machine-readable record may never appear in
the answer, even when it is the right series.

Users also describe what they need in their own words. "How many people go
hungry" and "prevalence of undernourishment" refer to the same topic, and a
keyword index treats them as unrelated.

## What good looks like

- Every dataset and indicator has a stable identifier and a complete record: title, description, unit, time coverage, geography, frequency, source, and contact.
- The catalog is published in a machine-readable form that crawlers and harvesters can read.
- Search finds results by meaning as well as by exact words, in the languages users use.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Fill in the description, unit, and coverage fields for every published series. Give each dataset its own page with a permanent URL. Publish an XML sitemap and a clear `robots.txt`. |
| **AI-ready** | Publish catalog records as DCAT or schema.org `Dataset` markup (JSON-LD) on each dataset page. Expose a catalog API or a harvestable feed. Add translated titles and descriptions for the main national and user languages. |
| **AI-native** | Add semantic search over the catalog using embeddings, combined with keyword search. Use AI to find and repair missing or inconsistent metadata, with human review. Offer the catalog to AI assistants through an agent interface (see [chapter 2](./retrieve.md)). |

## Implementation options

**Machine-readable catalog metadata**

- [DCAT](https://www.w3.org/TR/vocab-dcat-3/) (W3C) describes catalogs, datasets, and distributions. It is the base for many open data portals.
- [schema.org `Dataset`](https://schema.org/Dataset) markup in JSON-LD on dataset pages is read by general web crawlers and dataset search engines.
- [DDI](https://ddialliance.org/) describes surveys and microdata. [SDMX](https://sdmx.org/) describes aggregate statistical data and its structure.

**Search**

- Keyword search (BM25) remains valuable for exact codes, acronyms, and names.
- Dense retrieval with embedding models matches by meaning. Combining both usually gives the best results. Small open embedding models run on modest hardware.
- Multilingual embedding models support queries in languages other than the one the metadata is written in. Translated metadata still improves results.

**Metadata quality**

- Start with a completeness report: which required fields are empty, and for how many records.
- Use an LLM to draft missing descriptions or flag inconsistencies, and keep a person responsible for approval.

## World Bank examples

- [Generative AI for Metadata Quality](/docs/metadata-quality/generative-ai-for-metadata-quality) scores indicator metadata on completeness, semantic alignment, specificity, and consistency.
- [Metadata Augmentation](/docs/metadata-augmentation/) groups survey variables into DDI-style themes.
- [Data Discoverability](/docs/data-discoverability/) describes semantic search over indicator catalogs, and the [Pipeline Guide](/pift-toolkit/pipeline) shows how to fine-tune an embedding model for structured metadata records.

## How to test it

- **Completeness:** the share of records with every required field filled. Track it per field.
- **Known-item queries:** write 30 to 50 queries for which you know the correct series, using the words users use, which often differ from the official titles. Check whether the correct series appears in the top 5 and top 10.
- **Crawler view:** open a dataset page with a structured-data validator and confirm that the record is parsed.
- **Language check:** repeat the known-item queries in each supported language.

[Chapter 6](./evaluate.md) describes the measures in more detail.

## Checklist

- [ ] Stable identifier and permanent URL for each dataset
- [ ] Description, unit, coverage, and source populated
- [ ] Machine-readable catalog record (DCAT or schema.org) published
- [ ] Sitemap and crawler rules reviewed
- [ ] Metadata completeness measured per field
- [ ] Known-item search queries written and run
- [ ] Translated metadata for priority languages
- [ ] Semantic retrieval tested against the keyword baseline
