---
id: sustain
title: "9. Keep it sustainable"
sidebar_label: "9. Sustain"
sidebar_position: 9
description: Architecture choices, open standards, small and open models, cost, skills, and lifecycle management.
---

# 9. Keep it sustainable

**Question:** Can we maintain what we build?

## Why this matters

Pilots are easy to start and costly to keep running. Models change, vendors
change prices, and staff move on. An AI-ready dissemination system that depends
on one person or one provider can stop working within a year of launch.

## What good looks like

- The system is built from open standards and replaceable parts.
- Running costs are known and proportionate to the number of users.
- More than one person can operate and update each component.
- The metadata and data pipeline keeps working if the AI layer is switched off.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Document how data and metadata are produced and published. Assign backups for each task. Improve metadata and file formats first, since these keep their value regardless of tools. |
| **AI-ready** | Use open standards for catalog, structure, and API (chapters 1 to 3). Keep code and configuration in version control. Schedule reviews of metadata and the test suite. Estimate the cost per query before launch. |
| **AI-native** | Use small and open models where they reach the needed quality, and larger models where the task requires them. Monitor cost, latency, and quality together. Keep shared components with partner offices, and contribute fixes back. |

## Implementation options

- **Architecture:** keep the AI layer separate from the authoritative data store and API. If the AI layer is removed, the catalog and API continue to work.
- **Model size:** classification, tagging, and embedding tasks often work with small models. Open-weight models can run on one modest server. Test small models against the evaluation suite before choosing a larger one.
- **Cost control:** cache repeated queries, batch non-urgent work such as metadata review, and set usage limits.
- **Skills:** a small team that combines a metadata specialist, a data engineer, and someone responsible for evaluation covers most of the work in this guide. Training material can be shared across offices.
- **Lifecycle:** plan for model updates, index rebuilds, and the retirement of components. Re-run the evaluation suite after each.
- **Shared resources:** open-source tools and shared test sets let offices reuse each other's work.

## World Bank examples

- [Efficient and Inclusive AI Applications](/docs/inclusive-ai/) compares model sizes by task and describes batch processing and local models.
- The program publishes its methods and software as open resources in the [GitHub repository](https://github.com/worldbank/ai4data).
- The [PI-FT Deployment guide](/pift-toolkit/deployment) describes serving and refreshing a fine-tuned embedding model.

## How to test it

- **Switch-off test:** disable the AI layer in a test environment and confirm that search, catalog, and API still work.
- **Handover test:** ask a staff member who did not build a component to run an update from the documentation.
- **Cost review:** compare monthly cost with the number of queries served, and review quarterly.

## Checklist

- [ ] Production process documented, with backups for each task
- [ ] Open standards used for catalog, structure, and API
- [ ] Code and configuration in version control
- [ ] AI layer separable from the data store and API
- [ ] Cost per query estimated and monitored
- [ ] Small models compared against larger models on the test suite
- [ ] Review dates set for metadata, models, and the test suite
