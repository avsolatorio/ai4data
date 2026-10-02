---
id: index
title: Practical Guide to AI-Ready Data Dissemination
sidebar_label: Overview
sidebar_position: 0
description: A cookbook for national statistical offices that want their published statistics to be found, understood, and cited correctly by people and AI systems.
---

# Practical Guide to AI-Ready Data Dissemination

**A cookbook for national statistical offices**

:::note Working draft
This guide is a working draft from the AI for Data – Data for AI program. It
collects practical approaches and open standards. It is not an international
guideline. Comments and corrections are welcome as [GitHub issues](https://github.com/worldbank/ai4data/issues).
:::

People increasingly reach statistics through AI systems: search assistants,
chatbots, and code or analysis tools. These systems read the metadata, web
pages, and APIs that a statistical office publishes. When those are
incomplete, ambiguous, or hard to retrieve, the systems return the wrong
series, mix up definitions, or quote numbers without a source.

This guide is organized around the questions a statistical office can ask about its own
dissemination system. Each chapter explains why the question matters, what a
good answer looks like, how to reach it in steps, and how to test the result.

:::tip AI readiness does not require generative AI
Many of the highest-value steps are ordinary data practice: complete
metadata, open structured data, stable identifiers, and documented APIs. These
steps help human users and conventional search as well. A chatbot is an
optional later layer.
:::

## The questions

| Chapter | Question for a statistical office | Practical topics |
|---|---|---|
| [1. Find](./find.md) | Can people and AI find our statistics? | Metadata completeness, indexing, DCAT and schema.org, semantic search, multilingual metadata |
| [2. Retrieve](./retrieve.md) | Can AI retrieve our data reliably? | Open formats, APIs, stable identifiers, machine-readable access, MCP |
| [3. Understand](./understand.md) | Can AI understand what the numbers mean? | Units, concepts, classifications, methodological notes, SDMX, DDI, ontologies |
| [4. Ask](./ask.md) | Can users ask questions naturally? | Natural-language search, retrieval-augmented generation, conversational interfaces, multilingual access |
| [5. Trust](./trust.md) | Can answers be trusted? | Provenance, citation, numeric verification, grounding, freshness |
| [6. Evaluate](./evaluate.md) | Can we tell whether it works? | Retrieval benchmarks, hallucination tests, answer accuracy, multilingual evaluation |
| [7. Monitor use](./monitor-use.md) | Can we see how our data are used? | Dataset mention detection, citation tracking, downstream-use monitoring |
| [8. Govern](./govern.md) | Can we operate this responsibly? | Governance, human oversight, security, privacy, model choice, vendor dependence |
| [9. Sustain](./sustain.md) | Can we maintain it? | Architecture, open standards, small and open models, cost, skills, lifecycle |

These questions follow the same path as the program's overview of the
[AI-ready data framework](/docs/ai-ready-framework): find the right data,
retrieve it with context, interpret it, and verify the result.

## Maturity levels

Each chapter describes three levels. An office can sit at different levels in
different chapters, and the first level is a complete, useful state on its
own.

| Level | Meaning |
|---|---|
| **Foundational** | Data and metadata are published in open, documented formats that a person can download and that a crawler can read. |
| **AI-ready** | Data and metadata are structured, identified, and accessible through documented interfaces, so that software, including AI systems, can use them without manual steps. |
| **AI-native** | The office offers AI-oriented interfaces such as semantic search or an agent protocol, with provenance and verification built in. |

For access, as an example:

- **Foundational:** downloadable CSV or Excel files with a metadata page.
- **AI-ready:** a documented REST API, stable identifiers, and structured metadata.
- **AI-native:** semantic discovery, a standard agent interface such as MCP, and provenance on every answer.

## How each chapter is organized

1. **Why this matters.** The failure the chapter addresses.
2. **What good looks like.** The target state.
3. **Maturity levels.** Concrete steps at each level.
4. **Implementation options.** Open standards and patterns, with alternatives.
5. **World Bank examples.** Where the program has built or documented the approach.
6. **How to test it.** Checks that show whether the step worked.
7. **Checklist.** A short list to track progress.

## Where to start

- **Little technical capacity.** Begin with chapters [1](./find.md), [2](./retrieve.md), and [3](./understand.md) at the foundational level. Complete metadata and open formats come first.
- **An existing data portal or API.** Add machine-readable catalog metadata and stable identifiers (chapters 1 and 2), then run the tests in [chapter 6](./evaluate.md) against a few real questions.
- **Plans for a chatbot or assistant.** Read chapters [4](./ask.md), [5](./trust.md), and [8](./govern.md) before choosing a product.

To assess where an office stands before planning, see the program's
[AI-readiness assessment framework](/ai-readiness-assessment).

## How this guide relates to other resources

| Resource | Purpose |
|---|---|
| This guide | What a statistical office can do, and why, in a sequence of practical steps. |
| [Documentation](/docs/introduction) | How each program method and tool works. |
| [Research](/docs/data-discoverability/) | The evidence and methods behind the approaches. |
| [Code](https://github.com/worldbank/ai4data) | Open-source software to deploy or adapt. |
