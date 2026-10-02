---
id: ask
title: "4. Let users ask questions naturally"
sidebar_label: "4. Ask"
sidebar_position: 4
description: Natural-language search, retrieval-augmented generation, conversational interfaces, and multilingual access.
---

# 4. Let users ask questions naturally

**Question:** Can users ask questions in their own words?

## Why this matters

Most users do not know the official names of indicators or the structure of a
catalog. They type a question. Natural-language interfaces reduce the effort of
finding and reading statistics, and they are the form in which many users
already meet AI systems.

This chapter comes after findability, retrieval, and meaning because a
conversational interface depends on all three. A chatbot placed over incomplete
metadata produces confident answers from poor inputs.

## What good looks like

- A question returns the relevant series, with the unit, period, and source, in a few seconds.
- The system says when it has no matching data.
- Users can ask in the languages they use.
- The system's role is clear: it finds and explains published statistics and does not create new ones.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Provide search with filters by topic, country, and period. Add example queries on the search page. Review the logged search terms that returned no results. |
| **AI-ready** | Add semantic search that matches by meaning (chapter 1). Return structured answers that link to the series pages. Support the main user languages for queries. |
| **AI-native** | Offer a conversational interface that uses retrieval-augmented generation: it retrieves records and values first, and the language model writes the answer from them. Answers carry citations and numeric checks (chapter 5). |

## Implementation options

- **Search first:** a good semantic search page already answers many questions and carries lower risk than free-text generation.
- **Retrieval-augmented generation (RAG):** retrieve metadata and values from the authoritative source, then ask the model to answer only from what was retrieved. Return the retrieved records with the answer.
- **Tool use:** let the model call the data API or an MCP server for values, so that numbers come from the source and not from the model's memory.
- **Multilingual access:** use multilingual embedding models for queries, translate metadata for the priority languages, and test each language separately.
- **Model choice:** small and open models can handle routing and rephrasing tasks. See [chapter 9](./sustain.md).

## World Bank examples

- The [home page examples](/) show a multilingual search over indicator metadata and an assistant that answers through an MCP connection.
- [Efficient and Inclusive AI Applications](/docs/inclusive-ai/) discusses model choices and language coverage for low-resource settings.

## How to test it

- **Question set:** collect real questions from help-desk logs and search logs, and add questions in each supported language.
- **No-answer cases:** include questions the office cannot answer, and check that the system declines.
- **Review sample:** have subject-matter staff review a random sample of answers each month against a short rubric: correct series, correct numbers, correct period, correct caveats.

## Checklist

- [ ] Search logs reviewed for failed queries
- [ ] Semantic search in place before any chat interface
- [ ] Answers generated only from retrieved records and values
- [ ] The interface declines when no matching data exist
- [ ] Languages tested one by one
- [ ] Question set built from real user questions
- [ ] Monthly human review of a sample of answers
