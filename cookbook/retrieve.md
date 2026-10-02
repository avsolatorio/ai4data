---
id: retrieve
title: "2. Make your data retrievable"
sidebar_label: "2. Retrieve"
sidebar_position: 2
description: Open formats, stable identifiers, documented APIs, and agent interfaces such as MCP.
---

# 2. Make your data retrievable

**Question:** Can AI retrieve our data reliably?

## Why this matters

An AI system that cannot reach the numbers will use whatever it can reach,
such as an outdated table in a PDF, a copy on a third-party site, or a
remembered value from its training data. Reliable retrieval means the
authoritative values are available in a form that software can request
directly.

Data locked in PDFs, scanned tables, or interfaces that need a person to click
through forms are the most common barrier.

## What good looks like

- Data are available in open, structured formats.
- Every series, dataset, and release has a stable identifier that does not change when a page is redesigned.
- A documented interface lets software request a specific series, country, and period.
- Responses include the values together with their unit, period, and source.

## Maturity levels

| Level | Steps |
|---|---|
| **Foundational** | Publish data as CSV or another open format in addition to Excel and PDF. Use one tidy layout per file, with one header row and consistent codes. Keep download URLs stable. |
| **AI-ready** | Provide a documented REST API with an OpenAPI description. Support filtering by series, geography, and period. Return units, period, and a source URL with the values. Publish a bulk download. State the terms of use and rate limits. |
| **AI-native** | Expose an MCP server so AI assistants can discover and call data tools through a standard protocol. Return provenance with every response. Document which tools are read-only. |

## Implementation options

- **Formats:** CSV with a [CSV on the Web](https://www.w3.org/TR/tabular-data-primer/) or [Frictionless](https://frictionlessdata.io/) table schema; SDMX-CSV or SDMX-JSON; Parquet for large files.
- **APIs:** an SDMX web service for statistical data, or a REST API described by [OpenAPI](https://www.openapis.org/). Many offices already run one of these.
- **Identifiers:** persistent URLs, DOIs, or SDMX structure identifiers. Choose one scheme and keep it unchanged across site redesigns.
- **Agent interface:** the [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard. A single MCP server can serve any compatible AI client. It wraps the API that already exists and does not replace it.
- **Documents:** when data exist only in PDFs, extraction tools can recover tables and figures. Publish the extracted data next to the document, and mark them as extracted.

## World Bank examples

- [Model Context Protocol for AI-centric data dissemination](/docs/mcp/) describes the architecture and the tools an MCP server for official statistics can expose.
- [Data Snapshots](https://arxiv.org/abs/2606.06242) covers layout detection models that locate figures and tables in PDF documents so that the data inside them can be extracted.

## How to test it

- **Direct request:** retrieve a known value with a single API call, written from the documentation alone, by someone who did not build the API.
- **Stability:** request the same identifiers a month later and confirm that they still resolve.
- **Completeness of response:** check that every response includes the unit, the period, and the source.
- **Agent check:** if an MCP server exists, ask an assistant for five known values and compare them with the official figures.

## Checklist

- [ ] Open, tidy file formats available for each dataset
- [ ] Stable identifiers and download URLs
- [ ] Documented API with an OpenAPI or SDMX description
- [ ] Responses include unit, period, and source
- [ ] Bulk download available
- [ ] Terms of use and rate limits published
- [ ] Data in PDFs also published in structured form
- [ ] Agent interface (MCP) evaluated for the highest-demand datasets
