---
id: glossary
title: Glossary
sidebar_position: 12
description: Terms used in the Practical Guide to AI-Ready Data Dissemination.
---

# Glossary

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**Agent.** An AI system that can call tools, such as a data API, to complete a task. See MCP.

**BM25.** A keyword ranking function used by most search engines. It matches exact words and weights rare ones more.

**Citation validity.** The share of citations in generated answers that resolve to a real record and support the statement they are attached to.

**Code list.** A published table of the codes used in a dataset (for geography, periods, categories) with their labels. In SDMX, a code list is part of the data structure definition.

**Content-Oriented Guidelines.** The SDMX guidelines that define cross-domain concepts (`REF_AREA`, `TIME_PERIOD`, `OBS_VALUE`, `OBS_STATUS`, `UNIT_MEASURE`, and others) and cross-domain code lists for reuse across statistical domains.

**COUNTER Code of Practice for Research Data.** Rules for logging and reporting dataset views and downloads, including the separation of machine access from regular access.

**Croissant.** An MLCommons extension of schema.org for describing machine-learning datasets, including how to load them.

**Crosswalk.** A mapping from one classification to another, for example from a national occupation classification to ISCO.

**DataCite.** The registration agency and metadata schema for dataset DOIs. The World Bank Microdata Library assigns DataCite DOIs.

**DCAT.** Data Catalog Vocabulary, a W3C standard for describing datasets and catalogs in machine-readable form.

**DOI.** Digital Object Identifier, a persistent identifier that resolves to the dataset's page and carries citation metadata.

**DDI.** Data Documentation Initiative, a standard for documenting surveys and microdata, including variables and value labels.

**Dense retrieval.** Search that represents queries and records as numeric vectors (embeddings) and ranks by similarity of meaning.

**Embedding.** A numeric vector that represents the meaning of a text, produced by an embedding model.

**Grounding.** Restricting a language model to answer only from material retrieved for the question.

**Hallucination.** A statement in a generated answer that has no support in the retrieved material.

**JSON-LD.** A JSON format for linked data, used to embed schema.org records in web pages.

**Known-item question.** A test question for which the correct record is known in advance, used to score search.

**LLM.** Large language model. A model that generates text, used here for drafting metadata, answering questions, and grading.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**MCP.** Model Context Protocol, an open standard that lets AI clients discover and call tools exposed by a server, such as a statistics API.

**MRR.** Mean reciprocal rank. The average of 1 divided by the rank of the first correct result, across a question set.

**NADA.** The open-source data catalog from the International Household Survey Network, used for the World Bank Microdata Library. It supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC.

**OBS_STATUS.** The SDMX observation status attribute, with codes such as `A` normal, `P` provisional, `B` break in series, `E` estimated.

**nDCG@k.** Normalized discounted cumulative gain at k. A retrieval measure that credits partially relevant results and rewards placing the best ones first.

**OpenAPI.** A standard, machine-readable description of a REST API.

**Open-weight model.** A model whose weights are published, so that it can be run on the office's own hardware.

**Prompt injection.** Instructions hidden in content a system reads (a document, a web page, a user message) that try to change the system's behaviour.

**Provenance.** The record of where a value came from: series, release, source, and when it was retrieved.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Recall@k.** The share of questions for which the correct record appears in the top k results.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**SDMX.** Statistical Data and Metadata eXchange, a standard for aggregate statistical data, their structure, and code lists.

**Semantic search.** Search by meaning; see dense retrieval.

**Stable identifier.** An identifier for a series or dataset that does not change across releases or site redesigns.

**Statistical disclosure control.** Methods that prevent the identification of individuals from published data.

**Tidy data.** A table layout with one observation per row and one variable per column.

**Tool use.** The ability of a language model to call functions or APIs during an answer, so that values come from the source.

**World Bank metadata schemas.** JSON Schema definitions published by the Development Data Group for indicators, microdata, documents, geospatial data, tables, images, scripts, and videos, used by its catalogs and the Metadata Editor.

**XKOS.** The DDI Alliance extension of SKOS for statistical classifications and the correspondences between them.
