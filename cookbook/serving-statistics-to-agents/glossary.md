---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Serving Official Statistics to AI Agents.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task, here an assistant that calls the organization's server to answer a question about statistics.

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Client configuration.** The settings a user pastes into an assistant to connect it to the server: endpoint, transport, credentials.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables; the World Bank microdata schema is based on it.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**Guidance resource.** A resource that tells the model how to use the tools: search first, read metadata before interpreting, cite the source and release.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Manifest.** The design document listing the server's tools, inputs, outputs, examples, and resources.

**MCP.** The Model Context Protocol, an open protocol over JSON-RPC 2.0 through which AI applications discover and call tools, read resources, and use prompts offered by servers.

**OpenAPI.** The standard way to describe an API in a file: which addresses exist, what parameters they take, and what they return.

**Prompt injection.** Text placed in a document or a query that is written as an instruction to the model, in the hope that the model follows it; tested for before release.

**Provenance fields.** The fields every data response carries: series identifier, reference area, unit, release date, source URL, licence, citation, and observation status per value.

**Read-only.** A tool that reads and returns data and has no path to change anything; the property the server's safety rests on.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**SDMX.** The standard for exchanging aggregate statistical data and their structure; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this guide's data files.

**Streamable HTTP.** The transport for servers used over the network; stdio is the transport for servers run locally by a client.

**Structured content.** A tool result returned as JSON that conforms to the tool's output schema, alongside a text rendering.

**Structured output.** Model output returned as data in a declared shape (a JSON object with named fields) so that it can be checked and used by software.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; model prices and context limits are counted in tokens.

**Tool execution error.** A tool result marked as an error with a message the model can act on, as distinct from a protocol error.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.
