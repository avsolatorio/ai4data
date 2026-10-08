---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Serving Official Statistics to AI Agents.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task, here an assistant that calls the organization's server to answer a question about statistics.

**Annotation.** Optional properties on a tool that describe its behaviour (for example that it is read-only); hints that clients treat as untrusted unless the server is trusted.

**Client configuration.** The settings a user pastes into an assistant to connect it to the server: endpoint, transport, credentials.

**Guidance resource.** A resource that tells the model how to use the tools: search first, read metadata before interpreting, cite the source and release.

**Harness.** A program that sends questions to an agent connected to the server and logs the traces.

**Manifest.** The design document listing the server's tools, inputs, outputs, examples, and resources.

**MCP.** The Model Context Protocol, an open protocol over JSON-RPC 2.0 through which AI applications discover and call tools, read resources, and use prompts offered by servers.

**Provenance fields.** The fields every data response carries: series identifier, reference area, unit, release date, source URL, licence, citation, and observation status per value.

**Read-only.** A tool that reads and returns data and has no path to change anything; the property the server's safety rests on.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Streamable HTTP.** The transport for servers used over the network; stdio is the transport for servers run locally by a client.

**Structured content.** A tool result returned as JSON that conforms to the tool's output schema, alongside a text rendering.

**Tool execution error.** A tool result marked as an error with a message the model can act on, as distinct from a protocol error.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.
