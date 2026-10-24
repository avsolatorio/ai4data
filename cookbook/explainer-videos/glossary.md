---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in this guide.
---

# Glossary

**Agent.** An AI system that calls tools to complete a task; here an assistant that calls the organization's data API or MCP server to answer a question about statistics.

**AI-ready.** Data and metadata that software, including AI systems, can find, retrieve, interpret, and cite without manual steps. See the [AI-ready data framework](/docs/ai-ready-framework).

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Candidate.** A model shortlisted for a task, with its licence classified and its card checked, to be run on the suite.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Coding assistant.** An AI agent that reads and writes files and runs commands on a workstation, for example to draft a narration script or to build a web page.

**Concept explainer.** A short narrated video with animated scenes of its own that explains a method, an indicator, or a data release step by step.

**Contact sheet.** One image that holds small frames of a sequence, used to review every beat of a video's stage before the recording and every narration line after it.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables, value labels, and files; the World Bank microdata schema is based on it.

**Decision.** A curator's recorded verdict on a suggestion: accept, edit (with the final text), or reject (with a reason).

**Disclosure control.** The measurement and reduction of the risk that a released file identifies a person or business or reveals something about one: removal of identifiers, top-coding, coarsening of geography, suppression of rare combinations; applied before any release.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Fact sheet.** The table of every figure, definition, date, and claim a video will state, each with its source, the place in the source, and the person who checked it.

**Feature.** A column a model may use as input.

**Gap.** The distance between a dimension's current level and its target, measured in levels and in the number of questions below the target's value.

**Issue.** The review pipeline's output: a detected problem with a category, a severity from 1 to 5, the current value, and a proposed value.

**JSON-LD.** JSON with a vocabulary attached, so that each key has a defined meaning that different tools read the same way; the format in which schema.org and Croissant records are embedded in web pages.

**Label.** The column a model is trained to predict: the occupation code, the transcribed value, the series a question refers to.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Loudness.** The perceived level of a sound, measured in loudness units relative to full scale (LUFS) by the method of ITU-R BS.1770; videos normalized to the same integrated loudness play at the same perceived level.

**Map.** The table of model-assisted tasks with phase, role, review, data sensitivity, model location, and owner.

**Maturity level.** One of three states used in this guide: foundational, AI-ready, AI-native.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Metadata Editor.** The World Bank's open-source application for documenting data of all the types in the World Bank schemas, with templates and validation, publishing to NADA, and export to SDMX and, for some data types, schema.org, Croissant, and DCAT.

**Production guide.** An organization's written standard for its videos: the reference video, the theme, the scene sequence, the narration rules, the voice and loudness settings, the review checklist, and the history of review decisions.

**RAG.** Retrieval-augmented generation. A pattern where the system retrieves records first and the model writes the answer from them.

**Resource.** Context a server offers to the model or the user, such as code lists or a release calendar, identified by a URI.

**Roles.** What a model may do in production: flag, propose, code, explain, draft. It does not decide.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**schema.org Dataset.** A vocabulary for describing datasets on web pages, read by general crawlers and dataset search engines.

**schema.org VideoObject.** The schema.org type for describing a video on a web page, with properties for its duration, upload date, addresses, captions, transcript, sources, and licence.

**Sign-off.** The recorded decision of a named person that a video may be published, with the version of the files and the checks the decision rests on.

**Site walkthrough.** A narrated video that shows how to use a feature of a website, recorded from the live site with a pointer and captions.

**Split.** The division of a dataset into a training part (the model learns from it), a validation part (the model builder chooses settings on it), and a test part (touched once, to report a score).

**Spoken form.** The text of a narration line as the voice is to read it, with figures, years, and abbreviations written out in words; kept beside the caption text, which keeps the figures in digits.

**Stage page.** A web page made for recording an explainer video, with one section per scene at the size of the video frame and one animation hook per beat.

**Step script.** A file with one entry per beat of a video: the instruction that changes the screen, the caption text, the spoken form where it differs, and the timing.

**Suite.** A question set, the scripts that score it, the required scores, and the record of runs.

**Synthetic data.** Records generated by a model fitted to real data so that they have the same shape and similar relationships while containing no real person's record.

**Text-to-speech model.** A model that turns written text into speech; an open one runs on the organization's own workstation under a published licence.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; prices, context limits, and throughput are counted in tokens.

**Tool use.** The ability of a language model to call functions or APIs during an answer, so that values come from the source.

**Trace.** The logged record of one question: tools called, series used, answer, whether the agent declined, latency.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Video beat.** One change on the screen of an explainer video, with the one or two sentences of narration that describe it. A scene has one or more beats, and the step script has one step per beat.

**Video captions.** The text of a video's audio, timed to it and shown on request by the player, published as a caption file in a format such as WebVTT.

**Vocabulary.** A concept scheme with preferred labels, alternates, and URIs, to which free-text keywords and topics are mapped.

**WebVTT.** The W3C text format for captions and subtitles on the web, in which each cue has a start time, an end time, and one or more lines of text.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.

**Weight (survey).** The number of population members a record stands for in a survey, kept as a column for population estimates; not a training weight by default.

**World Bank metadata schemas.** JSON Schema definitions published by the Development Data Group for indicators, microdata, documents, geospatial data, tables, images, scripts, and videos, used by its catalogs and the Metadata Editor.
