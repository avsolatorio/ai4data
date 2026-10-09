---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Extracting Data from Documents.
---

# Glossary

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Bounding box.** The rectangle on a page that contains a detected region, given as normalized coordinates (0 to 1) of its top-left and bottom-right corners.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**DDI.** The Data Documentation Initiative, the standard for documenting surveys and their variables; the World Bank microdata schema is based on it.

**Derived column.** A table column computed from others, such as a rate from a numerator and a denominator; a check for extracted values.

**Document record.** The catalog record of a document in the World Bank document schema, with the series and surveys it draws on.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or a version of it and makes citations countable.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Fine-tuning.** Training an existing model further on the organization's own labelled data so that it follows the organization's conventions.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**GPU.** A graphics processing unit, the processor on which language models run fast; its memory decides which model fits.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Layout detection.** A model that finds regions of a page image and returns their class and bounding box.

**Manifest.** A file that lists the parts of a job or a release: for a job, the records and their status, so that a run can resume; for a release, every file with its checksum.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under its published licence.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Printed value label.** A number printed on a chart next to its bar or point, which makes the extracted value verifiable.

**Provenance.** The record of where a value came from: the series, the release, the document and page, the method, and the version, carried with the value.

**Rubric.** A written scale that says what each score means, so that two people, or a person and a model, score the same thing the same way.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**SDMX.** The standard for exchanging aggregate statistical data and their structure; its concept names (SERIES, REF_AREA, TIME_PERIOD, OBS_VALUE) are the column names of this guide's data files.

**Structured output.** A model's answer in a fixed, machine-readable form (a JSON object with named fields) that a script can validate and read.

**Table record.** The catalog record of a statistical table in the World Bank table schema, with columns, rows, sources, definitions, citation, and relations.

**Text layer.** The machine-readable text inside a PDF; absent in scanned documents until OCR produces it.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Tidy file.** A CSV with one row per value and provenance columns (document, page, bounding box, class, title, row, column, value, unit, status).

**Token.** The unit in which language models read and write text, roughly three quarters of a word; model prices and context limits are counted in tokens.

**Total row.** The row of a table that sums the others; a check for extracted values.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vision-language model.** A model that reads an image and returns text or structured output; used to extract chart data from snapshots.
