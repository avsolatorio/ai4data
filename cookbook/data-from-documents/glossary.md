---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Extracting Data from Documents.
---

# Glossary

**Bounding box.** The rectangle on a page that contains a detected region, given as normalized coordinates (0 to 1) of its top-left and bottom-right corners.

**Data snapshot.** The image of one table or figure cropped from a page, with its coordinates, class, and source document; the unit of extraction and of citation.

**Derived column.** A table column computed from others, such as a rate from a numerator and a denominator; a check for extracted values.

**Document record.** The catalog record of a document in the World Bank document schema, with the series and surveys it draws on.

**Estimated.** The status of an extracted value for which the document offers no check: no printed label, total, or derivation.

**Fine-tuning.** Training an existing model further on the organization's own labelled data so that it follows the organization's conventions.

**Flagged.** The status of an extracted value that failed a check and awaits correction with the page open.

**Inventory.** The list of documents with their type, year, page count, text-layer status, and counts of tables and figures.

**Layout detection.** A model that finds regions of a page image and returns their class and bounding box.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under its published licence.

**Printed value label.** A number printed on a chart next to its bar or point, which makes the extracted value verifiable.

**Table record.** The catalog record of a statistical table in the World Bank table schema, with columns, rows, sources, definitions, citation, and relations.

**Text layer.** The machine-readable text inside a PDF; absent in scanned documents until OCR produces it.

**Tidy file.** A CSV with one row per value and provenance columns (document, page, bounding box, class, title, row, column, value, unit, status).

**Token.** The unit in which language models read and write text, roughly three quarters of a word; model prices and context limits are counted in tokens.

**Total row.** The row of a table that sums the others; a check for extracted values.

**Verified.** The status of an extracted value that matched a printed label, a total, a derivation, or a catalog value.

**Vision-language model.** A model that reads an image and returns text or structured output; used to extract chart data from snapshots.
