---
id: standards
title: Standards used in this guide
sidebar_position: 90
hide_table_of_contents: true
description: The published standards and datasets each chapter builds on, with the ones the World Bank has adopted marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard for its own catalogs,
the guide uses that one.

## Documents and tables

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [World Bank document schema](https://github.com/worldbank/metadata-schemas) | Record for a document (Dublin Core based) with `data_sources`, `toc_structured`, identifiers, and citation | The Development Data Group's catalogs; the Metadata Editor; NADA | Chapter [1](./inventory.mdx) |
| [World Bank table schema](https://github.com/worldbank/metadata-schemas) | Record for a statistical table: columns, rows, data sources, definitions, licence, citation, relations | The same catalogs and tools | Chapter [5](./publish.mdx) |
| [Dublin Core](https://www.dublincore.org/specifications/dublin-core/) | Core metadata elements for documents | The basis of the document schema | Chapter 1 |
| [Metadata Editor](https://github.com/worldbank/metadata-editor) | Documentation of documents and tables with templates and validation; publishing to NADA | Used by the Development Data Group's curators | Chapters 1 and 5 |
| [NADA](https://nada.ihsn.org/) | Catalog for documents, tables, microdata, and indicators, with schema.org markup, search, API, and MCP interface | Runs the [Microdata Library](https://microdata.worldbank.org/) | Chapters 1, 5, and 6 |

## Layout detection and extraction

| Standard or resource | Use | In this guide |
|---|---|---|
| [Data Snapshots](https://arxiv.org/abs/2606.06242) (Dy and Solatorio) | Benchmark of open-source layout detection models on institutional documents | Chapter [2](./locate.mdx) |
| [data-snapshot dataset](https://huggingface.co/datasets/ai4data/data-snapshot) | 476 annotated regions (Figure, Table) with normalized XYXY bounding boxes, from UNHCR reports, World Bank policy research working papers, and refugee documents; MIT licence | Chapter 2: annotation format and test set |
| [COCO format](https://cocodataset.org/#format-data) | Interchange format for object detection annotations and tools | Chapter 2 |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF text, page rendering, and table extraction; used by the program's document parser | Chapters 1 and 3 |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Open-source OCR for scanned documents | Chapter 1 |

## Files, provenance, and citation

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [CSV on the Web](https://www.w3.org/TR/tabular-data-primer/) | Column types and annotations for tidy files | | Chapters 3 and 5 |
| [W3C PROV](https://www.w3.org/TR/prov-overview/) | Provenance of derived data (`wasDerivedFrom`) | | Chapter 5 |
| [schema.org Dataset](https://schema.org/Dataset) | Markup on table pages, with `isBasedOn` and `distribution` | NADA writes schema.org on catalog pages | Chapter 6 |
| [DataCite](https://schema.datacite.org/) | DOIs for documents and extracted tables | The Microdata Library and Documents and Reports assign DOIs | Chapter 6 |
| [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) | Licence inherited by extracted data | The default for [World Bank datasets](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) and publications | Chapter 5 |
| SDMX `OBS_STATUS` | The model for a status per value (verified, flagged, estimated) | | Chapter 4 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
