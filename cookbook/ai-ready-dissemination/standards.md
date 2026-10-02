---
id: standards
title: Standards used in this guide
sidebar_position: 11
hide_table_of_contents: true
description: The published standards each chapter builds on, with the ones the World Bank has adopted marked, and where each is used in the guide.
---

# Standards used in this guide

The recipes use published standards wherever one exists. Where the World
Bank's Development Data Group has adopted a standard for its own catalogs and
APIs, the guide uses that one, so that an office following the guide ends up
interoperable with the World Bank and with the other organizations that use
the same standards.

The running example reflects this: the catalog records use the field names of
the World Bank indicator metadata schema, and the values file uses SDMX
cross-domain concept names.

## Catalog records and discovery

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [World Bank metadata schemas](https://github.com/worldbank/metadata-schemas) | JSON Schema definitions for indicators, indicator databases, microdata, documents, geospatial data, tables, images, scripts, and videos, with a Python library and Excel templates (MIT licence) | Used by the Development Data Group's catalogs and the Metadata Editor; the microdata schema is based on DDI Codebook, documents on Dublin Core, geospatial on ISO 19115/19139, images on IPTC, with DataCite and provenance blocks | Chapters [1](./find.mdx) and [3](./understand.mdx): the record template and the completeness report |
| [NADA](https://nada.ihsn.org/) | Open-source data catalog from the International Household Survey Network | Runs the World Bank [Microdata Library](https://microdata.worldbank.org/); supports DDI Codebook, Dublin Core, ISO 19115/19139, and IPTC | Chapter 1: an implementation option for the catalog |
| [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/) | W3C vocabulary for catalogs, datasets, and distributions | Common in open data portals that harvest from each other | Chapter 1: machine-readable catalog |
| [schema.org Dataset](https://schema.org/Dataset) | Dataset markup on web pages, read by general crawlers and dataset search engines | | Chapter 1, recipe 1.2 |
| [Croissant 1.1](https://mlcommons.org/croissant/) | MLCommons extension of schema.org for machine-learning datasets: contents, provenance, usage restrictions, and how to load the data | | Chapter 1: AI-native option for datasets meant for model training |

## Aggregate statistical data

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [SDMX 3.1](https://sdmx.org/) | Data structure definitions, code lists, concept schemes, and the SDMX-ML, SDMX-JSON, and SDMX-CSV formats, with a REST API specification and the VTL validation language | The [World Development Indicators are available through an SDMX 2.1 REST API](https://datahelpdesk.worldbank.org/knowledgebase/articles/1886701-sdmx-api-queries) | Chapters [2](./retrieve.mdx) and [3](./understand.mdx): file layout, API, code lists |
| [SDMX Content-Oriented Guidelines](https://sdmx.org/guidelines/) | Cross-domain concepts (`REF_AREA`, `TIME_PERIOD`, `OBS_VALUE`, `OBS_STATUS`, `UNIT_MEASURE`, and others) and cross-domain code lists | | The values file in the running example and the API responses |
| [SDG global DSD](https://unstats.un.org/sdgs/iaeg-sdgs/sdmx-working-group/) | The SDMX data structure definition for the SDG indicators, maintained by the IAEG-SDGs SDMX working group | | Chapter 3: a published DSD to reuse for SDG reporting |

## Microdata

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [DDI Codebook](https://ddialliance.org/ddi-codebook) | Documentation of a survey or administrative dataset: study, files, variables, value labels | The Microdata Library and NADA; the World Bank microdata schema is based on it | Chapter 3: variables and value codes |
| [DDI Lifecycle](https://ddialliance.org/ddi-lifecycle) | Documentation across the data lifecycle, with reusable questions and concepts | | Chapter 3: for offices that manage questionnaires and concepts centrally |
| [DDI-CDI](https://ddialliance.org/ddi-cdi) | Cross-domain integration: describes data structures and provenance across data types | | Chapter 5: provenance for derived data |
| [XKOS](https://ddialliance.org/xkos) | Extension of SKOS for statistical classifications and correspondences between them | | Chapter 3, recipe 3.2: classifications and crosswalks |

## Files and APIs

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [SDMX-CSV](https://sdmx.org/) | CSV layout for SDMX data, one observation per row with dimension and attribute columns | | Chapter 2, recipe 2.1 |
| [CSV on the Web](https://www.w3.org/TR/tabular-data-primer/) and [Frictionless Table Schema](https://specs.frictionlessdata.io/table-schema/) | Column types and constraints for CSV files that are not SDMX | | Chapter 2, recipe 2.1 |
| [OpenAPI 3](https://www.openapis.org/) | Machine-readable description of a REST API | The [Data360 API](https://data360.worldbank.org/en/api) and the metadata schemas repository publish OpenAPI descriptions | Chapter 2, recipe 2.2, for APIs that are not SDMX |
| [Model Context Protocol](https://modelcontextprotocol.io/) | Open standard for AI clients to discover and call tools exposed by a server | The [Data360 MCP server](https://github.com/worldbank/data360-mcp) exposes search, metadata, data, and analysis tools | Chapter 2, recipe 2.3 |

## Identifiers, licences, and provenance

| Standard | Use | World Bank use | In this guide |
|---|---|---|---|
| [DataCite metadata schema](https://schema.datacite.org/) and DOIs | Persistent identifiers and citation metadata for datasets | The Microdata Library assigns DataCite DOIs (prefix 10.48529); the World Bank indicator schema has a `datacite` block | Chapter 5, recipe 5.1 |
| [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) | Open licence with attribution | The default licence for [World Bank datasets](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) | Chapters 1 and 5: licence in records and API responses |
| [W3C PROV](https://www.w3.org/TR/prov-overview/) | Vocabulary for the origin and derivation of data | | Chapter 5: provenance of derived values |
| [OAI provenance](https://www.openarchives.org/OAI/2.0/guidelines-provenance.htm) | Provenance of harvested metadata records | The `provenance` block of the World Bank schemas | Chapter 5: records harvested between catalogs |
| SDMX `OBS_STATUS` code list | Observation status codes (normal, provisional, break, estimated, and others) | | Chapters 2 and 3: status as data |

## Classifications and codes

| Standard | Use | In this guide |
|---|---|---|
| [ISO 3166](https://www.iso.org/iso-3166-country-codes.html) | Country codes | Chapter 3 |
| [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) | Dates and periods | Chapters 2 and 3 |
| [ISO 639](https://www.iso.org/iso-639-language-code) | Language codes for multilingual metadata | Chapters 1 and 6 |
| [ISIC](https://unstats.un.org/unsd/classifications/Econ/isic), [ISCO](https://www.ilo.org/public/english/bureau/stat/isco/), [COICOP](https://unstats.un.org/unsd/classifications/Econ/coicop), [ISCED](https://uis.unesco.org/en/topic/international-standard-classification-education-isced), [CPC](https://unstats.un.org/unsd/classifications/Econ/cpc) | International statistical classifications | Chapter 3, recipe 3.2 |
| [SKOS](https://www.w3.org/TR/skos-reference/) | Concept schemes and thesauri with multilingual labels | Chapter 3 |

## Usage metrics

| Standard | Use | In this guide |
|---|---|---|
| [COUNTER Code of Practice for Research Data](https://coprd.countermetrics.org/) | Rules for logging and reporting dataset views and downloads, including the separation of machine access from regular access, developed with [Make Data Count](https://makedatacount.org/) | Chapter [7](./monitor-use.mdx) |
| DataCite citations | Dataset citations collected through DOIs | Chapter 7 |

## Process, quality, and governance

| Standard | Use | In this guide |
|---|---|---|
| [GSBPM](https://unece.org/statistics/modernstats/gsbpm) | Generic Statistical Business Process Model; the program's [GSBPM mapping](/docs/gsbpm-mapping) places each workstream in it | Chapter [9](./sustain.mdx) |
| [GSIM](https://unece.org/statistics/modernstats/gsim) | Generic Statistical Information Model, the information objects behind GSBPM | Chapter 3 |
| [UN Fundamental Principles of Official Statistics](https://unstats.un.org/fpos/) | The principles under which statistical offices operate | Chapter [8](./govern.mdx) |
| [ISO/IEC 42001](https://www.iso.org/standard/42001) | Management system standard for AI | Chapter 8: a reference for offices that want a certifiable management system |
| [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | A voluntary framework for managing AI risk | Chapter 8 |

## Corrections

Standards change versions and URLs. If a reference here is wrong or out of
date, open an [issue](https://github.com/worldbank/ai4data/issues) with the
row and the corrected source.
