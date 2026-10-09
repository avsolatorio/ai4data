---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to AI-Ready Microdata Documentation.
---

# Glossary

**Access tier.** A class of release conditions: public use file, licensed file, or secure access, each with its own disclosure control and agreement.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Concept map.** A table that links each variable to the concept it measures and the classification it follows, with version, level, and URI.

**Data appraisal.** The DDI field for known quality issues of a study: coverage gaps, non-response, comparability limits.

**Data dictionary.** The documentation of a data file's variables: names, labels, types, value codes, missing codes, universes, questions.

**DCAT.** The web vocabulary for describing data catalogs and their datasets so that catalogs can be harvested.

**DDI Codebook.** The Data Documentation Initiative standard for documenting a study and its variables; the World Bank microdata schema is its JSON form.

**Derived variable.** A variable computed from others by a stated rule, for example labour force status from the employment questions.

**Disclosure control.** The steps that reduce the risk of identifying a person or business in released microdata: removal of identifiers, top-coding, coarsening of geography, suppression.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or a version of it and makes citations countable.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Fine-tuning.** Training an existing model further on the organization's own labelled data so that it follows the organization's conventions.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**Harmonized name.** A variable name used for the same concept across surveys, tied to one definition.

**ISCED.** The International Standard Classification of Education, whose levels classify educational attainment.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Known-variable question set.** Questions the way users ask for variables, each with the variable that should come first, used to score variable search.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Missing-value code.** A code in a variable that marks a missing, inapplicable, refused, or unknown response, which has to be labelled so that it is never read as a value.

**Model Context Protocol (MCP).** An open standard through which AI assistants discover and call an organization's tools and read its resources.

**MRR.** Mean reciprocal rank: 1 when the right item comes first, one half when it comes second, and so on, averaged over the questions.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Open-weight model.** A model whose weights can be downloaded and run on the organization's own infrastructure, under its published licence.

**OpenAPI.** The standard way to describe an API in a file: which addresses exist, what parameters they take, and what they return.

**Precision and recall.** Precision is the share of what a system found that is right; recall is the share of what is there that the system found. F1 is the average that combines the two.

**Provenance.** The record of where a value came from: the series, the release, the document and page, the method, and the version, carried with the value.

**Public use file.** A microdata file released to anyone under terms of use, after disclosure control.

**Question bank.** A repository of question wordings keyed by the concept each measures and the surveys that use each.

**Recall@k.** The share of questions for which the right item appears in the first k results.

**schema.org.** A vocabulary that search engines agree on for describing things on web pages; a Dataset record in it, embedded in a page, is how dataset search engines learn what the page is about.

**SKOS and XKOS.** SKOS is the web standard for publishing controlled vocabularies (concepts with labels and identifiers); XKOS extends it for statistical classifications and their correspondences.

**Study record.** The study-level documentation of a survey: title, abstract, dates, coverage, universe, design, access conditions.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Universe.** The population a variable applies to, as the question was asked.

**Variable group.** A DDI structure that organizes variables by theme for navigation.

**Weight.** A variable that scales each record to the population, with the calibration and the estimates it applies to documented.
