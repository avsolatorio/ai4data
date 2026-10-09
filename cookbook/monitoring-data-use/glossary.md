---
id: glossary
title: Glossary
sidebar_position: 91
description: Terms used in the Practical Guide to Monitoring Data Use.
---

# Glossary

**AI visibility check.** A fixed set of questions asked of AI assistants on a schedule, recording whether the organization is cited and whether the figures are right.

**API.** Application programming interface: the address at which a program asks a system for data and receives it in a structured form.

**Canonical name.** The one name, with its acronym and identifier, under which a dataset's mentions are counted.

**Checksum.** A fingerprint of a file's content (here SHA-256) that changes if one byte of the file changes; used to prove that a file is as released.

**Co-use.** The use of another organization's dataset in the same document as the organization's own.

**Confidence.** A number between 0 and 1 that a model returns with its output to say how sure it is; the basis of a threshold rule.

**Coverage.** The share of a reference list of known uses that the collected documents contain.

**DOI.** Digital object identifier, a persistent identifier registered with DataCite that resolves to a dataset or a version of it and makes citations countable.

**Embedding.** A list of numbers that a model assigns to a text so that texts with similar meaning have similar numbers; the basis of search by meaning.

**Fine-tuning.** Training an existing model further on the organization's own labelled data so that it follows the organization's conventions.

**Gate.** The rule that decides whether a change ships, applied to a comparison of the evaluation suite before and after the change.

**Harmonization.** The matching of mention variants to canonical identifiers.

**ISCO-08.** The International Standard Classification of Occupations, whose four-digit codes classify jobs; the label of the coding examples in this guide.

**Labelled sample.** Sentences labelled by a person with the dataset mention they contain, or none, used to measure the extractor.

**Language model.** A model that reads and writes text; used in these guides to draft, flag, code, explain, and translate, with a person deciding.

**Match type.** How a mention was matched: exact, phrase (contains), fuzzy, semantic, or none.

**Mention.** A reference to a dataset in a document, named or unnamed.

**Named-entity extraction.** A model that finds spans of text referring to entities of a given type, here dataset mentions, with a confidence.

**OCR.** Optical character recognition: reading text from the image of a page, which is what a scanned document needs before anything can be extracted from it.

**Precision.** The share of extracted mentions that are correct.

**Recall.** The share of labelled mentions that the extractor found.

**Reference list.** Known uses of the data, assembled from staff knowledge and citation tracking, against which coverage is measured.

**SKOS and XKOS.** SKOS is the web standard for publishing controlled vocabularies (concepts with labels and identifiers); XKOS extends it for statistical classifications and their correspondences.

**Threshold.** The value of a score (a confidence, a similarity, a garble rate) at which the rule changes what happens to an item; set from measurements.

**Token.** The unit in which language models read and write text, roughly three quarters of a word; model prices and context limits are counted in tokens.

**Typology of use.** The distinction between mention and use, and among primary, secondary, and background use.

**Use.** A document whose analysis, decision, or statement depends on the data, as opposed to a mention that cites them as context.

**Variant.** A way of naming a dataset other than its canonical name: an acronym, an informal name, a translation, a misspelling.

**Zero-shot model.** A model that performs a task from a description of the labels alone, without examples of that task in its training.
